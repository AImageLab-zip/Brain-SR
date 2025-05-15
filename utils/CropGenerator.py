import json
import numpy as np
import nibabel as nib
import os
import tifffile as tiff

from dataclasses import dataclass
from nibabel.affines import apply_affine
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from tqdm import tqdm
from typing import Optional


@dataclass
class CornerCoords:
    A: list
    B: list

@dataclass
class CropData:
    lr_data: np.ndarray
    lr_coords: CornerCoords
    hr_cords: CornerCoords
    hr_data: Optional[np.ndarray] = None

@dataclass
class CropConfig:
    lr_crop_size: int
    hr_crop_size: int
    stride: int
    white_threshold: float
    downsampled_sigma: float


class SingleImageCropGenerator():
    def __init__(self, lr_path, hr_affine_path, hr_path, out_path, crop_config: CropConfig):
        # Save paths
        self.lr_path = lr_path
        self.hr_affine_path = hr_affine_path
        self.hr_path = hr_path
        self.out_path = out_path

        self.image_idx = hr_path[-8:-4]

        self.crop_config = crop_config

        # Variables to save
        self.lr_image = None
        self.hr_image = None
        self.lr_affine = None
        self.hr_affine_inv = None
    
        # Intermediate variables
        self.crop_data = []

    def load_lr_image(self):
        print("Loading low-res image")

        proxy_images = nib.load(self.lr_path)
        images_array = np.array(proxy_images.dataobj)

        low_res_affine = proxy_images.affine

        self.lr_image = images_array
        self.lr_affine = low_res_affine

    def load_hr_image(self):
        print("Loading high-res image")

        with tiff.TiffFile(self.hr_path) as tif:
            high_res_image_page = tif.pages[0]

        high_res_array = high_res_image_page.asarray()
        high_res_array = np.flip(high_res_array, axis=0)
        
        self.hr_image = high_res_array
    
    def load_hr_affine(self):
        print("Loading high-res affine matrix")
        hr_affine =  np.array(json.load(open(self.hr_affine_path)))

        # only the inverse will be used (compute only once)
        self.hr_affine_inv = np.linalg.inv(hr_affine)
    
    def transform_coord(self, lr_coord: list) -> dict:
        # Using the affine matrixes, transform the coordinates from low-res to high-res

        global_coord = apply_affine(self.lr_affine, lr_coord)
        hr_coord = apply_affine(self.hr_affine_inv, global_coord).astype(int)

        return hr_coord
    
    def adjust_hr_coords(self, hr_coords: CornerCoords) -> CornerCoords:
        # Since the transformation matrix have a strange behavior, we need to adjust the coordinates

        img_shape_z = self.hr_image.shape[0]

        # Le coordinate di X e Z risultano invertite (qua tenute uguali ma quanado apro l'immagine li scambio)
        # Inoltre, un asse e' stato flippato, l'altro devo solo invertire le coordinate

        new_corner_cords = CornerCoords(
            A=[hr_coords.A[0], 0, img_shape_z - hr_coords.A[2]],
            B=[hr_coords.B[0], 0, img_shape_z - hr_coords.B[2]]
        )

        return new_corner_cords

    
    def generate_crop_coords(self):
        # Given the dimension of the low-res image, generate all the possible crop coordinates
        # that fit within the image dimensions

        lr_shape = self.lr_image.shape
        stride = self.crop_config.stride
        crop_size = self.crop_config.lr_crop_size

        lr_crop_coords = []
       
        for z in range(0, lr_shape[0] - crop_size + 1, stride):
            for x in range(0, lr_shape[2] - crop_size + 1, stride):
                
                cords = CornerCoords(
                    A=[x, 0, z],
                    B=[x + crop_size, 0, z + crop_size]
                )
                lr_crop_coords.append(cords)

        return lr_crop_coords
    
    def filter_crops(self, lr_crops_list):
        # Given the image data and the crops, filter out crops that are almost entirely white

        print("Loading low-res data and filtering crops")
        for lr_crop_coord in tqdm(lr_crops_list):

            crop = self.lr_image[lr_crop_coord.A[2]:lr_crop_coord.B[2], # z
                                 0,                                     # y (always 0)
                                 lr_crop_coord.A[0]:lr_crop_coord.B[0]] # x
            
            white_pixels = np.sum(crop == 65535)
            if white_pixels / (self.crop_config.lr_crop_size ** 2) > self.crop_config.white_threshold:
                continue
            
            hr_crop_coord = CornerCoords(
                A=self.transform_coord(lr_crop_coord.A),
                B=self.transform_coord(lr_crop_coord.B)
            )

            crop = CropData(
                lr_data=crop,
                lr_coords=lr_crop_coord,
                hr_cords=hr_crop_coord
            )
            self.crop_data.append(crop)

    def downsample_and_filter(self, hr_crop: np.ndarray) -> np.ndarray:

        # Apply Gaussian filter to smooth (and remove high frequency details)
        hr_crop_blurred = gaussian_filter(hr_crop, sigma=self.crop_config.downsampled_sigma)

        # Downsample to correct high-res size
        downsampled_crop = resize(hr_crop_blurred, 
                                  output_shape=(self.crop_config.hr_crop_size, self.crop_config.hr_crop_size), 
                                  anti_aliasing=False)

        return downsampled_crop

    def insert_hr_crop(self):
        # Load the HR data for each crop, then resize it and insert it in the crop data

        print("Loading high-res data and inserting crops")
        for crop in tqdm(self.crop_data):

            hr_coords = self.adjust_hr_coords(crop.hr_cords)

            hr_crop = self.hr_image[hr_coords.A[0]:hr_coords.B[0], # x
                                    hr_coords.A[2]:hr_coords.B[2]] # z
            
            crop.hr_data = self.downsample_and_filter(hr_crop)


    @staticmethod
    def normalize_to_save(img: np.ndarray, global_min: int, global_max: int) -> np.ndarray:
        img = np.clip(img, global_min, global_max)
        img_norm = (img - global_min) / (global_max - global_min)
        img_8bit = (img_norm * 255).astype(np.uint8)
        return img_8bit

        
    def save_crops(self):

        lr_folder = os.path.join(self.out_path, "low")
        hr_folder = os.path.join(self.out_path, "high")

        if not os.path.exists(lr_folder):
            os.makedirs(lr_folder)
        if not os.path.exists(hr_folder):
            os.makedirs(hr_folder)

        print("Saving crops")
        for idx, crop in tqdm(enumerate(self.crop_data)):
            filename = f"{self.image_idx}_{idx}.png"

            lr = np.flip(crop.lr_data.squeeze(), axis=0)
            lr_8bit = self.normalize_to_save(lr, global_min=0, global_max=65535)
            lr_rgb = np.stack([lr_8bit]*3, axis=-1)
            Image.fromarray(lr_rgb, mode='RGB').save(os.path.join(lr_folder, filename))

            hr = np.flip(crop.hr_data.squeeze(), axis=0)
            hr_8bit = self.normalize_to_save(hr, global_min=0, global_max=1)
            hr_rgb = np.stack([hr_8bit]*3, axis=-1)
            Image.fromarray(hr_rgb, mode='RGB').save(os.path.join(hr_folder, filename))

        print(f"Saved {len(self.crop_data)} crops to {self.out_path}")

    def generate_crops(self):

        # Load images and affine matrices
        self.load_lr_image()
        self.load_hr_affine()
        self.load_hr_image()
        
        # Compute possibile crop coordinates and filter them
        lr_crops_list = self.generate_crop_coords()
        self.filter_crops(lr_crops_list)

        # Insert high-res crops
        self.insert_hr_crop()

        # Save crops
        self.save_crops()



class DatasetCropsGenerator():
    def __init__(self, lr_folder, hr_folder, out_folder, crop_config: CropConfig):
        self.lr_folder = lr_folder
        self.hr_folder = hr_folder
        self.out_folder = out_folder

        self.crop_config = crop_config
        self.data = []

    def compute_list(self, exclude_ids=[]):
        
        image_ids = [f[-9:-5] for f in os.listdir(self.lr_folder) if f.endswith('.mnc')]
        image_ids = [image_id for image_id in image_ids if image_id not in exclude_ids]

        lr_files = [os.path.join(self.lr_folder, f"pm{image_id}o.mnc") for image_id in image_ids]
        hr_files = [os.path.join(self.hr_folder, f"B20_{image_id}.tif") for image_id in image_ids]
        hr_affine_files = [os.path.join(self.hr_folder, f"B20_{image_id}_affine.json") for image_id in image_ids]

        assert all(os.path.exists(f) for f in lr_files), "Some low-res files do not exist"
        assert all(os.path.exists(f) for f in hr_files), "Some high-res files do not exist"
        assert all(os.path.exists(f) for f in hr_affine_files), "Some high-res affine files do not exist"

        self.data = [{"lr": lr_file, "hr": hr_file, "hr_affine": hr_affine} 
                     for lr_file, hr_file, hr_affine in zip(lr_files, hr_files, hr_affine_files)]
    
    def random_select(self, n: int):

        selected = np.random.choice(len(self.data), size=n, replace=False)
        self.data = [self.data[i] for i in selected]

    def generate_crops(self):
        # Generate crops for each image in the dataset

        for image in self.data:
            lr_path = image["lr"]
            hr_affine_path = image["hr_affine"]
            hr_path = image["hr"]
            
            crop_generator = SingleImageCropGenerator(lr_path, hr_affine_path, hr_path, self.out_folder, self.crop_config)

            print("Generating crops for image:", lr_path)
            crop_generator.generate_crops()

            print("\n\n")

    

        

if __name__ == "__main__":

    validation_names = ['2251', '2447', '6899', '5048', '4449', '2803', '3305', '1901', '0199', '4950']

    lr_path = "/homes/gcasari/bigbrain/work_data/BigBrain/low_res_coronal_minc/"
    hr_path = "/homes/gcasari/bigbrain/work_data/BigBrain/high_res_aligned/"

    out_path = "/homes/gcasari/bigbrain/work_data/crops_datasets/train/"

    crop_config = CropConfig(
        lr_crop_size=128,
        hr_crop_size=512,
        stride=64,
        white_threshold=0.9,
        downsampled_sigma=2
    )

    crop_generator = DatasetCropsGenerator(lr_path, hr_path, out_path, crop_config)
    crop_generator.compute_list(exclude_ids=validation_names)
    crop_generator.random_select(90)
    crop_generator.generate_crops()

    print("Crops generation completed!")


    # # Example single usage
    # lr_path = "/homes/gcasari/bigbrain/work_data/example_data/original/pm2956o.mnc"
    # hr_affine_path = "/homes/gcasari/bigbrain/work_data/example_data/high-res/aligned/B20_2956_affine.json"
    # hr_path = "/homes/gcasari/bigbrain/work_data/example_data/high-res/aligned/B20_2956.tif"
    
    # out_path = "/homes/gcasari/bigbrain/crops/crop_test/"

    # crop_config = CropConfig(
    #     lr_crop_size=128,
    #     hr_crop_size=512,
    #     stride=64,
    #     white_threshold=0.9,
    #     downsampled_sigma=2
    # )

    # crop_generator = SingleImageCropGenerator(lr_path, hr_affine_path, hr_path, out_path, crop_config)
    # crop_generator.generate_crops()

