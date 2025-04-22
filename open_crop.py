import nibabel as nib
from nibabel.affines import apply_affine
import tifffile as tiff
import json
from io import BytesIO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def open_low_res(img_path: str) -> (np.ndarray, np.ndarray):
    proxy_images = nib.load(img_path)
    images_array = np.array(proxy_images.dataobj)

    low_res_affine = proxy_images.affine

    return images_array, low_res_affine


def open_high_res(img_path: str, alignment_matrix_path: str) -> (tiff.TiffPage, np.ndarray):
    with tiff.TiffFile(img_path) as tif:
        high_res_image_page = tif.pages[0]

    high_res_array = high_res_image_page#high_res_image_page.asarray()
    # Do not flip the image here since the computed indexes are in the original orientation

    transformation_matrix = np.array(json.load(open(alignment_matrix_path)))

    return high_res_array, transformation_matrix


def compute_y_start(high_res_affine: np.ndarray, low_res_affine: np.ndarray) -> int:
    # Dalla matrice high-res prendo il suo valore di y_start (in 1um)
    # e con la matrice di low-res trovo il valore corrispettivo in 20um

    y_start = high_res_affine[1, 3]
    y_start_20um = apply_affine(np.linalg.inv(low_res_affine), [1, y_start, 1])[1]
    y_start_index = int(round(y_start_20um))

    return y_start_index

def select_region_of_interest(y_start_index: int) -> dict:
    # TODO: provare con una y a caso e vedere se si allinea comunque

    # Nel codice originale venivano usati i punti della segmentazione
    # per definire dei bordi. In questo caso non ce li ho e li definisco io

    # hippo_points = hippo_mask[hippo_mask[:, 1] == y_start_index]
    # min_x, min_y = hippo_points[:, 0].min(), hippo_points[:, 2].min()
    # max_x, max_y = hippo_points[:, 0].max(), hippo_points[:, 2].max()

    #min_x, max_x = 3500, 3600
    #min_z, max_z = 3700, 3800
    min_x, max_x = 0, 4000
    min_z, max_z = 1000, 2000

    # FIXME: Perche definire 4 corner quando ne bastano 2?
    corners = {
        'A': [min_x, y_start_index, min_z],
        #'B': [max_x, y_start_index, min_z],
        #'C': [min_x, y_start_index, max_z],
        'D': [max_x, y_start_index, max_z]
    }

    return corners


def transform_corners(corners: dict, low_res_affine: np.ndarray, high_res_affine: np.ndarray) -> dict:
    # Trasformo i corner in high-res
    corners_global = {key: apply_affine(low_res_affine, value) for key, value in corners.items()}
    corners_1um = {key: apply_affine(np.linalg.inv(high_res_affine), value).astype(int) for key, value in corners_global.items()}

    print(f"Original corners: MinX: {corners['A'][0]}, MaxX: {corners['D'][0]}, MinZ: {corners['A'][2]}, MaxZ: {corners['D'][2]}")
    print(f"Global corners: MinX: {corners_global['A'][0]}, MaxX: {corners_global['D'][0]}, MinZ: {corners_global['A'][2]}, MaxZ: {corners_global['D'][2]}")
    print(f"Transformed corners: MinX: {corners_1um['A'][0]}, MaxX: {corners_1um['D'][0]}, MinZ: {corners_1um['A'][2]}, MaxZ: {corners_1um['D'][2]}")

    return corners_1um

def crop_low_res(low_res_img: np.ndarray, corners: dict) -> np.ndarray:
    # Crop low-res image and flip it

    min_x, max_x = corners['A'][0], corners['D'][0]
    min_z, max_z = corners['A'][2], corners['D'][2]

    print(f"Low res crop coordinates: {min_x}, {max_x}, {min_z}, {max_z}")

    low_res_crop = low_res_img[min_z:max_z, 0, min_x:max_x]

    return low_res_crop


def crop_high_res(high_res_img: np.ndarray, corners: dict) -> np.ndarray:

    min_x, max_x = corners['A'][0], corners['D'][0]
    min_z, max_z = corners['A'][2], corners['D'][2]

    print(f"High res crop coordinates: {min_x}, {max_x}, {min_z}, {max_z}")

    # TODO: gli indici ritornati dalla matrice di trasformazione per z
    #       hanno il min al posto del max -> capire come mai
    #       e' perche e' flippato? (pero' i valori sono corretti)

    high_res_img = high_res_img.asarray()

    # Flip the image
    high_res_flip = np.flip(high_res_img, axis=0)

    # Crop high-res image and flip it
    img_z_len = high_res_flip.shape[0]
    max_z = img_z_len - max_z
    min_z = img_z_len - min_z

    high_res_crop = high_res_flip[min_x:max_x, min_z:max_z]

    return high_res_crop


if __name__ == "__main__":

    image_id = "2956"

    low_res_path = f'/work/bolelli_synthetic/example_data/original/pm{image_id}o.mnc'
    high_res_path = f'/work/bolelli_synthetic/example_data/high-res/aligned/B20_{image_id}.tif'
    high_res_affine_path = f'/work/bolelli_synthetic/example_data/high-res/aligned/B20_{image_id}_affine.json'

    low_res_img, low_res_affine = open_low_res(low_res_path)
    high_res_img_page, high_res_affine = open_high_res(high_res_path, high_res_affine_path)

    # Estraggo il valore di y dell'immagine high-res
    # avendo gia' la coppia high/low il valore di y non serve (forse)
    y_start_index_20um = compute_y_start(high_res_affine, low_res_affine)

    corners_20um = select_region_of_interest(y_start_index_20um)
    corners_1um = transform_corners(corners_20um, low_res_affine, high_res_affine)

    # Crop low-res image
    low_res_crop = crop_low_res(low_res_img, corners_20um)

    # Load and crop high-res image
    high_res_crop = crop_high_res(high_res_img_page, corners_1um)

    # Visualize the results
    plt.imshow(low_res_crop, origin="lower")
    plt.title('Low Res Crop')
    plt.show()

    high_res_crop = high_res_crop[::8, ::8]
    plt.imshow(high_res_crop, origin="lower")
    plt.title('High Res Crop')
    plt.show()
