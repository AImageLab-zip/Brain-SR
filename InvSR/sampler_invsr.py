#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-13 16:59:27

import os, sys, math, random

import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from utils import util_net
from utils import util_image
from utils import util_common
from utils import util_color_fix
import torchvision.utils as vutils
from datapipe.datasets import get_transforms

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from datapipe.datasets import create_dataset
from diffusers import StableDiffusionInvEnhancePipeline, AutoencoderKL

#_positive= 'Cinematic, high-contrast, photo-realistic, 8k, ultra HD, ' +\
#           'meticulous detailing, hyper sharpness, perfect without deformations'
_positive= "High-resolution histological brain tissue, accurate cellular structures, sharp cortical layers, " \
            "precise anatomical detail, realistic staining patterns, microscopy-grade clarity, " \
            "no artifacts, no hallucinated features, no stylization"
_negative= 'Low quality, blurring, jpeg artifacts, deformed, over-smooth, cartoon, noisy,' +\
           'painting, drawing, sketch, oil painting'

class BaseSampler:
    def __init__(self, configs, only_encode=False):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
                configs.sampler_config.{start_timesteps, padding_mod, seed, sf, num_sample_steps}
            seed: int, random seed
        '''
        self.configs = configs

        self.setup_seed()

        self.build_model(only_encode=only_encode)

    def setup_seed(self, seed=None):
        seed = self.configs.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def write_log(self, log_str):
        print(log_str, flush=True)

    def build_model(self, only_encode: bool):
        # Build Stable diffusion
        params = dict(self.configs.sd_pipe.params)
        torch_dtype = params.pop('torch_dtype')
        params['torch_dtype'] = get_torch_dtype(torch_dtype)
        base_pipe = util_common.get_obj_from_str(self.configs.sd_pipe.target).from_pretrained(**params)

        # skippato
        if self.configs.get('scheduler', None) is not None:
            pipe_id = self.configs.scheduler.target.split('.')[-1]
            self.write_log(f'Loading scheduler of {pipe_id}...')
            base_pipe.scheduler = util_common.get_obj_from_str(self.configs.scheduler.target).from_config(
                base_pipe.scheduler.config
            )
            self.write_log('Loaded Done')
        # skippato
        if self.configs.get('vae_fp16', None) is not None:
            params_vae = dict(self.configs.vae_fp16.params)
            torch_dtype = params_vae.pop('torch_dtype')
            params_vae['torch_dtype'] = get_torch_dtype(torch_dtype)
            pipe_id = self.configs.vae_fp16.params.pretrained_model_name_or_path
            self.write_log(f'Loading improved vae from {pipe_id}...')
            base_pipe.vae = util_common.get_obj_from_str(self.configs.vae_fp16.target).from_pretrained(
                **params_vae,
            )
            self.write_log('Loaded Done')
        if self.configs.base_model in ['sd-turbo', 'sd2base'] :
            # Crea questa pipeline che e' quella per il img-to-img (lo standard era text-to-img)
            sd_pipe = StableDiffusionInvEnhancePipeline.from_pipe(base_pipe)
        else:
            raise ValueError(f"Unsupported base model: {self.configs.base_model}!")
        sd_pipe.to(f"cuda")
        if self.configs.sliced_vae:
            # Slicing e' tipo patch, ma il dato viene suddiviso e processato 1(o+) canali per volta
            sd_pipe.vae.enable_slicing()
        if self.configs.tiled_vae:
            # Divisioni in patch nel latent_space? non so dove venga utilizzato
            sd_pipe.vae.enable_tiling()
            sd_pipe.vae.tile_latent_min_size = self.configs.latent_tiled_size
            sd_pipe.vae.tile_sample_min_size = self.configs.sample_tiled_size
        if self.configs.gradient_checkpointing_vae:
            # Ottimizzazioni per la memoria
            self.write_log(f"Activating gradient checkpoing for vae...")
            sd_pipe.vae._set_gradient_checkpointing(sd_pipe.vae.encoder, True)
            sd_pipe.vae._set_gradient_checkpointing(sd_pipe.vae.decoder, True)

        if not only_encode:
            model_configs = self.configs.model_start
            params = model_configs.get('params', dict)
            # Carica il NoisePrediction con i relativi parametri
            model_start = util_common.get_obj_from_str(model_configs.target)(**params)
            model_start.cuda()
            # prende i pesi di noise-predictor-sd-turbo-v5
            ckpt_path = model_configs.get('ckpt_path')
            assert ckpt_path is not None
            self.write_log(f"Loading started model from {ckpt_path}...")
            state = torch.load(ckpt_path, map_location=f"cuda")
            if 'state_dict' in state:
                state = state['state_dict']
            # inserisce i pesi dentro al NoisePredictor
            util_net.reload_model(model_start, state)
            self.write_log(f"Loading Done")
            model_start.eval()
            # Inserisce il NoisePredictor dentro la pipeline generale
            setattr(sd_pipe, 'start_noise_predictor', model_start)

        self.sd_pipe = sd_pipe

class InvSamplerSR(BaseSampler):
    @torch.no_grad()
    def sample_func(self, im_cond):
        # Data l'immagine come tensore, effettua tutte le operazioni e ritorna l'upscalata

        '''
        Input:
            im_cond: b x c x h x w, torch tensor, [0,1], RGB
        Output:
            xt: h x w x c, numpy array, [0,1], RGB
        '''

        # FIXME: Cosa sono questi parametri positive e negative?
        if self.configs.cfg_scale > 1.0:
            negative_prompt = [_negative,]*im_cond.shape[0]
        else:
            negative_prompt = None

        # Calcolo dimensioni dell'HQ da ricostruire
        ori_h_lq, ori_w_lq = im_cond.shape[-2:]
        ori_w_hq = ori_w_lq * self.configs.basesr.sf # SF: scaling factor
        ori_h_hq = ori_h_lq * self.configs.basesr.sf
        vae_sf = (2 ** (len(self.sd_pipe.vae.config.block_out_channels) - 1)) # fattore downscale dato dal laten_space (8x)
        if hasattr(self.sd_pipe, 'unet'):
            diffusion_sf = (2 ** (len(self.sd_pipe.unet.config.block_out_channels) - 1)) # Altro fattore di scale dato dalla unet nel processare il dato nel latent space (8x)
        else:
            diffusion_sf = self.sd_pipe.transformer.patch_size

        mod_lq = vae_sf // self.configs.basesr.sf * diffusion_sf 
        # calcolo quanti pixel della LQ corrispondono a uno nel diffusion_latent (1 pixel latent_diff = 16x16 nella LQ)
        # prima 1 pixel LQ a quanti corrisponde nel latent_vae (che pero' si trova in HQ), e poi moltiplico ulteriorimente per lo scaling del diffusion 
        
        idle_pch_size = self.configs.basesr.chopping.pch_size # patch size impostata in cui verra' suddivisa l'immagine
        
        if min(im_cond.shape[-2:]) >= idle_pch_size:
            pad_h_up = pad_w_left = 0
        else:
            while min(im_cond.shape[-2:]) < idle_pch_size:
                pad_h_up = max(min((idle_pch_size - im_cond.shape[-2]) // 2, im_cond.shape[-2]-1), 0)
                pad_h_down = max(min(idle_pch_size - im_cond.shape[-2] - pad_h_up, im_cond.shape[-2]-1), 0)
                pad_w_left = max(min((idle_pch_size - im_cond.shape[-1]) // 2, im_cond.shape[-1]-1), 0)
                pad_w_right = max(min(idle_pch_size - im_cond.shape[-1] - pad_w_left, im_cond.shape[-1]-1), 0)
                im_cond = F.pad(im_cond, pad=(pad_w_left, pad_w_right, pad_h_up, pad_h_down), mode='reflect')
        
        if im_cond.shape[-2] == idle_pch_size and im_cond.shape[-1] == idle_pch_size:
            # Se l'immagine e' esattamente della patch size desiderata
            target_size = (
                im_cond.shape[-2] * self.configs.basesr.sf,
                im_cond.shape[-1] * self.configs.basesr.sf
            )
            res_sr = self.sd_pipe(
                image=im_cond.type(torch.float16),
                prompt=[_positive, ]*im_cond.shape[0],
                negative_prompt=negative_prompt,
                target_size=target_size,
                timesteps=self.configs.timesteps,
                guidance_scale=self.configs.cfg_scale,
                output_type="pt",    # torch tensor, b x c x h x w, [0, 1]
            ).images
        else: # Altrimenti spezzala in patch processabili
            if not (im_cond.shape[-2] % mod_lq == 0 and im_cond.shape[-1] % mod_lq == 0): # fa il pad della LQ per essere un multiplo della mod_lq
                target_h_lq = math.ceil(im_cond.shape[-2] / mod_lq) * mod_lq
                target_w_lq = math.ceil(im_cond.shape[-1] / mod_lq) * mod_lq
                pad_h = target_h_lq - im_cond.shape[-2]
                pad_w = target_w_lq - im_cond.shape[-1]
                im_cond= F.pad(im_cond, pad=(0, pad_w, 0, pad_h), mode='reflect')

            im_spliter = util_image.ImageSpliterTh(
                im_cond,
                pch_size=idle_pch_size,
                stride= int(idle_pch_size * 0.50), # fa uno stride di mezza patch cosi da avere overlap ed evitare border artifacts
                sf=self.configs.basesr.sf,
                weight_type=self.configs.basesr.chopping.weight_type,
                extra_bs=self.configs.basesr.chopping.extra_bs,
            )
            for patch_idx, (im_lq_pch, index_infos) in enumerate(im_spliter):
                print(f"Processing patches {patch_idx}-{patch_idx+self.configs.basesr.chopping.extra_bs} of {len(im_spliter)}", flush=True)
                target_size = (
                    im_lq_pch.shape[-2] * self.configs.basesr.sf,
                    im_lq_pch.shape[-1] * self.configs.basesr.sf,
                )

                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()

                # gli passo una batch di patch e lui ritorna gli upscalati
                res_sr_pch = self.sd_pipe(
                    image=im_lq_pch.type(torch.float16),
                    prompt=[_positive, ]*im_lq_pch.shape[0],
                    negative_prompt=negative_prompt,
                    target_size=target_size,
                    timesteps=self.configs.timesteps,
                    guidance_scale=self.configs.cfg_scale, # l'importanza di seguire i prompt (1 default, seguirli fortemente)
                    output_type="pt",    # torch tensor, b x c x h x w, [0, 1]
                ).images

                # end.record()
                # torch.cuda.synchronize()
                # print(f"Time: {start.elapsed_time(end):.6f}")
                
                im_spliter.update(res_sr_pch, index_infos) # salva le patch HQ nello splitter per poi combinarle
            res_sr = im_spliter.gather() # ritorna l'intera immagine HQ 

        pad_h_up *= self.configs.basesr.sf
        pad_w_left *= self.configs.basesr.sf
        res_sr = res_sr[:, :, pad_h_up:ori_h_hq+pad_h_up, pad_w_left:ori_w_hq+pad_w_left] # rimuovo il pad aggiunto prima

        if self.configs.color_fix:
            im_cond_up = F.interpolate(
                im_cond, size=res_sr.shape[-2:], mode='bicubic', align_corners=False, antialias=True
            )
            if self.configs.color_fix == 'ycbcr':
                res_sr = util_color_fix.ycbcr_color_replace(res_sr, im_cond_up)
            elif self.configs.color_fix == 'wavelet':
                res_sr = util_color_fix.wavelet_reconstruction(res_sr, im_cond_up)
            else:
                raise ValueError(f"Unsupported color fixing type: {self.configs.color_fix}")

        res_sr = res_sr.clamp(0.0, 1.0).cpu().permute(0,2,3,1).float().numpy()

        return res_sr

    def inference(self, in_path, out_path, bs=1):
        '''
        Inference demo.
        Input:
            in_path: str, folder or image path for LQ image
            out_path: str, folder save the results
            bs: int, default bs=1, bs % num_gpus == 0
        '''

        in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
        out_path = Path(out_path) if not isinstance(out_path, Path) else out_path

        if not out_path.exists():
            out_path.mkdir(parents=True)

        if in_path.is_dir():
            data_config = {'type': 'base',
                           'params': {'dir_path': str(in_path),
                                      'transform_type': 'default',
                                      'transform_kwargs': {
                                          'mean': 0.0,
                                          'std': 1.0,
                                          },
                                      'need_path': True,
                                      'recursive': False,
                                      'length': None,
                                      }
                           }
            dataset = create_dataset(data_config)
            self.write_log(f'Find {len(dataset)} images in {in_path}')
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=bs, shuffle=False, drop_last=False,
            )
            for data in tqdm(dataloader):
                res = self.sample_func(data['lq'].cuda())

                for jj in range(res.shape[0]):
                    im_name = Path(data['path'][jj]).stem
                    save_path = str(out_path / f"{im_name}.png")
                    util_image.imwrite(res[jj], save_path, dtype_in='float32')
        else:
            # Carica l'immagine e la trasforma in un tensore
            im_cond = util_image.imread(in_path, chn='rgb', dtype='float32')  # h x w x c
            im_cond = util_image.img2tensor(im_cond).cuda()                   # 1 x c x h x w

            # Qua dentro fa tutto
            image = self.sample_func(im_cond).squeeze(0)

            # Salva
            save_path = str(out_path / f"{in_path.stem}.png")
            util_image.imwrite(image, save_path, dtype_in='float32')

        self.write_log(f"Processing done, enjoy the results in {str(out_path)}")


    def encode_decode(self, in_path, out_path, bs=1):
        '''
        Inference demo.
        Input:
            in_path: str, folder or image path for LQ image
            out_path: str, folder save the results
            bs: int, default bs=1, bs % num_gpus == 0
        '''

        in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
        out_path = Path(out_path) if not isinstance(out_path, Path) else out_path

        if not out_path.exists():
            out_path.mkdir(parents=True)

        if in_path.is_dir():

            files = os.listdir(in_path)
            files = [f for f in files if f.endswith('.png')]

            for f in files:

                imgpath = os.path.join(in_path, f)
                im_base = util_image.imread(imgpath, chn='rgb', dtype='float32')
                t = get_transforms("default", {'mean':0.0, 'std':1.0})
                im_base = t(im_base)

                # aggiungo la dimensione batch
                im_base = im_base.unsqueeze(0)  # 1 x c x h x w
                im_base = im_base.cuda().to(dtype=torch.float32)

                im_latent = self.encode_first_stage(im_base.detach())
                x0_recon = self.decode_first_stage(im_latent).detach()
                #x0_recon = im_base

                # Save as output        
                x0_recon_norm = (x0_recon - x0_recon.min()) / (x0_recon.max() - x0_recon.min() + 1e-8)
                im_np = x0_recon_norm.squeeze(0).cpu().permute(1,2,0).numpy()

                im_name = Path(f).stem
                save_path = str(Path(out_path) / f"{im_name}.png")
                util_image.imwrite(im_np, save_path, dtype_in='float32')

                #log
                self.write_log(f"Processed {imgpath} -> {save_path}")

        #else:
        #    # Carica l'immagine e la trasforma in un tensore
        #    im_cond = util_image.imread(in_path, chn='rgb', dtype='float32')  # h x w x c
        #    im_cond = util_image.img2tensor(im_cond).cuda()                   # 1 x c x h x w

        #    # Qua dentro fa tutto
        #    image = self.sample_func(im_cond).squeeze(0)

        #    # Salva
        #    save_path = str(out_path / f"{in_path.stem}.png")
        #    util_image.imwrite(image, save_path, dtype_in='float32')

        self.write_log(f"Processing done, enjoy the results in {str(out_path)}")

    @torch.amp.autocast('cuda')
    def encode_first_stage(self, x, deterministic=False, center_input_sample=True):
        if center_input_sample:
            x = x * 2.0 - 1.0
        latents_mean = latents_std = None
        if hasattr(self.sd_pipe.vae.config, "latents_mean") and self.sd_pipe.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(self.sd_pipe.vae.config.latents_mean).view(1, -1, 1, 1)
        if hasattr(self.sd_pipe.vae.config, "latents_std") and self.sd_pipe.vae.config.latents_std is not None:
            latents_std = torch.tensor(self.sd_pipe.vae.config.latents_std).view(1, -1, 1, 1)

        if deterministic:
            partial_encode = lambda xx: self.sd_pipe.vae.encode(xx).latent_dist.mode()
        else:
            partial_encode = lambda xx: self.sd_pipe.vae.encode(xx).latent_dist.sample()

        trunk_size = 8 #self.configs.sd_pipe.vae_split
        if trunk_size < x.shape[0]:
            init_latents = torch.cat([partial_encode(xx) for xx in x.split(trunk_size, 0)], dim=0)
        else:
            init_latents = partial_encode(x)

        scaling_factor = self.sd_pipe.vae.config.scaling_factor
        if latents_mean is not None and latents_std is not None:
            latents_mean = latents_mean.to(device=x.device, dtype=x.dtype)
            latents_std = latents_std.to(device=x.device, dtype=x.dtype)
            init_latents = (init_latents - latents_mean) * scaling_factor / latents_std
        else:
            init_latents = init_latents * scaling_factor

        return init_latents

    @torch.amp.autocast('cuda')
    def decode_first_stage(self, z, clamp=True):
        z = z / self.sd_pipe.vae.config.scaling_factor

        trunk_size = 1
        if trunk_size < z.shape[0]:
            out = torch.cat(
                [self.sd_pipe.vae.decode(xx).sample for xx in z.split(trunk_size, 0)], dim=0,
            )
        else:
            out = self.sd_pipe.vae.decode(z).sample
        if clamp:
            out = out.clamp(-1.0, 1.0)
        return out

def get_torch_dtype(torch_dtype: str):
    if torch_dtype == 'torch.float16':
        return torch.float16
    elif torch_dtype == 'torch.bfloat16':
        return torch.bfloat16
    elif torch_dtype == 'torch.float32':
        return torch.float32
    else:
        raise ValueError(f'Unexpected torch dtype:{torch_dtype}')

if __name__ == '__main__':
    pass

