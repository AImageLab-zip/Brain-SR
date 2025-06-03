#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-10-26 20:20:36

import warnings
warnings.filterwarnings("ignore")

import argparse
from omegaconf import OmegaConf

from utils.util_common import get_obj_from_str
from utils.util_opts import str2bool

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
            "--save_dir",
            type=str,
            default="../work_data/logs/",
            help="Folder to save the checkpoints and training log",
            )
    parser.add_argument(
            "--resume",
            type=str,
            const=True,
            default=None,
            nargs="?",
            help="resume from the save_dir or checkpoint",
            )
    parser.add_argument(
            "--cfg_path",
            type=str,
            default="./configs/sd-turbo-sr-ldis.yaml",
            help="Configs of yaml file",
            )
    parser.add_argument(
            "--ldif",
            type=float,
            default=1.0,
            help="Loss coefficient for diffsuion in latent space",
            )
    parser.add_argument(
            "--llpips",
            type=float,
            default=0,
            help="Loss coefficient for latent lpips",
            )
    parser.add_argument(
            "--ldis",
            type=float,
            default=0,
            help="Loss coefficient for latent discriminator",
            )
    parser.add_argument(
            "--use_text",
            type=str2bool,
            default='False',
            help="Text Prompt",
            )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--remote-debug", action="store_true"
    )

    args = parser.parse_args()

    return args

def activate_remote_debug():

    import debugpy
    debugpy.listen(("0.0.0.0", 5678))  # Accept connections on all interfaces
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()

if __name__ == "__main__":
    args = get_parser()

    if args.remote_debug:
        print("Activating Remote Debugging!")
        activate_remote_debug()
        print("Remote Debugging Activated!")
    
    configs = OmegaConf.load(args.cfg_path)
    # Li setto dal config
    #if args.ldif > 0:
    #    configs.train.loss_coef.ldif = args.ldif
    #if args.ldis > 0:
    #    configs.train.loss_coef.ldis = args.ldis
    #if args.llpips > 0:
    #    configs.train.loss_coef.llpips = args.llpips
    configs.train.use_text = args.use_text
    configs.cfg_path = args.cfg_path

    # merge args to config
    for key in vars(args):
        if key in ['cfg_path', 'save_dir', 'resume', 'run_name']:
            configs[key] = getattr(args, key)

    trainer = get_obj_from_str(configs.trainer.target)(configs)
    trainer.train()
