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
            default="logs/",
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
            default="./configs/brain-sr.yaml",
            help="Configs of yaml file",
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

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_parser()
    
    configs = OmegaConf.load(args.cfg_path)
    configs.train.use_text = args.use_text
    configs.cfg_path = args.cfg_path

    # merge args to config
    for key in vars(args):
        if key in ['cfg_path', 'save_dir', 'resume', 'run_name']:
            configs[key] = getattr(args, key)

    trainer = get_obj_from_str(configs.trainer.target)(configs)
    trainer.train()
