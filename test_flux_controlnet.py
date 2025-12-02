#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import functools
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    T5EncoderModel,
)

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from diffusers.models.controlnet_flux import FluxControlNetModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
# from utils_self.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling, free_memory
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


from safetensors.torch import load_file
import glob
from utils_self.wavelet_color_fix import wavelet_color_fix, adain_color_fix
from diffusers.utils.torch_utils import randn_tensor
from ram.models.ram_lora import ram
from ram import inference_ram as inference
tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

logger = get_logger(__name__)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    #####
    ## additional 
    #####

    parser.add_argument("--image_path", type=str, default='preset/test_inp')
    parser.add_argument("--output_dir", type=str, default='preset/test_oup_c_flux')
    parser.add_argument("--negative_prompt", type=str, default='')
    parser.add_argument("--save_prompts", action="store_true")
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--control_scale", type=float, default=1.0)
    parser.add_argument("--align_method", choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument('--ram_path', type=str, default="preset/models/ram_swin_large_14m.pth", help='Path to RAM model')
    parser.add_argument('--dape_path', type=str, default="preset/models/DAPE.pth", help='Path to DPAE')
    
    
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default='preset/models/c-flux/model.safetensors',
        # default='preset/models/dp2o-flux/model.safetensors',
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='/home/notebook/data/group/LowLevelLLM/models/diffusion_models/FLUX.1-dev',
        # default='black-forest-labs/FLUX.1-dev',
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--num_double_layers",
        type=int,
        default=4,
        help="Number of double layers in the controlnet (default: 4).",
    )
    parser.add_argument(
        "--num_single_layers",
        type=int,
        default=0,
        help="Number of single layers in the controlnet (default: 4).",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.5,
        help="the guidance scale used for transformer.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def load_pipeline(args, accelerator):
    # Load the tokenizers
    # load clip tokenizer
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    # load t5 tokenizer
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    # load clip text encoder
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    # load t5 text encoder
    text_encoder_two = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )

    flux_controlnet = FluxControlNetModel.from_transformer(
            flux_transformer,
            attention_head_dim=flux_transformer.config["attention_head_dim"],
            num_attention_heads=flux_transformer.config["num_attention_heads"],
            num_layers=args.num_double_layers,
            num_single_layers=args.num_single_layers,
        )
    state_dict = load_file(args.controlnet_model_name_or_path)
    flux_controlnet.load_state_dict(state_dict)
    
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    flux_controlnet_pipeline = FluxControlNetPipeline(
        scheduler=noise_scheduler,
        vae=vae,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
        transformer=flux_transformer,
        controlnet=flux_controlnet,
    )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            flux_transformer.enable_xformers_memory_efficient_attention()
            flux_controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    vae.requires_grad_(False).to(accelerator.device, dtype=torch.float32)
    flux_transformer.requires_grad_(False).to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.requires_grad_(False).to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.requires_grad_(False).to(accelerator.device, dtype=weight_dtype)
    flux_controlnet.requires_grad_(False).to(accelerator.device, dtype=weight_dtype)

    return flux_controlnet_pipeline

def get_validation_prompt(args, validation_image, model_vlm, device='cuda'):

    lq = tensor_transforms(validation_image).unsqueeze(0).to(device)
    lq = ram_transforms(lq)
    res = inference(lq, model_vlm)
    validation_prompt = f"{res[0]}"

    return validation_prompt

def main(args):

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    if args.seed is not None:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        generator = None


    path = args.controlnet_model_name_or_path

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'===> {args.output_dir}.')


    pipeline = load_pipeline(args, accelerator)


    model_vlm = ram(pretrained=args.ram_path,
                pretrained_condition=args.dape_path,
                image_size=384,
                vit='swin_l')
    model_vlm.eval()
    model_vlm.to('cuda')


    from pathlib import Path
    base_path = Path(args.image_path)


    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.webp']
    image_list = []
    
    for ext in image_extensions:
        image_list.extend(base_path.glob(ext))
        image_list.extend(base_path.glob(ext.upper()))
    
    image_list = sorted(image_list)
    
    print(f"Found {len(image_list)} images to process")


    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    for idx, image_path in enumerate(image_list):
        if idx % num_processes != process_index:
            continue 

        validation_image = Image.open(str(image_path)).convert("RGB")
        validation_prompt = get_validation_prompt(args, validation_image, model_vlm)
        print(f"Process {process_index}, {image_path}, tag: {validation_prompt}")


        ori_width, ori_height = validation_image.size
        resize_flag = False
        rscale = args.upscale
        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            tmp_image = validation_image.resize((int(scale * ori_width), int(scale * ori_height)))
            validation_image = tmp_image
            resize_flag = True

        validation_image = validation_image.resize((validation_image.size[0] * rscale, validation_image.size[1] * rscale))
        validation_image = validation_image.resize((validation_image.size[0] // 8 * 8, validation_image.size[1] // 8 * 8))
        width, height = validation_image.size
        validation_image_tensor = (torch.from_numpy(np.array(validation_image)).permute(2, 0, 1).float() / 255.0) * 2 - 1
        validation_image_tensor = validation_image_tensor.unsqueeze(0)
        resize_flag = True
        

        basename = os.path.basename(image_path).split('.')[0]

        for sample_idx in range(args.sample_times):
            with torch.autocast("cuda"):
                prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(validation_prompt, prompt_2=validation_prompt)
                image_result = pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    control_image=validation_image_tensor,
                    num_inference_steps=args.num_inference_steps,
                    controlnet_conditioning_scale=args.control_scale,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    height=args.process_size,
                    width=args.process_size,
                ).images[0]

            if args.align_method == 'nofix':
                final_image = image_result
            else:
                if args.align_method == 'wavelet':
                    final_image = wavelet_color_fix(image_result, validation_image)
                elif args.align_method == 'adain':
                    final_image = adain_color_fix(image_result, validation_image)
                else:
                    final_image = image_result
            
            if resize_flag:
                final_image = final_image.resize((ori_width * rscale, ori_height * rscale))
            

            out_path = os.path.join(args.output_dir, f"{basename}.png")
            final_image.save(out_path)






    


    




if __name__ == "__main__":
    args = parse_args()
    main(args)
