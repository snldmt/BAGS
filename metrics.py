#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from skimage import metrics

import lpips
import json
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
from lpips.lpips import LPIPS


photometric = {
    "mse": None,
    "ssim": None,
    "psnr": None,
    "lpips": None
}

def compute_img_metric(im1t: torch.Tensor, im2t: torch.Tensor,
                       metric="mse", margin=0, mask=None):
    """
    im1t, im2t: torch.tensors with batched imaged shape, range from (0, 1)
    """
    if metric not in photometric.keys():
        raise RuntimeError(f"img_utils:: metric {metric} not recognized")
    if photometric[metric] is None:
        if metric == "mse":
            photometric[metric] = metrics.mean_squared_error
        elif metric == "ssim":
            photometric[metric] = metrics.structural_similarity
        elif metric == "psnr":
            photometric[metric] = metrics.peak_signal_noise_ratio
        elif metric == "lpips":
            photometric[metric] = LPIPS().cpu()

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] == 1:
            mask = mask.expand(-1, 3, -1, -1)
        mask = mask.permute(0, 2, 3, 1).numpy()
        batchsz, hei, wid, _ = mask.shape
        if margin > 0:
            marginh = int(hei * margin) + 1
            marginw = int(wid * margin) + 1
            mask = mask[:, marginh:hei - marginh, marginw:wid - marginw]

    # convert from [0, 1] to [-1, 1]
    im1t = (im1t * 2 - 1).clamp(-1, 1)
    im2t = (im2t * 2 - 1).clamp(-1, 1)

    if im1t.dim() == 3:
        im1t = im1t.unsqueeze(0)
        im2t = im2t.unsqueeze(0)
    im1t = im1t.detach().cpu()
    im2t = im2t.detach().cpu()

    if im1t.shape[-1] == 3:
        im1t = im1t.permute(0, 3, 1, 2)
        im2t = im2t.permute(0, 3, 1, 2)

    im1 = im1t.permute(0, 2, 3, 1).numpy()
    im2 = im2t.permute(0, 2, 3, 1).numpy()
    batchsz, hei, wid, _ = im1.shape
    if margin > 0:
        marginh = int(hei * margin) + 1
        marginw = int(wid * margin) + 1
        im1 = im1[:, marginh:hei - marginh, marginw:wid - marginw]
        im2 = im2[:, marginh:hei - marginh, marginw:wid - marginw]
    values = []

    for i in range(batchsz):
        if metric in ["mse", "psnr"]:
            if mask is not None:
                im1 = im1 * mask[i]
                im2 = im2 * mask[i]
            value = photometric[metric](
                im1[i], im2[i]
            )
            if mask is not None:
                hei, wid, _ = im1[i].shape
                pixelnum = mask[i, ..., 0].sum()
                value = value - 10 * np.log10(hei * wid / pixelnum)
        elif metric in ["ssim"]:
            # print(im1[i].shape, im2[i].shape)
            value, ssimmap = photometric["ssim"](
                im1[i], im2[i], multichannel=True, full=True
            )
            if mask is not None:
                value = (ssimmap * mask[i]).sum() / mask[i].sum()
        elif metric in ["lpips"]:
            value = photometric[metric](
                im1t[i:i + 1], im2t[i:i + 1]
            )
        else:
            raise NotImplementedError
        values.append(value)

    return sum(values) / len(values)


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in [e for e in os.listdir(renders_dir) if not e.startswith(".")]:
        render = Image.open(renders_dir / fname)

        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, scale, metric_method='mip-splatting'):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")


    for scene_dir in model_paths:
        # try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in [e for e in os.listdir(test_dir) if not e.startswith('.')]:
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ f"gt_{scale}"
                renders_dir = method_dir / f"test_preds_{scale}"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    if metric_method == 'mip-splatting':
                        ##### mip-splatting
                        if idx == 0:
                            print('********** mip-splatting metrics! ***************')
                        from utils.loss_utils import ssim
                        ssims.append(ssim(renders[idx], gts[idx]))
                        from utils.image_utils import psnr
                        psnrs.append(psnr(renders[idx], gts[idx]))
                        lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

                    elif metric_method == 'deblur-nerf':
                        ###### deblur-nerf
                        if idx == 0:
                            print('@@@@@@@@@@@ Deblur-NeRF metrics! @@@@@@@@@@@@@@@@')
                        psnr = compute_img_metric(renders[idx], gts[idx],metric='psnr')
                        psnrs.append(psnr)
                        ssim = compute_img_metric(renders[idx], gts[idx],metric='ssim')
                        ssims.append(ssim)
                        lpips = compute_img_metric(renders[idx], gts[idx],metric='lpips')
                        lpipss.append(lpips)
                    
                    else:
                        raise NotImplementedError
                    

                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print(torch.tensor(psnrs).mean().item(), torch.tensor(ssims).mean().item(), torch.tensor(lpipss).mean().item(), '\n\n')
                print("")

                full_dict[scene_dir][method].update({"PSNR": torch.tensor(psnrs).mean().item(),
                                                    "SSIM": torch.tensor(ssims).mean().item(),
                                                    "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                         "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--resolution', '-r', type=int, default=-1)
    parser.add_argument('--metric', '-t', type=str, default='mip-splatting')
    # parser.add_argument('--metric', '-t', type=str, default='deblur-nerf')

    
    args = parser.parse_args()
    evaluate(args.model_paths, args.resolution, args.metric)
