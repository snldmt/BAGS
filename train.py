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

import os
import numpy as np
import open3d as o3d
import cv2
import torch
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def tv_loss(grids):
    """
    https://github.com/apchenstu/TensoRF/blob/4ec894dc1341a2201fe13ae428631b58458f105d/utils.py#L139

    Args:
        grids: stacks of explicit feature grids (stacked at dim 0)
    Returns:
        average total variation loss for neighbor rows and columns.
    """
    number_of_grids = grids.shape[0]
    h_tv_count = grids[:, :, 1:, :].shape[1] * grids[:, :, 1:, :].shape[2] * grids[:, :, 1:, :].shape[3]
    w_tv_count = grids[:, :, :, 1:].shape[1] * grids[:, :, :, 1:].shape[2] * grids[:, :, :, 1:].shape[3]
    h_tv = torch.pow((grids[:, :, 1:, :] - grids[:, :, :-1, :]), 2).sum()
    w_tv = torch.pow((grids[:, :, :, 1:] - grids[:, :, :, :-1]), 2).sum()
    return 2 * (h_tv / h_tv_count + w_tv / w_tv_count) / number_of_grids


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


def get_2d_emb(batch_size, x, y, out_ch, device):
    out_ch = int(np.ceil(out_ch / 4) * 2)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, out_ch, 2).float() / out_ch))
    pos_x = torch.arange(x, device=device).type(inv_freq.type())*2*np.pi/x
    pos_y = torch.arange(y, device=device).type(inv_freq.type())*2*np.pi/y
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
    emb_x = get_emb(sin_inp_x).unsqueeze(1)
    emb_y = get_emb(sin_inp_y)
    emb = torch.zeros((x, y, out_ch * 2), device=device)
    emb[:, :, : out_ch] = emb_x
    emb[:, :, out_ch : 2 * out_ch] = emb_y
    return emb[None, :, :, :].repeat(batch_size, 1, 1, 1)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, resolution):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    scene = Scene(dataset)
    inp_shape = [len(scene.getTrainCameras()), int(np.round(scene.orig_h/resolution)), int(np.round(scene.orig_w/resolution))]

    kernel_size1 = dataset.kernel_size1
    kernel_size2 = dataset.kernel_size2
    kernel_size3 = dataset.kernel_size3
    print('kernel', kernel_size1, kernel_size2, kernel_size3)
    kernel_size_ss = dataset.kernel_size_ss
    print('kernel single scale', kernel_size_ss)


    gaussians = GaussianModel(dataset.sh_degree, inp_shape, 
                              ks1=kernel_size1, ks2=kernel_size2, ks3=kernel_size3, ks_ss=kernel_size_ss,
                              not_use_rgbd=opt.not_use_rgbd,not_use_pe=opt.not_use_pe)
    
    scene.load_gaussian(gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        print(checkpoint)
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras(scale=4.0).copy()
    testCameras = scene.getTestCameras(scale=4.0).copy()
    allCameras = trainCameras + testCameras
    
    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    upsample_iter = [3000, 6000]

    unfold1 = torch.nn.Unfold(kernel_size=(kernel_size1, kernel_size1),
                                padding=kernel_size1 // 2).cuda()
    unfold2 = torch.nn.Unfold(kernel_size=(kernel_size2, kernel_size2),
                                padding=kernel_size2 // 2).cuda()
    unfold3 = torch.nn.Unfold(kernel_size=(kernel_size3, kernel_size3),
                                padding=kernel_size3 // 2).cuda()    
    
    if kernel_size_ss != kernel_size3:
        opt.use_another_mlp = True
        unfold_ss = torch.nn.Unfold(kernel_size=(kernel_size_ss, kernel_size_ss),
                                    padding=kernel_size_ss // 2).cuda()
    else:
        unfold_ss = unfold3

    assert opt.ms_steps >= 6000
    print('************** position_lr_max_steps', opt.position_lr_max_steps)
    print('************** densify_until_iter', opt.densify_until_iter)
    print('************** init densify_grad_threshold', opt.init_dgt)
    print('************** densify_grad_threshold', opt.densify_grad_threshold)
    print('************** min_opacity', opt.min_opacity)
    print('************** ms_steps', opt.ms_steps)
    print('mask_loss', opt.use_mask_loss, 'depth_loss', opt.use_depth_loss, 'rgbtv_loss', opt.use_rgbtv_loss)


    for iteration in range(first_iter, opt.iterations + 1):        
        if iteration in upsample_iter:
            if iteration == upsample_iter[0]:
                print('CHANGE RESOLUTION')
                trainCameras = scene.getTrainCameras(scale=2.0).copy()
                testCameras = scene.getTestCameras(scale=2.0).copy() 
                allCameras = trainCameras + testCameras
                gaussians.compute_3D_filter(cameras=trainCameras)     
                viewpoint_stack = scene.getTrainCameras(scale=2.0).copy()                  
            else:
                print('CHANGE RESOLUTION')
                trainCameras = scene.getTrainCameras(scale=1.0).copy()
                testCameras = scene.getTestCameras(scale=1.0).copy() 
                allCameras = trainCameras + testCameras
                gaussians.compute_3D_filter(cameras=trainCameras)     
                viewpoint_stack = scene.getTrainCameras(scale=1.0).copy()      
        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        ori_iter = iteration
        if ori_iter > opt.ms_steps:
            if ori_iter == opt.ms_steps + 1:
                print('start training iterations for final scale, lr reset!')
            iteration = iteration - opt.ms_steps

        cur_lr = gaussians.update_learning_rate(iteration)
        # if iteration % 100 == 0:
        #     print('cur_lr:', cur_lr, 'iteration:', iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        cam_num = 1
        loss = 0
        # Pick a random Camera
        if not viewpoint_stack:
            if ori_iter >= upsample_iter[0] and ori_iter < upsample_iter[1]:
                viewpoint_stack = scene.getTrainCameras(scale=2.0).copy()
            elif ori_iter >= upsample_iter[1]:
                viewpoint_stack = scene.getTrainCameras(scale=1.0).copy()
            else:
                viewpoint_stack = scene.getTrainCameras(scale=4.0).copy()

        cam_indices = []
        for _ in range(cam_num):
            ind = randint(0, len(viewpoint_stack)-1)
            if ind not in cam_indices:
                cam_indices.append(ind)

        for cam_idx in cam_indices:
            viewpoint_cam = viewpoint_stack[cam_idx]

            # Render
            if (ori_iter - 1) == debug_from:
                pipe.debug = True

            #TODO ignore border pixels
            if dataset.ray_jitter:
                subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
                # subpixel_offset *= 0.0
            else:
                subpixel_offset = None
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset)
            image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            if iteration % 100 == 0:
                print(gaussians._xyz.shape,'NUM OF GAUSSIANS')
            if iteration > 250:
                shuffle_rgb = image.unsqueeze(0)
                shuffle_depth = depth.unsqueeze(0) - depth.min()
                shuffle_depth = shuffle_depth/shuffle_depth.max()

                pos_enc = get_2d_emb(1, shuffle_rgb.shape[-2], shuffle_rgb.shape[-1], 16, torch.device(0))

                if ori_iter < 3000:
                    kernel_weights, mask = gaussians.mlp_rgb_ms(cam_idx, pos_enc, torch.cat([shuffle_rgb,shuffle_depth],1).detach(),ori_iter)
                    patches = unfold1(shuffle_rgb)
                    patches = patches.view(1, 3, kernel_size1 ** 2, shuffle_rgb.shape[-2],
                                           shuffle_rgb.shape[-1])
                    kernel_weights = kernel_weights.unsqueeze(1)
                    rgb = torch.sum(patches * kernel_weights, 2)[0]
                    mask = mask[0]

                elif ori_iter >= 3000 and ori_iter < 6000:
                    kernel_weights, mask = gaussians.mlp_rgb_ms(cam_idx, pos_enc, torch.cat([shuffle_rgb,shuffle_depth],1).detach(),ori_iter)
                    patches = unfold2(shuffle_rgb)
                    patches = patches.view(1, 3, kernel_size2 ** 2, shuffle_rgb.shape[-2],
                                           shuffle_rgb.shape[-1])
                    kernel_weights = kernel_weights.unsqueeze(1)
                    rgb = torch.sum(patches * kernel_weights, 2)[0]
                    mask = mask[0]

                else:
                    if (ori_iter > opt.ms_steps) and opt.use_another_mlp:
                        kernel_weights, mask = gaussians.mlp_rgb_ss(cam_idx, pos_enc, torch.cat([shuffle_rgb,shuffle_depth],1).detach(),iteration)
                        patches = unfold_ss(shuffle_rgb)
                        patches = patches.view(1, 3, kernel_size_ss ** 2, shuffle_rgb.shape[-2],
                                                shuffle_rgb.shape[-1])
                    else:
                        kernel_weights, mask = gaussians.mlp_rgb_ms(cam_idx, pos_enc, torch.cat([shuffle_rgb,shuffle_depth],1).detach(),ori_iter)
                        patches = unfold3(shuffle_rgb)
                        patches = patches.view(1, 3, kernel_size3 ** 2, shuffle_rgb.shape[-2],
                                            shuffle_rgb.shape[-1])
                    
                    kernel_weights = kernel_weights.unsqueeze(1)
                    rgb = torch.sum(patches * kernel_weights, 2)[0] 
                    mask = mask[0]
                
                blur_image = mask*rgb + (1-mask)*image

                depthloss = opt.depth_loss_alpha * tv_loss(shuffle_depth) if opt.use_depth_loss else 0
                maskloss = opt.mask_loss_alpha * mask.mean() if opt.use_mask_loss else 0
                tvloss = opt.rgbtv_loss_alpha * tv_loss(shuffle_rgb) if opt.use_rgbtv_loss else 0

                if iteration % 1000 == 0:
                    print('depthloss',depthloss,'maskloss',maskloss,'tvloss:', tvloss)

                gt_image = viewpoint_cam.original_image.cuda()

                Ll1 = l1_loss(blur_image, gt_image)
                loss = loss + (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(blur_image, gt_image)) + tvloss + maskloss + depthloss
            else:
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                loss = loss + (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if ori_iter % 1000 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(1000)
            if ori_iter == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, ori_iter, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.kernel_size))
            
            if (ori_iter in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(ori_iter))
                scene.save(ori_iter)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:    
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None # reset interval = 3000
                    
                    if ori_iter <= opt.ms_steps:
                        dgt = opt.densify_grad_threshold if opt.init_dgt < 0 else opt.init_dgt                        
                        min_opacity = opt.min_opacity if opt.init_opacity < 0 else opt.init_opacity
                    else:
                        dgt = opt.densify_grad_threshold
                        min_opacity = opt.min_opacity

                    gaussians.densify_and_prune(dgt, min_opacity, scene.cameras_extent, size_threshold)


                    gaussians.compute_3D_filter(cameras=trainCameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < (opt.iterations - opt.ms_steps) - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)
        
            # Optimizer step
            if ori_iter < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (ori_iter in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(ori_iter))
                torch.save((gaussians.capture(), ori_iter), scene.model_path + "/chkpnt" + str(ori_iter) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 4_000, 6_000, 7_000, 9_000, 10_000, 12_000, 15_000, 18_000, 20_000, 22_000, 25_000, 28_000, 30_000, 32_000, 35_000, 38_000, 40_000, 42_000, 45_000, 48_000, 50_000, 60_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 4_000, 6_000, 7_000, 9_000, 10_000, 12_000, 15_000, 18_000, 20_000, 22_000, 25_000, 28_000, 30_000, 32_000, 35_000, 38_000, 40_000, 42_000, 45_000, 48_000, 50_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[6000,7000,9000,10000,12000,22000,32000,42000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    args.test_iterations.append(args.iterations)
    args.save_iterations.append(args.iterations)
   
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.resolution)

    # All done
    print("\nTraining complete.")
