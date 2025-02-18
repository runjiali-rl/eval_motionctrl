



import argparse
import datetime
import json
import math
import os
import sys
import time
from glob import glob
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor
from torcheval.metrics import PeakSignalNoiseRatio
import lpips

sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from sgm.util import default, instantiate_from_config


def calculate_score(pil_img1, pil_img2, device):
    """
    Compute PSNR and LPIPS between two PIL images.
    Note: FID is set to 0 for now.
    Assumes that global variables `loss_fn_alex`, `psnr_metric`, `fid_metric`, and `device`
    have been set in the process.
    """
    img1 = np.array(pil_img1)
    img2 = np.array(pil_img2)

    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions.")

    # Squeeze singleton channels if needed
    if img1.ndim == 3 and img1.shape[-1] == 1:
        img1 = img1.squeeze(axis=-1)
        img2 = img2.squeeze(axis=-1)

    # Determine data_range (this example does not use it further)
    if img1.dtype == np.uint8:
        data_range = 255.0
    elif img1.dtype == np.uint16:
        data_range = 65535.0
    elif np.issubdtype(img1.dtype, np.floating):
        data_range = 1.0
    else:
        raise ValueError("Unsupported dtype. Use uint8, uint16, or float.")

    # Convert images to torch tensors
    img1_tensor = torch.from_numpy(img1)
    img2_tensor = torch.from_numpy(img2)

    # Compute PSNR
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    psnr_metric = PeakSignalNoiseRatio()
    psnr_metric.update(img2_tensor.to(torch.float32), img1_tensor.to(torch.float32))
    psnr_value = psnr_metric.compute()
    psnr_metric.reset()

    # For now, we set FID to 0 (or implement if desired)
    fid_value = 0

    # Prepare tensors for LPIPS (channels-first, normalized to [-1,1])
    img1_tensor = img1_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    img2_tensor = img2_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    img1_tensor = (img1_tensor / 255.0) * 2.0 - 1.0
    img2_tensor = (img2_tensor / 255.0) * 2.0 - 1.0

    # Resize to 64x64 for LPIPS
    img1_tensor = F.interpolate(img1_tensor, size=(64, 64), mode='bilinear', align_corners=False)
    img2_tensor = F.interpolate(img2_tensor, size=(64, 64), mode='bilinear', align_corners=False)
    lpips_value = loss_fn_alex(img1_tensor, img2_tensor).item()

    return (psnr_value, lpips_value, fid_value)

def to_relative_RT2(org_pose, keyframe_idx=0, keyframe_zero=False):
        org_pose = org_pose.reshape(-1, 3, 4) # [t, 3, 4]
        R_dst = org_pose[:, :, :3]
        T_dst = org_pose[:, :, 3:]

        R_src = R_dst[keyframe_idx: keyframe_idx+1].repeat(org_pose.shape[0], axis=0) # [t, 3, 3]
        T_src = T_dst[keyframe_idx: keyframe_idx+1].repeat(org_pose.shape[0], axis=0)

        R_src_inv = R_src.transpose(0, 2, 1) # [t, 3, 3]
        
        R_rel = R_dst @ R_src_inv # [t, 3, 3]
        T_rel = T_dst - R_rel@T_src

        RT_rel = np.concatenate([R_rel, T_rel], axis=-1) # [t, 3, 4]
        RT_rel = RT_rel.reshape(-1, 12) # [t, 12]

        if keyframe_zero:
            RT_rel[keyframe_idx] = np.zeros_like(RT_rel[keyframe_idx])

        return RT_rel



def save_results(results, filename, fps=10):
    video = results.permute(1, 0, 2, 3, 4)  # [t, sample_num, c, h, w]
    frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(video.shape[1])) for framesheet in video]  # [3, 1*h, n*w]
    grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
    # Convert to numpy array in correct format for cv2
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = grid.shape[1:3]
    writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    
    # Write frames
    for frame in grid:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    
    writer.release()

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    ckpt: str,
    device: str,
    num_frames: int,
    num_steps: int,
):

    config = OmegaConf.load(config)
    config.model.params.ckpt_path = ckpt
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )

    model = instantiate_from_config(config.model)

    model = model.to(device).eval()    

    filter = None #DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter









def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_video_dir", type=str, default=None, help="processed video directory")
    parser.add_argument("--meta_file", type=str, default=None, help="meta file")
    parser.add_argument("--seed", type=int, default=23, help="seed for seed_everything")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--input", type=str, default=None, help="image path or folder")
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--savefps", type=int, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=6, help="control the fps")
    parser.add_argument("--motion", type=int, default=127, help="control the motion magnitude")
    parser.add_argument("--cond_aug", type=float, default=0.02, help="adding noise to input image")
    parser.add_argument("--decoding_t", type=int, default=1, help="frames num to decoding per time")
    parser.add_argument("--resize", action='store_true', default=False, help="resize all input to default resolution")
    parser.add_argument("--sample_num", type=int, default=1, help="frames num to decoding per time")
    parser.add_argument("--pose_dir", type=str, default='', help="checkpoint path")
    parser.add_argument("--height", type=int, default=576, help="frames num to decoding per time")
    parser.add_argument("--width", type=int, default=1024, help="frames num to decoding per time")
    parser.add_argument("--transform", action='store_true', default=False, help="resize all input to specific resolution")
    parser.add_argument("--save_images", action='store_true', default=False, help="save images")
    parser.add_argument("--speed", type=float, default=1.0, help="speed of camera motion")
    parser.add_argument("--rank", type=str, default="0", help="device")
    parser.add_argument("--num_gpus", type=int, default=2, help="number of gpus")
    return parser


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    #Fire(sample)
    parser = get_parser()
    args = parser.parse_args()
    ckpt = args.ckpt
    config = args.config
    input_path = args.input
    savedir = args.savedir
    savefps = args.savefps
    frames = args.frames
    fps = args.fps
    processed_video_dir = args.processed_video_dir
    meta_file = args.meta_file
    device = "cuda"
    mode = 'test'
    
    return_flag = True
    diffusion_num_frames = 14
    num_steps = args.ddim_steps
    
    processed_video_dir = os.path.join(processed_video_dir, mode)
    all_episode_psnr = []
    all_episode_lpips = []
    all_episode_fid = []
    model_config = default(config, "configs/inference/config_motionctrl_cmcm.yaml")
    model, filter = load_model(
        model_config,
        ckpt,
        device,
        diffusion_num_frames,
        num_steps,
    )
    episode_list = os.listdir(processed_video_dir)
    chunk_size = len(episode_list) // args.num_gpus
    print(f"rank {args.rank} has {chunk_size} episodes")
    rank = int(args.rank)
    rank_episode_list = episode_list[rank*chunk_size:(rank+1)*chunk_size]
    for episode_id in tqdm(rank_episode_list):
        output_folder = os.path.join(savedir, episode_id)
        os.makedirs(output_folder, exist_ok=True)
        images_dir = os.path.join(processed_video_dir, episode_id, 'images')
        all_image_paths = os.listdir(images_dir)
        timestamps = [int(os.path.splitext(image)[0]) for image in all_image_paths]
        timestamps.sort()

        all_image_paths = [os.path.join(images_dir, f"{timestamp}.png") for timestamp in timestamps]
        episode_psnr = []
        episode_lpips = []
        episode_fid = []
        if not os.path.exists(os.path.join(output_folder, 'final_0.png')):


            init_image_path = all_image_paths[0]
            episode_meta_file = os.path.join(meta_file, mode, f"{episode_id}.txt")
            camera_poses = []
            with open(episode_meta_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith('https://www.youtube.com/watch?v='):
                    continue
                info = line.split(' ')
                timestamp = int(info[0])
                if timestamp in timestamps:

                    camera_pose = np.array(info[7:19], dtype=np.float32)
                    camera_poses.append(camera_pose)

            camera_poses = np.array(camera_poses)
            camera_poses = to_relative_RT2(camera_poses)
            
            
            if return_flag:
                camera_poses = np.concatenate([camera_poses, camera_poses[::-1]], axis=0)
                all_image_paths = all_image_paths + all_image_paths[::-1]
            actual_num_frames = len(camera_poses)
        
            # camera_pose_path = os.path.join(episode_dir, 'poses', f"{timestamp}.json")

            camera_poses = torch.from_numpy(camera_poses)
            if actual_num_frames < diffusion_num_frames:
                num_additional_frames = diffusion_num_frames - actual_num_frames
                camera_poses = torch.cat([camera_poses, camera_poses[-1:].repeat(num_additional_frames,  1)], axis=0)
                generation_steps = 1
            elif actual_num_frames > diffusion_num_frames:
                generation_steps = (actual_num_frames - 1) // (diffusion_num_frames-1) + 1
                camera_poses = torch.cat([camera_poses, camera_poses[-1:].repeat(generation_steps*(diffusion_num_frames-1)+1 - actual_num_frames, 1)], axis=0)
                
            else:
                generation_steps = 1
            camera_poses = camera_poses.unsqueeze(0).repeat(2,1,1).to(device)

            
            
            
            diffusion_result_pil = []
            
            for i in range(generation_steps):

                if i == 0:
                    image_path = init_image_path
                else:
                    last_image_pil = diffusion_result_pil[-1]
                    image_path = os.path.join(output_folder, f'{i}.png')
                    last_image_pil.save(image_path)
                current_camera_traj = camera_poses[:, i*(diffusion_num_frames-1):(i+1)*(diffusion_num_frames-1)+1]
                image = Image.open(image_path)
                image = image.resize((args.width, args.height))
                image = ToTensor()(image)
                image = image * 2.0 - 1.0

                image = image.unsqueeze(0).to(device)
                H, W = image.shape[2:]
                assert image.shape[1] == 3
                scale_factor = 8
                C = 4
                shape = (diffusion_num_frames, C, H // scale_factor, W // scale_factor)

                
                value_dict = {}
                value_dict["motion_bucket_id"] = 127
                value_dict["fps_id"] = 6
                value_dict["cond_aug"] = 0.02
                value_dict["cond_frames_without_noise"] = image
                value_dict["cond_frames"] = image + 0.02 * torch.randn_like(image)

                with torch.no_grad():
                    with torch.autocast(device):
                        batch, batch_uc = get_batch(
                            get_unique_embedder_keys_from_conditioner(model.conditioner),
                            value_dict,
                            [1, diffusion_num_frames],
                            T=diffusion_num_frames,
                            device=device,
                        )
                        c, uc = model.conditioner.get_unconditional_conditioning(
                            batch,
                            batch_uc=batch_uc,
                            force_uc_zero_embeddings=[
                                "cond_frames",
                                "cond_frames_without_noise",
                            ],
                        )

                        for k in ["crossattn", "concat"]:
                            uc[k] = repeat(uc[k], "b ... -> b t ...", t=diffusion_num_frames)
                            uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=diffusion_num_frames)
                            c[k] = repeat(c[k], "b ... -> b t ...", t=diffusion_num_frames)
                            c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=diffusion_num_frames)

                        

                        additional_model_inputs = {}
                        additional_model_inputs["image_only_indicator"] = torch.zeros(
                            2, diffusion_num_frames
                        ).to(device)
                        #additional_model_inputs["image_only_indicator"][:,0] = 1
                        additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                        
                        additional_model_inputs["RT"] = current_camera_traj

                        def denoiser(input, sigma, c):
                            return model.denoiser(
                                model.model, input, sigma, c, **additional_model_inputs
                            )

                        results = []
                    
                        randn = torch.randn(shape, device=device)
                        samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                        model.en_and_decode_n_samples_a_time = 1
                        samples_x = model.decode_first_stage(samples_z)
                        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0) # [1*t, c, h, w]
                        results.append(samples)

                        samples = torch.stack(results, dim=0) # [sample_num, t, c, h, w]
                        samples = samples.data.cpu()
                        samples = samples[0]

                        if i == generation_steps - 1:
                            for j in range((actual_num_frames-1)%(diffusion_num_frames-1)+1):
                                if i !=0 and j == 0:
                                    continue # skip the repeated first frame
                                diffusion_result_pil.append(Image.fromarray((samples[j].permute(1, 2, 0) * 255).numpy().astype(np.uint8)))
                        else:
                            for j in range(diffusion_num_frames):
                                if i !=0 and j == 0:
                                    continue # skip the repeated first frame
                                diffusion_result_pil.append(Image.fromarray((samples[j].permute(1, 2, 0) * 255).numpy().astype(np.uint8)))
                      
            print(f"saving {len(diffusion_result_pil)} images to {output_folder}")
            for i, pil_img in enumerate(diffusion_result_pil):
                pil_img.save(os.path.join(output_folder, f'final_{i}.png'))
        else:
            diffusion_result_pil = [Image.open(os.path.join(output_folder, image)) for image in os.listdir(output_folder) if image.startswith('final_')]
            print(f"loading {len(diffusion_result_pil)} images from {output_folder}")
            if return_flag:
                all_image_paths = all_image_paths + all_image_paths[::-1]
        
        all_images = [Image.open(image_path).resize((args.width, args.height)) for image_path in all_image_paths]

        for i in range(len(diffusion_result_pil)):
            psnr_value, lpips_value, fid_value = calculate_score(diffusion_result_pil[i], all_images[i], device)
           
            episode_psnr.append(psnr_value)
            episode_lpips.append(lpips_value)
            episode_fid.append(fid_value)
        mean_psnr = np.mean(episode_psnr)   
        mean_lpips = np.mean(episode_lpips)
        mean_fid = np.mean(episode_fid)
        print(f"PSNR: {mean_psnr}, LPIPS: {mean_lpips}, FID: {mean_fid}")
        all_episode_psnr.append(mean_psnr)
        all_episode_lpips.append(mean_lpips)
        all_episode_fid.append(mean_fid)


    print(f"PSNR: {np.mean(all_episode_psnr)}, LPIPS: {np.mean(all_episode_lpips)}, FID: {np.mean(all_episode_fid)}")
