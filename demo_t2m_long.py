import numpy as np
import scipy as sp
import torch
import pickle
from utils.rotation_conversion import *
from utils.algo import *
from pathlib import Path
import clip
from torch.nn.utils.rnn import pad_sequence
from utils.algo import *

from model.SPDM.diffusion import DiffPhase
from model.TPDM.diffusion import TranPhase

transphase_model = TranPhase.load_from_checkpoint("./model/TPDM/lightning_logs/version_0/checkpoints/last.ckpt")
transphase_model = transphase_model.cuda()

diffphase_model = DiffPhase.load_from_checkpoint("./model/SPDM/lightning_logs/version_0/checkpoints/last.ckpt")
diffphase_model = diffphase_model.cuda()

### CLIP helper
def load_and_freeze_clip(clip_version="ViT-B/32", device='cpu'):
    clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                            jit=False)  # Must set jit=False for training
    clip.model.convert_weights(
        clip_model)  # Actually this line is unnecessary since clip by default already on float16

    # Freeze CLIP weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model

def encode_text(clip_model, raw_text, device='cpu'):
    # raw_text - list (batch_size length) of strings with input text prompts
    clip_model = clip_model.to(device)
    texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
    return clip_model.encode_text(texts).float()

### motion generation script
def dur2progress(left_length, right_length):
    left_half = int(left_length/2)
    right_half = int(right_length/2)
        # -1: left boundary, 0: at left_half, 1:right boundary
    left_progress_left = torch.linspace(-1,0,left_half+1)[:-1]                      # left,  include -1, exclude 0
    left_progress_right = torch.linspace(0,1,left_length-left_half)           # right, include 0,1
        # -1: left boundary (note that this is the last frame from left_mot), 0: at right_half, 1:right boundary
    right_progress_left = torch.linspace(-1,0,right_half+1)[:-1]                    # left, include -1, exclude 0
    right_progress_right = torch.linspace(0,1,right_length-right_half)        # right, include 0,1

        ## final progress tensor
    left_progress = torch.cat([left_progress_left, left_progress_right],dim=0)
    right_progress = torch.cat([right_progress_left, right_progress_right],dim=0)
    trans_progress = torch.cat([left_progress_right-1, torch.linspace(0,1,right_half+1)[1:]],dim=0)
    return left_progress, right_progress, trans_progress

def motion_generation(texts, timings):

    ### encode text
    cond_texts = encode_text(clip_model, texts, "cuda")
    cond_texts = cond_texts / cond_texts.norm(dim=-1, keepdim=True)
    cond_texts = diffphase_model.condlinear(cond_texts)

    ### generate meta (progress)
    # meta for semantics
    time_boundary = np.cumsum([0] + list(timings))
    progress_all = []
    for i in range(1, len(timings)):
        progress = dur2progress(timings[i - 1], timings[i])
        progress_all.append(list(progress))
    left_progress, right_progress, transitioning_progress = zip(*progress_all)
    semantics_progress = list(left_progress) + [right_progress[-1]]

    # meta for transitioning
    transitioning_boundary = ((time_boundary[1:] + time_boundary[:-1]) / 2).astype(np.int32)
    transitioning_timing = transitioning_boundary[1:] - transitioning_boundary[:-1]
    
    #############
    # get valid mask
    progress_all = []
    masks_all = []
    for progress in [semantics_progress, transitioning_progress]:
        lengths = [prog.shape[0] for prog in progress]
        progress = pad_sequence(progress, batch_first=True).float().to(cond_texts.device)
        masks = torch.zeros(
            (progress.shape[0], progress.shape[1]),
            device=cond_texts.device,
            dtype=torch.bool,
        )
        for b in range(len(lengths)):
            masks[b, : lengths[b]] = True
        progress_all.append(progress)
        masks_all.append(masks)

    semantics_progress, transitioning_progress = progress_all
    masks_semantics, masks_transitioning = masks_all

    #####################################################
    ### denoise start

    diffphase_model.scheduler.set_timesteps(99)
    timesteps = diffphase_model.scheduler.timesteps.to(cond_texts.device)
    code_semantics_pred = (
            torch.randn(
                (semantics_progress.shape[0], 4, diffphase_model.latent_dim)
            ).to(cond_texts.device)
            * diffphase_model.scheduler.init_noise_sigma
    )
    code_trans_pred = (
            torch.randn(
                (transitioning_progress.shape[0], 4, diffphase_model.latent_dim)
            ).to(cond_texts.device)
            * diffphase_model.scheduler.init_noise_sigma
    )

    code_semantics_pred0 = code_semantics_pred
    code_trans_pred0 = code_trans_pred

    for i in timesteps:
        t = torch.tensor([i], device=code_trans_pred.device, dtype=torch.long)

        code_left_pred, code_right_pred = (
            code_semantics_pred[:-1],
            code_semantics_pred[1:],
        )
        masks_left, masks_right = masks_semantics[:-1], masks_semantics[1:]
        left_progress, right_progress = (
            semantics_progress[:-1],
            semantics_progress[1:],
        )
        code_left_pred0, code_right_pred0 = (
            code_semantics_pred0[:-1],
            code_semantics_pred0[1:],
        )   

        # SPDM
        code_semantics_0T = diffphase_model.diffusion_step(
            t,
            code_semantics_pred,
            masks_semantics,
            semantics_progress,
            cond_texts.unsqueeze(1)
        )

        # TPDM
        code_left_0R = transphase_model.diffusion_step(
            t,
            code_left_pred,
            masks_left,
            left_progress,
            code_trans_pred0,
            masks_transitioning,
            transitioning_progress,
            decode_left=True,
            decode_trans=False,
        )

        code_right_0L = transphase_model.diffusion_step(
            t,
            code_right_pred,
            masks_right,
            right_progress,
            code_trans_pred0,
            masks_transitioning,
            transitioning_progress,
            decode_left=False,
            decode_trans=False,
        )

        code_trans_0R = transphase_model.diffusion_step(
            t,
            code_trans_pred,
            masks_transitioning,
            transitioning_progress,
            code_right_pred0,
            masks_right,
            right_progress,
            decode_left=False,
            decode_trans=True,
        )

        code_trans_0L = transphase_model.diffusion_step(
            t,
            code_trans_pred,
            masks_transitioning,
            transitioning_progress,
            code_left_pred0,
            masks_left,
            left_progress,
            decode_left=True,
            decode_trans=True,
        )

        ### phase mixing
        # semantic segments
        timestep_scale = (i / 1000) ** 3  # see paper
        trans_scale, text_scale = timestep_scale, 1 - timestep_scale
        code_left_0 = (code_semantics_0T[0] * text_scale + code_left_0R[0] * trans_scale) / (text_scale + trans_scale)  # leftmost
        code_right_0 = (code_semantics_0T[-1] * text_scale + code_right_0L[-1] * trans_scale) / (text_scale + trans_scale)  # rightmost
        code_inb_0 = (code_semantics_0T[1:-1] * text_scale + (code_left_0R[1:] + code_right_0L[:-1]) / 2 * trans_scale) / (text_scale + trans_scale)
        code_semantics_0 = torch.cat([code_left_0.unsqueeze(0), code_inb_0, code_right_0.unsqueeze(0)], dim=0)

        # transitioning segment
        timestep_scale = 1  # see paper
        code_trans_0 = (code_trans_0L + code_trans_0R) / 2
        code_semantics_0, code_trans_0 = code_semantics_0.detach(), code_trans_0.detach()

        ###
        # tidy up for next loop (including add noise)
        code_semantics_pred0 = diffphase_model.scheduler.step(
            code_semantics_0, i, code_semantics_pred
        ).pred_original_sample
        code_trans_pred0 = diffphase_model.scheduler.step(
            code_trans_0, i, code_trans_pred
        ).pred_original_sample

        code_semantics_pred = diffphase_model.scheduler.step(
            code_semantics_0, i, code_semantics_pred
        ).prev_sample
        code_trans_pred = diffphase_model.scheduler.step(
            code_trans_0, i, code_trans_pred
        ).prev_sample

    #####################################################
    ### merge to motion
    rep_semantics_0 = diffphase_model.PAE.latent_reparam(code_semantics_pred, semantics_progress)
    rec_semantics = diffphase_model.PAE.decode(rep_semantics_0, masks_semantics, semantics_progress).detach()

    rep_trans_0 = diffphase_model.PAE.latent_reparam(code_trans_pred, transitioning_progress)
    rec_trans = diffphase_model.PAE.decode(rep_trans_0, masks_transitioning, transitioning_progress).detach()

    # combine semantics segments
    full_motion = []
    for i in range(rec_semantics.shape[0]):
        full_motion.append(rec_semantics[i, : timings[i]])
    full_motion = torch.cat(full_motion, dim=0)

    # merge transitioning segments
    for i in range(rec_trans.shape[0]):
        trans_i = rec_trans[i, :transitioning_timing[i]]
        progress_i = transitioning_progress[i, :transitioning_timing[i]].unsqueeze(-1)
        weighting = torch.abs(progress_i)   #(1 -> 0 -> 1)

        start, end = transitioning_boundary[i], transitioning_boundary[i + 1]
        full_motion[start:end] = full_motion[start:end] * weighting + trans_i * (1 - weighting)

    full_motion = full_motion.detach().cpu()
    full_motion = torch.matmul(
        full_motion, diffphase_model.gmd_proj_inv.to(full_motion.device)
    )
    return full_motion


########################################################################################################


if __name__ == "__main__":
    clip_model = load_and_freeze_clip(device='cuda')

    text_batch = ["sit down", "stand up", "walk forward", "stand", "leap forward"]
    time_batch = [70, 45, 80, 45, 60]

    motion = motion_generation(text_batch, time_batch)
    trans, poses = motion_to_smpl(motion)
    save_motion("output_t2m.npz", trans, poses.reshape((-1, 22 * 3)))
