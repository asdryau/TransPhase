import numpy as np
import scipy as sp
import torch
import pickle
from utils.rotation_conversion import *
from utils.algo import *
from pathlib import Path
import clip

from torch.nn.utils.rnn import pad_sequence

from model.SPDM.diffusion import DiffPhase
from model.TPDM.diffusion import TranPhase

# torch.set_default_tensor_type('torch.cuda.FloatTensor')


batch_size = 8

phase_mix_basic = ["0", "05", "1", "0to1", "1to0"]
phase_mix_stage2 = ["1to0cube", "neg0to1cube"]
phase_mix_stage3 = ["1to0sq","1to0quat"]

phase_mix_all = phase_mix_basic + phase_mix_stage2 + phase_mix_stage3


### run  
transphase_model = TranPhase.load_from_checkpoint("./model/TPDM/lightning_logs/version_0/checkpoints/last.ckpt")
transphase_model = transphase_model.cuda()

diffphase_model = DiffPhase.load_from_checkpoint("./model/SPDM/lightning_logs/version_0/checkpoints/last.ckpt")
diffphase_model = diffphase_model.cuda()


def phase_mix_parameterization(phase_mix_id, step):
    # step from 1000 to 0
    all_code = [0, 0.5, 1, 1-(step/1000), (step/1000), \
                (step/1000)**3, 1-(1-(step/1000))**3,\
                # (1-(step/1000))**3, 1-(step/1000)**3,\
                (step/1000)**2, (step/1000)**4]
    assert len(all_code) == len(phase_mix_all)
    return all_code[phase_mix_id]

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

def motion_generation(texts, timings):
    time_boundary = np.cumsum([0] + list(timings))

    ### generate meta (progress)
    # meta for semantics
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
        progress = pad_sequence(progress, batch_first=True).float()
        masks = torch.zeros(
            (progress.shape[0], progress.shape[1]),
            device=progress.device,
            dtype=torch.bool,
        )
        for b in range(len(lengths)):
            masks[b, : lengths[b]] = True
        progress_all.append(progress)
        masks_all.append(masks)

    semantics_progress, transitioning_progress = progress_all
    masks_semantics, masks_transitioning = masks_all


    semantics_progress, transitioning_progress = (
        semantics_progress.cpu(),
        transitioning_progress.cpu(),
    )
    masks_semantics, masks_transitioning = (
        masks_semantics.cpu(),
        masks_transitioning.cpu(),
    )

    ### encode text
    cond_texts = encode_text(clip_model, texts, "cuda")
    cond_texts = cond_texts / cond_texts.norm(dim=-1, keepdim=True)
    cond_texts = diffphase_model.condlinear(cond_texts).cpu()

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

        code_semantics_0T = []
        for b in range(0,code_semantics_pred.shape[0],batch_size):
            # SPDM
            code_semantics_0T_b = diffphase_model.diffusion_step(
                t,
                code_semantics_pred[b:b+batch_size].cuda(),
                masks_semantics[b:b+batch_size].cuda(),
                semantics_progress[b:b+batch_size].cuda(),
                cond_texts.unsqueeze(1)[b:b+batch_size].cuda(),
            )
            code_semantics_0T.append(code_semantics_0T_b.detach().cpu())
        code_semantics_0T = torch.cat(code_semantics_0T, dim=0)

        # TPDM
        code_left_0R = []
        for b in range(0,code_left_pred.shape[0],batch_size):
            code_left_0R_b = transphase_model.diffusion_step(
                t,
                code_left_pred[b:b+batch_size].cuda(),
                masks_left[b:b+batch_size].cuda(),
                left_progress[b:b+batch_size].cuda(),
                code_trans_pred0[b:b+batch_size].cuda(),
                masks_transitioning[b:b+batch_size].cuda(),
                transitioning_progress[b:b+batch_size].cuda(),
                decode_left=True,
                decode_trans=False,
            )
            code_left_0R.append(code_left_0R_b.detach().cpu())
        code_left_0R = torch.cat(code_left_0R, dim=0)

        
        code_right_0L = []
        for b in range(0,code_right_pred.shape[0],batch_size):
            code_right_0L_b = transphase_model.diffusion_step(
                t,
                code_right_pred[b:b+batch_size].cuda(),
                masks_right[b:b+batch_size].cuda(),
                right_progress[b:b+batch_size].cuda(),
                code_trans_pred0[b:b+batch_size].cuda(),
                masks_transitioning[b:b+batch_size].cuda(),
                transitioning_progress[b:b+batch_size].cuda(),
                decode_left=False,
                decode_trans=False,
            )
            code_right_0L.append(code_right_0L_b.detach().cpu())
        code_right_0L = torch.cat(code_right_0L, dim=0)

        
        code_trans_0R = []
        for b in range(0,code_trans_pred.shape[0],batch_size):
            code_trans_0R_b = transphase_model.diffusion_step(
                t,
                code_trans_pred[b:b+batch_size].cuda(),
                masks_transitioning[b:b+batch_size].cuda(),
                transitioning_progress[b:b+batch_size].cuda(),
                code_right_pred0[b:b+batch_size].cuda(),
                masks_right[b:b+batch_size].cuda(),
                right_progress[b:b+batch_size].cuda(),
                decode_left=False,
                decode_trans=True,
            )
            code_trans_0R.append(code_trans_0R_b.detach().cpu())
        code_trans_0R = torch.cat(code_trans_0R, dim=0)


        code_trans_0L = []
        for b in range(0,code_trans_pred.shape[0],batch_size):
            code_trans_0L_b = transphase_model.diffusion_step(
                t,
                code_trans_pred[b:b+batch_size].cuda(),
                masks_transitioning[b:b+batch_size].cuda(),
                transitioning_progress[b:b+batch_size].cuda(),
                code_left_pred0[b:b+batch_size].cuda(),
                masks_left[b:b+batch_size].cuda(),
                left_progress[b:b+batch_size].cuda(),
                decode_left=True,
                decode_trans=True,
            )
            code_trans_0L.append(code_trans_0L_b.detach().cpu())
        code_trans_0L = torch.cat(code_trans_0L, dim=0)


        ### phase mixing
        # semantic segments
        timestep_scale = (i / 1000) ** 3  # see paper
        trans_scale, text_scale = timestep_scale, 1 - timestep_scale
        code_left_0 = (
                                code_semantics_0T[0] * text_scale + code_left_0R[0] * trans_scale
                        ) / (
                                text_scale + trans_scale
                        )  # leftmost
        code_right_0 = (
                                code_semantics_0T[-1] * text_scale + code_right_0L[-1] * trans_scale
                        ) / (
                                text_scale + trans_scale
                        )  # rightmost
        code_inb_0 = (
                                code_semantics_0T[1:-1] * text_scale
                                + (code_left_0R[1:] + code_right_0L[:-1]) / 2 * trans_scale
                        ) / (text_scale + trans_scale)
        code_semantics_0 = torch.cat(
            [code_left_0.unsqueeze(0), code_inb_0, code_right_0.unsqueeze(0)], dim=0
        )

        # transitioning segment
        timestep_scale = 1  # see paper
        code_trans_0 = (code_trans_0L + code_trans_0R) / 2

        code_semantics_0, code_trans_0 = (
            code_semantics_0.detach(),
            code_trans_0.detach(),
        )

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
    rec_semantics = []
    for b in range(0,code_semantics_pred.shape[0],batch_size):
        rep_semantics_0_b = diffphase_model.PAE.latent_reparam(
            code_semantics_pred[b:b+batch_size].cuda(), semantics_progress[b:b+batch_size].cuda()
        )
        rec_semantics_b = diffphase_model.PAE.decode(
            rep_semantics_0_b, masks_semantics[b:b+batch_size].cuda(), semantics_progress[b:b+batch_size].cuda()
        ).detach().cpu()
        rec_semantics.append(rec_semantics_b.detach().cpu())
    rec_semantics = torch.cat(rec_semantics, dim=0)

    rec_trans = []
    for b in range(0,code_trans_pred.shape[0],batch_size):
        rep_trans_0_b = diffphase_model.PAE.latent_reparam(
            code_trans_pred[b:b+batch_size].cuda(), transitioning_progress[b:b+batch_size].cuda()
        )

        rec_trans_b = diffphase_model.PAE.decode(
            rep_trans_0_b, masks_transitioning[b:b+batch_size].cuda(), transitioning_progress[b:b+batch_size].cuda()
        ).detach().cpu()
        rec_trans.append(rec_trans_b.detach().cpu())
    rec_trans = torch.cat(rec_trans, dim=0)

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
        # print(trans_i.shape, full_motion.shape, progress_i.shape)
        full_motion[start:end] = full_motion[start:end] * weighting + trans_i * (1 - weighting)

    full_motion = full_motion.detach().cpu()
    full_motion = torch.matmul(
        full_motion, diffphase_model.gmd_proj_inv.to(full_motion.device)
    )

    return full_motion


########################################################################################################
if __name__ == "__main__":

    output_path = Path(f"./evaluation/generated_repo/eval_t2m_long_concat_piecewise")
    output_path.mkdir(parents=True, exist_ok=True)

    with open('evaluation/evaluation_data.pkl', 'rb') as file:  
        save_pkl = pickle.load(file)

    clip_model = load_and_freeze_clip(device='cuda')

    keys_all, poses_all, trans_all, time_all, random_text_all, orig_text_pair_all =\
        save_pkl["keys"], save_pkl["poses"], save_pkl["trans"], save_pkl["time"], save_pkl["random_text"], save_pkl["orig_text_pair"]

    processed_data = list(zip(*[keys_all, poses_all, trans_all, time_all, random_text_all, orig_text_pair_all]))

    print("num:", len(processed_data))

    ### eval script
    text_batch = []
    time_batch = []
    keys_batch = []
    sum_length_list = []
    for index in range(len(processed_data)):
        # text is ignored
        keys, _, _, [pl, pe, se], _, texts = processed_data[index]
        text_batch = text_batch + list(texts)
        time_batch = time_batch + [int(pe-pl), int(se-pe)]
        keys_batch.append(keys)
        
    print(keys_batch)
    print(text_batch)
    print(time_batch)
    print(sum(time_batch))
    sum_length_list.append(sum(time_batch))
    time_boundary = np.cumsum([0] + list(time_batch))

    ### blend motion by modifying motion between pm to sm
    ### make sure pl-pm and sm-se is preserved
    full_motion = motion_generation(text_batch, time_batch)
    full_motion = torch.matmul(full_motion, diffphase_model.gmd_proj_inv.to(full_motion.device))
    full_motion = full_motion.detach().cpu()
    

    # save as full motion
    traj, mot = motion_to_smpl(full_motion)
    save_motion(output_path/f"full_motion_{str(time_boundary[-1])}.npz", traj, mot.reshape((-1,22*3)))

    # save as sliced motion pair for evaluation
    for i in range(len(keys_batch)):
        start, end = time_boundary[i], time_boundary[i + 2]
        key = keys_batch[i]
        motion_i = full_motion[start:end]
        traj, mot = motion_to_smpl(motion_i)
        save_motion(output_path/f"{key}.npz", traj, mot.reshape((-1,22*3)))


   