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

def dur2progress_inb(left_length, right_length, blend_frame):
    blend_frame_2 = int(blend_frame / 2)
        # -1: left boundary, 0: at left_half, 1:right boundary
    left_progress_left = torch.linspace(-1,0,left_length-blend_frame_2+1)[:-1]                      # left,  include -1, exclude 0
    left_progress_right = torch.linspace(0,1,blend_frame_2)                                         # right, include 0,1
        # -1: left boundary (note that this is the last frame from left_mot), 0: at right_half, 1:right boundary
    right_progress_left = torch.linspace(-1,0,blend_frame_2+1)[:-1]                                 # left, include -1, exclude 0
    right_progress_right = torch.linspace(0,1,right_length-blend_frame_2)                           # right, include 0,1

        ## final progress tensor
    left_progress = torch.cat([left_progress_left, left_progress_right],dim=0)
    right_progress = torch.cat([right_progress_left, right_progress_right],dim=0)
    trans_progress = torch.cat([left_progress_right-1, torch.linspace(0,1,blend_frame_2+1)[1:]],dim=0)
    return left_progress, right_progress, trans_progress

def motion_inbetweening(motion_left, motion_right, time, text):
    ### motion inbetweening by modifying motion between pm to sm
    ### make sure pl-pm and sm-se is preserved
    pl, pm, blend_start, pe, blend_end, sm, se = time
    
    ##########
    #   motion encoding (on the portion with valid frames)
    ##########
    motion_left = motion_left.unsqueeze(0)
    motion_right = motion_right.unsqueeze(0)

    left_progress, right_progress, _ = dur2progress(int(blend_start-pl), int(se-blend_end))
    masks_left = torch.ones_like(left_progress, dtype=torch.bool)
    masks_right = torch.ones_like(right_progress, dtype=torch.bool) 
    left_progress, right_progress = left_progress.unsqueeze(0).to(motion_left.device), right_progress.unsqueeze(0).to(motion_left.device)
    masks_left, masks_right = masks_left.unsqueeze(0).to(motion_right.device), masks_right.unsqueeze(0).to(motion_right.device)

    code_left = transphase_model.PAE.encode(motion_left, masks_left, left_progress)
    code_right = transphase_model.PAE.encode(motion_right, masks_right, right_progress)

    ##########
    #   motion in-betweening region
    ##########
        ### initialize mask on (trans1, inbetweening, trans2)
    trans1_progress, trans2_progress, inb_progress = dur2progress_inb(int(pe-pm), int(sm-pe), blend_frame)
    masks_trans1 = torch.ones_like(trans1_progress, dtype=torch.bool)
    masks_trans2 = torch.ones_like(trans2_progress, dtype=torch.bool) 
    masks_inb = torch.ones_like(inb_progress, dtype=torch.bool)
    trans1_progress, trans2_progress, inb_progress = trans1_progress.unsqueeze(0), trans2_progress.unsqueeze(0), inb_progress.unsqueeze(0)
    masks_trans1, masks_trans2, masks_inb = masks_trans1.unsqueeze(0), masks_trans2.unsqueeze(0), masks_inb.unsqueeze(0)
    trans1_progress, trans2_progress, inb_progress = trans1_progress.to(motion_left.device), trans2_progress.to(motion_left.device), inb_progress.to(motion_left.device)
    masks_trans1, masks_trans2, masks_inb = masks_trans1.to(motion_left.device), masks_trans2.to(motion_left.device), masks_inb.to(motion_left.device)

    # print("progress", trans1_progress.shape, trans2_progress.shape, inb_progress.shape)
    #####################################################
    ### denoise trans
    diffphase_model.scheduler.set_timesteps(99)
    timesteps = diffphase_model.scheduler.timesteps.to(code_left.device)

    code_trans1_pred = torch.randn_like(code_left).to(code_left.device) * diffphase_model.scheduler.init_noise_sigma
    code_trans2_pred = torch.randn_like(code_left).to(code_left.device) * diffphase_model.scheduler.init_noise_sigma
    code_inb_pred = torch.randn_like(code_left).to(code_left.device) * diffphase_model.scheduler.init_noise_sigma

    code_trans1_pred0 = code_trans1_pred
    code_trans2_pred0 = code_trans2_pred
    code_inb_pred0 = code_inb_pred
    
    null_text = diffphase_model.empty_clip_emb.unsqueeze(0).to(code_inb_pred)
    null_text = null_text / null_text.norm(dim=-1, keepdim=True)
    null_text = diffphase_model.condlinear(null_text)

    if text is not None:
        cond_text = encode_text(clip_model, [text], device=code_inb_pred.device).to(code_inb_pred)
        cond_text = cond_text / cond_text.norm(dim=-1, keepdim=True)
        cond_text = diffphase_model.condlinear(cond_text)
    else:
        cond_text = null_text


    for i in timesteps:
        if text is not None:
            semantics_timestep_scale = (i / 1000) ** 3
        else:
            semantics_timestep_scale = 1

        t = torch.tensor([i], device=code_inb_pred.device, dtype=torch.long)

        # SPDM
        code_inb_0T = diffphase_model.diffusion_step(t, code_inb_pred, masks_inb, inb_progress, cond_text)
        code_trans1_0T = diffphase_model.diffusion_step(t, code_trans1_pred, masks_trans1, trans1_progress, null_text)
        code_trans2_0T = diffphase_model.diffusion_step(t, code_trans2_pred, masks_trans2, trans2_progress, null_text)

        # TPDM
            # use preceding motion to predict transitioning phase
        code_trans1_0L = transphase_model.diffusion_step(t, code_trans1_pred, masks_trans1, trans1_progress, code_left, masks_left, left_progress, decode_left=True, decode_trans=True)
            # use succeeding motion to predict transitioning phase
        code_trans1_0R = transphase_model.diffusion_step(t, code_trans1_pred, masks_trans1, trans1_progress, code_inb_pred0, masks_inb, inb_progress, decode_left=False, decode_trans=True)

            # use transitioning motion to predict succeeding phase
        code_inb_0L = transphase_model.diffusion_step(t, code_inb_pred, masks_inb, inb_progress, code_trans1_pred0, masks_trans1, trans1_progress, decode_left=False, decode_trans=False)
            # use transitioning motion to predict preceding phase
        code_inb_0R = transphase_model.diffusion_step(t, code_inb_pred, masks_inb, inb_progress, code_trans2_pred0, masks_trans2, trans2_progress, decode_left=True, decode_trans=False)

            # use preceding motion to predict transitioning phase
        code_trans2_0L = transphase_model.diffusion_step(t, code_trans2_pred, masks_trans2, trans2_progress, code_inb_pred0, masks_inb, inb_progress, decode_left=True, decode_trans=True)
            # use succeeding motion to predict transitioning phase
        code_trans2_0R = transphase_model.diffusion_step(t, code_trans2_pred, masks_trans2, trans2_progress, code_right, masks_right, right_progress , decode_left=False, decode_trans=True)
        
        # phase mixing
            # inbetweening segment
        trans_scale, text_scale = semantics_timestep_scale, 1-semantics_timestep_scale
        code_inb_0 = (code_inb_0T * text_scale + (code_inb_0L + code_inb_0R) / 2 * trans_scale) / (text_scale + trans_scale)

            # transitioning segments
        timestep_scale = 1
        trans_scale, text_scale = timestep_scale, 1-timestep_scale
        code_trans1_0 = (code_trans1_0T * text_scale + (code_trans1_0L + code_trans1_0R) / 2 * trans_scale) / (text_scale + trans_scale)
        code_trans2_0 = (code_trans2_0T * text_scale + (code_trans2_0L + code_trans2_0R) / 2 * trans_scale) / (text_scale + trans_scale)
        
        code_inb_0, code_trans1_0, code_trans2_0 = code_inb_0.detach(), code_trans1_0.detach(), code_trans2_0.detach()
        #=====================================================
        ###
        # tidy up for next loop (including add noise)
        code_inb_pred0 = diffphase_model.scheduler.step(code_inb_0, i, code_inb_pred).pred_original_sample
        code_trans1_pred0 = diffphase_model.scheduler.step(code_trans1_0, i, code_trans1_pred).pred_original_sample
        code_trans2_pred0 = diffphase_model.scheduler.step(code_trans2_0, i, code_trans2_pred).pred_original_sample
        
        code_inb_pred = diffphase_model.scheduler.step(code_inb_0, i, code_inb_pred).prev_sample
        code_trans1_pred = diffphase_model.scheduler.step(code_trans1_0, i, code_trans1_pred).prev_sample
        code_trans2_pred = diffphase_model.scheduler.step(code_trans2_0, i, code_trans2_pred).prev_sample

    ###
    rep_inb_0 = diffphase_model.PAE.latent_reparam(code_inb_pred, inb_progress)
    rep_trans1_0 = diffphase_model.PAE.latent_reparam(code_trans1_pred, trans1_progress)
    rep_trans2_0 = diffphase_model.PAE.latent_reparam(code_trans2_pred, trans2_progress)

    rec_inb = diffphase_model.PAE.decode(rep_inb_0, masks_inb, inb_progress)[0].detach()
    rec_trans1 = diffphase_model.PAE.decode(rep_trans1_0, masks_trans1, trans1_progress)[0].detach()
    rec_trans2 = diffphase_model.PAE.decode(rep_trans2_0, masks_trans2, trans2_progress)[0].detach()
    return rec_trans1, rec_inb, rec_trans2

def combine_inbetweening(motion_left, rec_trans1, rec_inb, rec_trans2, motion_right, timing):
    pl, pm, blend_start, pe, blend_end, sm, se = timing
    sl = pe 
    
    left_progress, right_progress, _ = dur2progress(int(blend_start-pl), int(se-blend_end))
    trans1_progress, trans2_progress, inb_progress = dur2progress_inb(int(pe-pm), int(sm-pe), blend_frame)
    left_progress, right_progress, trans1_progress, trans2_progress, inb_progress \
        = left_progress.unsqueeze(0), right_progress.unsqueeze(0), trans1_progress.unsqueeze(0), trans2_progress.unsqueeze(0), inb_progress.unsqueeze(0)
    left_progress, right_progress, trans1_progress, trans2_progress, inb_progress \
        = left_progress.to(left_motion.device), right_progress.to(left_motion.device), trans1_progress.to(left_motion.device),\
          trans2_progress.to(left_motion.device), inb_progress.to(left_motion.device)

    length_left = left_progress.shape[1]
    length_trans1 = trans1_progress.shape[1]
    length_inb = inb_progress.shape[1]
    length_trans2 = trans2_progress.shape[1]
    length_right = right_progress.shape[1]

    #   (left, trans1, inb)
    left_half = int(length_left/2)
    inb_half = int(length_inb/2)
    right_half = int(length_right/2)
    
    result = torch.cat([motion_left, rec_inb, motion_right], dim=0)
    
    ################################################################
        #####   merge the 3 pieces into one (left, trans1, inb, trans2, right)
    alpha_left = torch.tensor(torch.linspace(1,0,left_half+1)[:-1]).unsqueeze(-1)
    alpha_right = torch.tensor(torch.linspace(0,1,inb_half)).unsqueeze(-1)
    alpha1 = torch.cat([alpha_left, alpha_right], dim=0).to(left_motion.device)

    alpha_left = torch.tensor(torch.linspace(1,0,inb_half+1)[:-1]).unsqueeze(-1)
    alpha_right = torch.tensor(torch.linspace(0,1,right_half)).unsqueeze(-1)
    alpha2 = torch.cat([alpha_left, alpha_right], dim=0).to(left_motion.device)
    
  
    alpha1 = torch.nn.functional.pad(alpha1, (0,0,pe-pm-alpha1.shape[0],0), value=1)[:pe-pm]
    alpha2 = torch.nn.functional.pad(alpha2, (0,0,sm-sl-alpha2.shape[0],0), value=1)[:sm-sl]
    result[pm:pe] = result[pm:pe]*alpha1 + rec_trans1*(1-alpha1)
    result[sl:sm] = result[sl:sm]*alpha2 + rec_trans2*(1-alpha2)
    return result


########################################################################################################


if __name__ == "__main__":
    clip_model = load_and_freeze_clip(device='cuda')
    blend_frame = 120

    data = np.load("./misc/test_motion1.npz")
    trans = torch.from_numpy(data["trans"])
    poses = torch.from_numpy(data["poses"]).reshape((-1,55,3))[:,:22]
    left_motion = motion_preprocess(trans, poses).cuda()
    left_motion = torch.matmul(left_motion.detach(), diffphase_model.gmd_proj.to(left_motion.device))

    data = np.load("./misc/test_motion2.npz")
    trans = torch.from_numpy(data["trans"])
    poses = torch.from_numpy(data["poses"]).reshape((-1,55,3))[:,:22]
    right_motion = motion_preprocess(trans, poses).cuda()
    right_motion = torch.matmul(right_motion.detach(), diffphase_model.gmd_proj.to(right_motion.device))

    # semantic condition
    # UMIB: text = None / CMIB: text = "walk forward"
    text = None

    # time specification    (different from evaluation experiment that do masking, we now want to extend the motion seuqnece by the blend frame length)
        # left_motion:  [pl, pm, blend_start]
        # right_motion: [blend_end, sm, se]
    blend_start, blend_end = left_motion.shape[0], left_motion.shape[0] + blend_frame
    pl, pm = 0, int(blend_start//2)
    sm, se = blend_end + int(right_motion.shape[0]//2), blend_end + right_motion.shape[0]
    pe = sl = blend_start + int(blend_frame//2)

    rec_trans1, rec_inb, rec_trans2 = motion_inbetweening(left_motion, right_motion, [pl, pm, blend_start, pe, blend_end, sm, se], text)
    rec_trans1, rec_inb, rec_trans2 = rec_trans1.detach(), rec_inb.detach(), rec_trans2.detach()

    # convert back to smpl
    left_motion = torch.matmul(left_motion.detach(), diffphase_model.gmd_proj_inv.to(left_motion.device))
    rec_trans1 = torch.matmul(rec_trans1, diffphase_model.gmd_proj_inv.to(rec_trans1.device))
    rec_inb = torch.matmul(rec_inb, diffphase_model.gmd_proj_inv.to(rec_inb.device))
    rec_trans2 = torch.matmul(rec_trans2, diffphase_model.gmd_proj_inv.to(rec_trans2.device))
    right_motion = torch.matmul(right_motion.detach(), diffphase_model.gmd_proj_inv.to(right_motion.device))

    # tidy up motion
    rec_trans1 = rec_trans1[:pe-pm].detach()
    rec_inb = rec_inb[:blend_end-blend_start].detach()
    rec_trans2 = rec_trans2[:sm-sl].detach()

    # combine motions
    result = combine_inbetweening(left_motion, rec_trans1, rec_inb, rec_trans2, right_motion, [pl, pm, blend_start, pe, blend_end, sm, se])
    trans, poses = motion_to_smpl(result.detach().cpu())
    save_motion("output_mib.npz", trans, poses.reshape((-1, 22 * 3)))