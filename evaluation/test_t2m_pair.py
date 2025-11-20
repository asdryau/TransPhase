import numpy as np
import scipy as sp
import torch
import pickle
from utils.rotation_conversion import *
from utils.algo import *
from pathlib import Path
import clip

from model.SPDM.diffusion import DiffPhase
from model.TPDM.diffusion import TranPhase

torch.set_default_tensor_type('torch.cuda.FloatTensor')

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

def motion_generation(text0, text1, time):

    ### blend motion by modifying motion between pm to sm
    ### make sure pl-pm and sm-se is preserved
    pl, pm, pe, sm, se = time
    
    left_progress, right_progress, trans_progress = dur2progress(int(pe-pl), int(se-pe))
    masks_left = torch.ones_like(left_progress, dtype=torch.bool)
    masks_right = torch.ones_like(right_progress, dtype=torch.bool) 
    masks_trans = torch.ones_like(trans_progress, dtype=torch.bool)
    # batchify progress and mask
    left_progress, right_progress, trans_progress = left_progress.unsqueeze(0), right_progress.unsqueeze(0), trans_progress.unsqueeze(0)
    masks_left, masks_right, masks_trans = masks_left.unsqueeze(0), masks_right.unsqueeze(0), masks_trans.unsqueeze(0)

    cond_texts0 = encode_text(clip_model, [text0], 'cuda')    #, device=code_trans_pred.device .to(code_trans_pred)
    cond_texts1 = encode_text(clip_model, [text1], 'cuda')
    cond_texts = torch.cat([cond_texts0,cond_texts1], dim=0)
    cond_texts = cond_texts / cond_texts.norm(dim=-1, keepdim=True)
    cond_texts = diffphase_model.condlinear(cond_texts)

    null_text = diffphase_model.empty_clip_emb.to(cond_texts)
    null_text = null_text / null_text.norm(dim=-1, keepdim=True)
    null_text = diffphase_model.condlinear(null_text)

    #####################################################
    ### denoise trans
    diffphase_model.scheduler.set_timesteps(99)
    timesteps = diffphase_model.scheduler.timesteps.to(cond_texts.device)
    code_left_pred = torch.randn((1,4,diffphase_model.latent_dim)).to(cond_texts.device) * diffphase_model.scheduler.init_noise_sigma
    code_right_pred = torch.randn((1,4,diffphase_model.latent_dim)).to(cond_texts.device) * diffphase_model.scheduler.init_noise_sigma
    code_trans_pred = torch.randn((1,4,diffphase_model.latent_dim)).to(cond_texts.device) * diffphase_model.scheduler.init_noise_sigma

    code_left_pred0 = code_left_pred
    code_right_pred0 = code_right_pred
    code_trans_pred0 = code_trans_pred

    for i in timesteps:
        t = torch.tensor([i], device=code_trans_pred.device, dtype=torch.long)
        # SPDM
        code_left_0T = diffphase_model.diffusion_step(t, code_left_pred, masks_left, left_progress, cond_texts[0:1].unsqueeze(0))
        code_right_0T = diffphase_model.diffusion_step(t, code_right_pred, masks_right, right_progress, cond_texts[1:2].unsqueeze(0))
        code_trans_0T = diffphase_model.diffusion_step(t, code_trans_pred, masks_trans, trans_progress, null_text[0:1].unsqueeze(0))
        # TPDM
        code_left_0R = transphase_model.diffusion_step(t, code_left_pred, masks_left, left_progress, code_trans_pred0, masks_trans, trans_progress, decode_left=True, decode_trans=False)
        code_right_0L = transphase_model.diffusion_step(t, code_right_pred, masks_right, right_progress, code_trans_pred0, masks_trans, trans_progress, decode_left=False, decode_trans=False)

        code_trans_0R = transphase_model.diffusion_step(t, code_trans_pred, masks_trans, trans_progress, code_right_pred0, masks_right, right_progress, decode_left=False, decode_trans=True)
        code_trans_0L = transphase_model.diffusion_step(t, code_trans_pred, masks_trans, trans_progress, code_left_pred0, masks_left, left_progress, decode_left=True, decode_trans=True)
        
        ### phase mixing
            # semantic segments
        timestep_scale = phase_mix_parameterization(phase_mix_semantics_id, i)
        trans_scale, text_scale = timestep_scale, 1-timestep_scale
        code_left_0 = (code_left_0T * text_scale + code_left_0R * trans_scale) / (text_scale + trans_scale)     # conditional --> blending
        code_right_0 = (code_right_0T * text_scale + code_right_0L * trans_scale) / (text_scale + trans_scale)  # conditional --> blending
            # transitioning segment
        timestep_scale = phase_mix_parameterization(phase_mix_transitions_id, i)
        trans_scale, text_scale = timestep_scale, 1-timestep_scale
        code_trans_0 = (code_trans_0T * text_scale + (code_trans_0L + code_trans_0R) / 2 * trans_scale) / (text_scale + trans_scale)

        code_left_0, code_right_0, code_trans_0 = code_left_0.detach(), code_right_0.detach(), code_trans_0.detach()
        ###
        # tidy up for next loop (including add noise)
        code_left_pred0 = diffphase_model.scheduler.step(code_left_0, i, code_left_pred).pred_original_sample
        code_right_pred0 = diffphase_model.scheduler.step(code_right_0, i, code_right_pred).pred_original_sample
        code_trans_pred0 = diffphase_model.scheduler.step(code_trans_0, i, code_trans_pred).pred_original_sample

        code_left_pred = diffphase_model.scheduler.step(code_left_0, i, code_left_pred).prev_sample
        code_right_pred = diffphase_model.scheduler.step(code_right_0, i, code_right_pred).prev_sample
        code_trans_pred = diffphase_model.scheduler.step(code_trans_0, i, code_trans_pred).prev_sample

    #####################################
    length_left = masks_left[0].sum()
    length_right = masks_right[0].sum()
    length_trans = masks_trans[0].sum()
    left_half = (length_left/2).long()
    # right_half = (length_right/2).long()
    trans_weight = trans_progress[0,:length_trans].unsqueeze(1)
    
    rep_left_0 = diffphase_model.PAE.latent_reparam(code_left_pred, left_progress)
    rep_right_0 = diffphase_model.PAE.latent_reparam(code_right_pred, right_progress) 
    rep_trans_0 = diffphase_model.PAE.latent_reparam(code_trans_pred, trans_progress)

    rec_left = diffphase_model.PAE.decode(rep_left_0, masks_left, left_progress)[0].detach()
    rec_right = diffphase_model.PAE.decode(rep_right_0, masks_right, right_progress)[0].detach()
    rec_trans = diffphase_model.PAE.decode(rep_trans_0, masks_trans, trans_progress)[0].detach()

    return rec_left, rec_right, rec_trans

def combine_t2m(rec_left, rec_trans, rec_right, time):
    ### blend motion by modifying motion between pm to sm
    ### make sure pl-pm and sm-se is preserved
    pl, pm, pe, sm, se = time
    
    left_progress, right_progress, trans_progress = dur2progress(int(pe-pl), int(se-pe))
    left_progress, right_progress, trans_progress = left_progress.unsqueeze(0), right_progress.unsqueeze(0), trans_progress.unsqueeze(0)

    length_left = left_progress.shape[1]
    length_right = right_progress.shape[1]
    length_trans = trans_progress.shape[1]
    
    left_half = int(length_left/2)
    right_half = int(length_right/2)
    
    result = torch.cat([rec_left, rec_right], dim=0)

    alpha_left = torch.tensor(torch.linspace(1,0,left_half+1)[:-1]).unsqueeze(-1)
    alpha_right = torch.tensor(torch.linspace(0,1,right_half)).unsqueeze(-1)
    alpha = torch.cat([alpha_left, alpha_right], dim=0)

    alpha = torch.nn.functional.pad(alpha, (0,0,sm-pm-alpha.shape[0],0), value=1)[:sm-pm]
    result[pm:sm] = result[pm:sm]*alpha + rec_trans*(1-alpha)
    return result

if __name__ == "__main__":
    phase_mix_semantics_id = 5
    phase_mix_transitions_id = 2

    ########################################################################################################
    output_path = Path(f"./evaluation/generated_repo/eval_t2m_piecewise-{phase_mix_all[phase_mix_semantics_id]}-{phase_mix_all[phase_mix_transitions_id]}_output")
    output_path.mkdir(parents=True, exist_ok=True)

    with open('evaluation/evaluation_data.pkl', 'rb') as file:  
        save_pkl = pickle.load(file)

    clip_model = load_and_freeze_clip(device='cuda')

    keys_all, poses_all, trans_all, time_all, random_text_all, orig_text_pair_all =\
        save_pkl["keys"], save_pkl["poses"], save_pkl["trans"], save_pkl["time"], save_pkl["random_text"], save_pkl["orig_text_pair"]

    processed_data = list(zip(*[keys_all, poses_all, trans_all, time_all, random_text_all, orig_text_pair_all]))
        
    print(output_path)
    print("num:", len(processed_data))
    ### eval script
    for k in processed_data:
        # text is ignored
        keys, _, _, [pl, pe, se], _, texts = k

        ### define motion segment
            # poses[pl:pe] is preceding motion
            # poses[pe:se] is succeeding motion (note that pe==sl)

        sl = pe
        pm = int((pl+pe)/2)
        sm = int((sl+se)/2)
        print(keys, "||", pl, pm, pe, "|", sl, sm, se, "||", texts)

        ### blend motion by modifying motion between pm to sm
        ### make sure pl-pm and sm-se is preserved
        rec_left, rec_right, rec_trans = motion_generation(texts[0], texts[1], [pl, pm, pe, sm, se])

        # convert back to smpl
        rec_left = torch.matmul(rec_left, diffphase_model.gmd_proj_inv.to(rec_left.device))
        rec_right = torch.matmul(rec_right, diffphase_model.gmd_proj_inv.to(rec_right.device))
        rec_trans = torch.matmul(rec_trans, diffphase_model.gmd_proj_inv.to(rec_trans.device))

        # tidy up motion
        rec_left = rec_left[:pe-pl].detach()
        rec_trans = rec_trans[:sm-pm].detach()
        rec_right = rec_right[:se-sl].detach()

        # combine motions
        result = combine_t2m(rec_left, rec_trans, rec_right, [pl, pm, pe, sm, se])
        traj, mot = motion_to_smpl(result.detach().cpu())
        save_motion(output_path/f"{keys}.npz", traj, mot.reshape((-1,22*3)))

        




    
    

