import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pathlib import Path

from utils.algo import *
from utils.rotation_conversion import *
import math
from diffusers import DDIMScheduler, DDPMScheduler

from model.PAE.model import MotionPAE

# loss_ce = nn.CrossEntropyLoss()
loss_mse = nn.MSELoss()
loss_l1 = nn.L1Loss()
# cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
loss_bce = nn.BCEWithLogitsLoss(reduction='none')
motion_features_all = []
styles_all = []
actions_all = []
motion_count = 0

### reference: MLD
def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, channel: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()

        self.linear_1 = nn.Linear(channel, time_embed_dim)
        self.act = None
        if act_fn == "silu":
            self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)
        return sample

class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb

class DiffPhase(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.num_train_timesteps = 1000
        self.num_inference_timesteps = 1000
        self.guidance_scale = 3.
        self.save_hyperparameters()

        # pretrained PAE
        PAE = MotionPAE.load_from_checkpoint("./model/PAE/lightning_logs/version_0/checkpoints/last.ckpt")

        # load pretrained autoencoder
        PAE = PAE.eval()
        for p in PAE.parameters():
            p.requires_grad = False
        self.PAE = PAE

        # model setting
        self.input_dim = PAE.input_dim
        self.latent_dim = PAE.latent_dim

        # GMD emphasis projection
        self.gmd_proj, self.gmd_proj_inv = PAE.gmd_proj, PAE.gmd_proj_inv

        self.enc_query = nn.Embedding(2+4, self.latent_dim)
        self.sequence_pos_encoder = PAE.sequence_pos_encoder#(self.latent_dim, 0.1)

        ###
        #    modules
        ###
        self.condlinear = nn.Linear(512, self.latent_dim)
        self.scheduler = DDIMScheduler(num_train_timesteps=self.num_train_timesteps, beta_start=0.00085, beta_end=0.012, \
                                       beta_schedule='scaled_linear', clip_sample=False, set_alpha_to_one=False, steps_offset=1,\
                                        prediction_type='epsilon')
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timesteps, beta_start=0.00085, beta_end=0.012, \
                                       beta_schedule='scaled_linear', variance_type='fixed_small', clip_sample=False,\
                                        prediction_type='epsilon')

        self.time_proj = Timesteps(512, True, 0)
        self.time_embedding = TimestepEmbedding(512, self.latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.denoiser = nn.TransformerEncoder(encoder_layer, num_layers=8)



        ## load clip_emb of "" for classifier free guidance
        if Path("data/label_clip_emb_BABELteach.npz").exists():
            data = np.load("data/label_clip_emb_BABELteach.npz")
            self.empty_clip_emb = torch.tensor(data["empty_clip_emb"]).float()



    def configure_optimizers(self):
        opt_0 = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return opt_0

    def loss_rec(self, motion_rec, motion_rot, len_mask):
        # (B,T=?,F=24*6)
        B,T,F = motion_rec.shape
        len_mask = len_mask.float()

        # mask
        motion_rec = motion_rec * len_mask.unsqueeze(-1)
        motion_rot = motion_rot * len_mask.unsqueeze(-1)
        # reshape
        motion_rec = motion_rec.reshape((B*T,-1))            #(B*T,J*6)
        motion_rot = motion_rot.reshape((B*T,-1))            #(B*T,J*6)
        len_mean = len_mask.reshape((-1)).mean()    #(B*T,)

        ### calc loss
        # recon loss on rot
        loss_rot = loss_l1(motion_rec, motion_rot) / len_mean
        return loss_rot

    def denoise(self, code_t, rep_t, len_mask, progress, time_emb, cond_emb):
        B = rep_t.shape[0]
        queries = torch.arange(6, device=rep_t.device).unsqueeze(0).long()    #(1,11)
        queries = queries.expand((B,6))                            #(B*11,)
        queries = self.enc_query(queries)

        queries[:,0:1] = queries[:,0:1] + time_emb
        queries[:,1:2] = queries[:,1:2] + cond_emb
        queries[:,2:6] = queries[:,2:6] + code_t

        token_mask = torch.ones((B, 6), dtype=bool, device=rep_t.device)

        rep_t = self.sequence_pos_encoder(rep_t, progress)
        # combine learned token to motion
        token_mask = torch.cat((token_mask, len_mask), dim=1)
        motion_seq = torch.cat((queries, rep_t), dim=1)    #(B,T+3,E)
        motion_seq = self.denoiser(src=motion_seq, src_key_padding_mask=~token_mask)

        latents = motion_seq[:,2:6]
        return latents               #(B,4,E)

    def diffusion_step(self, t, code_t, masks, progress, raw_clip_actions):
            # time emb and text emb
        time_emb = self.time_proj(t).type_as(code_t)
        time_emb = self.time_embedding(time_emb).unsqueeze(1)               #(B,1,E)

        ### diffusion training
            # expand to latent sine wave
        rep_t = self.PAE.latent_reparam(code_t, progress)   #(B,T1,E)
        code_0 = self.denoise(code_t, rep_t, masks, progress, time_emb, raw_clip_actions)

        return code_0


    def training_step(self, train_batch, batch_idx):
        mode = 'train'
        motion_left, masks_left, progress_left,\
        motion_right, masks_right, progress_right,\
        motion_trans, masks_trans, progress_trans,\
        raw_clip_actions = train_batch

        #(B,T,24,6), (B,T,3), (B,T), (B,T)
        motion_left, masks_left, progress_left = motion_left.detach(), masks_left.detach(), progress_left.detach()
        motion_right, masks_right, progress_right = motion_right.detach(), masks_right.detach(),progress_right.detach()
        motion_trans, masks_trans, progress_trans = motion_trans.detach(), masks_trans.detach(),progress_trans.detach()
        raw_clip_actions = raw_clip_actions.detach()    #(B,2,512)

        B = motion_left.shape[0]
        ### !!! GMD proj
        motion_left = torch.matmul(motion_left[:,:,:self.input_dim], self.gmd_proj.to(motion_left.device))
        motion_right = torch.matmul(motion_right[:,:,:self.input_dim], self.gmd_proj.to(motion_right.device))
        motion_trans = torch.matmul(motion_trans[:,:,:self.input_dim], self.gmd_proj.to(motion_trans.device))

        ### classifier free guidance: 10% random mask on raw_clip_actions
        rand_mask = (torch.rand(B, 2, device=motion_left.device) > 0.1).float()
        empty_clip_emb = self.empty_clip_emb.unsqueeze(0).expand(B,2,-1).to(motion_left)
        raw_clip_actions = raw_clip_actions * rand_mask.unsqueeze(-1) + empty_clip_emb * (1-rand_mask.unsqueeze(-1))

        ### encode motion
        code_left = self.PAE.encode(motion_left, masks_left, progress_left).detach()
        code_right = self.PAE.encode(motion_right, masks_right, progress_right).detach()
        # code_trans = self.PAE.encode(motion_trans, masks_trans, progress_trans).detach()

        # encoder condition
        raw_clip_actions = self.condlinear(raw_clip_actions)    #(B,2,E)

        ############
        ###    diffusion preparation
        ############

        # timestep
        t = torch.randint(0, self.num_train_timesteps, (B, ), device=code_left.device, dtype=torch.long)

            # code1 (left)
        code_left_noise = torch.randn_like(code_left)
        code_left_t = self.noise_scheduler.add_noise(code_left.clone(), code_left_noise, t)
            # code2 (right)
        code_right_noise = torch.randn_like(code_right)
        code_right_t = self.noise_scheduler.add_noise(code_right.clone(), code_right_noise, t)

        ############
        ###    diffusion step
        ############

        code_left_0 = self.diffusion_step(t, code_left_t, masks_left, progress_left, raw_clip_actions[:,0:1])
        code_right_0 = self.diffusion_step(t, code_right_t, masks_right, progress_right, raw_clip_actions[:,1:2])

        ### total loss
        loss_code_left = loss_l1(code_left_0, code_left_noise)
        loss_code_right = loss_l1(code_right_0, code_right_noise)


        loss = loss_code_left + loss_code_right
        self.log(f"{mode}/loss_code_left", loss_code_left)
        self.log(f"{mode}/loss_code_right", loss_code_right)
        self.log(f"{mode}/loss", loss)
        return loss


    def validation_step(self, train_batch, batch_idx):
        mode = 'valid'
        motion_left, masks_left, progress_left,\
        motion_right, masks_right, progress_right,\
        motion_trans, masks_trans, progress_trans,\
        raw_clip_actions = train_batch

        B = motion_left.shape[0]
        ### !!! GMD proj
        motion_left = torch.matmul(motion_left[:,:,:self.input_dim], self.gmd_proj.to(motion_left.device))
        motion_right = torch.matmul(motion_right[:,:,:self.input_dim], self.gmd_proj.to(motion_right.device))
        motion_trans = torch.matmul(motion_trans[:,:,:self.input_dim], self.gmd_proj.to(motion_trans.device))

        ### classifier free guidance: 10% random mask on raw_clip_actions
        rand_mask = (torch.rand(B, 2, device=motion_left.device) > 0.1).float()
        empty_clip_emb = self.empty_clip_emb.unsqueeze(0).expand(B,2,-1).to(motion_left)
        raw_clip_actions = raw_clip_actions * rand_mask.unsqueeze(-1) + empty_clip_emb * (1-rand_mask.unsqueeze(-1))

        ### encode motion
        code_left = self.PAE.encode(motion_left, masks_left, progress_left).detach()
        code_right = self.PAE.encode(motion_right, masks_right, progress_right).detach()
        # code_trans = self.PAE.encode(motion_trans, masks_trans, progress_trans).detach()

        # encoder condition
        raw_clip_actions = self.condlinear(raw_clip_actions)    #(B,2,E)

        ############
        ###    diffusion preparation
        ############

        # timestep
        t = torch.randint(0, self.num_train_timesteps, (B, ), device=code_left.device, dtype=torch.long)

            # code1 (left)
        code_left_noise = torch.randn_like(code_left)
        code_left_t = self.noise_scheduler.add_noise(code_left.clone(), code_left_noise, t)
            # code2 (right)
        code_right_noise = torch.randn_like(code_right)
        code_right_t = self.noise_scheduler.add_noise(code_right.clone(), code_right_noise, t)

        ############
        ###    diffusion step
        ############

        code_left_0 = self.diffusion_step(t, code_left_t, masks_left, progress_left, raw_clip_actions[:,0:1])
        code_right_0 = self.diffusion_step(t, code_right_t, masks_right, progress_right, raw_clip_actions[:,1:2])

        ### total loss
        loss_code_left = loss_l1(code_left_0, code_left_noise)
        loss_code_right = loss_l1(code_right_0, code_right_noise)


        loss = loss_code_left + loss_code_right
        self.log(f"{mode}/loss_code_left", loss_code_left)
        self.log(f"{mode}/loss_code_right", loss_code_right)
        self.log(f"{mode}/loss", loss)
        return loss

    # implemented in eval script
    def test_step(self, train_batch, batch_idx):
        mode = 'test'
        pass
