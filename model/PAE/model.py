import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pathlib import Path

from utils.algo import *
from utils.rotation_conversion import *

# loss_ce = nn.CrossEntropyLoss()
loss_mse = nn.MSELoss()
loss_l1 = nn.L1Loss()
# cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
loss_bce = nn.BCEWithLogitsLoss(reduction='none')
motion_features_all = []
styles_all = []
actions_all = []
motion_count = 0


# reference: https://github.com/korrawe/guided-motion-diffusion/blob/2f6264a9b793333556ef911981983082a1113049/data_loaders/humanml/data/dataset.py
def init_random_projection(scale):
    if Path(f"model/rand_proj_{scale}.npy").exists():
        proj_matrix = np.load(f"model/rand_proj_{scale}.npy")
        inv_proj_matrix = np.load(f"model/inv_rand_proj_{scale}.npy")
    else:
        proj_matrix = torch.normal(mean=0, std=1.0, size=(269, 269), dtype=torch.float)  # / np.sqrt(263)
        # scale first 10 values (rot spd, x spd, z spd)
        proj_matrix[range(10), :] *= scale
        proj_matrix = proj_matrix / np.sqrt(269 - 10 + 10 * scale**2)
        inv_proj_matrix = torch.inverse(proj_matrix)

        proj_matrix = proj_matrix.detach().cpu().numpy()
        inv_proj_matrix = inv_proj_matrix.detach().cpu().numpy()
        np.save(f"model/rand_proj_{scale}.npy", proj_matrix)
        np.save(f"model/inv_rand_proj_{scale}.npy", inv_proj_matrix)

    proj_matrix_th = torch.from_numpy(proj_matrix)
    inv_proj_matrix_th = torch.from_numpy(inv_proj_matrix)

    return proj_matrix_th, inv_proj_matrix_th

# ### reference: MDM
class PositionalEncoding_3way(nn.Module):
    def __init__(self, d_model, dropout=0.1, half_len=500):
        super(PositionalEncoding_3way, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe = torch.zeros(half_len*2+1, d_model)
        # position = torch.arange(-half_len, half_len+1, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0)    #(1,T,E)
        # self.mid = half_len
        # self.register_buffer('pe', pe)
        # self.register_buffer('div_term', div_term)

    def forward(self, x, progress):
        # X: (B,T,E)
        # progress is linear from -1 at motion start to 1 at motion end (motion is variable length)
        div_term = torch.exp(torch.arange(0, x.shape[2], 4, device=x.device).float() * (-np.log(10000.0) / x.shape[2])).detach()   #(E/2)
        # calc motion length of each segment
        left_valid = (progress < 0).float()      #(B,T)
        right_valid = (progress > 0).float()
        T_left = torch.sum((progress < 0) & (progress >= -1), dim=1, keepdim=True)
        T_right = torch.sum((progress > 0) & (progress <= 1), dim=1, keepdim=True)
        # length = T_right + T_left + 1

        position = (progress * left_valid * T_left + progress * right_valid * T_right).unsqueeze(-1).detach()   #(B,T,1)

        T = progress.shape[1]
        # preceding motion extend from frame = 0
            # 0 at start frame and then linear extrepolate
        left_position = torch.arange(T, device=progress.device).float()
        left_position = left_position #/ T_left
            # succeeding motion extend from frame = length-1
            # 0 at end frame and then linear extrepolate
        right_position = torch.arange(T, device=progress.device).float()
        right_position = (right_position - (T_right+T_left)) #/ T_right

        # add to x
        x[:,:,0::4] += torch.sin(position * div_term)
        x[:,:,1::4] += torch.cos(position * div_term)
        x[:,:,2::4] += torch.sin(left_position.unsqueeze(-1) * div_term)
        x[:,:,3::4] += torch.cos(right_position.unsqueeze(-1) * div_term)
        # x = x + self.pe[:, self.mid+shift:self.mid+x.shape[1]+shift, :x.shape[2]]
        return self.dropout(x)


class MotionPAE(pl.LightningModule):
    def __init__(self, latent_dim):
        super().__init__()

        # GMD emphasis projection
        self.gmd_proj, self.gmd_proj_inv = init_random_projection(15)

        # model setting
        self.input_dim = 269 #24*6
        self.latent_dim = latent_dim

        self.save_hyperparameters()

        self.sequence_pos_encoder = PositionalEncoding_3way(self.latent_dim, 0.1)

        ###
        #   Encoder
        ###
        self.enc_query = nn.Embedding(4, latent_dim)
        self.skelEmbedding = nn.Linear(self.input_dim, self.latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)

        ###
        #   Decoder
        ###

        decoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=8)

        self.finallayer = nn.Linear(self.latent_dim, self.input_dim)


    def latent_reparam(self, latents, progress):
        f = latents[:,0:1]     #(B,1,?)
        a = latents[:,1:2]     #(B,1,?)
        b = latents[:,2:3]     #(B,1,?)
        s = latents[:,3:4]     #(B,1,?)

        ### scale progress
        left_valid = (progress < 0).float()      #(B,T)
        right_valid = (progress > 0).float()
        T_left = torch.sum(left_valid, dim=1, keepdim=True)
        T_right = torch.sum(right_valid, dim=1, keepdim=True)
        # length = T_right + T_left + 1

        progress_2 = (progress * left_valid * T_left + progress * right_valid * T_right).detach()

        ### calc F, T
        progress = progress.unsqueeze(-1)  # (B,T) --> (B,T,1)
        progress_2 = progress_2.unsqueeze(-1)

        reparam_latents_1 = a * torch.sin(f * progress + s) + b         #(B,3,?) --> (B,T,?)
        reparam_latents_2 = a * torch.sin(f * progress_2 + s) + b         #(B,3,?) --> (B,T,?)

        reparam_latents = torch.cat([reparam_latents_1[:,:,0::2], reparam_latents_2[:,:,1::2]], dim=2)
        return reparam_latents

    def encode(self, motion_rot, len_mask, progress):
        # (B,T=?,F=24*6), (B,T)
        B,T = motion_rot.shape[:2]
        # process motion
        motion_vec = self.skelEmbedding(motion_rot)
        motion_vec = self.sequence_pos_encoder(motion_vec, progress)      #(B,T,E)

        ### generate learned token for ABS extraction
        queries = torch.arange(4, device=motion_rot.device).unsqueeze(0).long()    #(1,4)
        queries = queries.expand((B,4)).reshape((B*4))                              #(B*4,)
        queries = self.enc_query(queries).reshape((B,4,-1))

        # combine learned token to motion
        motion_seq = torch.cat((queries, motion_vec), dim=1)    #(B,T+4,E)
        token_mask = torch.ones((B, 4), dtype=bool, device=len_mask.device)
        len_mask = torch.cat((token_mask, len_mask), dim=1)
        # enc
        motion_seq = self.encoder(motion_seq, src_key_padding_mask=~len_mask)

        latents = motion_seq[:,:4]
        return latents         #(B,4,E)

    def decode(self, reparam_latents, len_mask, progress):
        # (B,T,?), (B,T)
        # decoder
        reparam_latents = self.sequence_pos_encoder(reparam_latents, progress)
        output = self.decoder(reparam_latents, src_key_padding_mask=~len_mask)    #(B,T,E)
        output = self.finallayer(output)                                                        #(B,T,F=24*6)

        ### pad to fix the predicted length
        output = torch.nn.functional.pad(output, (0,0,0,len_mask.shape[1]-output.shape[1]))

        # output = torch.nn.functional.pad(output, (0,0,0,len_mask.shape[1]-output.shape[1]))
        output = output * len_mask.float().unsqueeze(-1)
        return output


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
        loss_rot = loss_mse(motion_rec, motion_rot) / len_mean
        return loss_rot


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
        motion_left = torch.matmul(motion_left, self.gmd_proj.to(motion_left.device))
        motion_right = torch.matmul(motion_right, self.gmd_proj.to(motion_right.device))
        motion_trans = torch.matmul(motion_trans, self.gmd_proj.to(motion_trans.device))


        code_left = self.encode(motion_left, masks_left, progress_left)
        rep_left = self.latent_reparam(code_left, progress_left)
        rec_left = self.decode(rep_left, masks_left, progress_left)

        code_right = self.encode(motion_right, masks_right, progress_right)
        rep_right = self.latent_reparam(code_right, progress_right)
        rec_right = self.decode(rep_right, masks_right, progress_right)

        code_trans = self.encode(motion_trans, masks_trans, progress_trans)
        rep_trans = self.latent_reparam(code_trans, progress_trans)
        rec_trans = self.decode(rep_trans, masks_trans, progress_trans)

            # recon motion
        loss_rec_left = self.loss_rec(rec_left, motion_left, masks_left)
        loss_rec_right = self.loss_rec(rec_right, motion_right, masks_right)
        loss_rec_trans = self.loss_rec(rec_trans, motion_trans, masks_trans)

        loss = loss_rec_left + loss_rec_right + loss_rec_trans
        self.log(f"{mode}/loss", loss)
        self.log(f"{mode}/loss_rec_left", loss_rec_left)
        self.log(f"{mode}/loss_rec_right", loss_rec_right)
        self.log(f"{mode}/loss_rec_trans", loss_rec_trans)
        return loss


    def validation_step(self, train_batch, batch_idx):
        mode = 'valid'
        motion_left, masks_left, progress_left,\
        motion_right, masks_right, progress_right,\
        motion_trans, masks_trans, progress_trans,\
        raw_clip_actions = train_batch

        B = motion_left.shape[0]
        ### !!! GMD proj
        motion_left = torch.matmul(motion_left, self.gmd_proj.to(motion_left.device))
        motion_right = torch.matmul(motion_right, self.gmd_proj.to(motion_right.device))
        motion_trans = torch.matmul(motion_trans, self.gmd_proj.to(motion_trans.device))


        code_left = self.encode(motion_left, masks_left, progress_left)
        rep_left = self.latent_reparam(code_left, progress_left)
        rec_left = self.decode(rep_left, masks_left, progress_left)

        code_right = self.encode(motion_right, masks_right, progress_right)
        rep_right = self.latent_reparam(code_right, progress_right)
        rec_right = self.decode(rep_right, masks_right, progress_right)

        code_trans = self.encode(motion_trans, masks_trans, progress_trans)
        rep_trans = self.latent_reparam(code_trans, progress_trans)
        rec_trans = self.decode(rep_trans, masks_trans, progress_trans)

            # recon motion
        loss_rec_left = self.loss_rec(rec_left, motion_left, masks_left)
        loss_rec_right = self.loss_rec(rec_right, motion_right, masks_right)
        loss_rec_trans = self.loss_rec(rec_trans, motion_trans, masks_trans)

        loss = loss_rec_left + loss_rec_right + loss_rec_trans
        self.log(f"{mode}/loss", loss)
        self.log(f"{mode}/loss_rec_left", loss_rec_left)
        self.log(f"{mode}/loss_rec_right", loss_rec_right)
        self.log(f"{mode}/loss_rec_trans", loss_rec_trans)
        return loss

    def test_step(self, train_batch, batch_idx):
        mode = 'test'
        motion_left, masks_left, progress_left,\
        motion_right, masks_right, progress_right,\
        motion_trans, masks_trans, progress_trans,\
        raw_clip_actions = train_batch

        B = motion_left.shape[0]
        ### !!! GMD proj
        motion_left = torch.matmul(motion_left, self.gmd_proj.to(motion_left.device))
        motion_right = torch.matmul(motion_right, self.gmd_proj.to(motion_right.device))
        motion_trans = torch.matmul(motion_trans, self.gmd_proj.to(motion_trans.device))


        code_left = self.encode(motion_left, masks_left, progress_left)
        rep_left = self.latent_reparam(code_left, progress_left)
        rec_left = self.decode(rep_left, masks_left, progress_left)

        code_right = self.encode(motion_right, masks_right, progress_right)
        rep_right = self.latent_reparam(code_right, progress_right)
        rec_right = self.decode(rep_right, masks_right, progress_right)

        code_trans = self.encode(motion_trans, masks_trans, progress_trans)
        rep_trans = self.latent_reparam(code_trans, progress_trans)
        rec_trans = self.decode(rep_trans, masks_trans, progress_trans)

            # recon motion
        loss_rec_left = self.loss_rec(rec_left, motion_left, masks_left)
        loss_rec_right = self.loss_rec(rec_right, motion_right, masks_right)
        loss_rec_trans = self.loss_rec(rec_trans, motion_trans, masks_trans)

        loss = loss_rec_left + loss_rec_right + loss_rec_trans
        self.log(f"{mode}/loss", loss)
        self.log(f"{mode}/loss_rec_left", loss_rec_left)
        self.log(f"{mode}/loss_rec_right", loss_rec_right)
        self.log(f"{mode}/loss_rec_trans", loss_rec_trans)

        ### save for vis    (simple check to make sure PAE works)
        if batch_idx < 8:
            data_idx = -1
            ### !!! GMD proj
            rec_left = torch.matmul(rec_left, self.gmd_proj_inv.to(rec_left.device))
            motion_left = torch.matmul(motion_left, self.gmd_proj_inv.to(motion_left.device))
            m_len = masks_left[data_idx].sum()

            format_batch_index = "{:04d}".format(batch_idx)
            traj, mot = motion_to_smpl(rec_left[data_idx])
            mot = mot[:m_len].detach().cpu()        # T,24,3
            traj = traj[:m_len].detach().cpu()
            save_motion(f"output/{format_batch_index}_ref.npz", traj, mot.reshape((-1,22*3)))

            traj, mot = motion_to_smpl(motion_left[data_idx])
            mot = mot[:m_len].detach().cpu()        # T,24,3
            traj = traj[:m_len].detach().cpu()
            save_motion(f"output/{format_batch_index}_orig.npz", traj, mot.reshape((-1,22*3)))
        return loss
