import clip
from pathlib import Path
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import pickle
import json
from torch.nn.utils.rnn import pad_sequence
# from smplx import SMPL

from utils.rotation_conversion import *
from utils.algo import *

#####
#   DataModule
#####

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, split='train'):
        super().__init__()
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.split = split

        #### clip model
        self.clip_model = None

    ##### from MDM
    def load_and_freeze_clip(self, clip_version="ViT-B/32", device='cpu'):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text, device='cpu'):
        # raw_text - list (batch_size length) of strings with input text prompts
        self.clip_model.to(device)
        texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()
    #####

    def prepare_data(self):
        self.babelteach_datapath = Path("../DATA/babel-teach/")


    def read_babelteach(self, datapath):
        with open(datapath / f"babel-teach-{self.split}.pkl", 'rb') as handle:
            save_pkl = pickle.load(handle)
        motion_data, texts_data, durations = save_pkl
        filenames = motion_data.keys()

        motion_pair = []
        raw_clip_actions_pair = []
        metadata_pair = []
        # meta from json : "list" of [filename, start_time, end_time, action_name, action_index]
        # start_time, end_time are in frame index @30fps
        for filename in filenames:
            ### read data
            left_length, trans_length, right_length = durations[filename]
            trans_length_half = int(trans_length/2)
            start_time, mid_time, end_time = 0, left_length+trans_length_half, left_length+trans_length+right_length

            raw_label_pair = texts_data[filename]
            data = motion_data[filename]

            trans = data["trans"].float().cuda()
            poses = data["poses"].float().cuda()

            poses = poses.reshape((poses.shape[0], -1, 3))[:,:22]  #(T,J=22*E=3)

            print("filename", filename, poses.shape, start_time, end_time)

            # make start time=0
            trans, poses = trans[start_time:end_time], poses[start_time:end_time]
                # HumanML3D repr calculates velocity, 1 frame off
            motions = motion_preprocess(trans, poses)
                # handle 1 frame off by -1 on mid_time and end_time
            start_time, mid_time, end_time = start_time-start_time, mid_time-start_time-1, end_time-start_time-1

            motion_left = motions[start_time:mid_time]
            motion_right = motions[mid_time:end_time]
            motion_pair.append([motion_left.detach().cpu(), motion_right.detach().cpu()])


            raw_clip_action = self.encode_text(raw_label_pair, device='cuda')
            # print("raw_clip_action", raw_clip_action.shape)
            raw_clip_action = raw_clip_action / raw_clip_action.norm(dim=-1, keepdim=True)
            raw_clip_actions_pair.append(raw_clip_action)
                # save meta: filename, start_time, end_time, style_text, action_text
            metadata_pair.append([str(filename), [start_time, mid_time, end_time], raw_label_pair])
        return motion_pair, raw_clip_actions_pair, metadata_pair

    # setup() should contains code that will be run once per run.
    def setup(self, stage=None):
        ### calculate style/action label emb
        if Path("data/label_clip_emb_BABELteach.npz").exists():
            data = np.load("data/label_clip_emb_BABELteach.npz")
            self.empty_clip_emb = torch.tensor(data["empty_clip_emb"]).float()
        else:
            if self.clip_model is None:
                self.clip_model = self.load_and_freeze_clip(device='cuda')
            self.empty_clip_emb = self.encode_text([""], device='cuda')       ## (1, 512)
            self.empty_clip_emb = self.empty_clip_emb / self.empty_clip_emb.norm(dim=-1, keepdim=True)

            # save processed dataset
            print("label_clip_emb_BABELteach", self.empty_clip_emb.shape)
            np.savez("data/label_clip_emb_BABELteach.npz", \
                     empty_clip_emb=self.empty_clip_emb.detach().cpu().numpy())



        ### load xia dataset
        if Path(f"data/motion_CLIP_BABELteach_rel_{self.split}.pkl").exists():
            with open(f"data/motion_CLIP_BABELteach_rel_{self.split}.pkl", 'rb') as handle:
                save_pkl = pickle.load(handle)
            file_id, motions_left, motions_right, raw_clip_actions = save_pkl
            file_id, raw_clip_actions = torch.tensor(file_id).detach(), torch.tensor(raw_clip_actions).detach()

            ### load train/valid/test split
            metafile = open(f"data/meta_motion_CLIP_BABELteach_rel_{self.split}.json")
            meta = json.load(metafile)
            _, _, _, raw_text, split_idx_array = meta
            metafile.close()
            split_idx_array = torch.tensor(split_idx_array).detach()
        else:
            if self.clip_model is None:
                self.clip_model = self.load_and_freeze_clip(device='cuda')
            motion_babel, raw_clip_action_babel, metadata_babel = self.read_babelteach(self.babelteach_datapath)

            ###
            # process data
            motions = motion_babel
            # styles_label = [[-1,-1]] * len(motion_babel)
            # actions_label = actions_label_babel
            raw_clip_actions = raw_clip_action_babel
            metadata = metadata_babel
            f_name, time, raw_text = zip(*metadata)
            file_id = torch.arange(len(f_name)).long()

            ###
            # zip motions_left, motions_right in motions
            motions_left, motions_right = list(zip(*motions))


            raw_clip_actions = torch.stack(raw_clip_actions, dim=0)

            #### perform train/valid/test split
            d_size = len(f_name)
            rand_index = torch.randperm(d_size, dtype=torch.long)
                # split boundary
            split_idx_array = torch.zeros_like(rand_index)
            split_idx_array[:int(d_size*0.9)] = 0
            split_idx_array[int(d_size*0.9):] = 1
            split_idx_array = split_idx_array[rand_index].detach()

            ##########
            #   save everything
            ##########

            #### save processed dataset
            save_pkl = [file_id.detach().cpu().numpy(), motions_left, motions_right, raw_clip_actions.detach().cpu().numpy()]
            with open(f"data/motion_CLIP_BABELteach_rel_{self.split}.pkl", 'wb') as handle:
                pickle.dump(save_pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)

            #### save metadata
            print("file_id", file_id.shape, len(f_name), split_idx_array.shape)
            metadata = [file_id.detach().cpu().numpy().tolist(), f_name, time, raw_text, split_idx_array.detach().cpu().numpy().tolist()]
            metafile = open(f"data/meta_motion_CLIP_BABELteach_rel_{self.split}.json", "w")
            json.dump(metadata, metafile)
            metafile.close()

        ######################################
        # process and pad motion

        ### all poses and progress_T
        motion_data = []
        progress = []
        for mi in range(len(motions_left)):
            left_mot = motions_left[mi]
            right_mot = motions_right[mi]

            # total_mot = torch.cat([left_mot, right_mot], dim=0)         #(T1+T2, J, E)
            # total_tra = torch.cat([left_tra, right_tra], dim=0)         #(T1+T2, 3)
            # add one frame from left motion to right. That frame is the center frame of the transition
            right_mot = torch.cat([left_mot[-1:], right_mot],dim=0)     #(T2+1)

            left_half = int(left_mot.shape[0]/2)
            right_half = int(right_mot.shape[0]/2)
            # calc progress (left segment: the frame at left_half idx is labeled as progress 0)
                # -1: left boundary, 0: at left_half, 1:right boundary
            left_progress_left = torch.linspace(-1,0,left_half+1)[:-1]                      # left,  include -1, exclude 0
            left_progress_right = torch.linspace(0,1,left_mot.shape[0]-left_half)           # right, include 0,1
                # -1: left boundary (note that this is the last frame from left_mot), 0: at right_half, 1:right boundary
            right_progress_left = torch.linspace(-1,0,right_half+1)[:-1]                    # left, include -1, exclude 0
            right_progress_right = torch.linspace(0,1,right_mot.shape[0]-right_half)        # right, include 0,1
            left_progress = torch.cat([left_progress_left, left_progress_right],dim=0)
            right_progress = torch.cat([right_progress_left, right_progress_right],dim=0)

            ### transition
            # left segment[left_half] is the start, right segment[right half] is the end
                #left_mot include [left_half], right_mot exclude the prepend frame but include [right_half]
            trans_mot = torch.cat([left_mot[left_half:], right_mot[1:right_half+1]],dim=0)
                # left part labeled with -1,0 inclusive. right part is labeled with 1, but no 0.
                # -1: left_half, 0:prepended frame, 1: right half
            trans_progress = torch.cat([left_progress_right-1, torch.linspace(0,1,right_half+1)[1:]],dim=0)


            motion_data.append([left_mot, right_mot, trans_mot])
            progress.append([left_progress, right_progress, trans_progress])

        #### unzip
        data_left, data_right, data_trans = zip(*motion_data)
        progress_left, progress_right, progress_trans = zip(*progress)


        ### pad to tensor

        lengths_trans = [rot.shape[0] for rot in data_trans]
        data_trans = pad_sequence(data_trans, batch_first=True).float()
        progress_trans = pad_sequence(progress_trans, batch_first=True).float()
        masks_trans = torch.zeros((data_trans.shape[0], data_trans.shape[1]), device=data_trans.device, dtype=torch.bool)
        for b in range(len(lengths_trans)):
            masks_trans[b,:lengths_trans[b]] = True


        ### left poses
        lengths_left = [rot.shape[0] for rot in data_left]
        data_left = pad_sequence(data_left, batch_first=True).float()
        progress_left = pad_sequence(progress_left, batch_first=True).float()
        masks_left = torch.zeros((data_left.shape[0], data_left.shape[1]), device=data_left.device, dtype=torch.bool)
        for b in range(len(lengths_left)):
            masks_left[b,:lengths_left[b]] = True

        ### right pose
        lengths_right = [rot.shape[0] for rot in data_right]
        data_right = pad_sequence(data_right, batch_first=True).float()
        progress_right = pad_sequence(progress_right, batch_first=True).float()
        masks_right = torch.zeros((data_right.shape[0], data_right.shape[1]), device=data_right.device, dtype=torch.bool)
        for b in range(len(lengths_right)):
            masks_right[b,:lengths_right[b]] = True

        ######################################
        print("data_left", data_left.shape, masks_left.shape)
        self.raw_text = raw_text
        self.split_idx_array = split_idx_array
        self.test_raw_text = raw_text

        if self.split == "train":
            train_idx = (split_idx_array == 0)
            valid_idx = (split_idx_array == 1)

            self.train_dataset = TensorDataset(data_left[train_idx], masks_left[train_idx],  progress_left[train_idx],\
                                            data_right[train_idx], masks_right[train_idx], progress_right[train_idx],\
                                            data_trans[train_idx], masks_trans[train_idx], progress_trans[train_idx],\
                                            raw_clip_actions[train_idx])
            self.valid_dataset = TensorDataset(data_left[valid_idx], masks_left[valid_idx],  progress_left[valid_idx],\
                                            data_right[valid_idx], masks_right[valid_idx], progress_right[valid_idx],\
                                            data_trans[valid_idx], masks_trans[valid_idx], progress_trans[valid_idx],\
                                            raw_clip_actions[valid_idx])
            self.test_dataset = self.valid_dataset
            self.all_dataset = TensorDataset(data_left, masks_left, progress_left,\
                                            data_right, masks_right,progress_right,\
                                            data_trans, masks_trans,progress_trans,\
                                            raw_clip_actions)
        else:
            self.test_dataset = TensorDataset(data_left, masks_left, progress_left,\
                                            data_right, masks_right,progress_right,\
                                            data_trans, masks_trans,progress_trans,\
                                            raw_clip_actions)

        ### free clip model
        if self.clip_model is not None:
            del self.clip_model



    def train_dataloader(self):
        # handle dataset imbalance
        # sample_weight = self.train_sample_weight
        # sample_weight = sample_weight / sample_weight.sum()
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, sample_weight.shape[0], generator=torch.Generator(device='cuda'))

        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True)#, generator=torch.Generator(device='cuda'))#, sampler=sampler)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)