import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger

from model.SPDM.diffusion import *

batch_size = 2

from model.datamodule_babelteach_rel import *
dm = DataModule(batch_size=batch_size, split='train')

logger = TensorBoardLogger("./model/SPDM")

# initialize model
model = DiffPhase()

# model checkpoint
checkpoint_callback = ModelCheckpoint(monitor="valid/loss", save_last=True)

# Train
extra_trainer_args = {"precision":16}
if torch.cuda.is_available():# and not debug:
    extra_trainer_args["gpus"] = -1
    extra_trainer_args["strategy"] = "dp"
    print("cuda available! use all gpu in the machine")

max_epochs = 300000 # 400 per run is fine
check_val_every_n_epoch = 1


trainer = pl.Trainer(max_epochs=max_epochs, logger=logger, check_val_every_n_epoch=check_val_every_n_epoch, callbacks=[checkpoint_callback], **extra_trainer_args)  #gradient_clip_val=0.5,
trainer.fit(model, dm)
