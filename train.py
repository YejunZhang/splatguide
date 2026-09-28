#!/usr/bin/env python3
"""Train the SplatGuide diffusion model on raw multi-view images (WorldMirror runs on the fly).

    python train.py --base configs/splatguide.yaml -n splatguide
    python train.py --base configs/splatguide.yaml -n splatguide lightning.trainer.devices=4
    python train.py --resume logs/splatguide                # continue a run
"""
import argparse
import datetime
import glob
import os

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from training.data import SplatGuideDataModule
from training.engine import DiffusionEngine


def get_parser():
    parser = argparse.ArgumentParser(description="SplatGuide training")
    parser.add_argument("--base", nargs="*", default=[], metavar="config.yaml", help="configs, merged left to right")
    parser.add_argument("-n", "--name", type=str, default="", help="run name (default: name of the first config)")
    parser.add_argument("-r", "--resume", type=str, default="", help="resume from a log dir or a checkpoint inside it")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="start a new run from this checkpoint")
    parser.add_argument("-l", "--logdir", type=str, default="logs")
    parser.add_argument("-s", "--seed", type=int, default=23)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--projectname", type=str, default="splatguide")
    return parser


def latest_checkpoint(logdir: str) -> str:
    last = os.path.join(logdir, "checkpoints", "last.ckpt")
    if os.path.exists(last):
        return last
    return max(glob.glob(os.path.join(logdir, "checkpoints", "*.ckpt")), key=os.path.getmtime)


class SaveConfigCallback(Callback):
    def __init__(self, path: str, config):
        super().__init__()
        self.path = path
        self.config = config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config, self.path)


def main():
    opt, unknown = get_parser().parse_known_args()
    assert not (opt.name and opt.resume), "use -n with --resume_from_checkpoint to resume into a new log dir"
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    ckpt_path = opt.resume_from_checkpoint
    if opt.resume:
        logdir = os.path.dirname(os.path.dirname(opt.resume)) if os.path.isfile(opt.resume) else opt.resume.rstrip("/")
        ckpt_path = opt.resume if os.path.isfile(opt.resume) else latest_checkpoint(logdir)
        opt.base = sorted(glob.glob(os.path.join(logdir, "configs", "*.yaml"))) + opt.base
        print(f"Resuming from {ckpt_path}")
    else:
        name = opt.name or os.path.splitext(os.path.basename(opt.base[0]))[0]
        logdir = os.path.join(opt.logdir, name)

    seed_everything(opt.seed, workers=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = OmegaConf.merge(*[OmegaConf.load(c) for c in opt.base], OmegaConf.from_dotlist(unknown))
    model = DiffusionEngine(**config.model)
    data = SplatGuideDataModule(**config.data)

    if opt.wandb:
        logger = WandbLogger(name=os.path.basename(logdir), id=os.path.basename(logdir), project=opt.projectname, save_dir=logdir)
    else:
        logger = CSVLogger(save_dir=logdir, name="csv")
    callbacks = [
        SaveConfigCallback(os.path.join(logdir, "configs", f"{now}.yaml"), config),
        ModelCheckpoint(dirpath=os.path.join(logdir, "checkpoints"), **config.lightning.modelcheckpoint),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = Trainer(**config.lightning.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(model, data, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
