import fire
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger

from model.diffusion_v2 import DiffusionDataModule
from model.diffusion_v2.GaussianDiffusion import ExtinctPredictor

torch.set_float32_matmul_precision("medium")


def main(hdim: int):
    dm = DiffusionDataModule(
        data_type="peptide",
        batch_size=2048,
        num_workers=16,
        return_extinct=True,
    )

    logger = WandbLogger(
        project="DiffuseOpt",
        tags=["CLS_PRED"],
        offline=False,
    )

    check = ModelCheckpoint(
        every_n_epochs=1,
        save_top_k=-1,
        save_last=True,
        save_weights_only=True,
    )

    model = ExtinctPredictor(hdim=hdim)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=-1,
        logger=logger,
        max_epochs=200,
        inference_mode=True,
        precision="bf16-mixed",
        callbacks=[RichProgressBar(), check],
        gradient_clip_val=5.0,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    fire.Fire(main)
