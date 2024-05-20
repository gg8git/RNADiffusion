import lightning as L
import torch
from model.UNet1D import KarrasUnet1D
from model.GaussianDiffusion import GaussianDiffusion1D
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger

from data.diffusion_datamodule import DiffusionDataModule

torch.set_float32_matmul_precision("medium")


class Wrapper(L.LightningModule):
    def __init__(self):
        super().__init__()
        model = KarrasUnet1D(
            seq_len=64,
            dim=64,
            dim_max=128,
            channels=16,
            num_blocks_per_stage=5,
            num_downsamples=3,
            attn_res=(32, 16, 8),
            attn_dim_head=32,
            num_classes=2,
            self_condition=True,
        )

        self.diffusion = GaussianDiffusion1D(
            model,
            seq_length=64,
            timesteps=1000,
            objective="pred_noise",
        )

    def forward(self, x, labels):
        return self.diffusion(x, labels)

    def training_step(self, batch, batch_idx):
        z, labels = batch
        loss = self.forward(z, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.diffusion.parameters(), lr=3e-4)


def main():
    model = Wrapper()

    dm = DiffusionDataModule(
        data_dir="data",
        batch_size=64,
        num_workers=0,
    )

    logger = WandbLogger(
        project="RNA_Diffusion", offline=False, name="ClassConditioned_LowHigh"
    )

    check = ModelCheckpoint(every_n_train_steps=1024, save_top_k=-1, save_last=True)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=-1,
        logger=logger,
        max_epochs=20_000,
        inference_mode=True,
        precision="bf16-mixed",
        callbacks=[RichProgressBar(), check],
        gradient_clip_val=5.0,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=100,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
