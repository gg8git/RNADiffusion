import lightning as L
import torch
from denoising_diffusion_pytorch import GaussianDiffusion1D, KarrasUnet1D
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
            num_downsamples=3,
            attn_res=(32, 16, 8),
            attn_dim_head=32,
        )

        self.diffusion = GaussianDiffusion1D(
            model,
            seq_length=64,
            timesteps=1000,
            objective="pred_noise",
            auto_normalize=False,
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.diffusion(x)

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.diffusion.parameters(), lr=1e-4)


def main():
    model = Wrapper()

    dm = DiffusionDataModule(
        data_dir="data",
        batch_size=64,
        num_workers=0,
    )

    logger = WandbLogger(project="RNA_Diffusion", offline=False)

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
