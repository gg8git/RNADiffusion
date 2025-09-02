import math

import lightning as L
import selfies as sf
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger

from datamodules.diffusion_datamodule import DiffusionDataModule
from model.mol_score_model.conv_predictor import ConvScorePredictor
from utils.guacamol_utils import smile_to_guacamole_score

torch.set_float32_matmul_precision("medium")


class Wrapper(L.LightningModule):
    def __init__(self, task_id="pdop"):
        super().__init__()
        self.task_id = task_id

        self.predictor = ConvScorePredictor()

        self.vae_wrapper = VAEWrapper(path_to_vae_statedict="checkpoints/SELFIES_VAE/epoch=447-step=139328.ckpt")
        self.vae_wrapper.eval()

    def forward(self, z):
        return self.predictor(z)

    def training_step(self, batch, batch_idx):
        z = batch.transpose(1, 2)

        selfies = self.vae_wrapper.latent_to_selfies(z)
        target_scores = torch.zeros(len(selfies), device=z.device, dtype=torch.float32)
        for i, selfie in enumerate(selfies):
            smiles_str = sf.decoder(selfie)
            score = smile_to_guacamole_score(self.task_id, smiles_str)
            if (score is not None) and math.isfinite(score):
                target_scores[i] = score
            else:
                target_scores[i] = 0.0

        pred_scores = self.forward(z)

        loss = F.mse_loss(pred_scores, target_scores)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        z = batch.transpose(1, 2)

        selfies = self.vae_wrapper.latent_to_selfies(z)
        target_scores = torch.zeros(len(selfies), device=z.device, dtype=torch.float32)
        for i, selfie in enumerate(selfies):
            smiles_str = sf.decoder(selfie)
            score = smile_to_guacamole_score(self.task_id, smiles_str)
            target_scores[i] = score if score is not None and math.isfinite(score) else 0.0

        pred_scores = self(z)  # reuse forward()
        loss = F.mse_loss(pred_scores, target_scores)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Optional correlation
        if len(target_scores) > 1:
            corr = torch.corrcoef(torch.stack([pred_scores, target_scores]))[0, 1]
            self.log("val/corr", corr, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        z = batch.transpose(1, 2)

        selfies = self.vae_wrapper.latent_to_selfies(z)
        target_scores = torch.zeros(len(selfies), device=z.device, dtype=torch.float32)
        for i, selfie in enumerate(selfies):
            smiles_str = sf.decoder(selfie)
            score = smile_to_guacamole_score(self.task_id, smiles_str)
            target_scores[i] = score if score is not None and math.isfinite(score) else 0.0

        pred_scores = self(z)
        loss = F.mse_loss(pred_scores, target_scores)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Optional: correlation
        if len(target_scores) > 1:
            corr = torch.corrcoef(torch.stack([pred_scores, target_scores]))[0, 1]
            self.log("test/corr", corr, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.predictor.parameters(), lr=3e-4)


def main():
    model = Wrapper()

    dm = DiffusionDataModule(
        data_dir="data/selfies",
        batch_size=64,
        num_workers=0,
    )

    batch = next(iter(dm.train_dataloader()))
    import ipdb

    ipdb.set_trace()

    logger = WandbLogger(project="SELFIES_Diffusion", offline=False, name="ConvPredictor-pdop")

    check = ModelCheckpoint(
        monitor="val/loss",  # monitor validation loss
        mode="min",  # minimize val/loss
        save_top_k=3,  # keep only the best checkpoint
        save_last=True,  # also save the last epoch
        every_n_epochs=2,  # checkpoint less often for long training
    )

    early = EarlyStopping(monitor="val/loss", patience=20, mode="min")

    trainer = L.Trainer(
        accelerator="gpu",
        devices=-1,
        logger=logger,
        max_epochs=50,
        check_val_every_n_epoch=1,
        precision="bf16-mixed",
        inference_mode=True,
        callbacks=[TQDMProgressBar(), check, early],
        gradient_clip_val=5.0,
        num_sanity_val_steps=2,
        log_every_n_steps=50,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
