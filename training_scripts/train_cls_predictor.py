import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger

from datamodules import DiffusionDataModule, LatentDatasetClassifier
from model.mol_score_model.conv_classifier import ConvScoreClassifier

torch.set_float32_matmul_precision("medium")


class Wrapper(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.predictor = ConvScoreClassifier()

    def forward(self, z):
        # returns logits (B, 2)
        return self.predictor(z)

    def _shared_step(self, batch, stage: str):
        # datamodule provides: z: (B, 8, 16), y: (B,) in {0,1}
        z, y = batch
        logits = self(z)  # (B, 2)
        y = y.long()
        loss = F.cross_entropy(logits, y)  # CE expects raw logits

        # metrics
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        p1 = torch.softmax(logits, dim=1)[:, 1].mean()  # mean P(class=1) in batch

        self.log(f"{stage}/loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        self.log(f"{stage}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/p1_mean", p1, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)


def main():
    model = Wrapper()

    dm = DiffusionDataModule(
        data_dir="data/descriptors",
        batch_size=64,
        num_workers=0,
        dataset=LatentDatasetClassifier,
    )

    batch = next(iter(dm.train_dataloader()))
    import ipdb

    ipdb.set_trace()

    logger = WandbLogger(project="SELFIES_Diffusion", offline=False, name="ConvClassifier-qed")

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
