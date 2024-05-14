from dataclasses import asdict

import lightning as L
import schedulefree
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor

from data.vae_datamodule import VAEDataModule
from model.RNAVAE3 import Config, RNAVAE

torch.set_float32_matmul_precision("high")

torch._dynamo.config.cache_size_limit = 8192
torch._dynamo.config.accumulated_cache_size_limit = 8192


class Wrapper(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        if isinstance(config, dict):
            self.save_hyperparameters(config)
            config = Config(**config)
        else:
            self.save_hyperparameters(asdict(config))

        self.config = config
        self.model: RNAVAE = torch.compile(RNAVAE(config))

    def encode(self, tokens):
        return self.model.encoder(tokens)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        elbo, stats = self.model(batch)

        stats = {f"train/{k}": v for k, v in stats.items()}

        self.log_dict(stats, on_step=True, prog_bar=True)

        return elbo

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        elbo, stats = self.model(batch)

        stats = {f"val/{k}": v for k, v in stats.items()}
        self.log_dict(stats, prog_bar=True, sync_dist=True)

        return elbo

    def configure_optimizers(self):
        opt = schedulefree.AdamWScheduleFree(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.wd,
            betas=(0.9, 0.99),
            warmup_steps=2048,
        )
        return opt

    def on_validation_epoch_end(self) -> None:
        self.model.train()
        opt = self.optimizers()
        opt.train()

    def on_validation_epoch_start(self) -> None:
        self.model.eval()
        opt = self.optimizers()
        opt.eval()


def main(beta: float = 0.02):
    data_dir = "./data/"

    dm = VAEDataModule(
        data_dir,
        batch_size=512,
        num_workers=0,
    )

    config = Config(
        d_model=256,
        dim_ff=768,
        vocab=dm.vocab,
        beta=beta,
        wd=0.01,
        n_layers=8,
        n_bn=16,
        zdim=16,
        lr=1e-3,
    )

    model = Wrapper.load_from_checkpoint(
        "RNA_Diffusion/nb2skaky/checkpoints/last.ckpt", config=config
    )

    logger = WandbLogger(project="RNA_Diffusion", offline=False)

    check = ModelCheckpoint(every_n_train_steps=1024, save_top_k=-1, save_last=True)
    trainer = L.Trainer(
        accelerator="gpu",
        devices=-1,
        logger=logger,
        callbacks=[RichProgressBar(), check],
        max_epochs=20_000,
        inference_mode=True,
        precision="bf16-mixed",
        gradient_clip_val=5.0,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=100,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
