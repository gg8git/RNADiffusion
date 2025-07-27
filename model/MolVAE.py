import lightning.pytorch as pl
import selfies as sf
import torch
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR

from molformers.utils.PostCollapse import KLCalc
from molformers.utils.utils import count_parameters


class VAEModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters(model.hparams)
        self.model = model
        self.vocab = model.vocab

        self.val_kld = KLCalc()

        enc_params = count_parameters(self.model.encoder) + count_parameters(self.model.enc_neck)
        dec_params = count_parameters(self.model.decoder) + count_parameters(self.model.dec_neck) + count_parameters(self.model.dec_tok_deproj)

        print(f'Enc params: {enc_params:,}')
        print(f'Dec params: {dec_params:,}')

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)
        self.logvals(outputs, 'train')

        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch)
        self.logvals(outputs, 'validation')
        self.val_kld.update(outputs['mu_ign'], outputs['sigma_ign'])

    @torch.inference_mode()
    def sample(self, z: torch.Tensor, argmax=True, max_len=256):
        training = self.training
        self.eval()

        z = z.reshape(-1, self.model.n_acc, self.model.d_bnk).to(self.device)
        if hasattr(self, 'decoder_neck'):
            z = self.decoder_neck(z.flatten(1)).reshape(z.shape[0], self.n_bn, self.d_decoder)

        tokens = torch.full((z.shape[0], 1), fill_value=self.model.start_tok, device=self.device, dtype=torch.long)
        while True: # Loop until every molecule hits a stop token
            logits = self.model.decode(z, tokens)[:, -1:]
            if argmax:
                sample = logits.argmax(dim=-1)
            else:
                sample = Categorical(logits=logits).sample()

            tokens = torch.cat([tokens, sample], dim=-1)

            if (tokens == self.model.stop_tok).any(dim=-1).all() or tokens.shape[1] > max_len:
                break
        
        self.train(training)
        return tokens[:, 1:] # Cut out start token

    def logvals(self, logdict, split):
        for k, v in logdict.items():
            if k == 'z' or '_ign' in k:
                continue

            self.log(f'{split}/' + k, v, prog_bar=split == 'train', sync_dist=True)

    def selfie_to_tokens(self, selfie: str):
        """ Converts a *single* selfie string to a token encoding """

        selfie = f"[start]{selfie}[stop]"
        enc = sf.selfies_to_encoding(selfie, self.vocab, enc_type='label')
        return torch.tensor(enc, dtype=torch.long)

    def tokens_to_selfie(self, tokens, drop_after_stop=True) -> str:
        """ Converts a *single* token sequence to a selfie string """

        # returns single atom in case encoding to selfies fails
        try :
            selfie = sf.encoding_to_selfies(tokens.squeeze().tolist(), {v:k for k,v in self.vocab.items()}, 'label')
        except:
            selfie = '[C]'
        
        if drop_after_stop and '[stop]' in selfie:
            selfie = selfie[:selfie.find('[stop]')]
        if '[pad]' in selfie:
            selfie = selfie[:selfie.find('[pad]')]
        if '[start]' in selfie:
            selfie = selfie[selfie.find('[start]') + len('[start]'):]
        return selfie

    def on_validation_epoch_end(self) -> None:
        n_alive, mean, min_, max_, mean_sigma = self.val_kld.compute()
        self.log('validation/alive_n', n_alive, prog_bar=True)
        self.log('validation/alive_mean_kl', mean, prog_bar=True)
        self.log('validation/alive_min_kl', min_, prog_bar=True)
        self.log('validation/alive_max_kl', max_, prog_bar=True)
        self.log('validation/alive_sigma_mean', mean_sigma, prog_bar=True)
        self.val_kld.reset()


    def configure_optimizers(self):
        lr = 3e-4
        opt = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.95))

        lr_sched = CosineAnnealingLR(opt, T_max=622*1_000, eta_min=lr*0.1)

        return dict(
            optimizer=opt,
            lr_scheduler=dict(
                scheduler=lr_sched,
                interval='step',
                frequency=1
            )
        )