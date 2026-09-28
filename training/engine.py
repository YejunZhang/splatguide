import pytorch_lightning as pl
import torch
from einops import repeat
from torch.optim.lr_scheduler import LambdaLR

from recon.align import align_mean_transform
from recon.worldmirror import SceneEncoder
from seva.geometry import normalize_cameras, plucker_rays
from seva.model import SGMWrapper, Seva, SevaParams
from seva.sampling import DDPMDiscretization, DiscreteDenoiser
from training.loss import diffusion_loss


class DiffusionEngine(pl.LightningModule):
    """Trains the SEVA U-Net with the SplatGuide conditions. The frozen scene encoder (WorldMirror,
    VAE, CLIP) runs on the fly on the raw images of each sample; nothing is precomputed or stored."""

    def __init__(self, network: dict, base_learning_rate: float = 1e-6, warmup_steps: int = 1000):
        super().__init__()
        self.model = SGMWrapper(Seva(SevaParams(**network)))
        self.base_learning_rate = base_learning_rate
        self.warmup_steps = warmup_steps

    def on_fit_start(self):
        # Plain objects, not submodules: not trained, not in DDP, not in checkpoints.
        self.encoder = SceneEncoder(self.device)
        self.denoiser = DiscreteDenoiser(DDPMDiscretization(), device=self.device)

    @torch.no_grad()
    def get_conditions(self, images_wm: torch.Tensor, images: torch.Tensor, n_ref: int):
        """One sample (references first) -> clean latents, reference mask, raw cameras and the
        frame-flattened SEVA conditions."""
        scene = self.encoder(images_wm, images, n_ref, align_mean_transform)
        latents = self.encoder.ae.encode(images * 2.0 - 1.0)
        plucker = plucker_rays(normalize_cameras(scene["c2w"]), scene["K"])
        mask = torch.arange(len(images), device=self.device) < n_ref
        mask_map = repeat(mask, "n -> n 1 h w", h=plucker.shape[2], w=plucker.shape[3])
        cond = {
            "crossattn": repeat(scene["clip_features"], "d -> n 1 d", n=len(images)),
            "concat": torch.cat([mask_map, plucker, scene["rendered_latents"]], dim=1),
            "replace": torch.cat([latents, mask_map], dim=1),  # clean latent + mask
            "dense_vector": plucker,
            "tokens": scene["tokens"].flatten(0, 1)[None],  # [1, n_ref * 5, D]
        }
        return latents, mask, scene["c2w"], cond

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        latents, mask, c2w, cond = self.get_conditions(batch["images_wm"], batch["images"], batch["n_ref"])
        loss = diffusion_loss(self.model, self.denoiser, cond, latents, mask, c2w)
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Effective lr = base lr x #GPUs x grad accumulation (one sample per GPU).
        trainer = self.trainer
        lr = self.base_learning_rate * trainer.num_devices * trainer.num_nodes * trainer.accumulate_grad_batches
        print(f"Learning rate {lr:.2e}")
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        warmup = LambdaLR(opt, lambda n: min(1.0, 1e-6 + (1 - 1e-6) * n / self.warmup_steps))
        return [opt], [{"scheduler": warmup, "interval": "step"}]
