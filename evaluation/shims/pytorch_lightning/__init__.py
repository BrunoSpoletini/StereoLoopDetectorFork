"""Shim minimo de pytorch_lightning para cargar SALAD solo en inferencia (sin instalar lightning)."""
import torch


class LightningModule(torch.nn.Module):
    def save_hyperparameters(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass
