from contextlib import contextmanager

import torch


class ModelEMA:
    """Exponential moving average of model parameters and buffers."""

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(value)

    def state_dict(self):
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, state, model):
        self.decay = float(state["decay"])
        self.updates = int(state["updates"])
        device_by_name = {name: value.device for name, value in model.state_dict().items()}
        self.shadow = {
            name: value.to(device_by_name[name]) for name, value in state["shadow"].items()
        }

    @contextmanager
    def average_parameters(self, model):
        live = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        try:
            yield
        finally:
            model.load_state_dict(live)
