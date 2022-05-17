import torch

from .utils import tf

try:
    from torch import inference_mode as _im
except ImportError:
    torch.inference_mode = torch.no_grad

sim_engine = 'mujoco'  # bullet or mujoco
