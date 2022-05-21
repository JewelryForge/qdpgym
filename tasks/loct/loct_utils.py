import torch


class PolicyWrapper(object):
    def __init__(self, net):
        self.net = net
        self.device = next(net.parameters()).device

    def __call__(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.net(obs.reshape(-1, self.net.input_dim)).detach().cpu().numpy()
