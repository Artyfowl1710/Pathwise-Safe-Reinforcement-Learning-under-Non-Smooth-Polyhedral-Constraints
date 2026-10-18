import numpy as np, torch
from pifsrl.agents.safe_td3 import SafeTD3, TD3Config


def _agent(scale):
    a = SafeTD3.__new__(SafeTD3)
    torch.nn.Module.__init__(a)
    a.cfg = TD3Config(residual_scale=scale)
    return a


def test_compose_identity_when_disabled():
    a = _agent(0.0)
    r = torch.tensor([[0.3, -0.7]]); b = torch.tensor([[0.9, 0.9]])
    assert torch.equal(a.compose(r, b), r)


def test_compose_is_base_plus_scaled_residual_and_clamped():
    a = _agent(0.3)
    r = torch.tensor([[1.0, -1.0, 0.0]]); b = torch.tensor([[0.9, -0.9, 0.2]])
    out = a.compose(r, b)
    assert torch.allclose(out, torch.tensor([[1.0, -1.0, 0.2]]))          # clamp to the [-1,1] box
    assert torch.allclose(a.compose(torch.zeros(1, 3), b), b)             # zero residual == base policy exactly
