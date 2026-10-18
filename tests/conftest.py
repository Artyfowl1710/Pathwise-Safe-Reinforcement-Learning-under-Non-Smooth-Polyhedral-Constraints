"""Shared pytest fixtures.

The QP layer tests deliberately run in float64 (needed for tight agreement with
CVXPY's OSQP backend). Without resetting torch's *global* default dtype after
those tests, later test modules that rely on the normal float32 default would
silently break -- this fixture makes each test file dtype-independent of
execution order.
"""
import torch
import pytest


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    prev = torch.get_default_dtype()
    yield
    torch.set_default_dtype(prev)
