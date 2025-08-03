import sys
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Provide minimal stubs for heavy optional dependencies
torch_stub = types.ModuleType("torch")
torch_nn_stub = types.ModuleType("torch.nn")
torch_nn_attention_stub = types.ModuleType("torch.nn.attention")
flex_attention_stub = types.ModuleType("torch.nn.attention.flex_attention")
flex_attention_stub.or_masks = lambda *a, **k: None
flex_attention_stub.and_masks = lambda *a, **k: None
torch_nn_attention_stub.flex_attention = flex_attention_stub
torch_stub.nn = types.ModuleType("torch.nn")
torch_stub.nn.attention = torch_nn_attention_stub
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.nn", torch_stub.nn)
sys.modules.setdefault("torch.nn.attention", torch_nn_attention_stub)
sys.modules.setdefault(
    "torch.nn.attention.flex_attention", flex_attention_stub
)

from data.data_utils import len2weight

@pytest.mark.parametrize(
    "loss_reduction,x,expected",
    [
        ("token", 5, 1.0),
        ("sample", 5, 1/5),
        ("square", 4, 1/(4 ** 0.5)),
    ],
)
def test_len2weight_values(loss_reduction, x, expected):
    assert len2weight(x, loss_reduction) == pytest.approx(expected)

@pytest.mark.parametrize("loss_reduction", ["token", "sample", "square"])
def test_len2weight_zero(loss_reduction):
    assert len2weight(0, loss_reduction) == 0


def test_len2weight_invalid():
    with pytest.raises(NotImplementedError):
        len2weight(1, "unknown")

