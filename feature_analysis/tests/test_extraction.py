from contextlib import nullcontext
from types import SimpleNamespace

from salt_feature_analysis.extraction import _autocast_context


def test_autocast_disabled_uses_noop_context():
    torch = SimpleNamespace()
    assert isinstance(_autocast_context(torch, "cpu", False), nullcontext)


def test_autocast_falls_back_to_legacy_cuda_amp():
    marker = object()
    amp = SimpleNamespace(autocast=lambda enabled: marker)
    torch = SimpleNamespace(cuda=SimpleNamespace(amp=amp))
    assert _autocast_context(torch, "cuda", True) is marker
