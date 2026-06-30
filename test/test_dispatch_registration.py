# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
import torch_supa  # noqa: F401


def _is_known_external_dangling_impl(impl):
    return impl.startswith("name: torchvision::")


@pytest.mark.sanity
@pytest.mark.gcuSmoke
def test_find_dangling_impls():
    dangling_impls = torch._C._dispatch_find_dangling_impls()
    unexpected_impls = [impl for impl in dangling_impls if not _is_known_external_dangling_impl(impl)]
    assert not unexpected_impls, f"Expect zero torch-supa dangling impls, but found: {unexpected_impls}"
