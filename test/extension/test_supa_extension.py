# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import os
import pytest
import sys
import subprocess
from contextlib import contextmanager
import importlib
import torch
import torch_supa  # noqa
from torch_supa.testing.common_utils import assert_allclose
from torch_supa.utils.cpp_extension import load, load_inline

cwd = os.path.dirname(os.path.realpath(__file__))
resources = f"{cwd}/resources"

@pytest.mark.sanity
@pytest.mark.regression
@pytest.mark.single
@pytest.mark.pt20600
def test_load():
    vector_add_supa = load(
        name="vector_add_supa",
        sources=[
            f"{resources}/vector_add.cpp",
            f"{resources}/vector_add_kernel.su"
        ],
        verbose=True,
        extra_cflags=["-g", "-O0"],
        extra_supa_cflags=["-g"]
    )

    a = torch.randn((1024,), dtype=torch.float32).supa()
    b = torch.randn((1024,), dtype=torch.float32).supa()

    print("running kernel in extension...")
    c = vector_add_supa.vector_add(a, b)

    assert_allclose(c.cpu(), a.cpu() + b.cpu(), atol=1e-2, rtol=1e-5)


@pytest.mark.sanity
@pytest.mark.regression
@pytest.mark.single
@pytest.mark.pt20600
def test_load_inline():
    vector_add_supa = load_inline(
        name="vector_add_supa_inline",
        cpp_sources="""
namespace supa_kernel {
supaError_t vector_add(const float* a, const float* b, float* c, int n);
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.device().is_privateuseone(), "must be supa tensor");
  TORCH_CHECK(b.device().is_privateuseone(), "must be supa tensor");
  TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
  TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
  TORCH_CHECK(a.numel() == b.numel(), "a and b must have the same size");

  int n = a.numel();
  auto c = torch::empty_like(a);

  supa_kernel::vector_add(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vector_add", &vector_add, "Vector addition extension (SUPA)");
}
""",
        supa_sources="""
namespace {
__global__ void add(float *A, float *B, float *C, int N) {
  int i = threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}
}

namespace supa_kernel {

supaError_t vector_add(const float* a, const float* b, float* c, int n) {
  void* args[] = {
    const_cast<void*>(static_cast<const void*>(&a)),
    const_cast<void*>(static_cast<const void*>(&b)),
    const_cast<void*>(static_cast<const void*>(&c)),
    const_cast<void*>(static_cast<const void*>(&n))
  };

  auto err = supaLaunchKernel(add, dim3(1, 1, 1), dim3(n, 1, 1), args, 0, 0);
  return err;
}

}
        """,
        verbose=True
    )

    a = torch.randn((1024,), dtype=torch.float32).supa()
    b = torch.randn((1024,), dtype=torch.float32).supa()

    print("running kernel in extension...")
    c = vector_add_supa.vector_add(a, b)

    assert_allclose(c.cpu(), a.cpu() + b.cpu(), atol=1e-2, rtol=1e-5)


@pytest.mark.sanity
@pytest.mark.regression
@pytest.mark.single
@pytest.mark.pt20600
def test_building():
    result = subprocess.run(
        "python3 setup.py build",
        cwd=resources,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    assert result.returncode == 0, f"error when building extension, see:\n{result.stdout}"

    wd = next(
        (
            os.path.join(root, filename)
            for root, _, files in os.walk(resources)
            for filename in files
            if filename.startswith("suda") and filename.endswith(".so")
        ),
        "",
    )
    assert wd, "can't find target folder"
    wd = os.path.dirname(os.path.dirname(wd))

    @contextmanager
    def path_context():
        try:
            sys.path.insert(0, wd)
            yield
        finally:
            sys.path.remove(wd)

    @contextmanager
    def module_context(name):
        try:
            m = importlib.import_module(name)
            print("\nload module", m.__name__, m.__file__)
            yield m
        finally:
            m = sys.modules.pop(name, None)
            if m:
                del m

    with path_context():

        a = torch.randn((1024,), dtype=torch.float32).supa()
        b = torch.randn((1024,), dtype=torch.float32).supa()

        with module_context("torch_test_cpp_extension.supa") as supa:
            c = supa.vector_add(a, b)
            assert_allclose(c.cpu(), a.cpu() + b.cpu(), atol=1e-2, rtol=1e-5)


        with module_context("torch_test_cpp_extension.suda") as suda:
            c = suda.vector_add(a, b)
            assert_allclose(c.cpu(), a.cpu() + b.cpu(), atol=1e-2, rtol=1e-5)


@pytest.mark.sanity
@pytest.mark.regression
@pytest.mark.single
def test_building_dlink():
    result = subprocess.run(
        "python3 setup_dlink_test.py bdist_wheel",
        cwd=resources,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    assert result.returncode == 0, f"error when building extension, see:\n{result.stdout}"


@pytest.mark.sanity
@pytest.mark.regression
@pytest.mark.single
@pytest.mark.pt20600
def test_supa_class():
    """verify ability to access class binded by torch_supa"""
    modules = load_inline(
        name="supa_symbols",
        cpp_sources="""
#include <cstring>
#include <memory>

class supaDevicePropWrap : public supaDeviceProp {
public:
supaDevicePropWrap(const char* _name) {
    std::memset(this, 0, sizeof(supaDeviceProp));
    std::strncpy(name, _name, 256);
}
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("get_property", []() -> std::unique_ptr<supaDeviceProp> {return std::make_unique<supaDevicePropWrap>("Fake Property"); }, "Test get device property");
}
                    """,
                    verbose=True)
    c = modules.get_property()
    assert c.name == "Fake Property", "error when accessing supa class"
