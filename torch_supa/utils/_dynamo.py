# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import sys
import inspect

import torch
from torch._dynamo.utils import tensortype_to_dtype
from torch._dynamo.variables.torch import TorchCtxManagerClassVariable, TorchInGraphFunctionVariable
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.ctx_manager import AutocastModeVariable
from torch._dynamo.variables.user_defined import UserDefinedClassVariable
from torch._dynamo.variables.functions import SkipFunctionVariable
from torch._dynamo.variables.constant import ConstantVariable
from torch._dynamo.variables.tensor import TensorVariable
from torch._dynamo.variables.lists import TupleVariable
import torch_supa
from .utils import transfer_device_type, torch_version_ge, torch_version_le

class SUPATorchCtxManagerClassVariable(TorchCtxManagerClassVariable):
    def call_function(self, tx, args, kwargs):
        return SUPAAutocastModeVariable.create(self.value, args, kwargs)


class SUPAAutocastModeVariable(AutocastModeVariable):
    @staticmethod
    def create(func, args, kwargs):
        bound_args = inspect.signature(func).bind(*args, **kwargs)
        bound_args.apply_defaults()
        target_values = []
        kwargs.clear()

        for key in ["device_type", "dtype", "enabled", "cache_enabled"]:
            if key == "device_type" and func in [
                torch_supa.supa.amp.autocast,
            ]:
                arg = "supa" if func is torch_supa.supa.amp.autocast else "cpu"
            else:
                arg = bound_args.arguments[key]
            if isinstance(arg, VariableTracker):
                target_values.append(arg.as_python_constant())
            else:
                target_values.append(arg)

        var = AutocastModeVariable(
            target_values, initial_values=None, **kwargs)
        return var


def UserDefinedClassVariable__new__(cls, value, **kwargs):
    if value in [
        torch.supa.amp.autocast,
        torch_supa.supa.amp.autocast,
        torch.supa.amp.autocast_mode.autocast,
        torch_supa.supa.amp.autocast_mode.autocast,
    ]:
        return SUPATorchCtxManagerClassVariable(value, **kwargs)
    elif value in [
        torch.supa.Stream,
        torch_supa.supa.Stream,
        torch.supa.streams.Stream,
        torch_supa.supa.streams.Stream,
        torch_supa.supa.BoolTensor,
        torch_supa.supa.ByteTensor,
        torch_supa.supa.CharTensor,
        torch_supa.supa.DoubleTensor,
        torch_supa.supa.FloatTensor,
        torch_supa.supa.HalfTensor,
        torch_supa.supa.IntTensor,
        torch_supa.supa.LongTensor,
        torch_supa.supa.ShortTensor,
        torch_supa.supa.BFloat16Tensor,
        torch.device,
    ]:
        return TorchInGraphFunctionVariable(value, **kwargs)
    return cls.__new__raw(cls)


def SkipFunctionVariable__new__(cls, value, reason=None, **kwargs):
    if value in [
        torch.supa.stream,
        torch_supa.supa.stream,
        torch_supa.supa.utils.stream,
    ]:
        return TorchInGraphFunctionVariable(value, **kwargs)
    return cls.__new__raw(cls)


def TensorVariable_call_method(self, tx, name, args, kwargs):
    if (
        name == 'type'
        and self.dtype is not None
        and len(args) == 0
        and isinstance(self.device, torch.device)
        and self.device.type == torch.device("supa").type
    ):
        tensortype = next(k for k, v in tensortype_to_dtype.items() if self.dtype in v)
        constant_result = ConstantVariable.create(f"torch.supa.{tensortype.__name__}")

        if len(args) == 1:
            return constant_result.getitem_const(args[0])
        elif args:
            return TupleVariable([constant_result.getitem_const(a) for a in args])
        return constant_result
    else:
        return TensorVariable.call_method_raw(self, tx, name, args, kwargs)


class _InductorSupaRegistry:
    _disabled_register = False
    _has_inited = False

    @classmethod
    def register_inductor_supa(cls):
        if cls.has_initialized() or cls._disabled_register:
            return
        from torch_supa import _inductor  # noqa
        cls._has_inited = True

    @classmethod
    def disable_register(cls):
        cls._disabled_register = True

    @classmethod
    def enable_register(cls):
        cls._disabled_register = False

    @classmethod
    def has_initialized(cls):
        if cls._has_inited:
            return True
        # Maybe initialized by call `import torch_supa._inductor` manually.
        if 'torch_supa._inductor' in sys.modules:
            cls._has_inited = True
        return cls._has_inited


def is_inductor_supa_initialized():
    return _InductorSupaRegistry.has_initialized()


def disable_register_inductor_supa():
    _InductorSupaRegistry.disable_register()


def enable_register_inductor_supa():
    _InductorSupaRegistry.enable_register()


def register_inductor_supa():
    _InductorSupaRegistry.register_inductor_supa()


def patch_inductor_wrapper():
    if torch_version_ge(2, 5, 0):
        from torch._inductor.codegen import common
        _original_init_backend_registration = common.init_backend_registration
        common.init_backend_registration = lambda: (
            register_inductor_supa(),
            _original_init_backend_registration()
        )
    else:
        from torch._inductor.graph import GraphLowering
        _original_init_backend_registration = GraphLowering.init_backend_registration
        GraphLowering.init_backend_registration = lambda self: (
            register_inductor_supa(),
            _original_init_backend_registration(self)
        )


def patch_inductor_utils():
    if torch_version_ge(2, 4, 0):
        from torch._inductor.utils import is_gpu as origin_is_gpu

        def is_gpu(device):
            return origin_is_gpu(device) or device == "supa"

        torch._inductor.utils.is_gpu = is_gpu

    if torch_version_le(2, 6, 0):
        from torch._inductor.utils import is_big_gpu

        def _use_template_for_supa(layout, allowed_layout_dtypes) -> bool:
            return (
                layout.device.type == "supa"
                and layout.dtype in allowed_layout_dtypes
                and is_big_gpu(layout.device.index or 0)
            )

        torch._inductor.utils._use_template_for_cuda = _use_template_for_supa

def _add_dynamo_methods():
    UserDefinedClassVariable.__new__raw = UserDefinedClassVariable.__new__
    UserDefinedClassVariable.__new__ = UserDefinedClassVariable__new__
    SkipFunctionVariable.__new__raw = SkipFunctionVariable.__new__
    SkipFunctionVariable.__new__ = SkipFunctionVariable__new__
    TensorVariable.call_method_raw = TensorVariable.call_method
    TensorVariable.call_method = TensorVariable_call_method
    patch_inductor_utils()
    patch_inductor_wrapper()
