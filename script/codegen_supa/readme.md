# Codegen supa

code generator based on pytorch ATen code generator for supa
change its name to codegen supa in order not to conflict with torchgen under sys folder

## Base Version

|   |   |
|---|---|
|pytorch tag| v2.2.0, v2.2.0-rc7, v2.2.0-rc8 |
| commit hash | 8ac9b20d4b0 |
|commit log| Run docker release build on final tag (#117131) (#117182)|

## Structured code generation

in order to support structured code generation, must inplement `TORCH_IMPL_FUNC(<out-kind-name>)`<br/>
for example, structured functions of `addcdiv`

|    |   |
|---|---|
| addcdiv | functional kind |
| addcdiv_ | inplace kind |
| addcdiv.out | out kind |

and the out-kind-name is ```addcdiv_out```

👉 for some op, the map relation between out-kind-name and op name is not straightforward, for example: `avg_pool3d_backward.grad_input` has out-kind-name of `avg_pool3d_backward_out`

## Native OP Support
it can generate native function calling under RegisterSUPA/RegisterAutogradSUPA.cpp according to different situations.

examples like:
* unstructured
  * [both SUPA and Native](#both-supa-and-native-implements)
  * [Only Native implements](#only-native-implements)
  * [Only SUPA implements](#only-supa-implement)
* structured
  * [SUPA unstructured with Native structured](#supa-unstructured-implements-and-native-structured-implement)
  * [SUPA structured with Native structured](#supa-structured-and-native-structured-implements)
  * [Only SUPA structured](#only-supa-structured)
  * [Only Native structured](#only-native-structured)

### Unstructured Code
#### Both SUPA and Native Implements

```yaml
# yaml file
supported:
   - _adaptive_avg_pool2d
to_native:
   - _adaptive_avg_pool2d
```
```cpp
// cpp source
   Tensor SUPANativeFunctions::_adaptive_avg_pool2d(const at::Tensor& input, const IntArrayRef output_size) {xxxxxx}
```

code generated like
```cpp
   try { return SUPANativeFunctions::_adaptive_avg_pool2d();} // supa  implementation.
   catch (const Exception& ex) {LOG(ex);}
   try { return at::native::_adaptive_avg_pool2d();}          // native implementation
   catch (const Exception& ex) {LOG(ex);}
   return cpu_fallback_xxxx();                                // cpu fallback handler
```
#### Only Native Implements

```yaml
# yaml
supported:
  # - _adaptive_avg_pool2d
to_native:
  - _adaptive_avg_pool2d
```
```cpp
// no active supa implement in cpp source
   /*Tensor SUPANativeFunctions::_adaptive_avg_pool2d(const at::Tensor& input, const IntArrayRef output_size) {xxxxxx} */
```

code generated like
```cpp
   try { return at::native::_adaptive_avg_pool2d();}          // native implementation
   catch (const Exception& ex) {LOG(ex);}
   return cpu_fallback_xxxx();                                // cpu fallback handler
```
#### Only SUPA implements
```yaml
# yaml
supported:
   - _adaptive_avg_pool2d
to_native:
   # - _adaptive_avg_pool2d
```
```cpp
// cpp source
   Tensor SUPANativeFunctions::_adaptive_avg_pool2d(const at::Tensor& input, const IntArrayRef output_size) {xxxxxx}
```
```cpp
// code generated
   try { return SUPANativeFunctions::_adaptive_avg_pool2d();} // supa  implementation.
   catch (const Exception& ex) {LOG(ex);}
   return cpu_fallback_xxxx();                                // cpu fallback handler
```
### Structured Code
#### SUPA unstructured implements and Native structured implement
```yaml
   # yaml
supported:
    - add.Tensor
    - add_.Tensor
    - add.out
to_native:
    - add.out
```
```cpp
// cpp source
   Tensor SUPANativeFunctions::add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {}
   Tensor SUPANativeFunctions::add_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {}
   Tensor SUPANativeFunctions::add_(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {}
   TORCH_IMPL_FUNC(add_out)(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out) {}
```
```cpp
// code genereated
   at::Tensor wrapper_PrivateUse1_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
      try{ return at::supa::SUPANativeFunctions::add(self, other, alpha); }  // supa implement
      catch (const c10::Error& ex) { TORCH_WARN(ex.msg()) };
      structured_add_out_functional op;
      op.meta(self, other, alpha);
      try { op.impl(self, other, alpha, op.outputs_[0]);}    // native implement
      catch (const c10::Error& ex) { TORCH_WARN(ex.msg());
         return at::native::call_fallback_fn<&torch_supa::supa_cpu_fallback, ATEN_OP2(add, Tensor)>::call(self, other, alpha); //cpu fallback
      }
      return std::move(op.outputs_[0]);
   }
   at::Tensor & wrapper_PrivateUse1_add_out_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
      try{ return at::supa::SUPANativeFunctions::add_out(self, other, alpha, out); } // supa implement
      catch (const c10::Error& ex) { TORCH_WARN(ex.msg()) };
      structured_add_out_out op(out);
      op.meta(self, other, alpha);
      try { op.impl(self, other, alpha, op.maybe_get_output(0));}     // native implement
      catch (const c10::Error& ex) { TORCH_WARN(ex.msg());
         return at::native::call_fallback_fn<&torch_supa::supa_cpu_fallback, ATEN_OP2(add, out)>::call(self, other, alpha, out); //cpu fallback
         }
      if (op.proxy_outputs_[0].has_value()) op.outputs_[0].get().copy_(*op.proxy_outputs_[0]);
      return out;
   }
   // similar with inplace kind.
```
#### SUPA structured and Native structured implements.
```yaml
# yaml
supported:
    - add.out
to_native:
    - add.out
```
```cpp
// cpp source
   // do not provide any unstructured implement for add.Tensor, add_.Tensor or add.out.
   SUPA_IMPL_FUNC(add_out)(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out) {}
   TORCH_IMPL_FUNC(add_out)(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out) {}
```
```cpp
// code generated
at::Tensor wrapper_PrivateUse1_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
   structured_add_out_functional op;
   op.meta(self, other, alpha);
   try { op.impl_supa(self, other, alpha, op.outputs_[0]);}  // supa structured implement
   catch (const c10::Error& ex) { TORCH_WARN(ex.msg());
      try { op.impl(self, other, alpha, op.outputs_[0]);}    // native structured implement
      catch (const c10::Error& ex) { TORCH_WARN(ex.msg());
         return at::native::call_fallback_fn<&torch_supa::supa_cpu_fallback, ATEN_OP2(add, Tensor)>::call(self, other, alpha); // cpu fallback
      }
   }
   return std::move(op.outputs_[0]);
}
at::Tensor & wrapper_PrivateUse1_add_out_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
   structured_add_out_out op(out);
   op.meta(self, other, alpha);
   try { op.impl_supa(self, other, alpha, op.maybe_get_output(0));} // supa structured implement
   catch (const c10::Error& ex) { TORCH_WARN(ex.msg());
   try { op.impl(self, other, alpha, op.maybe_get_output(0));}     // native structured implement
   catch (const c10::Error& ex) { TORCH_WARN(ex.msg());
      return at::native::call_fallback_fn<&torch_supa::supa_cpu_fallback, ATEN_OP2(add, out)>::call(self, other, alpha, out); // cpu fallback
      }
   }
   if (op.proxy_outputs_[0].has_value()) op.outputs_[0].get().copy_(*op.proxy_outputs_[0]);
   return out;
}
// similar inplace code.
```

#### Only SUPA structured
```yaml
# yaml
supported:
    - add.out
to_native:
    # - add.out
```
```cpp
// cpp source
   SUPA_IMPL_FUNC(add_out)(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out) {}
```

#### Only Native structured
```yaml
# yaml
supported:
    # - add.out
to_native:
    - add.out
```
```cpp
// cpp source
   TORCH_IMPL_FUNC(add_out)(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out) {}
```

### ❗ NOTE ❗
1. `TORCH_IMPL_FUNC` and `SUPA_IMPL_FUNC` must used under namespace of `at::supa`
2. Structured function (`SUPA_IMPL_FUNC`) and unstructured function (`SUPANativeFunctions::xxx`) for same OP can't exists simultaneously.
3. for structured functions, it needs `<out-kind-name>` exists in br_natvie_functions.yaml and `SUPA_IMPL_FUNC(<out-kind-name>)` in `cpp` code.
4. if codegen finds `SUPA_IMPL_FUNC(<out-kind-name>)` marco in `cpp` code.
    1. `<functional-kind-name>`, `<inplace-kind-name>` in yaml are ignored.
    2. if either `SUPANativeFunctions::<functional>`, `SUPANativeFunctions::<inplace>` or `SUPANativeFunctions::<out>` code exists in `cpp` file, codegen reports error.
