#pragma once

// ${generated_comment}

#include <ATen/core/Tensor.h>
// #include <ATen/core/IListRef.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorIterator.h>
#include <ATen/NativeFunctions.h>

${NativeMetaFunctions_includes}

#define SUPA_IMPL_FUNC(x) void at::supa::structured_##x::impl_supa
#define SUPA_PRECOMPUTE_META_FUNC(name) \
  structured_##name::meta_return_ty at::supa::structured_##name::meta
#define SUPA_META_FUNC(name) void at::supa::structured_##name::meta

namespace at {
namespace supa {

${NativeMetaFunctions_declarations}
} // namespace supa
} // namespace at
