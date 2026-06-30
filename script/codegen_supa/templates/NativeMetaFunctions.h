#pragma once

// ${generated_comment}

#include <ATen/core/Tensor.h>
// #include <ATen/core/IListRef.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorIterator.h>
#include <ATen/NativeFunctions.h>

${NativeMetaFunctions_includes}

#define SUPA_IMPL_FUNC(x) void at::supa::structured_##x::impl_supa

namespace at {
namespace supa {

${NativeMetaFunctions_declarations}
} // namespace supa
} // namespace at
