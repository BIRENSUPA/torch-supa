// Copyright © 2023 Shanghai Biren Technology Co., Ltd. All rights reserved.
// Copyright (c) 2022, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/csrc/autograd/VariableTypeUtils.h>

#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <ATen/core/TorchDispatchUtils.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/RedispatchFunctions.h>
#include <torch/library.h>

#include "torch_supa/csrc/aten/autograd/FunctionsManual.h"
#include "torch_supa/csrc/aten/autograd/GradFnHolder.h"
#include "torch_supa/csrc/aten/autograd/generated/VariableType.h"
#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"
#include "torch_supa/csrc/aten/CustomRedispatch.h"

// ${generated_comment}

// NOTE [Sharded File]: on this file's split-into-shards state
//
// Back in the good old days, VariableType.cpp was generated as one
// file with every function in it, and everything was great and
// simple.
//
// However, this file was also very large (over 36,000 lines), and
// compiling it was very slow, and in fact was a significant
// bottleneck for incremental rebuilds. To address this, we now
// generate the file split across multiple shards, named
// VariableType_0.cpp and so on, which can be compiled in parallel.
//
// For ease of inspection and debugging, so that it's not necessary to
// go rooting around in multiple files, we also generate all the
// functions together in VariableTypeEverything.cpp. This generated
// file is only for convenience; it's not actually used in the
// build. If the file you're looking at now is one of the shards, you
// may want to switch over to the Everything variant to make you
// grepping smoother.

using namespace at;
using namespace at::supa::autograd::generated;
using namespace at::supa::autograd::generated::details;

namespace at {
namespace supa {
namespace autograd {

namespace VariableType {
namespace {
C10_UNUSED void reset_grad_accumulator(Variable& self) {
  AutogradMeta* meta = torch::autograd::impl::get_autograd_meta(self);
  if (meta != nullptr) {
    meta->grad_accumulator_.reset();
  }
}
} // namespace


${type_derived_method_definitions}

} // namespace VariableType

namespace {

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  ${wrapper_registrations_aten}
}

TORCH_LIBRARY_IMPL(supa, AutogradPrivateUse1, m) {
    ${wrapper_registrations_supa}
}

}
} // namespace autograd
} // namespace supa
} // namespace at
