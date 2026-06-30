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

#include <torch/csrc/dynamo/compiled_autograd.h>

#include "torch_supa/csrc/aten/autograd/FunctionsManual.h"
#include "torch_supa/csrc/aten/CustomFunctions.h"
#include "torch_supa/csrc/aten/core/TorchVersion.h"

// ${generated_comment}

// The manual function definitions that used to be here are now in torch/csrc/autograd/FunctionsManual.cpp
// This speeds up re-compilation and allow to share these implementations so that they can be
// used for forward mode AD formulas as well.

using namespace at::supa::autograd::generated::details;
using namespace at::supa::native::custom_ops;
using at::IntArrayRef;
using at::Scalar;
using at::Tensor;
using at::TensorList;

namespace at {
namespace supa {
namespace autograd {
namespace generated {

#if defined(TORCH_2_7_0) && TORCH_VER >= TORCH_2_7_0
static at::IValue compute_output_metadata(const torch::autograd::edge_list& next_edges) {
  auto output_metadata = torch::dynamo::autograd::IValuePacker<std::vector<std::optional<InputMetadata>>>::pack(
      torch::dynamo::autograd::get_input_metadata(next_edges));
  return output_metadata;
}

static C10_NOINLINE variable_list compiled_autograd_apply_functional(
    const PackedArgs& packed_args,
    const edge_list& next_edges,
    SwapSavedVariables& saved,
    const variable_list& grads,
    const std::string& name) {
  auto output_metadata = compute_output_metadata(next_edges);
  const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
  return pyinterface->call_function(
      saved.get_py_compiler(), "apply_functional", name, grads, packed_args.vec(), output_metadata);
}
#endif

${autograd_function_definitions}

} // namespace generated
} // namespace autograd
} // namespace supa
} // namespace at
