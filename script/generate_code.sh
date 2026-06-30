#!/bin/bash

# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ROOT="$(cd "$(dirname "$0")/.." ; pwd -P)"

if [ $# -lt 1 ]
then
   echo "Usage $0 <torch_ver>"
   exit -1
fi
TORCH_VER=${1}

failure_check()
{
  if [ $? -ne 0 ]; then
    echo "$1"
    exit 1
  fi
}

cd $ROOT/script

python3 -m codegen_native.gen \

source_yaml="$ROOT/torch_supa/csrc/aten/supa_native_functions.yaml"

python3 -m codegen_supa.gen \
  --torch-version ${TORCH_VER} \
  --install-dir ${ROOT}/torch_supa/csrc/aten/ \
  --custom-function-yaml ${source_yaml} \
  --impl-path ${ROOT}/torch_supa/csrc/aten/ \
  --custom-ops-dir ${ROOT}/torch_supa/utils/ 

failure_check "Failed to generate SUPA derivative stubs."
