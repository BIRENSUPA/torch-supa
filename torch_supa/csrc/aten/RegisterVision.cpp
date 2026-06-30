/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <torch/library.h>

#include <c10/util/Exception.h>

namespace vision::ops {
#ifdef TORCH_SUPA_OP_DIR
void torchvision_register_all();
__attribute__((constructor)) void torchvision_init(void) {
   torchvision_register_all();
}
#endif
}
