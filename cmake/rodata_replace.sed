# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.


# keep the same size; eg: sizeof(CUDNN_BACKEND) == sizeof(SUDNN_BACKEND)
# Without 'g' flag: replace only first match per line
s/CUDNN_BACKEND/SUDNN_BACKEND/g
s/cudnnCreate/sudnnCreate/g
s/cudnnConvolutionBiasActivationForward/sudnnConvolutionBiasActivationForward/g
s/cudaMemcpy/supaMemcpy/g
s/cudaHostRegister/supaHostRegister/g
s/cudaGetDevice/supaGetDevice/g
s/cublasLtMatmul/sublasLtMatmul/g
s/nvrtc/brrtc/g
