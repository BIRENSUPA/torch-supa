.. _native-api-torch_storage:

torch.Storage
================


.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.TypedStorage``
     - 是
     - 
   * - ``torch.TypedStorage.bfloat16``
     - 
     - 
   * - ``torch.TypedStorage.bool``
     - 是
     - 
   * - ``torch.TypedStorage.byte``
     - 是
     - 
   * - ``torch.TypedStorage.char``
     - 是
     - 
   * - ``torch.TypedStorage.clone``
     - 是
     - 
   * - ``torch.TypedStorage.complex_double``
     - 
     - 
   * - ``torch.TypedStorage.complex_float``
     - 
     - 
   * - ``torch.TypedStorage.copy_``
     - 是
     - 
   * - ``torch.TypedStorage.cpu``
     - 是
     - 
   * - ``torch.TypedStorage.cuda``
     - 是
     - 
   * - ``torch.TypedStorage.data_ptr``
     - 是
     - 
   * - ``torch.TypedStorage.device``
     - 
     - 
   * - ``torch.TypedStorage.double``
     - 是
     - 
   * - ``torch.TypedStorage.dtype``
     - 
     - 
   * - ``torch.TypedStorage.element_size``
     - 是
     - 
   * - ``torch.TypedStorage.fill_``
     - 是
     - 
   * - ``torch.TypedStorage.float``
     - 是
     - 
   * - ``torch.TypedStorage.from_buffer``
     - 是
     - 
   * - ``torch.TypedStorage.from_file``
     - 是
     - 
   * - ``torch.TypedStorage.get_device``
     - 是
     - 
   * - ``torch.TypedStorage.half``
     - 是
     - 
   * - ``torch.TypedStorage.hpu``
     - 
     - 
   * - ``torch.TypedStorage.int``
     - 是
     - 
   * - ``torch.TypedStorage.is_cuda``
     - 
     - 
   * - ``torch.TypedStorage.is_hpu``
     - 
     - 
   * - ``torch.TypedStorage.is_pinned``
     - 是
     - 
   * - ``torch.TypedStorage.is_shared``
     - 是
     - 
   * - ``torch.TypedStorage.is_sparse``
     - 
     - 
   * - ``torch.TypedStorage.long``
     - 是
     - 
   * - ``torch.TypedStorage.nbytes``
     - 是
     - 
   * - ``torch.TypedStorage.pickle_storage_type``
     - 是
     - 
   * - ``torch.TypedStorage.pin_memory``
     - 是
     - 
   * - ``torch.TypedStorage.resize_``
     - 是
     - 
   * - ``torch.TypedStorage.share_memory_``
     - 
     - 
   * - ``torch.TypedStorage.short``
     - 是
     - 
   * - ``torch.TypedStorage.size``
     - 是
     - 
   * - ``torch.TypedStorage.tolist``
     - 是
     - 
   * - ``torch.TypedStorage.type``
     - 是
     - 
   * - ``torch.TypedStorage.untyped``
     - 是
     - 
   * - ``torch.UntypedStorage``
     - 是
     - 
   * - ``torch.UntypedStorage.bfloat16``
     - 
     - 
   * - ``torch.UntypedStorage.bool``
     - 是
     - 
   * - ``torch.UntypedStorage.byte``
     - 是
     - 
   * - ``torch.UntypedStorage.byteswap``
     - 
     - 
   * - ``torch.UntypedStorage.char``
     - 是
     - 
   * - ``torch.UntypedStorage.clone``
     - 是
     - 
   * - ``torch.UntypedStorage.complex_double``
     - 
     - 
   * - ``torch.UntypedStorage.complex_float``
     - 
     - 
   * - ``torch.UntypedStorage.copy_``
     - 是
     - 
   * - ``torch.UntypedStorage.cpu``
     - 是
     - 
   * - ``torch.UntypedStorage.cuda``
     - 是
     - 
   * - ``torch.UntypedStorage.data_ptr``
     - 是
     - 
   * - ``torch.UntypedStorage.device``
     - 
     - 
   * - ``torch.UntypedStorage.double``
     - 是
     - 
   * - ``torch.UntypedStorage.element_size``
     - 是
     - 
   * - ``torch.UntypedStorage.fill_``
     - 是
     - 
   * - ``torch.UntypedStorage.float``
     - 是
     - 
   * - ``torch.UntypedStorage.from_buffer``
     - 
     - 
   * - ``torch.UntypedStorage.from_file``
     - 
     - 
   * - ``torch.UntypedStorage.get_device``
     - 是
     - 
   * - ``torch.UntypedStorage.half``
     - 是
     - 
   * - ``torch.UntypedStorage.hpu``
     - 
     - 
   * - ``torch.UntypedStorage.int``
     - 是
     - 
   * - ``torch.UntypedStorage.is_cuda``
     - 
     - 
   * - ``torch.UntypedStorage.is_hpu``
     - 
     - 
   * - ``torch.UntypedStorage.is_pinned``
     - 是
     - 
   * - ``torch.UntypedStorage.is_shared``
     - 是
     - 
   * - ``torch.UntypedStorage.is_sparse``
     - 
     - 
   * - ``torch.UntypedStorage.is_sparse_csr``
     - 
     - 
   * - ``torch.UntypedStorage.long``
     - 是
     - 
   * - ``torch.UntypedStorage.mps``
     - 
     - 
   * - ``torch.UntypedStorage.nbytes``
     - 是
     - 
   * - ``torch.UntypedStorage.new``
     - 是
     - 
   * - ``torch.UntypedStorage.pin_memory``
     - 是
     - 
   * - ``torch.UntypedStorage.resize_``
     - 是
     - 
   * - ``torch.UntypedStorage.share_memory_``
     - 
     - 
   * - ``torch.UntypedStorage.short``
     - 是
     - 
   * - ``torch.UntypedStorage.size``
     - 是
     - 
   * - ``torch.UntypedStorage.tolist``
     - 是
     - 
   * - ``torch.UntypedStorage.type``
     - 是
     - 
   * - ``torch.UntypedStorage.untyped``
     - 是
     - 
   * - ``torch.DoubleStorage``
     - 是
     - 
   * - ``torch.DoubleStorage.dtype``
     - 
     - 
   * - ``torch.FloatStorage``
     - 是
     - 
   * - ``torch.FloatStorage.dtype``
     - 
     - 
   * - ``torch.HalfStorage``
     - 是
     - 
   * - ``torch.HalfStorage.dtype``
     - 
     - 
   * - ``torch.LongStorage``
     - 是
     - 
   * - ``torch.LongStorage.dtype``
     - 
     - 
   * - ``torch.IntStorage``
     - 是
     - 
   * - ``torch.IntStorage.dtype``
     - 
     - 
   * - ``torch.ShortStorage``
     - 是
     - 
   * - ``torch.ShortStorage.dtype``
     - 
     - 
   * - ``torch.CharStorage``
     - 是
     - 
   * - ``torch.CharStorage.dtype``
     - 
     - 
   * - ``torch.ByteStorage``
     - 是
     - 
   * - ``torch.ByteStorage.dtype``
     - 
     - 
   * - ``torch.BoolStorage``
     - 是
     - 
   * - ``torch.BoolStorage.dtype``
     - 
     - 
   * - ``torch.BFloat16Storage``
     - 
     - 
   * - ``torch.BFloat16Storage.dtype``
     - 
     - 
   * - ``torch.ComplexDoubleStorage``
     - 
     - 
   * - ``torch.ComplexDoubleStorage.dtype``
     - 
     - 
   * - ``torch.ComplexFloatStorage``
     - 
     - 
   * - ``torch.ComplexFloatStorage.dtype``
     - 
     - 
   * - ``torch.QUInt8Storage``
     - 
     - 
   * - ``torch.QUInt8Storage.dtype``
     - 
     - 
   * - ``torch.QInt8Storage``
     - 
     - 
   * - ``torch.QInt8Storage.dtype``
     - 
     - 
   * - ``torch.QInt32Storage``
     - 
     - 
   * - ``torch.QInt32Storage.dtype``
     - 
     - 
   * - ``torch.QUInt4x2Storage``
     - 
     - 
   * - ``torch.QUInt4x2Storage.dtype``
     - 
     - 
   * - ``torch.QUInt2x4Storage``
     - 
     - 
   * - ``torch.QUInt2x4Storage.dtype``
     - 
     - 
