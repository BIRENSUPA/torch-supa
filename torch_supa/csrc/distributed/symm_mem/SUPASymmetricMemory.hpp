/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <ATen/ATen.h>
#include "torch_supa/csrc/core/supa/SUPAAllocatorConfig.h"
#include "torch_supa/csrc/distributed/symm_mem/SUPASymmetricMemoryTypes.hpp"

#include <torch/csrc/distributed/c10d/Store.hpp>
#include "torch_supa/csrc/core/supa/TorchVersion.h"

#if TORCH_VER >= TORCH_2_8_0
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#elif TORCH_VER >= TORCH_2_5_0
#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>
#endif

namespace c10d::supa::symmetric_memory {

#if TORCH_VER >= TORCH_2_6_0

using ::c10d::symmetric_memory::SymmetricMemory;
using ::c10d::symmetric_memory::SymmetricMemoryAllocator;

// Resource wrapper that owns a (vaddr, allocation handle) pair. Upon
// destruction, it unmaps the vaddr and releases the allocation handle.
struct AllocationRef : public c10::intrusive_ptr_target {
  void* ptr;
  HandleType handle;
  size_t block_size;
  int device_idx;
  bool is_multicast;

  AllocationRef(void* ptr, HandleType handle, size_t block_size, int device_idx, bool is_multicast = false);

  ~AllocationRef();
};

// Forward declaration of SUPAPeerAllocInfo
class SUPAPeerAllocInfo;

class SUPASymmetricMemory : public SymmetricMemory {
 public:
  // This is mostly a shallow copy that shares the pointer to
  // `SUPAPeerAllocInfo` which corresponds to the base Block. The
  // SUPASymmetricMemory handle is specified by the offset to the base ptr.
  SUPASymmetricMemory(const c10::intrusive_ptr<SUPAPeerAllocInfo>& pai, size_t offset);

  ~SUPASymmetricMemory() override{};

  std::vector<void*> get_buffer_ptrs() override;
  std::vector<void*> get_signal_pad_ptrs() override;
  void** get_buffer_ptrs_dev() override;
  void** get_signal_pad_ptrs_dev() override;
  size_t get_buffer_size() override;
#if TORCH_VER >= TORCH_2_9_0
  size_t get_offset() override;
  c10::Device get_device() override;
  bool world_within_direct_access() override;
#endif
#if TORCH_VER >= TORCH_2_5_0 && TORCH_VER < TORCH_2_10_0
  size_t get_signal_pad_size() override;
#endif
#if TORCH_VER >= TORCH_2_5_0 && TORCH_VER < TORCH_2_9_0
  at::Tensor get_buffer(int rank, c10::IntArrayRef sizes, c10::ScalarType dtype, int64_t storage_offset) override;
#endif
#if TORCH_VER >= TORCH_2_6_0 && TORCH_VER < TORCH_2_9_0
  at::Tensor get_signal_pad(
      int rank,
      c10::IntArrayRef sizes,
      std::optional<c10::ScalarType> dtype,
      int64_t storage_offset) override;
#endif

  bool has_multicast_support() override;
  void* get_multicast_ptr() override;

  void barrier(int channel, size_t timeout_ms) override;
  void put_signal(int dst_rank, int channel, size_t timeout_ms) override;
  void wait_signal(int src_rank, int channel, size_t timeout_ms) override;

  int get_rank() override;
  int get_world_size() override;

 private:
  int local_device_idx_;
  int rank_;
  int world_size_;
  c10::intrusive_ptr<SUPAPeerAllocInfo> pai_;
  size_t offset_{0}; // in byte
};

// A class to hold the base pointers and signal pad pointers for a group of
// peers. One `SUPAPeerAllocInfo` object can be shared by multiple
// `SUPASymmetricMemory` objects when latter reside on the same allocation
// and rendezvous over the same group. (The `SUPASymmetricMemory` objects may
// have different offsets compared to the base address.)
class SUPAPeerAllocInfo : public c10::intrusive_ptr_target {
 public:
  SUPAPeerAllocInfo(
      std::vector<c10::intrusive_ptr<AllocationRef>> alloc_refs,
      std::vector<void*> buffers,
      std::vector<void*> signal_pads,
      HandleType mc_handle,
      void* mc_addr,
      size_t buffer_size,
      int local_device_idx,
      int rank,
      int world_size);

 private:
  std::vector<c10::intrusive_ptr<AllocationRef>> alloc_refs_;
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  [[maybe_unused]] HandleType mc_handle_;
  void* mc_addr_;
  size_t buffer_size_;
  int local_device_idx_;
  int rank_;
  int world_size_;
  void** buffers_dev_;
  void** signal_pads_dev_;

  friend class SUPASymmetricMemory;
};

// Metadata associated with each allocation performed by
// `SUPASymmetricMemoryAllocator`.
struct Block : public c10::intrusive_ptr_target {
  c10::intrusive_ptr<AllocationRef> alloc_ref;
  int device_idx;
  size_t block_size;
  size_t buffer_size;
  size_t signal_pad_offset;
  std::optional<std::string> default_group_name;
  std::map<std::string, c10::intrusive_ptr<SUPAPeerAllocInfo>> symm_mems;

  Block(
      c10::intrusive_ptr<AllocationRef> alloc_ref,
      int device_idx,
      size_t block_size,
      size_t buffer_size,
      size_t signal_pad_offset,
      const std::optional<std::string>& group_name);
};

class SUPASymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(size_t size, int device_idx, const std::optional<std::string>& group_name) override;

  void free(void* ptr) override;
  size_t get_alloc_size(void* ptr) override;
  c10::intrusive_ptr<SymmetricMemory> rendezvous(void* ptr, const std::optional<std::string>& group_name) override;
  bool has_multicast_support(int device_idx) override;
#if TORCH_VER >= TORCH_2_9_0
  c10::DeviceType supported_device_type() override;
  std::string name() override;
#endif

 private:
  c10::intrusive_ptr<Block> find_block(void* ptr);
  c10::intrusive_ptr<Block> find_block_covering(void* ptr, size_t& offset);

  std::shared_mutex mutex_;
  std::unordered_map<void*, c10::intrusive_ptr<Block>> ptr_to_block_;
  c10::supa::SUPACachingAllocator::Expandable_Segments_Handle_Type handle_type_ =
      c10::supa::SUPACachingAllocator::Expandable_Segments_Handle_Type::UNSPECIFIED;
};

#endif

} // namespace c10d::supa::symmetric_memory
