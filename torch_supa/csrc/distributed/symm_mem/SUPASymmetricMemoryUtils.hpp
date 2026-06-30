/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include "torch_supa/csrc/core/supa/TorchVersion.h"
#include "torch_supa/csrc/distributed/supa/Utils.hpp"
#include "torch_supa/csrc/distributed/symm_mem/SUPASymmetricMemoryTypes.hpp"

#include <torch/csrc/distributed/c10d/Store.hpp>
#if TORCH_VER >= TORCH_2_8_0
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#elif TORCH_VER >= TORCH_2_5_0
#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>
#endif

namespace c10d::supa {
namespace symmetric_memory {

bool device_has_multicast_support(int device_idx);

bool allow_overlapping_devices();

// Query environment variable to get the backend used for SUPA Symmetric Memory.
std::string getSymmMemBackendSUPA();

class IpcChannel {
 public:
  IpcChannel();
  IpcChannel(const IpcChannel&) = delete;
  IpcChannel& operator=(const IpcChannel&) = delete;
  IpcChannel(IpcChannel&&) = delete;
  IpcChannel& operator=(IpcChannel&&) = delete;
  ~IpcChannel();

  void send_fd(int dst_pid, int fd) const;
  int recv_fd();

  std::vector<int> all_gather_fds(int rank, const std::vector<int>& pids, int fd);

  int broadcast_fds(int rank, int src_rank, const std::vector<int>& pids, int fd);

 private:
  static std::string get_socket_name(int pid);

  std::string socket_name_;
  int socket_;
};

// A set of store-based exchange methods with a preset prefix typically type of
// the SymmetricMemory.  Most used as static instances at respective
// SymmetricMemory implementation files.
class StoreExchange {
 public:
  StoreExchange(const std::string& store_prefix) : store_prefix_(store_prefix) {}

  // Put template function in header file so that compiler can easily access it.
  template <typename T>
  std::vector<T> all_gather(const c10::intrusive_ptr<c10d::Store>& store, int rank, int world_size, T val) {
    static_assert(std::is_trivially_copyable_v<T>);

    std::vector<std::string> peer_keys;
    peer_keys.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
      std::ostringstream oss;
      oss << store_prefix_ << '/' << seq_id_ << '/' << r;
      peer_keys.push_back(oss.str());
    }
    ++seq_id_;

    {
      std::vector<uint8_t> payload(reinterpret_cast<uint8_t*>(&val), reinterpret_cast<uint8_t*>(&val) + sizeof(T));
      store->set(peer_keys[rank], payload);
    }

    std::vector<T> peer_vals;
    peer_vals.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
      if (r == rank) {
        peer_vals.push_back(val);
        continue;
      }
      store->wait({peer_keys[r]});
      auto payload = store->get(peer_keys[r]);
      TORCH_CHECK(payload.size() == sizeof(T));
      T peer_val{};
      std::memcpy(&peer_val, payload.data(), sizeof(T));
      peer_vals.push_back(peer_val);
    }
    return peer_vals;
  }

  void barrier(const c10::intrusive_ptr<c10d::Store>& store, int rank, int world_size) {
    // TODO: implement an efficient one?
    all_gather(store, rank, world_size, 0);
  }

 private:
  const std::string store_prefix_;
  size_t seq_id_ = 0;
};

// Returns a pointer of virtual address that is mapped to the physical memory
// held by the handle.
void map_block(void** ptr, c10d::supa::symmetric_memory::HandleType handle, size_t size, int device_idx);

} // namespace symmetric_memory
} // namespace c10d::supa
