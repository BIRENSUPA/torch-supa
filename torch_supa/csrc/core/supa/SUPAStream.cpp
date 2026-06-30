/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright (c) 2023-2024 Shanghai Biren Technology Co., Ltd. All rights
 * reserved.
 */

#include "torch_supa/csrc/core/supa/SUPAStream.h"
#include <c10/core/impl/GPUTrace.h>
#include <c10/util/CallOnce.h>
#include <c10/util/irange.h>
#include <array>
#include <atomic>
#include <cstdint>
#include "torch_supa/csrc/core/supa/SUPAException.h"
#include "torch_supa/csrc/core/supa/SUPAFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAGuard.h"

namespace c10::supa {

namespace {

// Global stream state and constants
c10::once_flag init_flag;
DeviceIndex num_gpus = -1;
constexpr int kStreamsPerPoolBits = 5;
constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
constexpr unsigned int kDefaultFlags = supaStreamNonBlocking;
constexpr int kStreamTypeBits = 4;

int max_stream_priorities;

std::array<c10::once_flag, C10_COMPILE_TIME_MAX_SUPA_GPUS> device_flags;
std::array<
    std::array<std::atomic<uint32_t>, C10_COMPILE_TIME_MAX_SUPA_GPUS>,
    c10::supa::max_compile_time_stream_priorities>
    priority_counters;

std::array<
    std::array<std::array<supaStream_t, kStreamsPerPool>, C10_COMPILE_TIME_MAX_SUPA_GPUS>,
    c10::supa::max_compile_time_stream_priorities>
    streams;

// Note [HIP Lazy Streams]
// ~~~~~~~~~~~~~~~~~~~~~~~
// For ROCm/HIP, each stream is lazily initialized rather than creating all
// streams when the first stream is requested. HIP streams are not as
// lightweight as SUPA streams; the pooling strategy can affect performance.
// Rather than changing the pooling implementation, ROCm/HIP will lazy init
// each stream when it is first requested.

// Note [StreamId assignment]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// How do we assign stream IDs?
//
// -- 54 bits --  -- 5 bits -----  -- 4 bits --     --1 bit --
// zeros          stream id index  StreamIdType     Ext/native stream
//                ignored for ext   ignored for ext
// for external stream, StreamID is a supaStream_t pointer
// this means that last bit will always be 0
// so when constructing StreamId for a native stream we set last bit to 1
// to distinguish between native and external streams
//
//
// We are obligated to treat the stream ID 0 as the default stream, per the
// invariant specified in c10::Stream, so this is one exception to
// "last bit = 1 for native streams". However, all other numbers are entirely
// an internal implementation detail, we reserve the right to renumber streams
// however we like.
//
// Note that it is really important that the MSB is zero; StreamId is a
// *signed* integer, and unsigned to signed conversion outside of the
// bounds of signed integer representation is undefined behavior.  You
// could work around this with something like
// https://stackoverflow.com/questions/13150449/efficient-unsigned-to-signed-cast-avoiding-implementation-defined-behavior
// but it seems a bit overkill for this.
//
// Also, external managed stream pointers (supaStream_t) can be directly stored
// in the Id field so in this case, we need to check the stream alignment.

class StreamIdType {
  // StreamIdType encodes whether this stream is DEFAULT, EXTernal or
  // for all other native streams, the stream priority (higher value is higher
  // priority)
 private:
  uint8_t stream_type;

 public:
  static const uint8_t DEFAULT = 0x0;
  static const uint8_t EXT = 0xF;

  StreamIdType(const uint8_t _stream_type) : stream_type(_stream_type) {}

  bool isExt() const {
    return EXT == stream_type;
  }

  bool isDefault() const {
    return DEFAULT == stream_type;
  }

  uint8_t getStreamType() const {
    return stream_type;
  }
};

std::ostream& operator<<(std::ostream& stream, StreamIdType s) {
  if (s.isDefault()) {
    stream << "DEFAULT";
  } else if (s.isExt()) {
    stream << "EXT";
  } else {
    stream << "PRIORITY " << int(s.getStreamType());
  }
  return stream;
}

// StreamId is 64-bit, so we can just rely on regular promotion rules.
// We rely on streamIdIndex and streamIdType being non-negative;
// see Note [Hazard when concatenating signed integers]
inline StreamIdType streamIdType(c10::StreamId s) {
  // Externally allocated streams have their id being the supaStream_ptr
  // so the last bit will be 0
  if ((!(s & 1)) && s) {
    return StreamIdType(StreamIdType::EXT);
  }
  // last bit is external/internal stream, the mask should start from second
  // rightmost bit
  int mask_for_type = (1 << kStreamTypeBits) - 1;
  auto val = (s >> 1) & mask_for_type;
  TORCH_INTERNAL_ASSERT(val || !(s & 1), "invalid StreamId", s);
  return StreamIdType(val);
}

inline size_t streamIdIndex(StreamId s) {
  return static_cast<size_t>((s >> (kStreamTypeBits + 1)) & ((1 << kStreamsPerPoolBits) - 1));
}

c10::StreamId makeStreamId(StreamIdType st, size_t si) {
  if (st.isDefault()) {
    return static_cast<StreamId>(0);
  }
  return (static_cast<StreamId>(si) << (kStreamTypeBits + 1)) | static_cast<StreamId>(st.getStreamType() << 1) | 1;
}

// Thread-local current streams
thread_local std::unique_ptr<StreamId[]> current_streams = nullptr;

void initGlobalStreamState() {
  num_gpus = device_count();
  // Check if the number of GPUs matches the expected compile-time max number
  // of GPUs.
  TORCH_CHECK(
      num_gpus <= C10_COMPILE_TIME_MAX_SUPA_GPUS,
      "Number of SUPA devices on the machine is larger than the compiled "
      "max number of gpus expected (",
      C10_COMPILE_TIME_MAX_SUPA_GPUS,
      "). Increase that and recompile.");

  int leastPriority = -1;
  int greatestPriority = -1;
  C10_SUPA_CHECK(supaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

  // greatestPriority is negative
  auto range = leastPriority - greatestPriority + 1;
  max_stream_priorities =
      range >= c10::supa::max_compile_time_stream_priorities ? c10::supa::max_compile_time_stream_priorities : range;
}

// Init a single SUPA or HIP stream
// See Note [HIP Lazy Streams]
void initSingleStream(int p, DeviceIndex device_index, int i) {
  SUPAGuard device_guard(device_index);
  auto& stream = streams.at(p).at(device_index).at(i);
  auto pri = -p; // lower number is higher priority

  C10_SUPA_CHECK(supaStreamCreateWithPriority(&stream, kDefaultFlags, pri));
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
#if TORCH_VER >= TORCH_2_4_0
    (*interp)->trace_gpu_stream_creation(c10::kPrivateUse1, reinterpret_cast<uintptr_t>(stream));
#else
    (*interp)->trace_gpu_stream_creation(reinterpret_cast<uintptr_t>(stream));
#endif
    priority_counters.at(p).at(device_index) = 0;
  }
}

// Creates the low and high priority stream pools for the specified device
// Warning: only call once per device!
void initDeviceStreamState(DeviceIndex device_index) {
  // Switches to the requested device so streams are properly associated
  // with it.
  SUPAGuard device_guard{device_index};
  for (const auto i : c10::irange(kStreamsPerPool)) {
    for (const auto p : c10::irange(max_stream_priorities)) {
      initSingleStream(p, device_index, i);
    }
  }
}

// Init front-end to ensure initialization only occurs once
void initSUPAStreamsOnce() {
  // Inits default streams (once, globally)
  c10::call_once(init_flag, initGlobalStreamState);

  if (current_streams) {
    return;
  }

  // Inits current streams (thread local) to default streams
  // NOLINTNEXTLINE(*-arrays)
  current_streams = std::make_unique<StreamId[]>(num_gpus);
  for (const auto i : c10::irange(num_gpus)) {
    current_streams[i] = makeStreamId(StreamIdType::DEFAULT, 0);
  }
}

// Helper to verify the GPU index is valid
inline void check_gpu(DeviceIndex device_index) {
  TORCH_CHECK(
      device_index >= 0 && device_index < num_gpus,
      "Device index value ",
      static_cast<int>(device_index),
      " is out of index range [0, ",
      static_cast<int>(num_gpus),
      ")");
}

// Helper to determine the index of the stream to return
// Note: Streams are returned round-robin (see note in SUPAStream.h)
uint32_t get_idx(std::atomic<uint32_t>& counter) {
  auto raw_idx = counter++;
  return raw_idx % kStreamsPerPool;
}

SUPAStream SUPAStreamForId(DeviceIndex device_index, StreamId stream_id) {
  return SUPAStream(
      SUPAStream::UNCHECKED, Stream(Stream::UNSAFE, c10::Device(DeviceType::PrivateUse1, device_index), stream_id));
}

} // anonymous namespace

// =================== SUPAStream =============================
supaStream_t SUPAStream::stream() const {
  c10::DeviceIndex device_index = stream_.device_index();
  StreamId stream_id = stream_.id();
  StreamIdType st = streamIdType(stream_id);
  size_t si = streamIdIndex(stream_id);
  if (st.isDefault()) {
    TORCH_INTERNAL_ASSERT(
        si == 0,
        "Unrecognized stream ",
        stream_,
        " (I think this should be the default stream, but I got a non-zero index ",
        si,
        ").",
        " Did you manufacture the StreamId yourself?  Don't do that; use the",
        " official API like c10::supa::getStreamFromPool() to get a new stream.");
    // should not return nullptr, supa driver could not handle it
    return 0;
  }
  if (st.isExt()) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<supaStream_t>(stream_id);
  }
  auto streamType = st.getStreamType();
  TORCH_INTERNAL_ASSERT(
      streamType >= 1 && streamType <= max_stream_priorities,
      "Unrecognized stream ",
      stream_,
      " (I didn't recognize the stream type, ",
      st,
      " with the value ",
      streamType,
      ")");
  return streams.at(st.getStreamType() - 1).at(device_index).at(si);
}

std::tuple<int, int> SUPAStream::priority_range() {
  // Note: this returns the range of priority **supported by PyTorch**, not
  // the range of priority **supported by SUPA**. The former is a subset of
  // the latter.
  int least_priority = 0;
  int greatest_priority = 0;
  C10_SUPA_CHECK(supaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));

  TORCH_INTERNAL_ASSERT(least_priority == 0, "Unexpected SUPA stream priority range");

  TORCH_INTERNAL_ASSERT(greatest_priority <= -1, "Unexpected SUPA stream priority range");
  greatest_priority = std::max(-c10::supa::max_compile_time_stream_priorities + 1, greatest_priority);
  return std::make_tuple(least_priority, greatest_priority);
}

// Returns a stream from the requested pool
// Note: when called the first time on a device, this will create the
// stream pools for that device.
SUPAStream getStreamFromPool(const int priority, DeviceIndex device_index) {
  initSUPAStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
    c10::supa::SetTargetDevice();
  }
  check_gpu(device_index);
  // SUPA-only: Initializes the stream pools (once)
  c10::call_once(device_flags.at(device_index), initDeviceStreamState, device_index);
  auto pri_idx = std::clamp(-priority, 0, max_stream_priorities - 1);
  const auto idx = get_idx(priority_counters.at(pri_idx).at(device_index));
  StreamIdType id_type = StreamIdType(pri_idx + 1);
  return SUPAStreamForId(device_index, makeStreamId(id_type, idx));
}

SUPAStream getStreamFromPool(const bool isHighPriority, DeviceIndex device) {
  initSUPAStreamsOnce();
  int priority = isHighPriority ? -max_stream_priorities + 1 : 0;
  return getStreamFromPool(priority, device);
}

SUPAStream getStreamFromExternal(supaStream_t ext_stream, DeviceIndex device_index) {
  // The stream pointer will be the actual id
  return SUPAStreamForId(device_index, reinterpret_cast<int64_t>(ext_stream));
}

SUPAStream getDefaultSUPAStream(DeviceIndex device_index) {
  initSUPAStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
    c10::supa::SetTargetDevice();
  }
  check_gpu(device_index);
  return SUPAStreamForId(device_index, makeStreamId(StreamIdType::DEFAULT, 0));
}

SUPAStream getCurrentSUPAStream(DeviceIndex device_index) {
  initSUPAStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
    c10::supa::SetTargetDevice();
  }
  check_gpu(device_index);
  return SUPAStreamForId(device_index, current_streams[device_index]);
}

void setCurrentSUPAStream(SUPAStream stream) {
  initSUPAStreamsOnce();
  current_streams[stream.device_index()] = stream.id();
}

std::ostream& operator<<(std::ostream& stream, const SUPAStream& s) {
  return stream << s.unwrap();
}

SUPAStream::SUPAStream(c10::Stream stream) : stream_(stream) {
  TORCH_CHECK(stream_.device_type() == c10::DeviceType::PrivateUse1);
}

bool SUPAStream::query() const {
  supaError_t err = supaStreamQuery(stream());

  if (err == supaSuccess) {
    return true;
  }

  if (err != supaErrorNotReady) {
    C10_SUPA_CHECK(err);
  }
  (void)supaGetLastError();

  return false;
}

int SUPAStream::priority() const {
  c10::DeviceGuard guard{stream_.device()};
  int priority = 0;
  C10_SUPA_CHECK(supaStreamGetPriority(stream(), &priority));
  return priority;
}

void SUPAStream::synchronize() const {
  c10::DeviceGuard guard{stream_.device()};
  c10::supa::stream_synchronize(stream());
}

} // namespace c10::supa
