/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>

#include <torch_supa/csrc/core/supa/SUPAMacros.h>
#include <torch_supa/csrc/core/supa/SUPAStream.h>
#include <torch_supa/csrc/core/supa/impl/SUPAGuardImpl.h>

#include <cstddef>

namespace c10::supa {

// This code is kind of boilerplatey.  See Note [Whither the DeviceGuard
// boilerplate]

/// A variant of DeviceGuard that is specialized for SUPA.  It accepts
/// integer indices (interpreting them as SUPA devices) and is a little
/// more efficient than DeviceGuard (it compiles to straight line
/// supaSetDevice/supaGetDevice calls); however, it can only be used
/// from code that links against SUPA directly.
struct SUPAGuard {
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit SUPAGuard() = delete;

  /// Set the current SUPA device to the passed device index.
  explicit SUPAGuard(DeviceIndex device_index) : guard_(device_index) {}

  /// Sets the current SUPA device to the passed device.  Errors if the passed
  /// device is not a SUPA device.
  explicit SUPAGuard(Device device) : guard_(device) {}

  // Copy is not allowed
  SUPAGuard(const SUPAGuard&) = delete;
  SUPAGuard& operator=(const SUPAGuard&) = delete;

  // Move is not allowed (there is no uninitialized state)
  SUPAGuard(SUPAGuard&& other) = delete;
  SUPAGuard& operator=(SUPAGuard&& other) = delete;
  ~SUPAGuard() = default;

  /// Sets the SUPA device to the given device.  Errors if the given device
  /// is not a SUPA device.
  void set_device(Device device) {
    guard_.set_device(device);
  }

  /// Sets the SUPA device to the given device.  Errors if the given device
  /// is not a SUPA device.  (This method is provided for uniformity with
  /// DeviceGuard).
  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  /// Sets the SUPA device to the given device index.
  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set upon construction of the guard
  Device original_device() const {
    return guard_.original_device();
  }

  /// Returns the last device that was set via `set_device`, if any, otherwise
  /// the device passed during construction.
  Device current_device() const {
    return guard_.current_device();
  }

 private:
  /// The guard for the current device.
  c10::impl::InlineDeviceGuard<impl::SUPAGuardImpl> guard_;
};

/// A variant of OptionalDeviceGuard that is specialized for SUPA.  See
/// SUPAGuard for when you can use this.
struct OptionalSUPAGuard {
  /// Create an uninitialized OptionalSUPAGuard.
  explicit OptionalSUPAGuard() = default;

  /// Set the current SUPA device to the passed Device, if it is not nullopt.
  explicit OptionalSUPAGuard(optional<Device> device_opt) : guard_(device_opt) {}

  /// Set the current SUPA device to the passed device index, if it is not
  /// nullopt
  explicit OptionalSUPAGuard(optional<DeviceIndex> device_index_opt) : guard_(device_index_opt) {}

  // Copy is not allowed
  OptionalSUPAGuard(const OptionalSUPAGuard&) = delete;
  OptionalSUPAGuard& operator=(const OptionalSUPAGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalSUPAGuard(OptionalSUPAGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalSUPAGuard& operator=(OptionalSUPAGuard&& other) = delete;
  ~OptionalSUPAGuard() = default;

  /// Sets the SUPA device to the given device, initializing the guard if it
  /// is not already initialized.  Errors if the given device is not a SUPA
  /// device.
  void set_device(Device device) {
    guard_.set_device(device);
  }

  /// Sets the SUPA device to the given device, initializing the guard if it is
  /// not already initialized.  Errors if the given device is not a SUPA device.
  /// (This method is provided for uniformity with OptionalDeviceGuard).
  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  /// Sets the SUPA device to the given device index, initializing the guard if
  /// it is not already initialized.
  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set immediately prior to initialization of the
  /// guard, or nullopt if the guard is uninitialized.
  optional<Device> original_device() const {
    return guard_.original_device();
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  optional<Device> current_device() const {
    return guard_.current_device();
  }

  /// Restore the original SUPA device, resetting this guard to uninitialized
  /// state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalDeviceGuard<impl::SUPAGuardImpl> guard_;
};

/// A variant of StreamGuard that is specialized for SUPA.  See SUPAGuard
/// for when you can use this.
struct SUPAStreamGuard {
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit SUPAStreamGuard() = delete;

  /// Set the current SUPA device to the device associated with the passed
  /// stream, and set the current SUPA stream on that device to the passed
  /// stream. Errors if the Stream is not a SUPA stream.
  explicit SUPAStreamGuard(Stream stream) : guard_(stream) {}

  /// Copy is disallowed
  SUPAStreamGuard(const SUPAStreamGuard&) = delete;
  SUPAStreamGuard& operator=(const SUPAStreamGuard&) = delete;

  /// Move is disallowed, as SUPAStreamGuard does not have an uninitialized
  /// state, which is required for moves on types with nontrivial destructors.
  SUPAStreamGuard(SUPAStreamGuard&& other) = delete;
  SUPAStreamGuard& operator=(SUPAStreamGuard&& other) = delete;
  ~SUPAStreamGuard() = default;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Errors if the stream passed is not a SUPA stream.
  ///
  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  ///
  /// WARNING: reset_stream does NOT preserve previously set streams on
  /// different devices.  If you need to set streams on multiple devices
  /// on SUPA, use SUPAMultiStreamGuard instead.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the SUPA stream that was set at the time the guard was
  /// constructed.
  SUPAStream original_stream() const {
    return SUPAStream(SUPAStream::UNCHECKED, guard_.original_stream());
  }

  /// Returns the most recent SUPA stream that was set using this device guard,
  /// either from construction, or via set_stream.
  SUPAStream current_stream() const {
    return SUPAStream(SUPAStream::UNCHECKED, guard_.current_stream());
  }

  /// Returns the most recent SUPA device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  Device current_device() const {
    return guard_.current_device();
  }

  /// Returns the SUPA device that was set at the most recent reset_stream(),
  /// or otherwise the device at construction time.
  Device original_device() const {
    return guard_.original_device();
  }

 private:
  c10::impl::InlineStreamGuard<impl::SUPAGuardImpl> guard_;
};

/// A variant of OptionalStreamGuard that is specialized for SUPA.  See
/// SUPAGuard for when you can use this.
struct OptionalSUPAStreamGuard {
  /// Create an uninitialized guard.
  explicit OptionalSUPAStreamGuard() = default;

  /// Set the current SUPA device to the device associated with the passed
  /// stream, and set the current SUPA stream on that device to the passed
  /// stream. Errors if the Stream is not a SUPA stream.
  explicit OptionalSUPAStreamGuard(Stream stream) : guard_(stream) {}

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream,
  /// if the passed stream is not nullopt.
  explicit OptionalSUPAStreamGuard(optional<Stream> stream_opt) : guard_(stream_opt) {}

  /// Copy is disallowed
  OptionalSUPAStreamGuard(const OptionalSUPAStreamGuard&) = delete;
  OptionalSUPAStreamGuard& operator=(const OptionalSUPAStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalSUPAStreamGuard(OptionalSUPAStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalSUPAStreamGuard& operator=(OptionalSUPAStreamGuard&& other) = delete;
  ~OptionalSUPAStreamGuard() = default;

  /// Resets the currently set SUPA stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Initializes the guard if it was not previously initialized.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the SUPA stream that was set at the time the guard was most
  /// recently initialized, or nullopt if the guard is uninitialized.
  optional<SUPAStream> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return make_optional(SUPAStream(SUPAStream::UNCHECKED, r.value()));
    }
    return nullopt;
  }

  /// Returns the most recent SUPA stream that was set using this stream guard,
  /// either from construction, or via reset_stream, if the guard is
  /// initialized, or nullopt if the guard is uninitialized.
  optional<SUPAStream> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return make_optional(SUPAStream(SUPAStream::UNCHECKED, r.value()));
    }
    return nullopt;
  }

  /// Restore the original SUPA device and stream, resetting this guard to
  /// uninitialized state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalStreamGuard<impl::SUPAGuardImpl> guard_;
};

/// A variant of MultiStreamGuard that is specialized for SUPA.
struct SUPAMultiStreamGuard {
  explicit SUPAMultiStreamGuard(ArrayRef<SUPAStream> streams) : guard_(unwrapStreams(streams)) {}

  /// Copy is disallowed
  SUPAMultiStreamGuard(const SUPAMultiStreamGuard&) = delete;
  SUPAMultiStreamGuard& operator=(const SUPAMultiStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  SUPAMultiStreamGuard(SUPAMultiStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  SUPAMultiStreamGuard& operator=(SUPAMultiStreamGuard&& other) = delete;
  ~SUPAMultiStreamGuard() = default;

 private:
  c10::impl::InlineMultiStreamGuard<impl::SUPAGuardImpl> guard_;

  static std::vector<Stream> unwrapStreams(ArrayRef<SUPAStream> supaStreams) {
    std::vector<Stream> streams;
    streams.reserve(supaStreams.size());
    for (const SUPAStream& supaStream : supaStreams) {
      streams.push_back(supaStream);
    }
    return streams;
  }
};
} // namespace c10::supa
