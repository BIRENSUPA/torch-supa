/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/*
 * Copyright (c) 2025 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#include <atomic>
#include <cstring>
#include <functional>
#include <vector>

#pragma once
namespace torch_supa::utils {

template <typename T>
class BaseRingBuffer {
 protected:
  std::size_t buffer_length_;
  std::atomic<size_t> head_;
  std::atomic<size_t> tail_;
  explicit BaseRingBuffer(std::size_t size) : buffer_length_(size), head_(0), tail_(0) {}

 public:
  BaseRingBuffer(const BaseRingBuffer&) = delete;
  BaseRingBuffer& operator=(const BaseRingBuffer&) = delete;
  BaseRingBuffer(BaseRingBuffer&&) = delete;
  BaseRingBuffer& operator=(BaseRingBuffer&&) = delete;
  virtual ~BaseRingBuffer() = default;
  virtual void write_buffer(const T& value, size_t currentTail) = 0;
  virtual void read_buffer(T& value, size_t currentHead) = 0;

  bool push(const T& value) {
    size_t currentTail = tail_.load(std::memory_order_relaxed);
    size_t nextTail = (currentTail + 1) % buffer_length_;
    size_t currentHead = head_.load(std::memory_order_acquire);

    // Check if the buffer is full
    if (nextTail == currentHead) {
      // Force write by moving the head forward
      size_t newHead = (currentHead + 1) % buffer_length_;
      // Use CAS to update the head
      while (!head_.compare_exchange_weak(currentHead, newHead, std::memory_order_release, std::memory_order_relaxed)) {
        if (currentHead == newHead) {
          // Another producer has already moved the head, break the loop
          break;
        }
      }
    }

    // Write the value to the buffer
    write_buffer(value, currentTail);

    // Use CAS to update the tail
    while (!tail_.compare_exchange_weak(currentTail, nextTail, std::memory_order_release, std::memory_order_relaxed)) {
      // The tail has been changed by another producer, retry
    }

    return true;
  }

  bool pop(T& value) {
    size_t currentHead = head_.load(std::memory_order_relaxed);
    size_t currentTail = tail_.load(std::memory_order_acquire);

    // Check if the buffer is empty
    if (currentHead == currentTail) {
      return false;
    }

    // Read the value from the buffer
    read_buffer(value, currentHead);
    size_t newHead = (currentHead + 1) % buffer_length_;

    // Use CAS to update the head
    while (!head_.compare_exchange_weak(currentHead, newHead, std::memory_order_release, std::memory_order_relaxed)) {
      // The head has been changed by another consumer, retry
    }

    return true;
  }
};

template <typename T>
class RingBuffer : public BaseRingBuffer<T> {
 private:
  std::vector<T> buffer_;

 public:
  explicit RingBuffer(std::size_t size) : BaseRingBuffer<T>(size), buffer_(size) {}

  void write_buffer(const T& value, size_t currentTail) override {
    buffer_[currentTail] = value;
  }

  void read_buffer(T& value, size_t currentHead) override {
    value = buffer_[currentHead];
  }
};

class RingBufferForShm : public BaseRingBuffer<char*> {
 private:
  char* buffer_;
  std::size_t msg_size_ = 0;

 public:
  RingBufferForShm(std::size_t msg_size, std::size_t buffer_length, void* addr)
      : BaseRingBuffer<char*>(buffer_length), msg_size_(msg_size) {
    buffer_ = new (addr) char[msg_size_ * buffer_length_]; // NOLINT(cppcoreguidelines-owning-memory)
    std::memset(buffer_, ' ', msg_size_ * buffer_length_);
  }

  void write_buffer(char* const& value, size_t currentTail) override {
    strncpy(buffer_ + currentTail * msg_size_, value, msg_size_);
  }

  void read_buffer(char*& value, size_t currentHead) override {
    strncpy(value, buffer_ + currentHead * msg_size_, msg_size_);
  }

  void dump(const std::function<void(const char*)>& callback) const {
    size_t head = head_.load(std::memory_order_acquire);
    size_t tail = tail_.load(std::memory_order_acquire);

    // empty buffer
    if (head == tail) {
      return;
    }

    for (size_t i = head; i != tail; i = (i + 1) % buffer_length_) {
      const char* msg = buffer_ + i * msg_size_;
      callback(msg);
    }
  }
};

} // namespace torch_supa::utils
