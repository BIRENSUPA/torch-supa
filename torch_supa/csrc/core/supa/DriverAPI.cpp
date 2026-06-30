/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include "torch_supa/csrc/core/supa/DriverAPI.h"
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <dlfcn.h>
#include <supa_runtime.h>
#include "torch_supa/csrc/core/supa/SUPAException.h"

namespace c10::supa {

namespace {

void* get_symbol(const char* name, int version);

DriverAPI create_driver_api() {
  void* handle_1 = DriverAPI::get_brml_handle();
  DriverAPI r{};

#define LOOKUP_LIBSUPA_ENTRY_WITH_VERSION_REQUIRED(name, version)            \
  r.name##_ = reinterpret_cast<decltype(&name)>(get_symbol(#name, version)); \
  TORCH_INTERNAL_ASSERT(r.name##_, "Can't find ", #name);
  C10_LIBSUPA_DRIVER_API_REQUIRED(LOOKUP_LIBSUPA_ENTRY_WITH_VERSION_REQUIRED)
#undef LOOKUP_LIBSUPA_ENTRY_WITH_VERSION_REQUIRED

// Users running older drivers may not have these symbols,
// they would be resolved into nullptr, but we guard their usage at runtime
// to ensure safe fallback behavior.
#define LOOKUP_LIBSUPA_ENTRY_WITH_VERSION_OPTIONAL(name, version) \
  r.name##_ = reinterpret_cast<decltype(&name)>(get_symbol(#name, version));
  C10_LIBSUPA_DRIVER_API_OPTIONAL(LOOKUP_LIBSUPA_ENTRY_WITH_VERSION_OPTIONAL)
#undef LOOKUP_LIBSUPA_ENTRY_WITH_VERSION_OPTIONAL

  if (handle_1) {
#define LOOKUP_BIREN_ML_ENTRY(name)                      \
  r.name##_ = ((decltype(&name))dlsym(handle_1, #name)); \
  TORCH_INTERNAL_ASSERT(r.name##_, "Can't find ", #name, ": ", dlerror())
    C10_BRML_DRIVER_API(LOOKUP_BIREN_ML_ENTRY)
#undef LOOKUP_BIREN_ML_ENTRY
  }

  if (handle_1) {
#define LOOKUP_BIREN_ML_ENTRY_OPTIONAL(name) r.name##_ = ((decltype(&name))dlsym(handle_1, #name));
    C10_BRML_DRIVER_API(LOOKUP_BIREN_ML_ENTRY_OPTIONAL)
#undef LOOKUP_BIREN_ML_ENTRY_OPTIONAL
  }
  return r;
}

void* get_symbol(const char* name, int version) {
  void* out = nullptr;
  supaDriverEntryPointQueryResult qres{};

  // SUPA runtime supports version-based lookup.
  if (auto st = supaGetDriverEntryPointByVersion(name, &out, version, supaEnableDefault, &qres);
      st == supaSuccess && qres == supaDriverEntryPointSuccess && out) {
    return out;
  }

  // Fallback to the legacy entry point lookup.
  if (auto st = supaGetDriverEntryPoint(name, &out, supaEnableDefault, &qres);
      st == supaSuccess && qres == supaDriverEntryPointSuccess && out) {
    return out;
  }

  // If the symbol cannot be resolved, report and return nullptr;
  // the caller is responsible for checking the pointer.
  LOG(INFO) << "Failed to resolve symbol " << name;
  return nullptr;
}

} // namespace

void* DriverAPI::get_brml_handle() {
  static void* biren_ml_handle = dlopen("libbiren-ml.so", RTLD_LAZY);
  return biren_ml_handle;
}

C10_EXPORT DriverAPI* DriverAPI::get() {
  static DriverAPI singleton = create_driver_api();
  return &singleton;
}

} // namespace c10::supa
