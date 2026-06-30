/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/* Copyright © 2026 Shanghai Biren Technology Co., Ltd. All rights reserved. */
/* replace device ctor with our new ctor which handles 'cuda' to 'supa'*/
#include <dlfcn.h>
#include <link.h>
#include <sys/mman.h>
#include <array>
#include <cstring>
#include <string>

#include <torch/csrc/Exceptions.h>

#include "torch_supa/csrc/utils/logger/Logger.h"

namespace torch_supa::utils {

using DeviceCtorType = void (*)(void* self, const std::string& device_string);

namespace {
DeviceCtorType original_device_ctor = nullptr;

void supa_device_ctor(void* self, const std::string& device_string) {
  if (std::strncmp(device_string.c_str(), "cuda", 4) == 0) {
    std::string new_device_string(device_string);
    new_device_string.replace(0, 4, "supa");
    original_device_ctor(self, new_device_string);
  } else {
    original_device_ctor(self, device_string);
  }
}

/**
 * Find the base address of a loaded shared library
 */
void* find_library_base(const char* lib_name) {
  void* handle = dlopen(lib_name, RTLD_LAZY);
  if (!handle) {
    return nullptr;
  }

  link_map* map = nullptr;
  if (dlinfo(handle, RTLD_DI_LINKMAP, &map) != 0) {
    dlclose(handle);
    return nullptr;
  }

  void* base = reinterpret_cast<void*>(map->l_addr);
  dlclose(handle);
  return base;
}

/**
 * @brief Find the GOT entry for a given symbol in a shared library
 *
 * @param lib_base: base address of lib calls symbol
 * @param symbol_name : the name of function.
 * @return Found address
 */
void** find_got_entry(void* lib_base, const char* symbol_name) {
  if (!lib_base) {
    return nullptr;
  }

  // Get the ELF header
  ElfW(Ehdr)* ehdr = static_cast<ElfW(Ehdr)*>(lib_base);
  ElfW(Phdr)* phdr = static_cast<ElfW(Phdr)*>(static_cast<void*>(static_cast<char*>(lib_base) + ehdr->e_phoff));

  // Find the dynamic segment
  ElfW(Dyn)* dyn = nullptr;
  for (int i = 0; i < ehdr->e_phnum; ++i) {
    if (phdr[i].p_type == PT_DYNAMIC) {
      dyn = static_cast<ElfW(Dyn)*>(static_cast<void*>(static_cast<char*>(lib_base) + phdr[i].p_vaddr));
      break;
    }
  }

  if (!dyn) {
    return nullptr;
  }

  // Parse dynamic section to find symbol table, string table, and relocation table
  ElfW(Sym)* symtab = nullptr;
  const char* strtab = nullptr;
  ElfW(Rela)* rela = nullptr;
  size_t rela_size = 0;

  for (ElfW(Dyn)* d = dyn; d->d_tag != DT_NULL; ++d) {
    switch (d->d_tag) {
      case DT_SYMTAB:
        symtab = reinterpret_cast<ElfW(Sym)*>(d->d_un.d_ptr);
        break;
      case DT_STRTAB:
        strtab = reinterpret_cast<const char*>(d->d_un.d_ptr);
        break;
      case DT_JMPREL:
        rela = reinterpret_cast<ElfW(Rela)*>(d->d_un.d_ptr);
        break;
      case DT_PLTRELSZ:
        rela_size = d->d_un.d_val;
        break;
    }
  }

  if (!symtab || !strtab || !rela) {
    return nullptr;
  }

#define R_SYM(type) _ElfW(ELF, __ELF_NATIVE_CLASS, R_SYM)(type)

  // Search for the symbol in relocation entries
  for (size_t i = 0; i < rela_size / sizeof(ElfW(Rela)); ++i) {
    uint32_t sym_idx = R_SYM(rela[i].r_info);
    ElfW(Sym)* sym = &symtab[sym_idx];
    const char* name = strtab + sym->st_name;

    if (name && strcmp(name, symbol_name) == 0) {
      return reinterpret_cast<void**>(static_cast<char*>(lib_base) + rela[i].r_offset);
    }
  }

  return nullptr;
}

inline bool change_protection(void* addr, size_t size, bool enable_write) {
  static size_t page_size = sysconf(_SC_PAGESIZE);
  void* page = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(addr) & ~(page_size - 1));

  return mprotect(page, page_size, PROT_READ | (enable_write ? PROT_WRITE : PROT_EXEC)) == 0;
}

/**
 * @brief modify GOT to hijack calling to symbol.
 * @details there are 2 tables for a shared library: PLT (Procedure Linkage Table) and GOT(Global Offset Table)
 *          ld.so fills GOT table with actual addresses of function.
 *          from a.so, it spends 2 steps before calling a function in b.so:
 *          1. searchs the entry offset by PLT
 *           2. get actual function address in GOT.
 */
bool hook_via_got(const char* lib_name, const char* mangled_symbol) {
  void* lib_base = find_library_base(lib_name);
  if (!lib_base) {
    TORCH_SUPA_ERROR("Failed to find libary {} to hook", lib_name);
    return false;
  }

  void** got_entry = find_got_entry(lib_base, mangled_symbol);
  if (!got_entry) {
    TORCH_SUPA_INFO("Failed to find GOT entry for: {}. skip it.", mangled_symbol);
    return false;
  }

  original_device_ctor = reinterpret_cast<DeviceCtorType>(*got_entry);

  if (!change_protection(got_entry, sizeof(void*), true)) {
    TORCH_SUPA_ERROR("Failed to make GOT writable");
    return false;
  }

  *got_entry = reinterpret_cast<void*>(&supa_device_ctor);

  change_protection(got_entry, sizeof(void*), false);

  TORCH_SUPA_VERBOSE("Successfully replaced with new address {} for symbol {}", *got_entry, mangled_symbol);
  return true;
}
} // namespace

/**
 * @brief replace default ctor c10::Device::Device(const std::string&) with new ctor,
 *         which replace 'cuda' to 'supa' if need.
 */
bool initDeviceWrap(void) {
  // Mangled name for c10::Device(const std::string&)
  const std::array<const char*, 3> mangled_names = {
      // c10::Device::Device(basic_string) - complete object constructor
      "_ZN3c106DeviceC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE",
      // Alternative: without __cxx11 namespace
      "_ZN3c106DeviceC1ERKSs",
      nullptr};
  const char* target_so =
      "libtorch_python.so"; // shared library who calls c10::Device(), it contains all torch operators.

  for (int i = 0; mangled_names.at(i); ++i) {
    if (hook_via_got(target_so, mangled_names.at(i))) {
      return true;
    }
  }
  return false;
}

} // namespace torch_supa::utils
