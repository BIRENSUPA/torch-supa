/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <execinfo.h>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "torch_supa/csrc/debugger/SignalHandler.h"
#include "torch_supa/csrc/utils/EnvConfig.h"

namespace torch_supa {
namespace utils {

typedef void (*signalHandler_t)(int, siginfo_t*, void*);
static signalHandler_t k_previous_segment_fault_handler = NULL;
static signalHandler_t k_previous_abnormal_termination_handler = NULL;

static void PrintBacktrace() {
  char** strings = NULL;
  int i = 0;
  int size = 0;
  enum Constexpr { MAX_SIZE = 1024 };
  void* array[MAX_SIZE];
  printf("\n------------backtrace:------------\n");
  size = backtrace(array, MAX_SIZE);
  if (size > 0) {
    strings = backtrace_symbols(array, size);
    if (strings != NULL) {
      for (i = 0; i < size; i++) {
        printf("%s\n", strings[i]);
      }
      puts("");
      free(strings); // NOLINT(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory)
    }
  }
}

static void segmentFaultHandler(int sig, siginfo_t* si, void* other) {
  if (si) {
    printf("\nCaught segmentation fault at address %p\n", si->si_addr);
  } else {
    printf("\nGot SIGSEGV, siginfo null\n");
  }
  PrintBacktrace();
  if (k_previous_segment_fault_handler) {
    k_previous_segment_fault_handler(sig, si, other);
  }
}

static void abnormalTerminationHandler(int sig, siginfo_t* si, void* other) {
  printf("\nGot SIGABRT\n");
  PrintBacktrace();
  if (k_previous_abnormal_termination_handler) {
    k_previous_abnormal_termination_handler(sig, si, other);
  }
}

void initSignalHandler() {
  if (!EnvConfig::IsEnableSignalHandling()) {
    return;
  }

  struct sigaction act = {0};
  struct sigaction old_act = {0};

  memset(&act, 0, sizeof(act));
  sigemptyset(&act.sa_mask);

  act.sa_flags = SA_ONESHOT | SA_NOMASK;
  act.sa_sigaction = segmentFaultHandler;

  if (sigaction(SIGSEGV, &act, &old_act) == -1) {
    perror("sigsegv: sigaction");
  } else {
    k_previous_segment_fault_handler = old_act.sa_sigaction;
  }

  memset(&act, 0, sizeof(act));
  sigemptyset(&act.sa_mask);
  act.sa_flags = SA_ONESHOT | SA_NOMASK;
  act.sa_sigaction = abnormalTerminationHandler;

  if (sigaction(SIGABRT, &act, &old_act) == -1) {
    perror("SIGABRT: sigaction");
  } else {
    k_previous_abnormal_termination_handler = old_act.sa_sigaction;
  }
}

} // namespace utils
} // namespace torch_supa
