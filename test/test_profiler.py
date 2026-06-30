# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import torch
import pytest 
import os
import json
from torch.testing._internal.common_utils import TestCase, TemporaryFileName

from torch.autograd.profiler import KinetoStepTracker
from torch._C._profiler import (
    _ExperimentalConfig,
)

@pytest.mark.sanity
@pytest.mark.regression
class TestProfiler(TestCase):
    def sub_op(self, device="cpu"):
        a = torch.rand(5, 5).to(device)
        b = torch.randn(5, 5).to(device)
        c = torch.sub(a, b)

    def test_cpu_op_profiler(self):
        with torch.autograd.profiler.profile(use_supa=False) as prof:
            self.sub_op()
        found_sub = False

        for e in prof.function_events:
            if "sub" in e.name:
                found_sub = True
        self.assertTrue(found_sub)

    def test_supa_op_profiler(self):
        # test basic function for supa op
        if torch.supa.is_available():
            device = "supa:0"
        else:
            return
        with torch.autograd.profiler.profile(use_supa=True) as prof:
            self.sub_op(device)
        found_sub = False
        print(prof.function_events)
        for e in prof.function_events:
            if "sub" in e.name:
                found_sub = True
        self.assertTrue(found_sub)

    def test_memory_profiler(self):
        # test momory usage
        def run_profiler(creat_tensor, metric):
            # collecting allocs / deallocs
            with torch.autograd.profiler.profile(profile_memory=True, record_shapes=False) as prof:
                input_x = None
                with torch.profiler.record_function("user_allocate"):
                    input_x = creat_tensor()
                with torch.profiler.record_function("user_deallocate"):
                    del input_x
            print(prof)
            return prof.key_averages()

        def check_metrics(stats, metric, allocs=None, deallocs=None):
            stat_metrics = {}
            for stat in stats:
                stat_metrics[stat.key] = getattr(stat, metric)
            if allocs is not None:
                for alloc_fn in allocs:
                    self.assertTrue(alloc_fn in stat_metrics)
                    self.assertTrue(stat_metrics.get(alloc_fn, 0) > 0)
            if deallocs is not None:
                for dealloc_fn in deallocs:
                    self.assertTrue(dealloc_fn in stat_metrics)
                    self.assertTrue(stat_metrics.get(dealloc_fn, 0) < 0)

        def create_cpu_tensor():
            return torch.rand(1000, 1000, dtype=torch.float32)

        def create_supa_tensor():
            return torch.rand(200, 300).supa()

        stats = run_profiler(create_cpu_tensor, "cpu_memory_usage")
        check_metrics(
            stats,
            "cpu_memory_usage",
            allocs=[
                "aten::empty",
                "aten::rand",
                "user_allocate",
            ],
            deallocs=[
                "user_deallocate",
            ],
        )

        if torch.supa.is_available():
            stats = run_profiler(create_supa_tensor, "device_memory_usage")
            check_metrics(
                stats,
                "device_memory_usage",
                allocs=[
                    "user_allocate",
                    "aten::to",
                    "aten::_to_copy",
                ],
                deallocs=[
                    "user_deallocate",
                ],
            )
            check_metrics(
                stats,
                "cpu_memory_usage",
                allocs=[
                    "aten::rand",
                    "aten::empty",
                ],
            )

@pytest.mark.sanity
@pytest.mark.regression
class TestNewProfilerInf(TestCase):
    def sub_op(self, device="cpu"):
        a = torch.rand(5, 5).to(device)
        b = torch.randn(5, 5).to(device)
        c = torch.sub(a, b)

    def test_verify_supported_activity(self):
        supported_activites = torch.profiler.supported_activities()
        self.assertIn(torch.profiler.ProfilerActivity.SUPA, supported_activites)

    def test_cpu_op_profiler(self):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            self.sub_op()
        found_sub = False

        for e in prof.events():
            if "sub" in e.name:
                found_sub = True
        self.assertTrue(found_sub)

    @pytest.mark.skip(reason="cmodel could not grep gpu activities")
    def test_supa_op_profiler(self):
        # test basic function for supa op
        if torch.supa.is_available():
            device = "supa:0"
        else:
            return
        folder = "profiler_log"
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.SUPA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(folder, worker_name="trace"),
        ) as prof:
            self.sub_op(device)
        found_sub = False

        for e in prof.events():
            if "sub" in e.name:
                found_sub = True
        self.assertTrue(found_sub, "'sub' is not found in event names")

        # get fiel name of last created.
        files = os.listdir(folder)
        files = sorted(files, key=lambda x: os.path.getctime(os.path.join(folder, x)))
        file = files[-1]

        try:
            with open(os.path.join(folder, file), "r") as fi:
                ret = json.load(fi)
        except Exception as e:
            self.fail(e)

        kernel = [x for x in ret["traceEvents"] if x.get("cat") == "kernel"]
        self.assertGreater(len(kernel), 0, "no kernel found in exported trace file. missing supti?")

    def test_profiler_schedule(self):
        self.assertTrue(torch.supa.is_available())
        input_x = torch.rand(200, 300).supa()
        input_y = torch.rand(200, 300).supa()
        wait = 1
        warmup = 1
        active = 8

        initial_step = KinetoStepTracker.current_step()
        # collecting allocs / deallocs
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.SUPA],
            profile_memory=True,
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
        ) as prof:
            for i in range(10):
                input_y = input_x + input_y
                prof.step()
            with torch.profiler.record_function("user_deallocate"):
                del input_x

        self.assertEqual(KinetoStepTracker.current_step(), initial_step + 10)

    def test_export_stacks(self):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.SUPA],
            with_stack=True, 
            use_supa=True, 
            with_flops=True,
            experimental_config=_ExperimentalConfig(verbose=True)
        ) as prof:
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.mm(x, y)
            z = z + y

        with TemporaryFileName(mode="w+") as fname:
            prof.export_stacks(fname)
            with open(fname) as f:
                lines = f.readlines()
            assert len(lines) > 0, "Empty stacks file"
            for line in lines:
                is_int = False
                try:
                    assert int(line.split(" ")[-1]) > 0, "Invalid stacks record"
                    is_int = True
                except ValueError:
                    pass
                assert is_int, "Invalid stacks record"

    def test_is_profiler_enabled(self):
        self.assertFalse(torch.autograd.profiler._is_profiler_enabled)

        with torch.profiler.profile() as p:
            self.assertTrue(torch.autograd.profiler._is_profiler_enabled)

        self.assertFalse(torch.autograd.profiler._is_profiler_enabled)

        with torch.autograd.profiler.profile() as p:
            self.assertTrue(torch.autograd.profiler._is_profiler_enabled)

        self.assertFalse(torch.autograd.profiler._is_profiler_enabled)
