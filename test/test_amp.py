# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pickle
from itertools import chain
import torch

from torch_supa.supa.amp import GradScaler
from torch.testing._internal.common_utils import TestCase
from torch_supa.testing.common_utils import assertRtolEqual 

class TestAmp(TestCase):
    def test_grad_scaling_scale(self, device=torch.device("supa")):
        scaler = GradScaler(init_scale=2.0)
        t0 = torch.full((1,), 4.0, dtype=torch.bfloat16, device=device)
        t1 = torch.full((1,), 4.0, dtype=torch.bfloat16, device=device)
        # Create some nested iterables of tensors on different devices.
        outputs = (t1.clone(), (t0.clone(), t1.clone()), [t0.clone(), (t1.clone(), t0.clone())])
        outputs = scaler.scale(outputs)
        self.assertTrue(
            outputs[0] == 8.0
            and outputs[1][0] == 8.0
            and outputs[1][1] == 8.0
            and outputs[2][0] == 8.0
            and outputs[2][1][0] == 8.0
            and outputs[2][1][1] == 8.0
        )
        self.assertTrue(scaler._scale.device == t1.device)

    # test scaler load state_dict
    def test_grad_scaling_state_dict(self, device="supa"):
        for lazy_init_scale in True, False:
            s0 = GradScaler(init_scale=3.0, growth_factor=4.0, backoff_factor=0.5, growth_interval=2)
            s1 = GradScaler(init_scale=6.0, growth_factor=7.0, backoff_factor=0.8, growth_interval=1)

            # sets a random value for load_state_dict to overwrite
            s1._init_growth_tracker = 7

            if lazy_init_scale:
                # Dummy scale() call to ensure the scale tensor is lazily initialized.
                s1.scale(torch.full((1,), 4.0, dtype=torch.float32, device="supa"))
                self.assertTrue(isinstance(s1._scale, torch.supa.FloatTensor))

            s1.load_state_dict(s0.state_dict())

            self.assertTrue(s1.get_scale() == 3.0)
            self.assertTrue(s1.get_growth_factor() == 4.0)
            self.assertTrue(s1.get_backoff_factor() == 0.5)
            self.assertTrue(s1.get_growth_interval() == 2)
            self.assertTrue(s1._init_growth_tracker == 0)

    def _create_scaling_models_optimizers(self, device="supa"):
        # Create a module+optimizer that will use scaling, and a control module+optimizer
        # that will not use scaling, against which the scaling-enabled module+optimizer can be compared.
        mod_control = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False), torch.nn.Linear(8, 8, bias=False)).to(
            device=device
        )
        mod_scaling = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False), torch.nn.Linear(8, 8, bias=False)).to(
            device=device
        )
        for c, s in zip(mod_control.parameters(), mod_scaling.parameters()):
            s.data.copy_(c.data)

        opt_control = torch.optim.SGD(mod_control.parameters(), lr=1.0)
        opt_scaling = torch.optim.SGD(mod_scaling.parameters(), lr=1.0)

        ret = (mod_control, mod_scaling, opt_control, opt_scaling)
        return ret

    def _create_scaling_case(self, device="supa", dtype=torch.float):
        data = [
            (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
            (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
            (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
            (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
        ]

        loss_fn = torch.nn.MSELoss().supa()

        skip_iter = 2

        return self._create_scaling_models_optimizers(device=device) + (data, loss_fn, skip_iter)

    # _run_scaling_case generalizes some single-optimizer test logic to avoid too much copy-pasting below.
    def _run_scaling_case(self, run, unskipped, skipped, atol=1e-7):
        # Ensure scaling can be disabled without changing user control flow.
        for enabled in True, False:
            mod_control, mod_scaling, opt_control, opt_scaling, data, loss_fn, skip_iter = self._create_scaling_case()

            # For functionality, test with a modest initial scale, and an unrealistically-large growth factor
            # so any potential errors with the growth factor handling will be magnified.
            scaler = GradScaler(init_scale=128.0, growth_factor=2.0, enabled=enabled, growth_interval=1)

            _ = run(data, mod_control, opt_control, scaler, loss_fn, skip_iter, False)
            ret = run(data, mod_scaling, opt_scaling, scaler, loss_fn, skip_iter, True)

            # Allows run() to optionally return a different scaler instance.
            scaler = ret if ret else scaler

            if enabled:
                net_growth = scaler.get_growth_factor() ** unskipped if unskipped > 0 else 1.0
                net_backoff = scaler.get_backoff_factor() ** skipped if skipped > 0 else 1.0
                self.assertTrue(scaler.get_scale() == (128.0 * net_growth * net_backoff))
            else:
                self.assertTrue(scaler.get_scale() == 1.0)

            for c, s in zip(mod_control.parameters(), mod_scaling.parameters()):
                c = c.cpu().to(torch.float).detach().numpy()
                s = s.cpu().to(torch.float).detach().numpy()
                assertRtolEqual(c, s, atol)

    # Compares no scaling + no autocasting against scaling + autocasting.
    def test_grad_scaling_autocast(self, device="supa"):
        try_pickle = False

        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            for i, (input_data, target) in enumerate(data):
                optimizer.zero_grad()
                with torch.autocast("supa", enabled=try_scaling_api):
                    output = model(input_data)
                    loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float("inf"))
                    scaler.step(optimizer)
                    scaler.update()
                    if try_pickle:
                        scaler = pickle.loads(pickle.dumps(scaler))
                else:
                    loss.backward()
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()
            return scaler

        # sets atol=1e-2 because we're comparing pure fp32 arithmetic vs a mixture of bf16 and fp32
        self._run_scaling_case(run, unskipped=3, skipped=1, atol=1e-2)
        # this will be picked up by try_pickle within run():
        try_pickle = True
        self._run_scaling_case(run, unskipped=3, skipped=1, atol=1e-2)

    def test_grad_scaling_clipping(self, device="supa"):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            max_norm = 0.2  # A reasonable value that actually has an effect, based on printouts of grads
            for i, (input_data, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input_data)
                loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm * scaler.get_scale())
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float("inf"))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(run, unskipped=3, skipped=1, atol=1e-5)

    def test_grad_scaling_clipping_separate_unscale(self, device="supa"):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            max_norm = 0.2  # A reasonable value that actually has an effect, based on printouts of grads
            for i, (input_data, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input_data)
                loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float("inf"))
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(run, unskipped=3, skipped=1)

    def test_grad_scaling_penalty(self, device="supa"):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            for i, (input_data, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input_data)
                loss = loss_fn(output, target)

                if try_scaling_api:
                    grad_params = torch.autograd.grad(scaler.scale(loss), model.parameters(), create_graph=True)
                    inv_scale = 1.0 / scaler.get_scale()
                    grad_params = [p * inv_scale for p in grad_params]
                else:
                    grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)

                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()
                loss = loss + grad_norm

                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float("inf"))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(run, unskipped=3, skipped=1)

    def test_grad_scaling_accumulation(self, device="supa"):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            iters_to_accumulate = 2
            for i, (input_data, target) in enumerate(data):
                output = model(input_data)
                loss = loss_fn(output, target)
                loss = loss / iters_to_accumulate
                if try_scaling_api:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (i + 1) % iters_to_accumulate == 0:
                    if try_scaling_api:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()

        self._run_scaling_case(run, unskipped=2, skipped=0)

    def test_grad_scaling_multiple(self, device="supa"):
        # Tests gradient scaling with 2 models and 2 optimizers that both receive gradients from 2 losses.
        # Some of the logic here cannot reuse the generic helper functions created for the 1-optimizer cases.
        for enabled in True, False:
            mod_control0, mod_scaling0, opt_control0, opt_scaling0, data, loss_fn, skip_iter = (
                self._create_scaling_case()
            )
            mod_control1, mod_scaling1, opt_control1, opt_scaling1 = self._create_scaling_models_optimizers()

            scaler = GradScaler(init_scale=128.0, growth_factor=2.0, enabled=enabled, growth_interval=1)

            def run(model0, model1, optimizer0, optimizer1, try_scaling_api):
                for i, (input_data, target) in enumerate(data):
                    optimizer0.zero_grad()
                    optimizer1.zero_grad()
                    output0 = model0(input_data)
                    output1 = model1(input_data)
                    loss0 = loss_fn(0.3 * output0 + 0.7 * output1, target)
                    loss1 = loss_fn(0.6 * output0 - 0.4 * output1, target)

                    if try_scaling_api:
                        scaler.scale(loss0).backward(retain_graph=True)
                        scaler.scale(loss1).backward()
                        if i == skip_iter and scaler.is_enabled():
                            model1[1].weight.grad.data.fill_(float("inf"))

                        # As an additional stress test, separately unscale for one of the optimizers.
                        scaler.unscale_(optimizer0)

                        scaler.step(optimizer0)
                        scaler.step(optimizer1)
                        scaler.update()
                    else:
                        loss0.backward(retain_graph=True)
                        loss1.backward()
                        if (not scaler.is_enabled()) or (i != skip_iter):
                            optimizer0.step()
                            optimizer1.step()

            run(mod_control0, mod_control1, opt_control0, opt_control1, False)
            run(mod_scaling0, mod_scaling1, opt_scaling0, opt_scaling1, True)

            # The loss scale should have been multiplied by the growth factor 3 times and the backoff factor once.
            self.assertTrue(
                scaler.get_scale() == (128.0 * scaler.get_growth_factor() ** 3 * scaler.get_backoff_factor() ** 1)
                if enabled
                else 1.0
            )

            # TODO: Amp need to refresh the verification because OP normal_ was not implemented before
            for c, s in zip(
                chain(mod_control0.parameters(), mod_control1.parameters()),
                chain(mod_scaling0.parameters(), mod_scaling1.parameters()),
            ):
                c = c.cpu().to(torch.float).detach().numpy()
                s = s.cpu().to(torch.float).detach().numpy()
                assertRtolEqual(c, s, 1e-1)
