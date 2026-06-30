# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.


import gc
import os
import tempfile
import torch

from torch.testing._internal.common_utils import TestCase


class TestSupaGraph(TestCase):
    def test_graph_is_current_stream_capturing(self):
        self.assertFalse(torch.supa.is_current_stream_capturing())

        s = torch.supa.Stream()
        with torch.supa.stream(s):
            g = torch.supa.SUPAGraph()
            self.assertFalse(torch.supa.is_current_stream_capturing())
            g.capture_begin()
            self.assertTrue(torch.supa.is_current_stream_capturing())
            g.capture_end()


    def test_graph_capture_simple(self):
        s = torch.supa.Stream()

        with torch.supa.stream(s):
            a = torch.full((1000,), 1, device="supa")
            g = torch.supa.SUPAGraph()
            torch.supa.empty_cache()
            g.capture_begin()
            b = a
            for _ in range(10):
                b = b + 1
            g.capture_end()
        torch.supa.current_stream().wait_stream(s)

        g.replay()
        self.assertEqual(b.sum().item(), 11000.0)


    def test_graphsafe_set_get_rng_state(self):
        # Define a function to create generator states, with optional graph registration
        def create_states(generator):
            """Initializes generator states and registers them with a SUPA graph if provided."""
            # Ensure the SUPA generator is initialized
            torch.rand(1, device="supa")
            generator.manual_seed(0)

            # Save the current state of the generator
            old_state = generator.graphsafe_get_state()
            # Create and save a cloned state of the generator
            new_state = generator.clone_state()
            # Return the original generator and its two states
            return generator, old_state, new_state

        def register_states_to_graph(generator_state, graph):
            generator, old_state, new_state = generator_state
            graph.register_generator_state(old_state)
            graph.register_generator_state(new_state)

        # Define a function to perform specific RNG actions using the generator's states
        def perform_random_generation_steps(generator_state):
            generator, old_state, new_state = generator_state
            random_values = []

            # Generate random numbers with the new generator state
            generator.graphsafe_set_state(new_state)
            random_values.append(torch.rand(5, device="supa", generator=generator))

            # Generate random numbers twice with the old generator state
            generator.graphsafe_set_state(old_state)
            random_values.extend(
                [torch.rand(5, device="supa", generator=generator) for _ in range(2)]
            )

            return random_values

        # Define a function to retrieve the final offsets of the original and new generator states
        def get_final_offsets_of_states(generator_state):
            generator, old_state, new_state = generator_state
            old_state_offset = old_state.get_offset()
            new_state_offset = new_state.get_offset()
            return old_state_offset, new_state_offset

        # Set up and test a new SUPA generator
        generator = torch.Generator(device="supa")
        generator_state = create_states(generator)

        # Set up and test the default SUPA generator with a SUPA Graph
        g = torch.supa.SUPAGraph()
        s = torch.supa.Stream()
        default_generator = torch.supa.default_generators[0]
        default_generator_state = create_states(default_generator)
        register_states_to_graph(default_generator_state, g)

        # Perform random number generation within a SUPA graph
        with torch.supa.stream(s):
            g.capture_begin()
            graphed_random_values = perform_random_generation_steps(
                default_generator_state
            )
            g.capture_end()

        # Synchronize the streams and replay the graph
        torch.supa.current_stream().wait_stream(s)
        for _ in range(3):
            random_values = perform_random_generation_steps(generator_state)
            g.replay()
            offset = get_final_offsets_of_states(generator_state)
            graph_offset = get_final_offsets_of_states(default_generator_state)

            # Compare the final offsets of states for both generators to ensure consistency
            self.assertEqual(offset, graph_offset)
            # Compare the states generated outside and inside the graph
            self.assertEqual(random_values, graphed_random_values)


    def test_memory_stats_of_multiple_generators_and_graphs(self):
        # Function to clear SUPA cache and collect garbage
        def clear_supa_cache():
            gc.collect()
            torch.supa.empty_cache()

        # Executes a simple graph task which includes capturing and executing a random number generation within a SUPA graph.
        def simple_graph_task(graph):
            s = torch.supa.Stream()
            with torch.supa.stream(s):
                graph.capture_begin()
                torch.rand(1, device="supa")
                graph.capture_end()
            torch.supa.current_stream().wait_stream(s)
            graph.replay()  # Replays the captured operations

        def get_memory_stats():
            stats = torch.supa.memory_stats()
            num_blocks = stats["active.all.current"]
            total_size = stats["active_bytes.all.current"]
            return num_blocks, total_size

        def test(num_graphs, num_generators):
            baseline = get_memory_stats()
            baseline_num_blocks, baseline_total_size = baseline

            # Allocate SUPA graphs
            graphs = [torch.supa.SUPAGraph() for _ in range(num_graphs)]

            # Allocate and manage generator states
            default_generator = torch.supa.default_generators[0]
            generators = [default_generator.graphsafe_get_state()]

            # Starts from 1 as one state is already added
            for _ in range(1, num_generators):
                generators.append(default_generator.clone_state())

            for graph in graphs:
                for generator_state in generators:
                    graph.register_generator_state(generator_state)
                simple_graph_task(graph)

            # Assert conditions after graph tasks
            num_blocks, total_size = get_memory_stats()
            # The allocated blocks should only be proportional to the number of generators
            expected_blocks_diff = 2 * num_generators
            expected_size_diff = 2 * 512 * num_generators  # Each block's size is 512

            self.assertEqual(
                (num_blocks - baseline_num_blocks),
                expected_blocks_diff,
                "Unexpected number of active blocks.",
            )
            self.assertEqual(
                (total_size - baseline_total_size),
                expected_size_diff,
                "Unexpected total memory size.",
            )

            # Cleanup graphs and clear SUPA cache
            while graphs:
                graph = graphs.pop()
                del graph
            clear_supa_cache()

            # Assert that memory stats return to baseline after cleanup
            self.assertEqual(
                get_memory_stats(),
                baseline,
                "Memory stats do not match baseline after cleanup.",
            )

        # Running the test function with different parameters
        test(1, 1)
        test(3, 2)
        test(10, 20)


    def test_graph_capture_reset_recapture(self):
        s = torch.supa.Stream()

        with torch.supa.stream(s):
            a = torch.full((1000,), 1, device="supa")
            g = torch.supa.SUPAGraph()
            torch.supa.empty_cache()
            g.capture_begin()
            b = a
            for _ in range(10):
                b = b + 1
            g.capture_end()
        torch.supa.current_stream().wait_stream(s)

        g.replay()

        self.assertEqual(b.sum().item(), 11000.0)

        g.reset()

        with torch.supa.stream(s):
            g.capture_begin()
            b.fill_(2.0)
            for _ in range(10):
                b = b + 2
            g.capture_end()
        torch.supa.current_stream().wait_stream(s)

        g.replay()
        self.assertEqual(b.sum().item(), 22000.0)

        g.reset()
        del g


    def test_graph_debugdump(self):
        torch.supa.empty_cache()
        x = torch.randn(10240000, device="supa")
        y = torch.rand_like(x)
        g = torch.supa.SUPAGraph()
        g.enable_debug_mode()
        s0 = torch.supa.Stream()
        s1 = torch.supa.Stream()
        s0.wait_stream(torch.supa.current_stream())
        with torch.supa.stream(s0):
            g.capture_begin()
            z = x + y
            with torch.supa.stream(s1):
                s1.wait_stream(s0)
                w = z + y
            s0.wait_stream(s1)
            g.capture_end()
        s0.synchronize()
        torch.supa.synchronize()
        with tempfile.TemporaryDirectory() as tempdir:
            g.debug_dump(os.path.join(tempdir, "out_multi_stream.dot"))


    def test_graph_capture_oom(self):
        oom_regex = (
            "out of memory"
        )
        with self.assertRaisesRegex(RuntimeError, oom_regex):
            with torch.supa.graph(torch.supa.SUPAGraph()):
                torch.zeros(2**40, device="supa")


    def test_graph_rng_functional(self):
        ops_with_kwargs = (
            (torch.nn.functional.dropout, {"p": 0.1}),
            (torch.nn.functional.rrelu, {"training": True}),
        )
        size = 10000

        def run(op, kwargs):
            a = torch.randn((size,), device="supa", dtype=torch.float)

            # Control
            torch.supa.manual_seed(5)
            eager_out = a
            for _ in range(6):
                eager_out = op(eager_out, **kwargs)

            graph_in = a.clone()
            stream = torch.supa.Stream()
            stream.wait_stream(torch.supa.current_stream())
            with torch.supa.stream(stream):
                torch.supa.manual_seed(5)

                g = torch.supa.SUPAGraph()
                torch.supa.empty_cache()
                g.capture_begin()
                graph_out = graph_in
                for _ in range(2):
                    graph_out = op(graph_out, **kwargs)
                g.capture_end()
            torch.supa.current_stream().wait_stream(stream)

            # Runs a graphed->eager->graphed sequence of RNG ops.
            # replay() plays 2 invocations of the op, so the sequence has 6
            # invocations total, matching Control.
            # replay() reads from graph_in and writes to graph_out.
            g.replay()
            out = op(graph_out, **kwargs)
            out = op(out, **kwargs)
            graph_in.copy_(out)
            g.replay()

            # If replay() updated RNG state correctly, graph_out
            # should now hold data equal to eager_out.
            try:
                self.assertEqual(eager_out, graph_out)
            except Exception as e:
                raise RuntimeError("Failed on ", op) from e

            # Do the same operations varying seeds
            seeds = [6, 128, 9999]

            for seed in seeds:
                torch.supa.manual_seed(seed)
                graph_in.copy_(a)
                for _ in range(3):
                    g.replay()

                # If the random seed was not updated then the graph would
                # generate the same output as in previous check.
                try:
                    self.assertNotEqual(eager_out, graph_out)
                except Exception as e:
                    raise RuntimeError("Failed on ", op) from e

                # Now repeat the same operations in non-graphed mode.
                torch.supa.manual_seed(seed)
                for _ in range(3):
                    eager_out.copy_(a)
                    eager_out = op(eager_out, **kwargs)
                    eager_out = op(eager_out, **kwargs)

                # In the end, graph_out and eager_out must be equal
                # as they went under the same set of operations.
                try:
                    self.assertEqual(eager_out, graph_out)
                except Exception as e:
                    raise RuntimeError("Failed on ", op) from e

            # We hold references to all tensors used across streams up til this sync,
            # so no need to call record_stream on those tensors.
            torch.supa.synchronize()

        for op, kwargs in ops_with_kwargs:
            run(op, kwargs)
