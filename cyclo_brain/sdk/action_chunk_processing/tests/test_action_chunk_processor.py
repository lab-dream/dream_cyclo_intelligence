#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SDK_ROOT = Path(__file__).resolve().parents[1]
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from action_chunk_processing import ActionChunkProcessor  # noqa: E402
from action_chunk_processing.spline import quintic_spline_multi  # noqa: E402


class ActionChunkProcessorTests(unittest.TestCase):
    def test_quintic_endpoint_derivatives_and_joint_bounds(self) -> None:
        start = np.array([-0.4, 0.8, -0.2])  # arm, gripper (rad), lift (m)
        end = np.array([0.6, 0.2, -0.1])
        zero = np.zeros(3)
        for t, expected in ((0.0, start), (0.25, end)):
            result = quintic_spline_multi(t, 0.0, 0.25, start, zero, zero,
                                          end, zero, zero)
            np.testing.assert_allclose(result[0], expected, atol=1e-12)
            np.testing.assert_allclose(result[1:], 0.0, atol=1e-10)
        positions = np.array([quintic_spline_multi(
            t, 0.0, 0.25, start, zero, zero, end, zero, zero)[0]
            for t in np.linspace(0.0, 0.25, 101)])
        self.assertTrue(np.all(positions >= np.minimum(start, end) - 1e-12))
        self.assertTrue(np.all(positions <= np.maximum(start, end) + 1e-12))

    def test_quintic_resampling_and_fixed_target_use_same_curve(self) -> None:
        chunk = np.array([[0.0, -0.2], [1.0, -0.1]])
        for target in (None, 5):
            processor = ActionChunkProcessor(
                inference_hz=1.0, control_hz=4.0, alignment_mode="none",
                interpolation_method="quintic", target_chunk_size=target)
            produced = processor.push_actions(chunk)
            positions = np.array([processor.pop_action() for _ in range(produced)])
            self.assertEqual(produced, 4 if target is None else 5)
            # At 1/4 of the interval, quintic differs from linear interpolation.
            np.testing.assert_allclose(positions[1], [0.103515625, -0.1896484375])
            np.testing.assert_allclose(positions[2], [0.5, -0.15])
            if target is not None:
                np.testing.assert_allclose(positions[-1], chunk[-1])

    def test_quintic_chunk_transition_uses_smooth_blend_weight(self) -> None:
        processor = ActionChunkProcessor(
            inference_hz=1.0, control_hz=10.0, alignment_mode="none",
            interpolation_method="quintic")
        processor._last_output_action = np.array([0.0])
        processor.push_actions(np.array([[1.0], [1.0]]))
        np.testing.assert_allclose(processor.pop_action(), [17 / 81])
        np.testing.assert_allclose(processor.pop_action(), [64 / 81])
        np.testing.assert_allclose(processor.pop_action(), [1.0])

    def test_dynamic_resampling_preserves_legacy_16_step_timing(self) -> None:
        processor = ActionChunkProcessor(
            inference_hz=15.0,
            control_hz=100.0,
            alignment_mode="none",
        )
        chunk = np.arange(16, dtype=np.float64).reshape(16, 1)

        produced = processor.push_actions(chunk)

        self.assertEqual(produced, 100)
        first = processor.pop_action()
        last = None
        for _ in range(produced - 1):
            last = processor.pop_action()

        np.testing.assert_allclose(first, np.asarray([0.0]))
        np.testing.assert_allclose(last, np.asarray([14.85]))
        self.assertIsNone(processor.pop_action())

    def test_dynamic_resampling_scales_with_chunk_length(self) -> None:
        for source_count, expected_count in ((32, 207), (100, 660)):
            with self.subTest(source_count=source_count):
                processor = ActionChunkProcessor(
                    inference_hz=15.0,
                    control_hz=100.0,
                    alignment_mode="none",
                )
                chunk = np.zeros((source_count, 2), dtype=np.float64)

                produced = processor.push_actions(chunk)

                self.assertEqual(produced, expected_count)

    def test_dynamic_resampling_scales_with_inference_hz(self) -> None:
        chunk = np.zeros((16, 2), dtype=np.float64)

        slow_source = ActionChunkProcessor(
            inference_hz=10.0,
            control_hz=100.0,
            alignment_mode="none",
        )
        fast_source = ActionChunkProcessor(
            inference_hz=20.0,
            control_hz=100.0,
            alignment_mode="none",
        )

        self.assertEqual(slow_source.push_actions(chunk), 150)
        self.assertEqual(fast_source.push_actions(chunk), 75)

    def test_fixed_target_chunk_size_remains_available(self) -> None:
        processor = ActionChunkProcessor(
            inference_hz=15.0,
            control_hz=100.0,
            target_chunk_size=100,
            alignment_mode="none",
        )
        chunk = np.zeros((32, 2), dtype=np.float64)

        produced = processor.push_actions(chunk)

        self.assertEqual(produced, 100)

    def test_empty_buffer_does_not_repeat_last_action(self) -> None:
        processor = ActionChunkProcessor(
            inference_hz=1.0,
            control_hz=1.0,
            postprocess=False,
        )
        processor.push_actions(np.asarray([[1.0, 2.0]], dtype=np.float64))

        first = processor.pop_action()
        second = processor.pop_action()

        np.testing.assert_allclose(first, np.asarray([1.0, 2.0]))
        self.assertIsNone(second)

    def test_async_chunk_alignment_uses_scheduled_start_delay(self) -> None:
        processor = ActionChunkProcessor(
            inference_hz=10.0,
            control_hz=10.0,
            chunk_align_window_s=0.3,
        )
        processor.push_actions(np.arange(11, dtype=np.float64).reshape(11, 1))

        for _ in range(7):
            processor.pop_action()

        produced = processor.push_actions(
            np.arange(5, 16, dtype=np.float64).reshape(11, 1),
            scheduled_start_delay_s=0.4,
        )

        self.assertGreater(produced, 0)
        for _ in range(3):
            last_old_action = processor.pop_action()
        first_new_action = processor.pop_action()

        np.testing.assert_allclose(last_old_action, np.asarray([9.0]))
        self.assertGreater(first_new_action[0], last_old_action[0])

    def test_sync_chunk_can_skip_alignment(self) -> None:
        processor = ActionChunkProcessor(
            inference_hz=10.0,
            control_hz=10.0,
            chunk_align_window_s=0.3,
        )
        processor._last_output_action = np.asarray([20.0])

        produced = processor.push_actions(
            np.arange(20, 31, dtype=np.float64).reshape(11, 1),
            align=False,
        )
        first_new_action = processor.pop_action()

        self.assertEqual(produced, 10)
        np.testing.assert_allclose(first_new_action, np.asarray([20.0]))

    def test_late_async_chunk_falls_back_instead_of_dropping_all(self) -> None:
        processor = ActionChunkProcessor(
            inference_hz=15.0,
            control_hz=100.0,
            chunk_align_window_s=0.3,
        )
        processor.push_actions(np.arange(16, dtype=np.float64).reshape(16, 1))
        while processor.pop_action() is not None:
            pass

        produced = processor.push_actions(
            np.arange(16, 32, dtype=np.float64).reshape(16, 1),
            scheduled_start_delay_s=1.5,
        )

        self.assertGreater(produced, 0)

    def test_late_async_chunk_never_drops_all_when_window_covers_chunk(self) -> None:
        processor = ActionChunkProcessor(
            inference_hz=10.0,
            control_hz=10.0,
            chunk_align_window_s=10.0,
        )
        processor._last_output_action = np.asarray([99.0])

        produced = processor.push_actions(
            np.arange(5, dtype=np.float64).reshape(5, 1),
            scheduled_start_delay_s=99.0,
        )

        self.assertGreater(produced, 0)


if __name__ == "__main__":
    unittest.main()
