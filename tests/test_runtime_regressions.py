from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

from hand_pipeline import om_runtime
from hand_pipeline import tracking
from hand_pipeline import two_stage
from hand_pipeline.roi import NormalizedRect


class _FakeAclRt:
    def __init__(self) -> None:
        self.context_index = 0
        self.stream_index = 0
        self.reset_calls = 0

    def set_device(self, _device_id: int) -> int:
        return 0

    def create_context(self, _device_id: int):
        self.context_index += 1
        return f"context-{self.context_index}", 0

    def create_stream(self):
        self.stream_index += 1
        return f"stream-{self.stream_index}", 0

    def destroy_stream(self, _stream) -> int:
        return 0

    def destroy_context(self, _context) -> int:
        return 0

    def reset_device(self, _device_id: int) -> int:
        self.reset_calls += 1
        return 0


class _FakeAcl:
    def __init__(self, init_result: int = 0) -> None:
        self.init_result = init_result
        self.init_calls = 0
        self.finalize_calls = 0
        self.rt = _FakeAclRt()

    def init(self) -> int:
        self.init_calls += 1
        return self.init_result

    def finalize(self) -> int:
        self.finalize_calls += 1
        return 0


class PersistentAclRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_acl = om_runtime.acl
        om_runtime._ACL_RUNTIME_REFCOUNT = 0
        om_runtime._ACL_RUNTIME_INITIALIZED_HERE = False
        om_runtime._ACL_RUNTIME_CAN_FINALIZE = False

    def tearDown(self) -> None:
        om_runtime.acl = self.original_acl
        om_runtime._ACL_RUNTIME_REFCOUNT = 0
        om_runtime._ACL_RUNTIME_INITIALIZED_HERE = False
        om_runtime._ACL_RUNTIME_CAN_FINALIZE = False

    def test_last_runtime_finalizes_module_owned_acl(self) -> None:
        fake_acl = _FakeAcl()
        om_runtime.acl = fake_acl

        first = om_runtime.PersistentAclRuntime()
        second = om_runtime.PersistentAclRuntime()
        self.assertEqual(fake_acl.init_calls, 1)

        first.release()
        self.assertEqual(fake_acl.finalize_calls, 0)
        second.release()

        self.assertEqual(fake_acl.rt.reset_calls, 1)
        self.assertEqual(fake_acl.finalize_calls, 1)

    def test_external_acl_is_not_finalized(self) -> None:
        fake_acl = _FakeAcl(init_result=om_runtime.ACL_ALREADY_INITIALIZED)
        om_runtime.acl = fake_acl

        runtime = om_runtime.PersistentAclRuntime()
        runtime.release()

        self.assertEqual(fake_acl.rt.reset_calls, 0)
        self.assertEqual(fake_acl.finalize_calls, 0)

    def test_finalize_disabled_keeps_acl_initialized(self) -> None:
        fake_acl = _FakeAcl()
        om_runtime.acl = fake_acl

        runtime = om_runtime.PersistentAclRuntime(finalize_on_release=False)
        runtime.release()

        self.assertEqual(fake_acl.rt.reset_calls, 0)
        self.assertEqual(fake_acl.finalize_calls, 0)


class TrackingLimitTests(unittest.TestCase):
    @staticmethod
    def _rect(x_center: float) -> NormalizedRect:
        return NormalizedRect(x_center=x_center, y_center=0.5, width=0.1, height=0.1, rotation=0.0)

    def test_previous_rects_are_prioritized_when_limiting(self) -> None:
        palm_first = self._rect(0.1)
        palm_second = self._rect(0.3)
        previous = self._rect(0.8)

        limited = tracking._limit_associated_rects(
            [palm_first, palm_second, previous],
            {id(previous)},
            max_hands=2,
        )

        self.assertEqual(limited, [palm_first, previous])

    def test_zero_max_hands_returns_no_rects(self) -> None:
        rect = self._rect(0.1)
        self.assertEqual(tracking._limit_associated_rects([rect], set(), max_hands=0), [])


class OmPipelineInitializationTests(unittest.TestCase):
    def test_landmark_failure_releases_detector_and_runtime(self) -> None:
        runtime = mock.Mock()
        detector = mock.Mock()

        with mock.patch.object(two_stage, "PersistentAclRuntime", return_value=runtime):
            with mock.patch.object(two_stage, "_OmDetectorRunner", return_value=detector):
                with mock.patch.object(
                    two_stage,
                    "_OmLandmarkRunner",
                    side_effect=RuntimeError("landmark load failed"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "landmark load failed"):
                        two_stage.OmHandPipeline(
                            Path("detector.om"),
                            Path("landmark.om"),
                            device_id=0,
                            score_threshold=0.5,
                            nms_iou=0.3,
                            max_hands=2,
                            min_hand_score=0.5,
                            max_det=20,
                        )

        detector.close.assert_called_once_with()
        runtime.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
