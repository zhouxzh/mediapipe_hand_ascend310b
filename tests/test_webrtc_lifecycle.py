from __future__ import annotations

import queue
from pathlib import Path
import threading
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import webrtc_hand_om_app
from scripts.webrtc_hand_om_app import HandOmVideoTrack
from webrtc_app import cann_encoder
from webrtc_app.v4l2_raw import V4l2RawCapture


class WebrtcTrackCleanupTests(unittest.TestCase):
    @staticmethod
    def _track() -> HandOmVideoTrack:
        track = HandOmVideoTrack.__new__(HandOmVideoTrack)
        track._state_lock = threading.Lock()
        track._frame_condition = threading.Condition()
        track._pipeline_stop = threading.Event()
        track._closed = False
        track._inference_thread = None
        track._infer_queue = None
        track.pipeline = mock.Mock()
        track.realtime_graph = object()
        track.cap = None
        track.capture_impl = None
        track.jpegd = None
        track._publish_stats = mock.Mock()
        return track

    def test_cleanup_defers_pipeline_release_when_worker_is_alive(self) -> None:
        track = self._track()
        pipeline = track.pipeline
        track._join_pipeline_threads = mock.Mock(return_value=False)

        track._cleanup()

        pipeline.close.assert_not_called()
        self.assertIs(track.pipeline, pipeline)

    def test_worker_exit_releases_deferred_pipeline(self) -> None:
        track = self._track()
        pipeline = track.pipeline
        track._closed = True
        track._inference_thread = threading.current_thread()
        track._run_inference_loop = mock.Mock()

        track._inference_loop()

        pipeline.close.assert_called_once_with()
        self.assertIsNone(track.pipeline)
        self.assertIsNone(track.realtime_graph)


class DatasetVideoSourceTests(unittest.TestCase):
    def test_video_listing_returns_supported_files_with_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_dir = root / "data" / "PianoVAM_v1" / "Video"
            video_dir.mkdir(parents=True)
            (video_dir / "b.MKV").write_bytes(b"b" * 2048)
            (video_dir / "a.mp4").write_bytes(b"a" * 1024)
            (video_dir / "notes.txt").write_text("ignore", encoding="utf-8")

            videos = webrtc_hand_om_app.list_dataset_videos(video_dir=video_dir, root=root)

        self.assertEqual([item["name"] for item in videos], ["a.mp4", "b.MKV"])
        self.assertEqual(videos[0]["path"], "data/PianoVAM_v1/Video/a.mp4")
        self.assertEqual(videos[0]["size_bytes"], 1024)

    def test_file_source_rewinds_and_requests_tracker_reset_at_eof(self) -> None:
        track = HandOmVideoTrack.__new__(HandOmVideoTrack)
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        track.camera_backend = webrtc_hand_om_app.CAMERA_BACKEND_OPENCV
        track.cap = mock.Mock()
        track.cap.read.side_effect = [(False, None), (True, frame)]
        track._source_is_file = True
        track._source_loop_count = 0
        track._pipeline_reset_requested = threading.Event()
        track._prediction_lock = threading.Lock()
        track._last_predictions = [{"hand": 1}]
        track._last_graph_streams = {"hand_rects": [1]}
        track._last_debug = {"debug": True}
        track._last_prediction_at = 1.0
        track._last_prediction_frame_index = 10
        track._capture_error = ""
        track._last_capture_ms = 0.0
        track._capture_fps_start = 0.0
        track._capture_fps_frames = 0
        track._capture_fps = 0.0

        output, nv12 = track._read_capture_frame()

        self.assertIs(output, frame)
        self.assertIsNone(nv12)
        track.cap.set.assert_called_once_with(webrtc_hand_om_app.cv2.CAP_PROP_POS_FRAMES, 0)
        self.assertEqual(track._source_loop_count, 1)
        self.assertTrue(track._pipeline_reset_requested.is_set())
        self.assertEqual(track._last_predictions, [])

    def test_requested_reset_clears_pipeline_and_graph_state(self) -> None:
        track = HandOmVideoTrack.__new__(HandOmVideoTrack)
        track._pipeline_reset_requested = threading.Event()
        track._pipeline_reset_requested.set()
        track.pipeline = mock.Mock()
        track.realtime_graph = mock.Mock()
        track.realtime_graph.STREAM_NAMES = ("palms", "hands")

        reset = track._reset_pipeline_if_requested()

        self.assertTrue(reset)
        track.pipeline.reset.assert_called_once_with()
        self.assertEqual(track.realtime_graph.last_streams, {"palms": [], "hands": []})


class V4l2CleanupTests(unittest.TestCase):
    def test_stop_defers_device_cleanup_while_capture_thread_is_alive(self) -> None:
        capture = V4l2RawCapture()
        capture._running.set()
        capture._thread = mock.Mock()
        capture._thread.is_alive.return_value = True

        with mock.patch.object(capture, "_stop_stream") as stop_stream:
            with mock.patch.object(capture, "_close_device_resources") as close_resources:
                capture.stop()

        stop_stream.assert_called_once_with()
        capture._thread.join.assert_called_once_with(timeout=2.0)
        close_resources.assert_not_called()


class CannVencTimeoutTests(unittest.TestCase):
    def test_timeout_destroys_channel_before_releasing_frame_resources(self) -> None:
        events: list[str] = []
        media = mock.Mock()
        media.dvpp_malloc.side_effect = [(101, cann_encoder.ACL_SUCCESS), (202, cann_encoder.ACL_SUCCESS)]
        media.dvpp_create_pic_desc.return_value = 303
        media.dvpp_create_stream_desc.return_value = 404
        media.venc_send_frame.return_value = cann_encoder.ACL_SUCCESS
        media.dvpp_destroy_pic_desc.side_effect = lambda _value: events.append("pic_desc")
        media.dvpp_free.side_effect = lambda _value: events.append("buffer")
        media.dvpp_destroy_stream_desc.side_effect = lambda _value: events.append("stream_desc")
        runtime = mock.Mock()
        runtime.set_context.return_value = cann_encoder.ACL_SUCCESS
        runtime.memcpy.return_value = cann_encoder.ACL_SUCCESS

        venc = cann_encoder.CannVenc.__new__(cann_encoder.CannVenc)
        venc._running = True
        venc._ctx = object()
        venc._stride = 16
        venc._out_buf_size = 6
        venc._frame_config = object()
        venc._channel_desc = object()
        venc.width = 2
        venc.height = 2
        venc._cb_queue = mock.Mock()
        venc._cb_queue.empty.return_value = True
        venc._cb_queue.get.side_effect = queue.Empty
        venc._active_callback_done = None
        venc.destroy = mock.Mock(side_effect=lambda: events.append("destroy"))

        nv12 = np.zeros((3, 2), dtype=np.uint8)
        with mock.patch.object(cann_encoder, "_acl_media", media):
            with mock.patch.object(cann_encoder, "_acl_rt", runtime):
                with self.assertRaisesRegex(RuntimeError, "callback timed out"):
                    venc._encode_frame(nv12, force_keyframe=False, pre_padded=False)

        self.assertEqual(events[0], "destroy")
        self.assertEqual(events.count("pic_desc"), 1)
        self.assertEqual(events.count("buffer"), 2)
        self.assertEqual(events.count("stream_desc"), 1)
        self.assertIsNone(venc._active_callback_done)


if __name__ == "__main__":
    unittest.main()
