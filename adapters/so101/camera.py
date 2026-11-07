"""OpenCV camera adapter for SO-101 wrist and column cameras.

Config keys
-----------
device_alias : path or alias, e.g. /dev/v4l/by-id/...
role         : "wrist" | "column" (or any string role)
width        : capture width in pixels
height       : capture height in pixels
fourcc       : FourCC string, e.g. "MJPG" or "YUYV"
fps          : target frame rate (default 30)
"""

from __future__ import annotations

import time
import numpy as np

from interfaces.hardware import CameraInterface, Frame


class OpenCVCameraAdapter(CameraInterface):
    """CameraInterface backed by OpenCV VideoCapture."""

    def __init__(
        self,
        device_alias: str,
        role: str,
        width: int,
        height: int,
        fourcc: str,
        fps: int = 30,
    ) -> None:
        self._device = device_alias
        self._role = role
        self._width = width
        self._height = height
        self._fourcc = fourcc
        self._fps = fps
        self._cap = None

    @property
    def role(self) -> str:
        return self._role

    def start(self) -> None:
        import cv2
        cap = cv2.VideoCapture(self._device)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera: {self._device}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self._fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._cap = cap

    def capture(self) -> Frame:
        import cv2
        if self._cap is None:
            raise RuntimeError("call start() before capture()")
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"frame read failed: {self._device}")
        h, w = frame.shape[:2]
        return Frame(
            data=frame,
            width=w,
            height=h,
            timestamp_s=time.monotonic(),
            camera_id=self._device,
            role=self._role,
            encoding="bgr",
        )

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_alive(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
