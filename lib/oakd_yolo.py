##!/usr/bin/env python3

import contextlib
import datetime
import json
import os
import time
from pathlib import Path
from typing import Any, List, Tuple, Union

import blobconverter
import cv2
import depthai as dai
import numpy as np

BLACK = (255, 255, 255)
DISPLAY_WINDOW_SIZE_RATE = 2.0

DETECTION_TARGET = ["face", "phone", "right-hand", "left-hand"]


class OakdYolo(object):
    def __init__(self, config_path: str, model_path: str, fps: int = 10, save_fps: int = 0) -> None:
        if not Path(config_path).exists():
            raise ValueError(
                "Path {} does not poetry exist!".format(config_path))
        with Path(config_path).open() as f:
            config = json.load(f)
        nnConfig = config.get("nn_config", {})

        # parse input shape
        if "input_size" in nnConfig:
            self.width, self.height = tuple(
                map(int, nnConfig.get("input_size").split("x"))
            )

        # extract metadata
        metadata = nnConfig.get("NN_specific_metadata", {})
        self.classes = metadata.get("classes", {})
        self.coordinates = metadata.get("coordinates", {})
        self.anchors = metadata.get("anchors", {})
        self.anchorMasks = metadata.get("anchor_masks", {})
        self.iouThreshold = metadata.get("iou_threshold", {})
        self.confidenceThreshold = metadata.get("confidence_threshold", {})

        print(metadata)
        self.cam_fps = fps
        self.save_fps = save_fps
        if self.save_fps > self.cam_fps:
            print("[WARNING] save_fps should be smaller than camera FPS!")
        # parse labels
        nnMappings = config.get("mappings", {})
        self.labels = nnMappings.get("labels", {})

        self.nnPath = Path(model_path)
        # get model path
        if not self.nnPath.exists():
            print(
                "No blob found at {}. Looking into DepthAI model zoo.".format(
                    self.nnPath
                )
            )
            self.nnPath = str(
                blobconverter.from_zoo(
                    model_path, shaves=6, zoo_type="depthai", use_cache=True
                )
            )

        self._stack = contextlib.ExitStack()
        self._pipeline = self._create_pipeline()
        self._device = self._stack.enter_context(
            dai.Device(self._pipeline, usb2Mode=True)
        )
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        self.qControl = self._device.getInputQueue("control")
        self.qRgb = self._device.getOutputQueue(
            name="rgb", maxSize=4, blocking=False)
        self.qIsp = self._device.getOutputQueue(name="isp")
        self.qDet = self._device.getOutputQueue(
            name="nn", maxSize=4, blocking=False)
        self.counter = 0
        self.startTime = time.monotonic()
        self.frame_name = 0
        self.dir_name = ""
        self.path = ""
        self.num = 0
        self.counter = 0
        if self.save_fps > 0:
            self.qRaw = self._device.getOutputQueue(
                name="raw", maxSize=4, blocking=False
            )
            self.setup_save()
        self.start_image_save_time = time.time()
        self.last_image_save_time = time.time()

    def close(self) -> None:
        # self._device.close()
        # self.end = True
        if self.save_fps > 0:
            self.make_video()

    def set_camera_brightness(self, brightness: int) -> None:
        ctrl = dai.CameraControl()
        ctrl.setBrightness(brightness)
        self.qControl.send(ctrl)

    def get_labels(self) -> List[str]:
        return self.labels

    def _create_pipeline(self) -> dai.Pipeline:
        # OAK-Dのセットアップ
        # Create pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        controlIn = pipeline.create(dai.node.XLinkIn)
        camRgb = pipeline.create(dai.node.ColorCamera)
        detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        nnOut = pipeline.create(dai.node.XLinkOut)
        controlIn.setStreamName("control")
        xoutRgb.setStreamName("rgb")
        nnOut.setStreamName("nn")

        # Properties
        controlIn.out.link(camRgb.inputControl)
        camRgb.setPreviewKeepAspectRatio(False)
        camRgb.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setPreviewSize(1920, 1080)
        camRgb.setFps(self.cam_fps)

        xoutIsp = pipeline.create(dai.node.XLinkOut)
        xoutIsp.setStreamName("isp")
        camRgb.isp.link(xoutIsp.input)

        manip = pipeline.create(dai.node.ImageManip)
        manip.setMaxOutputFrameSize(self.width * self.height * 3)  # 640x640x3
        manip.initialConfig.setResizeThumbnail(self.width, self.height)
        camRgb.preview.link(manip.inputImage)
        if self.save_fps > 0:
            xoutRaw = pipeline.create(dai.node.XLinkOut)
            xoutRaw.setStreamName("raw")
            camRgb.video.link(xoutRaw.input)

        # Network specific settings
        detectionNetwork.setConfidenceThreshold(self.confidenceThreshold)
        detectionNetwork.setNumClasses(self.classes)
        detectionNetwork.setCoordinateSize(self.coordinates)
        detectionNetwork.setAnchors(self.anchors)
        detectionNetwork.setAnchorMasks(self.anchorMasks)
        detectionNetwork.setIouThreshold(self.iouThreshold)
        detectionNetwork.setBlobPath(self.nnPath)
        detectionNetwork.setNumInferenceThreads(2)
        detectionNetwork.input.setBlocking(False)

        # Linking
        manip.out.link(detectionNetwork.input)
        detectionNetwork.passthrough.link(xoutRgb.input)
        detectionNetwork.out.link(nnOut.input)
        return pipeline

    def frame_norm(self, frame: np.ndarray, bbox: Tuple[float]) -> List[int]:
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def get_frame(self) -> Union[np.ndarray, List[Any]]:
        try:
            inRgb = self.qRgb.get()
            inIsp = self.qIsp.get()
            inDet = self.qDet.get()
        except BaseException:
            raise
        if inIsp is not None:
            frame = inRgb.getCvFrame()
        if inDet is not None:
            detections = inDet.detections
            self.counter += 1
            width = frame.shape[1]
            height = frame.shape[1] * 9 / 16
            brank_height = width - height
            frame = frame[
                int(brank_height / 2): int(frame.shape[0] - brank_height / 2), 0:width
            ]
            for detection in detections:
                # Fix ymin and ymax to cropped frame pos
                detection.ymin = (width / height) * detection.ymin - (
                    brank_height / 2 / height
                )
                detection.ymax = (width / height) * detection.ymax - (
                    brank_height / 2 / height
                )
        if self.save_fps > 0:
            self.save_image(frame)
        return frame, detections

    def display_frame(
        self, name: str, frame: np.ndarray, detections: List[Any]
    ) -> None:
        if frame is not None:
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1] * DISPLAY_WINDOW_SIZE_RATE),
                    int(frame.shape[0] * DISPLAY_WINDOW_SIZE_RATE),
                ),
            )
            for detection in detections:

                if self.labels[detection.label] or DETECTION_TARGET:
                    bbox = self.frame_norm(
                        frame,
                        (detection.xmin, detection.ymin,
                         detection.xmax, detection.ymax),
                    )
                    cv2.putText(
                        frame,
                        self.labels[detection.label],
                        (bbox[0] + 10, bbox[1] + 20),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        (0, 255, 255),
                    )
                    cv2.putText(
                        frame,
                        f"{int(detection.confidence * 100)}%",
                        (bbox[0] + 10, bbox[1] + 40),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        (0, 255, 255),
                    )
                    cv2.rectangle(
                        frame, (bbox[0], bbox[1]), (bbox[2],
                                                    bbox[3]), (0, 255, 255), 2
                    )
            cv2.putText(
                frame,
                "NN fps: {:.2f}".format(
                    self.counter / (time.monotonic() - self.startTime)
                ),
                (2, frame.shape[0] - 4),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.4,
                (255, 255, 255),
            )
            # Show the frame
            cv2.imshow(name, frame)

    def setup_save(self) -> None:
        """
        @brief 保存方法のセットアップ
        """
        now = datetime.datetime.now()
        date = now.strftime("%Y%m%d%H%M")
        self.path = os.getcwd() + "/images/" + date
        print(self.path)
        os.makedirs(self.path, exist_ok=True)

    def save_image(self, frame: np.ndarray) -> None:
        """
        @brief 画像の保存
        """
        now = datetime.datetime.now()
        date = now.strftime("%Y%m%d%H%M")
        frameRaw = self.qRaw.get()
        if (time.time() - self.last_image_save_time) >= (
            1 / self.save_fps
        ) and self.qRaw is not None:
            frame = frameRaw.getCvFrame()
            file_path = "{}/{}_{}.jpg".format(self.path,
                                              date, str(self.num).zfill(3))
            cv2.imwrite(file_path, frame)
            print("save to: " + file_path)
            self.num += 1
            self.last_image_save_time = time.time()

    def make_video(self) -> None:
        """
        @brief 動画の作成
        """
        image_files = sorted([f for f in os.listdir(self.path) if f.endswith(".jpg")])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = len(image_files) / (
            self.last_image_save_time - self.start_image_save_time
        )
        print(
            "file num: "
            + str(len(image_files))
            + " time: "
            + str(self.last_image_save_time - self.start_image_save_time)
        )
        image_path = os.path.join(self.path, image_files[0])
        frame = cv2.imread(image_path)
        self.video_writer = cv2.VideoWriter(
            self.path + "/color.mp4", fourcc, fps, (frame.shape[1], frame.shape[0])
        )
        for image_file in image_files:
            image_path = os.path.join(self.path, image_file)
            frame = cv2.imread(image_path)
            if frame is not None:
                self.video_writer.write(frame)
        self.video_writer.release()
        print("save video to " + self.path + "/color.mp4")
