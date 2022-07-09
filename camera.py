from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class OpticalFlowInfo:
    feature_params: dict
    lk_params: dict
    color: np.ndarray
    mask: np.ndarray
    last_gray: np.ndarray
    current_gray: np.ndarray
    last_frame: np.ndarray
    current_frame: np.ndarray
    last_features: Any
    current_features: Any
    flow: Any

    def __init__(self) -> None:
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.2,
                                   minDistance=7,
                                   blockSize=7)

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.color = np.random.randint(0, 255, (300, 3))


class VideoCap:
    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.OFInfo = OpticalFlowInfo()
        self.n_frames = 0
        self.optical_flow_sparse()

    def optical_flow_sparse_setup(self, frame=None):
        if frame is None:
            ret, frame = self.vs.read()
            if not ret:
                raise Exception("couldn't grab image frame")

        self.OFInfo = OpticalFlowInfo()
        self.OFInfo.mask = np.zeros_like(frame)
        self.OFInfo.last_frame = frame
        self.OFInfo.last_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.OFInfo.last_features = cv2.goodFeaturesToTrack(self.OFInfo.last_gray, mask=None,
                                                            **self.OFInfo.feature_params)
        return

    def optical_flow_sparse(self, frame: np.ndarray = None) -> None:
        if self.n_frames % 1000 == 0:
            self.optical_flow_sparse_setup(frame=frame)
            self.n_frames = 0
        self.n_frames += 1
        if frame is None:
            ret, frame = self.vs.read()
            if not ret:
                raise Exception("couldn't grab image frame")
        try:
            self.OFInfo.current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.OFInfo.current_features, st, err = cv2.calcOpticalFlowPyrLK(self.OFInfo.last_gray,
                                                                             self.OFInfo.current_gray,
                                                                             self.OFInfo.last_features, None,
                                                                             **self.OFInfo.lk_params)
            good_new = self.OFInfo.current_features[st == 1]  # and err<10]
            good_old = self.OFInfo.last_features[st == 1]

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.astype(np.int16).ravel()
                c, d = old.astype(np.int16).ravel()
                cv2.line(self.OFInfo.mask, (a, b), (c, d), self.OFInfo.color[i].tolist(), 2)
                # cv2.imshow('Mask', mask)
                cv2.circle(frame, (a, b), 5, self.OFInfo.color[i % 300].tolist(), -1)
            self.OFInfo.current_frame = cv2.add(frame, self.OFInfo.mask)
            self.OFInfo.last_gray = self.OFInfo.current_gray.copy()
            self.OFInfo.last_features = good_new.reshape(-1, 1, 2)
        except:
            pass
        return

    def __del__(self):
        self.vs.release()

    def get_frame(self):
        self.optical_flow_sparse()
        ret, jpg = cv2.imencode(".jpg", self.OFInfo.current_frame)
        return jpg.tobytes()