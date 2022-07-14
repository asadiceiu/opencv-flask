from dataclasses import dataclass
from typing import Any
from datetime import datetime

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
    def __init__(self, video_path=None, refresh_timeout: int = 500):
        """
        Creates a VideoCap object with the given video path.
        :param video_path:
        :param refresh_timeout:
        """
        self.refresh_timeout = refresh_timeout
        self.vs = cv2.VideoCapture(video_path) if video_path is not None else cv2.VideoCapture(0)
        self.OFInfo = OpticalFlowInfo()
        self.n_frames = 0
        self.optical_flow_sparse()
        self.writer = None
        self.start_writing = False

    def setup_writer(self) -> None:
        """
        Sets up the writer for the video.
        :return:
        """
        frame_height = int(self.vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(self.vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(self.vs.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # creates a file name with current date and time from datetime module
        file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".avi"
        print("Video recording: " + file_name)
        self.writer = cv2.VideoWriter(file_name, fourcc, fps, (frame_width, frame_height))
        self.start_writing = True
        return


    def optical_flow_sparse_setup(self, frame=None):
        """
        Sets up the optical flow sparse algorithm.
        :param frame:
        :return:
        """
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
        """
        Performs optical flow sparse algorithm.
        :param frame:
        :return:
        """
        if self.n_frames % self.refresh_timeout == 0:
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
                cv2.line(self.OFInfo.mask, (a, b), (c, d), self.OFInfo.color[i].tolist(), 1)
                # cv2.imshow('Mask', mask)
                cv2.circle(frame, (a, b), 3, self.OFInfo.color[i % 300].tolist(), -1)
            self.OFInfo.current_frame = cv2.add(frame, self.OFInfo.mask)
            self.OFInfo.last_gray = self.OFInfo.current_gray.copy()
            self.OFInfo.last_features = good_new.reshape(-1, 1, 2)
        except:
            pass
        return

    def __del__(self):
        self.vs.release()
        if self.writer is not None:
            self.writer.release()
        self.start_writing = False

    def write_frame(self, frame: np.ndarray) -> None:
        """
        Writes the given frame to the video.
        :param frame:
        :return:
        """
        if self.writer is None:
            self.setup_writer()
            self.start_writing = True
        self.writer.write(frame)
        return

    def stop_writer(self) -> None:
        """
        Stops the writer.
        :return:
        """
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.start_writing = False
        return

    def get_frame(self):
        """
        Returns the current frame without any modification to it.
        :return:
        """
        ret, frame = self.vs.read()
        if not ret:
            raise Exception("couldn't grab image frame")
        ret, jpg = cv2.imencode(".jpg", frame)
        if self.start_writing:
            self.write_frame(frame)
        return jpg.tobytes()

    def get_opticalflow(self):
        """
        Returns the current optical flow frame.
        :return:
        """
        self.optical_flow_sparse()
        ret, jpg = cv2.imencode(".jpg", self.OFInfo.current_frame)
        if self.start_writing:
            self.write_frame(self.OFInfo.current_frame)
        return jpg.tobytes()
