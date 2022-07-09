import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
from datetime import datetime
from utilities import OpticalFlowInfo, get_video_handles


def optical_flow_sparse_setup(frame: np.ndarray) -> OpticalFlowInfo:
    OFInfo = OpticalFlowInfo()
    OFInfo.mask = np.zeros_like(frame)
    OFInfo.last_frame = frame
    OFInfo.last_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    OFInfo.last_features = cv2.goodFeaturesToTrack(OFInfo.last_gray, mask=None, **OFInfo.feature_params)
    return OFInfo


def optical_flow_sparse(current_frame: np.ndarray, OFInfo: OpticalFlowInfo, nframes: int = 0) -> OpticalFlowInfo:
    if nframes % 1000 == 0:
        OFInfo = optical_flow_sparse_setup(current_frame)

    OFInfo.current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    OFInfo.current_features, st, err = cv2.calcOpticalFlowPyrLK(OFInfo.last_gray, OFInfo.current_gray,
                                                                OFInfo.last_features, None, **OFInfo.lk_params)
    good_new = OFInfo.current_features[st == 1]  # and err<10]
    good_old = OFInfo.last_features[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.astype(np.int16).ravel()
        c, d = old.astype(np.int16).ravel()
        cv2.line(OFInfo.mask, (a, b), (c, d), OFInfo.color[i].tolist(), 2)
        # cv2.imshow('Mask', mask)
        cv2.circle(current_frame, (a, b), 5, OFInfo.color[i % 300].tolist(), -1)
    OFInfo.current_frame = cv2.add(current_frame, OFInfo.mask)
    OFInfo.last_gray = OFInfo.current_gray.copy()
    OFInfo.last_features = good_new.reshape(-1, 1, 2)
    return OFInfo


def main():
    cap = cv2.VideoCapture(0)
    status, frame = cap.read()

    OFInfo = optical_flow_sparse_setup(frame)

    start_time_py = time.time()
    start_time = datetime.now()
    nframes = 0
    while True:
        status, frame = cap.read()
        if not status:
            print("Stream Ended: Exiting")
            break
        nframes += 1

        fps = round(nframes / (time.time() - start_time_py), 2)
        # frame = draw_insects(frame,find_insects_bg(frame))
        OFInfo = optical_flow_sparse(frame, OFInfo, nframes)
        frame = OFInfo.current_frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        # writer.write(frame)
        cv2.imshow("BG", frame)
        if cv2.waitKey(1) & 0xff == 27:
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
