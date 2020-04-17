
import signal
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import insightface
import mxnet as mx
import numpy as np


from loguru import logger

model = None

# CPU Only
ctx_id = -1

"""
Works about the same as Pytorch/TF models.

Easy integration with face recognition and compare embeddings.

Grab insightface.app.FaceAnalysis and use embeddings to compare facial similarity.
"""


def make_model(nms_threshold: float = 0.4):
    global model
    model = insightface.app.FaceAnalysis().det_model
    model.prepare(ctx_id=ctx_id, nms=nms_threshold)
    return model

def signal_handler(sig, frame):
    logger.debug("Shutting down")
    exit(0)

def get_faces(frame: np.ndarray, **kwargs) -> List[np.ndarray]:
    faces, _ = model.detect(frame)
    return faces


@logger.catch
def main(input: str, resize: float = 1.0) -> None:
    assert input.isdigit(), "Debug with webcam plz"
    assert resize > 0, "Resize limit not allowed"

    # Setup
    signal.signal(signal.SIGINT, signal_handler)
    cap = cv2.VideoCapture(int(input))

    # Model init
    make_model()

    # Name of bounding box
    name = "Person"

    # FPS Stuff
    counter = 0
    calculated_FPS_array = list()

    # Start Inference timer
    start_time = time.time()

    while True:
        ret, frame_original = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB)

        assert len(frame.shape) == 3, "Improper frame shape"

        if resize != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize, interpolation=cv2.INTER_AREA)

        # Model inference
        faces = get_faces(frame)

        # Compute FPS
        counter += 1
        if (time.time() - start_time) > 1:
            fps = round(counter / (time.time() - start_time), 2)
            logger.info(f"FPS: {fps}")
            counter = 0
            start_time = time.time()
            calculated_FPS_array.append(fps)

        # Display the results
        for idx, location in enumerate(faces):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            # location = face
            if resize == 1.0:
                rescaled_location = location.astype(int)
            else:
                rescaled_location = (location * (1 / resize)).astype(int)       

            # Get coordinates
            x1 = rescaled_location[0]
            x2 = rescaled_location[2]
            y1 = rescaled_location[1]
            y2 = rescaled_location[3]

            # Draw a box around the face
            cv2.rectangle(frame_original, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame_original, (x1, y1 - 35), (x2, y1), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_original, name, (x1 + 6, y1 - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow("Video", frame_original)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release handle to the webcam
    cap.release()
    cv2.destroyAllWindows()

    try:
        calculated_FPS_array.pop(0)  # -> First run has model overhead
        logger.info(f"Average inference: {np.asarray(calculated_FPS_array).mean()}")
    except Exception:
        pass


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input source (webcam or path)")
    parser.add_argument("--resize", type=float, default=1.0, help="Resize image to increase inference speed")
    kwargs = vars(parser.parse_args())

    main(**kwargs)
