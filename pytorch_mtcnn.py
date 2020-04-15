import signal
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from facenet_pytorch import MTCNN
from loguru import logger

import cv2

model: MTCNN = None


def get_faces(frame: np.ndarray, **kwargs) -> List[np.ndarray]:
    boxes, _ = model.detect(frame)

    if boxes is None:
        # logger.debug("No boxes found")
        return list()
    return boxes


def signal_handler(sig, frame):
    logger.debug("Shutting down")
    exit(0)


@logger.catch
def main(input: str, gray: bool = False, resize: float = 1.0) -> None:
    assert input.isdigit(), "Debug with webcam plz"
    assert resize > 0, "Resize limit not allowed"

    # Setup
    signal.signal(signal.SIGINT, signal_handler)
    cap = cv2.VideoCapture(int(input))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Running on device: {device}")

    # Model init
    global model
    model = MTCNN(keep_all=True, device=device)
    model.eval()

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
        locations = get_faces(frame)

        # Compute FPS
        counter += 1
        if (time.time() - start_time) > 1:
            fps = round(counter / (time.time() - start_time), 2)
            logger.info(f"FPS: {fps}")
            counter = 0
            start_time = time.time()
            calculated_FPS_array.append(fps)

        # Display the results
        for location in locations:

            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
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
