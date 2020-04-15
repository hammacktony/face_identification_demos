from pathlib import Path
from typing import List, Tuple

import cvlib
import face_recognition
import numpy as np

import cv2


def get_faces(frame: np.ndarray, **kwargs) -> List[Tuple[int, ...]]:
    return cvlib.detect_face(frame, **kwargs)[0]


def get_embeddings(frame: np.ndarray, locations: List[Tuple[int, ...]]) -> List[np.ndarray]:
    return face_recognition.face_encodings(frame, locations)


def main(input: str, gray: bool = False, resize: float = 1.0) -> None:
    assert input.isdigit(), "Debug with webcam plz"
    assert resize > 0, "Resize limit not allowed"

    cap = cv2.VideoCapture(int(input))

    while True:
        ret, frame_original = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB)
        if resize != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)

        locations = get_faces(frame)
        name = "Person"

        # Display the results
        for x1, y1, x2, y2 in locations:

            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            rescale = int(1 / resize)

            x1 *= rescale
            x2 *= rescale
            y1 *= rescale
            y2 *= rescale

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


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input source (webcam or path)")
    parser.add_argument("--resize", type=float, default=1.0, help="Resize image to increase inference speed")
    kwargs = vars(parser.parse_args())

    main(**kwargs)
