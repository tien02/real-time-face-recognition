from PIL import Image
import numpy as np
import torch
import cv2 as cv
import yaml
import torchvision
from imutils.video import WebcamVideoStream, FPS
from facenet_pytorch import MTCNN, InceptionResnetV1

from utils import load_config, putFPS, putBoundingBox

if __name__ == "__main__":

    print("##########################")
    print("### Press ESC to exit ####")
    print("##########################")

    device = "cpu"
    config = load_config()
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Initialize the detector
    detector = MTCNN(device=device, thresholds= [0.7, 0.7, 0.8] ,keep_all=True)
    detector = torch.compile(detector)

    # Initalize the video capture & FPS
    vid = WebcamVideoStream()
    vid.stream.set(cv.CAP_PROP_FRAME_WIDTH, config["FRAME"]["WIDTH"])
    vid.stream.set(cv.CAP_PROP_FRAME_HEIGHT, config["FRAME"]["HEIGHT"])
    vid.start()
    fps = FPS().start()

    # Working
    while True:
        frame = vid.read()
        fps.update()
        fps.stop()

        # Draw FPS on the frame
        image = putFPS(
            frame,
            str(int(fps.fps())),
            config["FRAME"]["HEIGHT"],
            config["FRAME"]["WIDTH"],
            True
        )

        # Face Detector finding face in 

        boxes, confidences = detector.detect(
            Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        )

        # Draw bounding boxes & texts to the detected faces
        if boxes is not None:
            for box, conf in zip(boxes, confidences):
                putBoundingBox(frame, int(box[0]), int(box[1]), int(box[2]), int(box[3]), str(round(conf * 100, 2)) + "%")
        cv.imshow("frame", image)
        if cv.waitKey(1) & 0xFF==27:
            break
    vid.stop()
    cv.destroyAllWindows()
    print()
    print("##########################")
    print("#### Finish streaming ####")
    print("##########################")
