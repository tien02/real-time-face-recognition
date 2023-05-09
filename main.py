import os
import src
import sys
import torch
import cv2 as cv
import argparse
import numpy as np
from insightface.utils import face_align
from imutils.video import WebcamVideoStream, FPS
from torchvision.transforms import Resize

# Load model
detector, recognizer = src.loadDetectorRecognizer(keep_all=True)
classifier, input_name = src.loadClassifier()
config = src.load_config()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True, action='store', help='Perform on image / video', choices=['img', 'vid'])
    parser.add_argument('-p', '--path', type=str, action='store', help='path to the image')
    parser.add_argument('-s', '--save', action='store_true', help="Save image after drawing")
    opt = parser.parse_args()
    return opt


def process_face(frame:np.ndarray) -> np.ndarray:
    # Detect face
    faces, _, boxes, _ = detector(frame)

    if faces is not None:
    # Get face embedding
        faces = Resize((112, 112), antialias=True)(faces)
        if faces.dim() == 3:
            faces = torch.unsqueeze(faces, dim=0)
        faces = faces.detach().numpy().astype(np.float32)

        # Classify faces
        for face, box in zip(faces, boxes):
            out = recognizer.forward(np.expand_dims(face, axis=0))

            # Predict identity
            pred = classifier.run(None, {input_name: out})
            pred_label = pred[0][0]
            pred_acc = str(round(pred[1][0][pred_label] * 100, 2)) + "%"

            # Draw bounding box
            frame = src.putBoundingBox(frame, box.astype(np.int32), pred_label + " " + pred_acc)

    return frame


def video(save:bool = False):
    print("##########################")
    print("### Press ESC to exit ####")
    print("##########################")

    # Initalize the video capture & FPS
    vid = WebcamVideoStream()

    if not vid.stream.isOpened():
        print("Can not read video file")
        sys.exit()

    vid.stream.set(cv.CAP_PROP_FRAME_WIDTH, config["FRAME"]["WIDTH"])
    vid.stream.set(cv.CAP_PROP_FRAME_HEIGHT, config["FRAME"]["HEIGHT"])
    vid.start()
    fps = FPS().start()

    record = None
    if save:
        video_name = "webcam_output"
        if not os.path.exists("./result"):
            os.makedirs("result")
        record = cv.VideoWriter(f'result/{video_name}.avi', 
                         cv.VideoWriter_fourcc(*'MJPG'),
                         30, (config["FRAME"]["WIDTH"], config["FRAME"]["HEIGHT"]))

    # Working
    while True:
        frame = vid.read()
        fps.update()
        fps.stop()
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Process face
        frame = process_face(img)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        
        # Draw FPS on the frame
        frame = src.putFPS(
            frame,
            str(int(fps.fps())),
            config["FRAME"]["HEIGHT"],
            config["FRAME"]["WIDTH"],
            True
        )

        # Show video
        cv.imshow("frame", frame)

        # Save video
        if record is not None:
            record.write(frame)

        if cv.waitKey(1) & 0xFF==27:
            break

    vid.stop()
    if record is not None:
        record.release()
    cv.destroyAllWindows()

    print()
    print("##########################")
    print("#### Finish streaming ####")
    print("##########################")


def image(path:str, save:bool):
    # Read input 
    img = cv.imread(path)
    if img is None:
        print(f"Can not read image from {path}")
        sys.exit()
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Process
    img = process_face(img)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    # Show image after drawing
    cv.imshow("Ouput", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Save image after drawing 
    if save:
        file_name, file_extension = path.split(".")
        save_path = file_name + "_output." + file_extension
        cv.imwrite(save_path, img)


def main(task:str, path:str = None, save:bool = False):
    if task == 'img':
        image(path=path, save=save)
    elif task == 'vid':
        video(save = save)
    else:
        print(f"No support for '{task}'")

if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))