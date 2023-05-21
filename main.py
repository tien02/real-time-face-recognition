import os
import src
import sys
import argparse
import cv2 as cv
import numpy as np

from insightface.utils import face_align
from utils import load_data, cosine_similarity
from imutils.video import WebcamVideoStream, FPS

# Load model
detector, recognizer = src.loadDetectorRecognizer(keep_all=True)
classifier, input_name = src.loadClassifier()
config = src.load_config()
data = load_data(config['RECOGNIZER']['data_dir'])
values = np.concatenate(data['embedding'])


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True, action='store', help='Perform on image / video', choices=['img', 'vid'])
    parser.add_argument('-p', '--path', type=str, action='store', help='path to the image')
    parser.add_argument('-s', '--save', action='store_true', help="Save image after drawing")
    opt = parser.parse_args()
    return opt


def predict_id(query):
    if config['CLASSIFIER']['use_model'] == "ml_model":
        pred = classifier.run(None, {input_name: query})
        pred_label = pred[0][0]
        pred_acc_logits = pred[1][0][pred_label]

        if pred_acc_logits >= config['CLASSIFIER']['threshold']: 
            return_tag = pred_label + " " + str(round(pred[1][0][pred_label] * 100, 2)) + "%"
        else:
            return_tag = "unknown"

    else:
        vector_sim = cosine_similarity(query, values)
        pred_id = np.argmax(vector_sim, axis=1)[0]
        pred_logits = vector_sim[0,pred_id] 
        if pred_logits <= config['CLASSIFIER']['threshold']:
            return_tag = 'unknown'
        else:
            return_tag = data['name'][pred_id] + " " + str(round(pred_logits * 100, 2)) + "%"

    return return_tag


def process_face(frame:np.ndarray) -> np.ndarray:
    # Detect face
    boxes, _, landmarks = detector.detect(frame, landmarks=True)
    if boxes is not None and len(boxes) > 0:
        # Classify faces
        try:
            for box, marks in zip(boxes, landmarks):
                box = np.array(box).astype(np.int32)

                # Crop the face in frame
                crop_face = frame[box[1] : box[3], box[0] : box[2]]
                crop_face_h, crop_face_w, _ = crop_face.shape

                # Resize the crop face to the Recognizer need
                resize_crop_face = cv.resize(crop_face, (112,112))
                resize_crop_h, resize_crop_w, _ = resize_crop_face.shape

                # Scale the landmark by the factor of downscale size
                crop_marks = marks.copy()
                crop_marks[:, 0] = crop_marks[:, 0] - box[0]
                crop_marks[:, 1] = crop_marks[:, 1] - box[1]
                scale_factor = np.array((resize_crop_w, resize_crop_h)) / np.array((crop_face_w, crop_face_h))
                new_marks = np.multiply(crop_marks, scale_factor)

                # Face Alignment
                face_aligned = face_align.norm_crop(resize_crop_face, new_marks.astype(np.float64))
                face_aligned = face_aligned.astype(np.uint8)
                face_aligned = np.transpose(face_aligned, (2,0,1)).astype(np.float32)

                # Extract Face Embedding
                out = recognizer.forward(np.expand_dims(face_aligned, axis=0))

                # Predict identity
                return_tag = predict_id(out)

                # Draw bounding box
                frame = src.putBoundingBox(frame, box.astype(np.int32), return_tag)

                # Draw landmarks
                frame = cv.circle(frame, center=marks[0, :].astype(np.int32), radius=5, color=(0,255,0),thickness=-1)
                frame = cv.circle(frame, center=marks[1, :].astype(np.int32), radius=5, color=(255,0,0),thickness=-1)
                frame = cv.circle(frame, center=marks[2, :].astype(np.int32), radius=5, color=(0,0,255),thickness=-1)
                frame = cv.circle(frame, center=marks[3, :].astype(np.int32), radius=5, color=(0,0,0),thickness=-1)
                frame = cv.circle(frame, center=marks[4, :].astype(np.int32), radius=5, color=(255,255,255),thickness=-1)
            
        except:
            pass
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
                        10, (config["FRAME"]["WIDTH"], config["FRAME"]["HEIGHT"]))

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