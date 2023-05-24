import os
import sys
import cv2 as cv
import numpy as np

from termcolor import colored
from facenet_pytorch import MTCNN
from insightface.utils import face_align
from imutils.video import WebcamVideoStream, FPS

import src

if __name__ == "__main__":

    print("##########################")
    print("### Press ESC to exit ####")
    print("##########################\n")

    config = src.load_config()
    data_dir = config['RECOGNIZER']['data_dir']
    if not os.path.exists(data_dir):
        print(f"Can not find {data_dir}\nCreating{data_dir}")
        
    # Init hyperparameter
    count = 0
    leap = 1
    flag = True

    # Init Detector
    print(colored("Detector initialize", 'blue'), end=' - ')
    detector = MTCNN(keep_all=False, post_process=False)
    print(colored("Done", 'green'))

    # Initalize the video capture & FPS
    vid = WebcamVideoStream()
    if not vid.stream.isOpened():
        print("Can not read from webcam")
        sys.exit()
    
    # Set up save path
    username = input(colored("User name: ", "blue")).strip()
    user_dir = os.path.join(data_dir, username)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    else:
        print(f"Already exist {username} in {data_dir}")
        sys.exit()
    
    # Set up camera and FPS
    vid.stream.set(cv.CAP_PROP_FRAME_WIDTH, config["FRAME"]["WIDTH"])
    vid.stream.set(cv.CAP_PROP_FRAME_HEIGHT, config["FRAME"]["HEIGHT"])
    vid.start()
    fps = FPS().start()

    # Working
    while True:
        frame = vid.read()
        fps.update()
        fps.stop()
        img = cv.cvtColor(frame.copy(), cv.COLOR_BGR2RGB)

        # Draw FPS on the frame
        frame = src.putFPS(
            frame,
            str(int(fps.fps())),
            config["FRAME"]["HEIGHT"],
            config["FRAME"]["WIDTH"],
            True
        )

        frame = src.putTrainingData(
            frame,
            str(count + 1),
            config["FRAME"]["HEIGHT"],
            config["FRAME"]["WIDTH"],
            True
        )

        # Detect face
        boxes, probs, landmarks = detector.detect(img, landmarks=True)
        if boxes is not None and len(boxes) > 0:
            # Classify faces
            try:
                for box, prob, marks in zip(boxes, probs, landmarks):
                    # Draw bounding box
                    frame = src.putBoundingBox(frame, box.astype(np.int32), str(round(prob * 100, 2)) + "%")
                    frame = cv.circle(frame, center=marks[0, :].astype(np.int32), radius=5, color=(0,255,0),thickness=-1)
                    frame = cv.circle(frame, center=marks[1, :].astype(np.int32), radius=5, color=(255,0,0),thickness=-1)
                    frame = cv.circle(frame, center=marks[2, :].astype(np.int32), radius=5, color=(0,0,255),thickness=-1)
                    frame = cv.circle(frame, center=marks[3, :].astype(np.int32), radius=5, color=(0,0,0),thickness=-1)
                    frame = cv.circle(frame, center=marks[4, :].astype(np.int32), radius=5, color=(255,255,255),thickness=-1)
            
                if (leap % 5 == 0) and (flag is True):
                    box = np.array(box).astype(np.int32)

                    # Crop the face in frame
                    crop_face = img[box[1] : box[3], box[0] : box[2]]
                    crop_face_h, crop_face_w, _ = crop_face.shape

                    # Resize the crop face to the Recognizer need
                    resize_crop_face = cv.resize(crop_face, (config['RECOGNIZER']['crop_img_size'],config['RECOGNIZER']['crop_img_size']))
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

                    save_file = os.path.join(user_dir, f"{username}_{count + 1}.jpeg")

                    cv.imwrite(save_file, cv.cvtColor(face_aligned, cv.COLOR_RGB2BGR))
                    count += 1
            except:
                pass

            if count >= config['RECOGNIZER']['max_training_img']:
                flag = False
            
            leap +=1

        cv.imshow("frame", frame)

        if not flag:
            print(colored("Gather enough data", "blue"))
            break
        
        if cv.waitKey(1) & 0xFF==27:
            break
    vid.stop()
    cv.destroyAllWindows()
    print()
    print("##########################")
    print("#### Finish streaming ####")
    print("##########################")