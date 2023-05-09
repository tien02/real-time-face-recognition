import os
import sys
import torch
import cv2 as cv
import numpy as np
from imutils.video import WebcamVideoStream, FPS
from termcolor import colored
from src import myMTCNN
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
    detector = myMTCNN(keep_all=False, margin=20, post_process=False)
    print(colored("Done", 'green'))

    # Initalize the video capture & FPS
    vid = WebcamVideoStream()
    if not vid.stream.isOpened():
        print("Can not read from webcam")
        sys.exit()
    
    # Set up save path
    username = input(colored("User name: ", "blue"))
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

        # Detect faces
        faces, probs, boxes, _ = detector(img)

        # Draw bounding box 
        if boxes is not None:
            boxes = boxes[0].astype(np.int32)    
            frame = src.putBoundingBox(frame, boxes.astype(np.int32), str(round(probs * 100,2)))
            
            if (leap % 5 == 0) and (flag is True):
                faces = torch.permute(faces, (1,2,0)).detach().numpy().astype(np.uint8)

                save_file = os.path.join(user_dir, f"{username}_{count + 1}.jpeg")

                cv.imwrite(save_file, cv.cvtColor(faces, cv.COLOR_RGB2BGR))
                count += 1

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