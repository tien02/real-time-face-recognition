import os 
import glob
import shutil
import cv2 as cv
import numpy as np
from PIL import Image
from insightface.utils import face_align

from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query, HTTPException, File, UploadFile, BackgroundTasks

import src
from make_identifier import create_classifier, create_embedding

app = FastAPI()

origins = [
    # "http://localhost.tiangolo.com",
    # "https://localhost.tiangolo.com",
    # "http://localhost",
    # "http://localhost:8000",
    "*",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model
detector, recognizer = src.loadDetectorRecognizer(keep_all=True)
classifier, input_name = src.loadClassifier()
config = src.load_config()


@app.get("/")
def root():
    '''
    Greeting!!!
    '''
    if os.path.exists(config['RECOGNIZER']['data_dir']):
        return {
            "message": "Welcome to Face Recognition API."
        }
    else:
        return {
            "message": f"Error when trying to connect {config['RECOGNIZER']['data_dir']}, there is no database available."
        }


@app.get('/img-db-info')
def get_img_db_info():
    '''
    Get database information, return all files in the database
    '''
    user_lst = next(os.walk(config['RECOGNIZER']['data_dir']))[1]
    return {
        'all_identity': user_lst,
        'number_of_identity': len(user_lst)
    }


@app.get('/show_img/{name}')
def show_img(name: str | None = None):
    '''
    Return image file from the given name

    Arguments:  
        name(str): image file
    '''
    if name is None:
        raise HTTPException(status_code=404, detail='Client should provide name')
    
    id_dir = os.path.join(config['RECOGNIZER']['data_dir'], name)
    if not os.path.exists(id_dir):
        raise HTTPException(status_code=404, detail=f'{name} not in the database')

    lst_imgs = []
    for im in glob.glob(os.path.join(id_dir, '*')):
        if '.pkl' in im:
            continue
        else:
            lst_imgs.append(im)
    
    img_file = np.random.choice(lst_imgs, 1)
    return FileResponse(img_file[0])


@app.get('/prepapre-resource/')
def prepare_resource():
    '''
    Retrain model
    '''
    create_embedding(force=True)


@app.post('/register')
def face_register(
    background_tasks: BackgroundTasks,
    img_files: list[UploadFile] | None = File(..., description="Upload Image"),
    to_gray: bool | None = Query(
            default=False, 
            description="Whether save image in gray scale or not"),
    username: str  = Query(
        ...,
        description="File's name to be save, file extension can be available or not",
    ),):
    '''
    Add new user to the database by face registering. Resize image if necessary.

     Arguments:  
        img_files(File): upload image file
        username(string): user name to be saved
    '''
    save_dir = os.path.join(config['RECOGNIZER']['data_dir'], username)
    # Raise error if there is duplicate
    if os.path.exists(save_dir):
        raise HTTPException(status_code=409, detail=f"{username} has already in the database.")
    else:
        os.makedirs(save_dir)
    
    # Save image to database
    for img_file in img_files:
        
        # Process to valid save name
        if '/' in img_file.filename:    
            save_img_dir = os.path.join(save_dir, img_file.filename.split('/')[-1])
        elif "\\" in img_file.filename:
            save_img_dir = os.path.join(save_dir, img_file.filename.split("\\")[-1])
        else:
            save_img_dir = os.path.join(save_dir, img_file.filename)

        if to_gray is False:
            with open(save_img_dir, "wb") as w:
                shutil.copyfileobj(img_file.file, w)

        else:
            try:
                image = Image.open(img_file.file)
                if image.mode in ("RGBA", "P"):
                    image = image.convert("RGB")

                np_image = np.array(image)
                np_image = cv.cvtColor(np_image, cv.COLOR_RGB2BGR)

                np_image = cv.cvtColor(np_image, cv.COLOR_BGR2GRAY)

            except:
                raise HTTPException(status_code=500, detail="Something went wrong when saving the image")
            finally:
                img_file.file.close()
                image.close()
    background_tasks.add_task(create_embedding)
    if config['CLASSIFIER']['use_model'] == "ml_model":
        background_tasks.add_task(create_classifier)
    return {
        "message": f"{username} is created.",
    }


@app.post("/recognition/")
def face_recognition(
    img_file:UploadFile =  File(...,description="Query image file"),
    to_gray: bool | None = Query(
            default=False, 
            description="Whether save image in gray scale or not"),
    return_img_file:bool = Query(default=False, description="Whether return username or image after process face recognizer"),
):
    '''
    Do Face Recognition task, give the image which is 
    the most similar with the input image from the 
    database - in this case is a folder of images

    Arguments:  
        img_file(File): image file
        return_img_file(bool): Decide whether return only image file (img) or image file with extension (img.[jpg|jpeg])
    Return:
        Return path to the most similar image file
    '''
    if len(next(os.walk(config['RECOGNIZER']['data_dir']))[1]) == 0:
        return {
            "message": "No user found in the database."
        }

    # Convert image to gray (if necessary) then save it
    image = Image.open(img_file.file)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    np_image = np.array(image)
    if to_gray:
        np_image = cv.cvtColor(np_image, cv.COLOR_BGR2GRAY)

    # Face detection - recognition
    boxes, _, landmarks = detector.detect(np_image, landmarks=True)

    if boxes is not None and len(boxes) > 0:
        # Classify faces
        for box, marks in zip(boxes, landmarks):
            box = np.array(box).astype(np.int32)

            # Crop the face in np_image
            crop_face = np_image[box[1] : box[3], box[0] : box[2]]
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
            pred = classifier.run(None, {input_name: out})
            pred_label = pred[0][0]
            pred_acc_logits = pred[1][0][pred_label]

            print(pred_label)
            print(pred_acc_logits)

            if pred_acc_logits >= config['CLASSIFIER']['threshold']: 
                return_name = pred_label
                # acc = str(round(pred[1][0][pred_label] * 100, 2)) + "%"
                pred_user = return_name
            else:
                pred_user = "unknown"

            # Draw bounding box
            if return_img_file:
                np_image = src.putBoundingBox(np_image, box.astype(np.int32), pred_user)

                # Draw landmarks
                np_image = cv.circle(np_image, center=marks[0, :].astype(np.int32), radius=5, color=(0,255,0),thickness=-1)
                np_image = cv.circle(np_image, center=marks[1, :].astype(np.int32), radius=5, color=(255,0,0),thickness=-1)
                np_image = cv.circle(np_image, center=marks[2, :].astype(np.int32), radius=5, color=(0,0,255),thickness=-1)
                np_image = cv.circle(np_image, center=marks[3, :].astype(np.int32), radius=5, color=(0,0,0),thickness=-1)
                np_image = cv.circle(np_image, center=marks[4, :].astype(np.int32), radius=5, color=(255,255,255),thickness=-1)

                np_image = cv.cvtColor(np_image, cv.COLOR_RGB2BGR)
                success, im = cv.imencode('.png', np_image)
                headers = {'Content-Disposition': 'inline; filename="test.png"'}
                return Response(im.tobytes() , headers=headers, media_type='image/png')
            else:
                return {
                    'result': pred_user,
                }


@app.put('/change-username')
def change_img_name(
    background_tasks: BackgroundTasks,
    old_name:str = Query(..., description="User name to be change"), 
    new_name:str = Query(..., description="New name")
    ):
    '''
    Change file name in database

    Arguments:
        old_name (str) Path to the source name (e.g: user1)
        new_name (str) Name to be change (e.g: user2)
    '''
    # Check old name valid
    old_id_dir = os.path.join(config['RECOGNIZER']['data_dir'], old_name)
    if not os.path.exists(old_id_dir):
        raise HTTPException(status_code=404, detail=f'{old_name} not in the database')

    # Check new name valid
    new_id_dir = os.path.join(config['RECOGNIZER']['data_dir'], new_name)
    if os.path.exists(new_id_dir):
        raise HTTPException(status_code=404, detail=f'{new_name} already in database')

    # remove representation
    os.remove(os.path.join(config['RECOGNIZER']['data_dir'], old_name, f'{old_name}_representation.pkl'))

    # change dir name
    os.rename(old_id_dir, new_id_dir)

    background_tasks.add_task(create_embedding)

    return {
        "message": f"Already change {old_name} to {new_name}"
    }


@app.delete('/del-one-user')
def del_img(name:str = Query(..., description="User name to be removed")):
    '''
    Delete single image file in database

    Arguments:
        img_path (str) Path to the image (e.g: images/img1.jpeg)
    '''
    id_dir = os.path.join(config['RECOGNIZER']['data_dir'], name)
    if not os.path.exists(id_dir):
        raise HTTPException(status_code=404, detail=f'{name} not in the database')
    
    shutil.rmtree(id_dir, ignore_errors=True)

    return {
        "message": f"{name} has been deleted!"
    }


@app.delete('/reset-db')
def del_db():
    '''
    Delete all file in database ~ Delete database
    '''
    if not os.path.exists(config['RECOGNIZER']['data_dir']):
        raise HTTPException(status_code=404, detail='No database available')
    
    for files in os.listdir(config['RECOGNIZER']['data_dir']):
        shutil.rmtree(os.path.join(config['RECOGNIZER']['data_dir'], files), ignore_errors=True)
    
    return {
        "message": "All files have been deleted!"
    }