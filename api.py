import os 
import glob
import shutil
import cv2 as cv
import shutil
from PIL import Image
import src
from make_identifier import create_classifier, create_embedding
from utils import load_data, cosine_similarity

from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query, HTTPException, File, UploadFile, BackgroundTasks

import numpy as np

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
# data = load_data(config['RECOGNIZER']['data_dir'])
# values = np.concatenate(data['embedding'])


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


@app.post('/register')
def face_register(
    background_tasks: BackgroundTasks,
    img_files: list[UploadFile] | None = File(..., description="Upload Image"),
    to_gray: bool | None = Query(
            default=True, 
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

                cv.imwrite(save_img_dir, np_image)
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
            default=True, 
            description="Whether save image in gray scale or not"),
    return_image_name:bool = Query(default=True, description="Whether return only image name or full image path"),
):

    '''
    Do Face Recognition task, give the image which is 
    the most similar with the input image from the 
    database - in this case is a folder of images

    Arguments:  
        img_file(File): image file
        return_image_name(bool): Decide whether return only image file (img) or image file with extension (img.[jpg|jpeg])
    Return:
        Return path to the most similar image file
    '''

    # Check if database is empty
    empty = check_empty_db()
    if empty:
        return "No image found in the database"

    if len(os.listdir(config.DB_PATH)) == 0:
        return {
            "message": "No image found in the database."
        }
    
    # Save query image to ./query
    if not os.path.exists("query"):
        os.makedirs("query")

    if '/' in img_file.filename:    
        query_img_path = os.path.join("query", img_file.filename.split('/')[-1])
    elif "\\" in img_file.filename:
        query_img_path = os.path.join("query", img_file.filename.split("\\")[-1])
    else:
        query_img_path = os.path.join("query", img_file.filename)

    # Convert image to gray (if necessary) then save it
    if to_gray:
        image = Image.open(img_file.file)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        np_image = np.array(image)
        np_image = cv.cvtColor(np_image, cv.COLOR_BGR2GRAY)

        cv.imwrite(query_img_path, np_image)
    else:
        with open(query_img_path, "wb") as w:
            shutil.copyfileobj(img_file.file, w)

    # Face detection - recognition
    # try:
    df = DeepFace.find(img_path=query_img_path, 
                        db_path = config.DB_PATH, 
                        model_name = config.MODELS[config.MODEL_ID], 
                        distance_metric = config.METRICS[config.METRIC_ID], 
                        detector_backend = config.DETECTORS[config.DETECTOR_ID], 
                        silent = True, align = True, prog_bar = False, enforce_detection=False)
    # except:
    #     return {
    #         'error': "Error happening when trying to detecting face or recognition"
    #     }
    
    # Remove query image
    os.remove(query_img_path)

    # If faces are detected/recognized
    if not df.empty:
        path_to_img, metric = df.columns
        ascending = True
        if config.METRIC_ID == 0:
            ascending = False
        df = df.sort_values(by=[metric], ascending=ascending)
        value_img_path = df[path_to_img].iloc[0]

        if return_image_name:
            return_value = value_img_path.split(os.path.sep)[-1]
            return_value = return_value.split(".")[0]
            return {
                "result": return_value,
            }
        else:
            return {
                "result": value_img_path,
            }
    else:
        return {
            "result": "No faces have been found"
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