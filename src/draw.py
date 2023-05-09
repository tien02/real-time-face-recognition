import cv2 as cv
import numpy as np

from .loader import load_config

config = load_config()

def putFPS(
    frame: np.ndarray,
    fps: str,
    frameHeight: int,
    frameWidth: int,
    background: bool = False,
) -> np.ndarray:
    x_coor, y_coor = int(frameWidth * 0.05), int(frameHeight * 0.2)
    text = f"FPS:{fps}"
    if background:
        text_w, text_h = cv.getTextSize(
            text,
            config["FPS"]['TEXT']["font"],
            config["FPS"]['TEXT']["fontScale"],
            config["FPS"]['TEXT']["thickness"],
        )[0]
        frame = cv.rectangle(
            frame,
            (
                x_coor - config['FPS']['BACKGROUND']['padding'],
                y_coor + config['FPS']['BACKGROUND']['padding'],
            ),
            (
                x_coor + text_w + config['FPS']['BACKGROUND']['padding'],
                y_coor - text_h - config['FPS']['BACKGROUND']['padding'],
            ),
            tuple(config['FPS']['BACKGROUND']['color']),
            -1,
        )

    frame = cv.putText(
        frame,
        text,
        (x_coor, y_coor),
        config["FPS"]['TEXT']["font"],
        config["FPS"]['TEXT']["fontScale"],
        tuple(config["FPS"]['TEXT']["color"]),
        config["FPS"]['TEXT']["thickness"],
    )

    return frame


def putBoundingBox(frame:np.ndarray, box: list | np.ndarray, text:str, background:bool = True) -> np.ndarray:
    xmin, ymin, xmax, ymax = box
    frame = cv.rectangle(
                    frame,
                    (xmin, ymin),
                    (xmax, ymax),
                    tuple(config["BB"]["BOX"]["color"]),
                    config["BB"]["BOX"]["thickness"],
                )
    
    if background:
       text_w, text_h = cv.getTextSize(
            text,
            config["BB"]['TEXT']["font"],
            config["BB"]['TEXT']["fontScale"],
            config["BB"]['TEXT']["thickness"],
        )[0]
       frame = cv.rectangle(
            frame,
            (
                xmin - config["BB"]['BACKGROUND']['padding'],
                ymin + config["BB"]['BACKGROUND']['padding'],
            ),
            (
                xmin + text_w + config["BB"]['BACKGROUND']['padding'],
                ymin - text_h - config["BB"]['BACKGROUND']['padding'],
            ),
            tuple(config["BB"]['BACKGROUND']['color']),
            -1,
        ) 

    frame = cv.putText(
        frame,
        str(text),
        (xmin, ymin),
        config["BB"]["TEXT"]["font"],
        config["BB"]["TEXT"]["fontScale"],
        tuple(config["BB"]["TEXT"]["color"]),
        config["BB"]["TEXT"]["thickness"],
    )
    return frame


def putTrainingData(
        frame: np.ndarray,
        count: str,
        frameHeight: int,
        frameWidth: int,
        background: bool = False,
) -> np.ndarray:
    x_coor, y_coor = int(frameWidth * 0.05), int(frameHeight * (1 - 0.2))
    text = f"Count:{count}/{config['RECOGNIZER']['max_training_img']}"
    if background:
        text_w, text_h = cv.getTextSize(
            text,
            config["FPS"]['TEXT']["font"],
            config["FPS"]['TEXT']["fontScale"],
            config["FPS"]['TEXT']["thickness"],
        )[0]
        frame = cv.rectangle(
            frame,
            (
                x_coor - config['FPS']['BACKGROUND']['padding'],
                y_coor + config['FPS']['BACKGROUND']['padding'],
            ),
            (
                x_coor + text_w + config['FPS']['BACKGROUND']['padding'],
                y_coor - text_h - config['FPS']['BACKGROUND']['padding'],
            ),
            tuple(config['FPS']['BACKGROUND']['color']),
            -1,
        )

    frame = cv.putText(
        frame,
        text,
        (x_coor, y_coor),
        config["FPS"]['TEXT']["font"],
        config["FPS"]['TEXT']["fontScale"],
        tuple(config["FPS"]['TEXT']["color"]),
        config["FPS"]['TEXT']["thickness"],
    )

    return frame

