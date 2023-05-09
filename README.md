# Real time face recognition

Detect and recognize face in realtime.

**TO DO:**


* Due to poor classifier performance, intend to experiment classifier with Deep Learning, Image Augmentation

* Add `face-alignment` before `arc-face` feature extractor

* Multiprocessing improve speed


* Demo web-base application

* Containerize with Docker

# Demo

![](./assets/demo.png)
![](assets/demo2.png)

# Run code

Expect `Python >= 3.10`

1. Install dependencies
```
pip install -r requirements.txt
```

2. Prepare data

There are two options

* Manually creating `facebank` data.

* Get faces from webcam by running:

```
python create_data.py
```

Add `unknown` classes if necessary.

3. Create `face embedding`

Create face embedding for each image in the dataset. **Expect 1 face each images**. Embedding from `ArcFace` is a vector `(1,512)`.
```
python create_embedding.py
```

4. Train a Classifier

Create classifier for face recognition, using machine learning algorithm. See detail in the file:

```
python create_classifier.py
```

5. Inference

Run the following command for details information:

```
python main.py --help
```

* `-t` tag for task, `img` for inference on image, `vid` for inference on video.

* `-p` path to the image file.

* `-s` save the image/video after processing.