# Face Matching Project

## Details
For advanced details, read [this file](Face_Matching_Project.pdf)

## Overview
This repository contains a data science project focused on face recognition. The primary objective is to develop a system that can identify all the images in a database where a specific person appears, given a reference face image of that individual.

For example, consider a company that hosts an event and captures photographs during the occasion. The goal is to enable the company to efficiently identify and provide the relevant images to each employee, under the assumption that the company has access to the employees' phone numbers and face images.

To achieve this, the project leverages the following key components:

#### DeepFace Library:
The system utilizes the [DeepFace](https://github.com/serengil/deepface)
 library, a state-of-the-art deep learning-based face recognition framework, to perform the core face detection and recognition tasks.

#### Ensemble Learning of Face Recognition Models: 
The project employs an ensemble learning approach, combining multiple face recognition models to improve the overall accuracy and robustness of the face identification process.

#### Machine Learning Techniques:
The system incorporates advanced machine learning techniques, such as XGBoost

## Installation
To set up the project environment, follow these steps:

 1. **Install deepface**: 
    ```sh
    pip install deepface
    ```

2. **Clone yolo-v8 repo**:
    ```sh
    git clone https://github.com/akanametov/yolov8-face
    pip install ultralytics
    ```

3. **Install Dependencies**:
   ```sh
   pip install scipy pandas scikit-learn
   pip install xgboost
    ```

## Usage

Here are the steps to follow if you want to train your own XGBoost model:

1. Ensure that you have a folder containing images from the event in the main directory.
2. Open the terminal and navigate to the "yolov8-face" directory. Then, run the following command:
   ```bash
   yolo task=detect mode=predict model=yolov8m-face.pt conf=0.85 imgsz=1280 line_width=1 max_det=1000 source=event_images save_crop=True
3. Use the "create_dataset.py" script to generate the dataset.
4. Finally, run the "main.py" script to train your XGboost model.

To ensure that your dataset is accurate, it is recommended that you divide the "create_dataset.py" script into two parts: person clustering and CSV file creation. After the first part is completed, you may need to manually correct any minor errors in the clustered dataset.

## Credits
Acknowledgment is due to the [DeepFace](https://github.com/serengil/deepface) library, upon which this code is built, as well as [YOLO-V8-face](https://github.com/akanametov/yolov8-face).
