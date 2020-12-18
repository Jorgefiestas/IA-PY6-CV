# AI P6 - Computer Vision

*Members:*
- Martin Carrasco
- Jorge Fiestas

## Introduction

In this proyect we implemented a live mask recognition software that works with the webcam using OpenCV and TensorFlow. For the recognition of the faces in the webcam we used a pretrained openCV classifier to detect faces. Then we trained CNN model to classify masked and unmasked faces. We run each of the faces detected through this model.

In order to train the model we used the following Kaggle dataset: https://www.kaggle.com/omkargurav/face-mask-dataset

## Workflow

As in order to detect the faces in the webcam capture we use a pre-trained calssifier, we only need to train a CNN to differentiate faces with facemasks and faces without. For this reason the workflow is extremely simple:

1. Train the CNN
2. Run the mask detection app

## Running the application:

**Dependencies:**

- OpenCV2
- pythonCV
- TensorFlow
- Keras
- Numpy
- Pandas
- Seaborn
- Matplotlib

**Training the model:**

In order to train the model we need to run the following command in the `src` folder:

```bash
python3 model.py
```

**Executing the webcam mask detection:**

In order to run the application we only need to run the command in the `src` folder:

```bash
python3 main.py
```

