# AI P6 - Computer Vision

*Members:*
- Martin Carrasco
- Jorge Fiestas

** Please check the [Github Repo](https://github.com/Jorgefiestas/IA-PY6-CV)**

## Introduction

In this proyect we implemented a live mask recognition software that works with the webcam using OpenCV and TensorFlow. For the recognition of the faces in the webcam we used a pretrained openCV classifier to detect faces. Then we trained CNN model to classify masked and unmasked faces. We run each of the faces detected through this model in order to differentiate masked and unmasked faces.


## Workflow

As in order to detect the faces in the webcam capture we use a pre-trained calssifier, we only need to train a CNN to differentiate faces with facemasks and faces without. For this reason the workflow is extremely simple:

1. Train the CNN to detect mask vs unmasked faces
2. Run the mask detection app

For every frame that the webcam captures our application first detects all the faces in the image using a pre-trained openCV classifier. Then each of this faces is passed to our CNN model to detect which have facemasks and which don't. Drawing the red or green square around the face is something that can be done very easily using openCV.

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

## Future Works

Currently the application works pretty well when dealing with a small number of faces, but introducing more faces in the app can lead to lag and slow running. This is probably due to the complexity of the CNN model, which is relatively slow generating a large overhead. A possible improvement is to find a simpler architecture or model with similar accuracy that can predict a face faster.

Also having a better dataset could help to further improve acurracy, however in order to be sure about this we would need to actually test it.

## Resources

- [Dataset](https://www.kaggle.com/omkargurav/face-mask-dataset)
- [Tensor Flow to Detect Facemasks](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)
- [Face Classification SKimage](https://scikit-image.org/docs/dev/auto_examples/applications/plot_haar_extraction_selection_classification.html)
- [Feature Extraction SKimage](https://analyticsindiamag.com/image-feature-extraction-using-scikit-image-a-hands-on-guide/)
