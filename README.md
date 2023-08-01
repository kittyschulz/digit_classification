# Classification and Detection with Convolutional Neural Networks

## Introduction

Detecting and classifying digits from natural images is an important task in computer vision. It makes it possible to extract meaningful data from images, and has applications such as identifying license plates for automated tolling or reading serial numbers on scientific samples to help maintain chain-of-custody. In the future, this task may also become important for autonomous vehicles to obtain information from their environment, such as speed limits or highway signs. 

We adopt a two stage pipeline where we first propose regions which are likely to contain digits using the feature extractor Maximally Stable Extremal Regions (MSER). Then we train a convolutional neural network (CNN) to classify each region proposal as either a digit from 0 to 9 or background.

Read the [full write-up here](https://drive.google.com/file/d/15p9nqY72T4ghOvcmj4kNbOYogzQ3MVLt/view?usp=sharing).

## Requirements

The pipeline requires the following libraries to run:

os
cv2
numpy
torch
torchvision
matplotlib

To run the demo in `run.py`, some files are also required to be saved in the local directory:
* A directory, `./weights` containing three files for model weights: `vgg.pth`, `vgg_imagenet.pth`, and `cnn.pth`. The weights can be downloaded from Gatech Box [here](https://gatech.box.com/s/4hpnvcb1uwjc6tpw4395x8543jgsej0x).
* A directory containing `./example_images` containing example images. The example images can be downloaded from GatechBox [here](https://gatech.box.com/s/24xvjqw8zuitltwesu1z5vttmf6c5l2e).

## Demos

A demo video is available [here](https://drive.google.com/file/d/1ew_krJDEt7YrclJcCV2HAWtV7qCTybWZ/view?usp=sharing).

A Jupyter Notebook, `demo.ipynb`, has also been provided in the project folder. The notebook allows the user to visualize both the region proposal and classification stage for a specified image. It also makes it easier for a user to test each of the three different classification models, the simple CNN, VGG16, and VGG16 with Imagenet. To use a different model, a string should be passed to the argument `architecture` in the `classify()` function. The function will accept 'cnn', 'vgg', and 'vgg+imagenet', with the default being the latter.

## Usage

A demo of the pipeline can be run on the five example images by executing the file `run.py`.

```
$ python3 build_model.py
```

The demo will read five images from the directory `./example_images` and save the outputs to the directory `./graded_images`. If the images do not exist in the `./example_images` directory, they can be downloaded from GatechBox [here](https://gatech.box.com/s/24xvjqw8zuitltwesu1z5vttmf6c5l2e).

The pipeline will load the model weights for VGG16 pretrained with Imagenet and fine tuned on SVHN and DTD. These model weights should be located in the directory `./weights`. If this directory or the `*.pth` files do not exist, the weights can be downloaded from Gatech Box [here](https://gatech.box.com/s/4hpnvcb1uwjc6tpw4395x8543jgsej0x).
