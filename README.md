# Text2VideoGAN
A pytorch implementation of a text to videos GAN that makes use of MoCoGAN to create new text, of Caffe library with pretrained S2VT to get the description of videos and of UCF-101 Dataset to train models.

## Build
In this section, the passages to use the developed model will be shown.

### Dependencies
This project uses Anaconda to create an isolated environment for executing the model.

### Build
1- The first step is to run `conda env create -f environment.yml`. This will create a new environment and will install all of the dependencies needed to execute the model in a fresh environment.
2- If you have to install other dependencies, and you want to publish then the new project, it is recommended to use `conda env export > environment.yml` command and then to make the final user repeat the first command to install a new environment.
