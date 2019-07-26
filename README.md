# Text2VideoGAN
A pytorch implementation of a text to videos GAN that makes use of MoCoGAN to create new text, of Caffe library with pretrained S2VT to get the description of videos and of UCF-101 Dataset to train models. An LSTM model is trained on the results of S2VT to classificate user input to classes of UCF_101

## 1 - Text2VideoGAN
In this notebook the results can be seen. Here it is explained how to use the project and how to generate videos.

## 1.1 - Exploratory Data Analysis
Here the transformations for videos in UCF_101 are shown with results. The statistics of the dataset are computed in the last cell.

## 2 - Text To Class
Here the process of loading the LSTM model pretrained and the usage of it to classificate NL are shown.

## Usage
To use notebooks, mocogan you can import the environment called `mocogan.yml` found into the directory called `Environments`.
To do so, simply use command `conda env create -f Environments/mocogan.yml`.

Same thing apply if you want to use the s2vt captioner: `conda env create -f Environments/s2vt_captioner.yml`.