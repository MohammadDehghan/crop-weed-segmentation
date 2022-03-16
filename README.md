# Plant segmentation system for crop/weed discrimination in image processing

## Abstract
In recent years, one of the most significant issues in agriculture is to increase the use of agricultural land. Therefore, the removal of weeds from agricultural land, which is a difficult and costly task, plays an important role in increasing the use of agricultural land. Automatic weed and crop discrimination is a topic we will address. To do this we will train a neural network that will receive a patch of an image as input and its output will be semantic segmentation of the input image. The database we use is a collection of photos taken from a carrot farm taken by a robot. Lastly, we evaluate the performance of the trained neural network, which has an average accuracy of approximately 79%.

## Dataset

This article describes the production of this database, and based on that, a complete database in this field can be downloaded from this github page. This database contains 60 images including 162 product images and 332 weed images. For each image, there is a binary image that provides the user with the result of the Mask Vegetation of the plants from the ground.

## Extracted Features Description

Table 6 gives a description of all the extracted features.

<img src="https://github.com/Dehghan99/crop-weed-segmentation/blob/main/figures/features%20description.png" alt="drawing" width="800"/>

## Feature Selection

In order to further improve the results, I used different Evolutionary algorithms like GA, PSO, and SA to select the most important features. Each FSS folder contains the code for feature selection. By feature selection, average accuracy increases from 79 to 83.7, while other metrics such as F1-score, Precision, and Recall improve as well.
