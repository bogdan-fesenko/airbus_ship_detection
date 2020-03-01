# Airbus ship detection


The goal of the task is to create an algorithm for segmentation of ships in satellite images. Using this algorithm, for example, it is possible to detect contraband by comparing recognized ships with the online base of GPS coordinates of registered ships. As, for example, in medicine domain, the minimization of the False Negative is also more important in this task. It's easier to recheck every positive prediction for correctness instead of missing any of it. Accordingly, F-score over IoU thresholds with β=2 metric was selected for evaluation (weights recall higher). 


## Solution

Since predicting at least one pixel as a ship in an image without ships, we get a score of 0 for this picture, our algorithm must accurately understand if there is at least one ship in the picture. To do this, we will train a classifier that will predict the presence of at least one ship in the picture. This will reduce the False Negative errors and also save the time of the prediction, since we will not use segmentation network on pictures that are marked by the classifier as 'non-ship' images. Unet architecture was chosen for the segmentation network. Resnet34 for the classifier and Unet34 for the segmentation network were chosen. Accordingly, we can use our trained classifier network as pretrained encoder part for the Unet network instead of using Imagenet weights. Unet was first trained on only ship images (choosing only crops with ships) for faster training. Then dataset was balanced as approximitely 0.75 images w/ ships and 0.25 w/o ships for further training. 

**Submission:**
|Private score| Public score |
|--|--|
| 0.82353 |  0.68276|

This was achieved with training segmentation network for only ~400k images at total and having classifier with 97.5% accuracy only (256x256 resized prediction)

### Prerequisites
For training classifier Resnet34 network from [https://github.com/qubvel/classification_models](https://github.com/qubvel/classification_models) (need to installed for cls. training) repo was used and [https://github.com/qubvel/segmentation_models](https://github.com/qubvel/segmentation_models) fot the Unet34 segmentation network. 
For training the classifier you need to install 'classification_models' repo (described above). On the contrary, for training the segmentation model, the code for building the network is in the notebook.



### Installing


Requirements.txt file was created manually and only for prediction notebook.
```
pip install -r requirements.txt
```
### Trained models

Download trained models
Classification: https://drive.google.com/open?id=1yosonQpB8IQ7SfdX8_V7n8vOgYZ_JqtK . Trained for few epochs on 256 resized images, 97.5% accuracy achieved on default unbalanced dataset.
Segmentation: https://drive.google.com/open?id=1PSgK4hySLrL7qILVqzwW1IKSDKPneuCf . Trained on 256x256 crops for few hours (1-2 epoch total because very low I/O ops. on the Kaggle Kernels) and finetuned on 384x384 crops, then for less than epoch on the fullsize 768x768 images. 


## Running the tests

Tehre are ShipSegmentor class in the 'Predict.ipynb' notebook that can be used both for prediction and visualization. For prediction create instance of this class with specified paths as in the example provided in the notebook. Then run code below for creating submission file. For this you also must download the trained models and specify paths for them. File has_ship_or_not already has classifier prediction so you can use it instead of classification from scratch. 

    ship_segmentor.predict_ship_masks_to_csv()

### Visualisation

Visualize both train and test images prediction using ShipSegmentor class. 

```
ship_segmentator.visualize_ship_segmentation(n_images=16, is_train=False, shuffle=True)
```

