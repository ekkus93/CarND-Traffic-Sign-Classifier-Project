---
output:
  html_document: default
  pdf_document: default
---
# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**


[//]: # (Image References)

[image1]: ./writeup_images/sample_train_images.png "Sample Training Images"
[image2]: ./writeup_images/sample_train_gray_images.png "Sample Training Images (Grayscale)"
[image3]: ./writeup_images/number_of_traffic_signs_per_label.png "Number of Traffic Signs per Label"
[image4]: ./writeup_images/sample_train_rotate_images.png "Sample Training Images (Rotate)"
[image5]: ./writeup_images/sample_train_shift_images.png "Sample Training Images (Shift)"

[image6]: ./writeup_images/sample_train_shear_images.png "Sample Training Images (Shear)"
[image7]: ./writeup_images/extra_traffic_signs.png "Extra Traffic Signs"
[image8]: ./writeup_images/ensemble_individual_accuracies.png 

[image9]: ./writeup_images/extra_30.png
[image10]: ./writeup_images/extra_30_activations.png

[image11]: ./writeup_images/extra_bicycle.png
[image12]: ./writeup_images/simple_model_accuracies.png
[image13]: ./writeup_images/voting_model_accuracies.png
[image14]: ./writeup_images/msl_model_accuracies.png
[image15]: ./writeup_images/wiki_right_of_way.png

[image17]: ./writeup_images/top_softmax_bicycle.png
[image18]: ./writeup_images/extra_peds.png
[image19]: ./writeup_images/extra_peds_activations.png
[image20]: ./writeup_images/extra_roundabout.png
[image21]: ./writeup_images/extra_roundabout_activations.png
[image22]: ./writeup_images/extra_no_entry.png
[image23]: ./writeup_images/extra_no_entry_activations.png

---
### Introduction

The purpose of this project is to build a model to classify German traffic signs. The model is a convolutional neural network based on LeNet.  The image data will be augmented and an additional enhanced grayscale layer used as the input.  This will produce an acceptable model but we will use an ensemble model to improve the accuracy even further.  The ensemble model will be an easy way to improve the accuracy without having to make a lot of structural changes to the model or modifications to the data.

Here is a link to my [project code](https://github.com/ekkus93/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

#### Basic summary of the data set. 

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is a sample of the traffic light images from the training dataset. 

![][image1]


Here is a breakdown of the number of images per label from the training dataset.

![][image3]

The images are imbalanced. The traffic sign with least number of examples is "Go straight or left" (37) with 180 images.
The traffic sign with most number of examples is "Speed limit (50km/h)" (2) with 2010 images.

### Design and Test a Model Architecture

#### Preprocessing the Image Data

##### Balancing Images

The images will be augmented so each label will end up having the same number of images as the label with the most images (2010).  The main purpose of balancing the data is to not bias the prediction towards more common traffic signs.  We will also need the extra data for the ensemble model.

If a label has less than 2010 images, an images for that label will be picked at random.  The random image will have one of the following transformations applied to it:

1. Rotation 

The range for the random rotation between -20.0 and 20.0 degrees. Some of the traffic signs such as "Turn right ahead" and "Turn left ahead" depend on direction so we will limit the rotation to that range.

![][image4]

2. Shift 

The images can be shifted +/-5 up or down and left or right.

![][image5]

3. Shear

The images can be sheared from -0.2 to 2.0 radians.

![][image6]

##### Grayscale layer

In addition to the RBG color layers, an additional grayscale layer was added to each of the images.  The grayscale layer was sharpened using histogram equalization. The color infotmation is important for distinguishing different traffic signs. The sharpened grayscale layer is important for different signs with text such as speed limit signs.

![][image2]

#### 2. The Model
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------|:---------------------------------------------| 
| Input         		| 32x32x4 RGBL image   							|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x16 	|
| Leaky ReLU					|	alpha = 0.00001											|
| Max pooling	      	| 2x2 stride,  outputs 16x16x16 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x32 	|
| Leaky ReLU					|	alpha = 0.00001											|
| Max pooling	      	| 2x2 stride,  outputs 8x8x32 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 8x8x64 	|
| Leaky ReLU					|	alpha = 0.00001											|
| Max pooling	      	| 2x2 stride,  outputs 4x4x64 				|
| Flattened  | Output = 1024  |
| Fully connected		| Output = 512        									|
| Leaky ReLU					|	alpha = 0.00001											|
| Dropout  | keep_prob = 0.5  |
| Fully connected		| Output = 256        									|
| Leaky ReLU					|	alpha = 0.00001											|
| Dropout  | keep_prob = 0.5  |
| Fully connected		| Output = 43        									|
 
There are 3 convolutional/max pooling layers.  Leaky ReLU's with an alpha of 1e-05 were used instead of regular ReLU's to try to avoid the “dying ReLU” problem.  Dropout was added to the fully connected layers to help prevent overfitting.

#### 3. Training
##### Simple Model
The model was trained for 100 epochs with a batch size of 125.  All of the training data was used.  An Adam Optimizer with a learning rate of 1e-3 was used.  In additional to this, L2 regularization was added with a beta of 1e-4 to help prevent overfitting.  Out of the 100 epochs, the model with the best validation accuracy was saved as the final model.

##### Ensemble Model
For the ensemble model, the simple model was used with the same training parameters but with 10 versions of that model.  The only change was the training data.  90% of the training data was randomly chosen to train each model.  The models vote on the best answer.  Majority wins for the predicted label.

The reason for using slightly different datasets is create randomly different models.  One model might be better at predicting certain types of labels compared to other models.  The models should have some overlapping specializations.  Voting should help these overlapping models predict the correct label.  Multiple models also helps prevent overfitting.

#### 4. Solution and Validation

##### Single Model

* training set accuracy of 1.000
* validation set accuracy of 0.985 
* test set accuracy of 0.966

![] [image12]

Initially, I tried using only RGB color layers for the input.  The test accuracy was a little below 93% for 30 epochs.  I tried training longer with lower training rates.  The model would overfit but the test validation would still stay below 93%.  Dropout and L2 regulization was added to deal with the overfitting.  The test validation accuracy increased over 93% but only slightly. From evaluating the misclassified traffic signs in the validation set, I could see that it was having the most trouble with different speed limit signs. The numbers were blurry. To deal with this problem, I added the histogram equalized grayscale layer.  This increased the test accuracy to 96.6%. 

There is a little bit of overfitting with the Train Accuracy being at 100%.  The Validation Accuracy is slightly lower than the Train Accuracy and the Test Accuracy is slightly lower than the Validation Accuracy.

##### Ensemble Model

A test accuracy of 96.6% was good but I thought that it could be better.  An ensemble model seemed like an easy way to improve the accuracy without doing a lot of changes to the model.  Only a slight modification was needed to be done with dividing up the training data.  10 models were used, each training on a random 90% of the training data.  The main purpose of this was to make sure that models would be different.  Hopefully, some models would be better at predicting certain types of traffic signs and the majority of the models would pick the best label.  This should help improve the accuracy. The disadvantage of this is that the training time will increase by 9 times. Predictions will take about 10 times as long compared to using a single model.

Here are the accuracies for all of the 10 models:

![] [image8]

They are similar to that of the single model.  They are also similar to each other.  The validation accuracies range from 97.3-98.5%.  The test accuracies range from 95.7-96.8%.

###### Voting vs. Mean Softmax Logits

I tried two different ways of ensembling the individual models: a Voting model and a Mean Softmax Logits model.  By a Voting model, each model gets one vote with its best prediction.  For the Mean Softmax Logits model, the softmax logits are calculated for each of the models then mean of these softmax logits.  Best label is the prediction.

| Data Type	| Voting Accuracy | Mean Softmax Logits Accuracy |
|:----------|----------------:|-----------------------------:| 
| train	    | 1.000           | 1.000                        |
| valid	    | 0.993           | 0.992                        |
| test	    | 0.984           | 0.984                        |

![] [image13] 

![] [image14]

Both models were pretty similar with a validation accuracy of 99.2-99.3% and a test accuracy of 98.4%.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.993
* test set accuracy of 0.984

![] [image13]

### Test a Model on New Images

Here are ten German traffic signs that I found on the web:

![][image7]

As part of the preprocessing, I manually cropped all of the images.  

Here are the results of the prediction:

#### Voting Model

| Image			        |     Prediction	        					| 
|:---------------------|:---------------------------------------------| 
| (1) Speed limit (30km/h) |	(1) Speed limit (30km/h) |
| (27) Pedestrians |	(27) Pedestrians |
| (28) Children crossing |	(28) Children crossing |
| (40) Roundabout mandatory	| (40) Roundabout mandatory |
| (35) Ahead only	| (35) Ahead only |
| (23) Slippery road |	(23) Slippery road |
| (17) No entry	| (17) No entry |
| (24) Road narrows on the right | (24) Road narrows on the right |
| (29) Bicycles crossing |	(28) Children crossing |
| (18) General caution |	(18) General caution |

#### Mean Softmax Logits Model

| Image			        |     Prediction	        					| 
|:---------------------|:---------------------------------------------| 
| (1) Speed limit (30km/h) |	(1) Speed limit (30km/h) |
| (27) Pedestrians |	(27) Pedestrians |
| (28) Children crossing |	(28) Children crossing |
| (40) Roundabout mandatory	| (40) Roundabout mandatory |
| (35) Ahead only	| (35) Ahead only |
| (23) Slippery road |	(23) Slippery road |
| (17) No entry	| (17) No entry |
| (24) Road narrows on the right | (24) Road narrows on the right |
| (29) Bicycles crossing |	(28) Children crossing |
| (18) General caution |	(18) General caution |

The both models was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of 90%. The one traffic sign which was mislabeled was "(29) Bicycles crossing".

#### Mislabeled "Bicycles crossing" Traffic Sign

The extra "Bicycles crossing" image looks slightly different from the other ones training data.  It has a person riding the bicycle instead of just the bicycle by itself.  The sign is also rotated slightly clockwise.  I was curious if the model would still be able to recognize the bicycle and make the connection that it was a "Bicycles crossing" traffic sign.  

Both models mislabeled the sign as a "Children crossing" traffic sign.

Here is a comparison of the mislabeled "Bicycles crossing" image with sample "Bicycles crossing" and "Children crossing" images from the validation data: 

![] [image11]

The sample images for "Children crossing" don't really look that similar to the "Bicycles crossing" image.  Besides both having some black figures in the middle of a red traffic sign, they aren't that similar.  Maybe looking at the Top Softmax Probabilities might shed some light on to why this sign was mislabeled.

### Top Softmax Probabilities

For the mislabeled "Pedestrian" traffic sign, here are the top 5 Softmax Probabilities for each of the models:

![] [image16]


Here are the top 5 Softmax Probabilities for each of the models for the "Bicycles crossing" traffic sign:

![] [image17]

6 of out 10 of models did have "Pedestrian" as one of the top 5 labels but the probabilities weren't large enough to make it the best prediction.  

### Visualize the Neural Network's State with Test Images

Here are some examples visualizatize activation layers.  Below are the first activation layer of the single model using different traffic signs.  They were chosen for their different shapes and colors.

#### Speed limit (30km/h)

![][image9]

![][image10]

******

#### Pedestrians

![][image18]

![][image19]

******

#### Roundabout mandatory

![][image20]

![][image21]

******

#### No entry

![][image22]

![][image23]

Some generalizations can be made about some of the feature maps.  Feature Map 9 seems to be activate for the color red.  The red triangular and circular shapes of 3 out of 4 traffic signs light up while it is pretty much dark for the blue "Roundabout mandatory" sign.  Feature Map 12 is for blue.  The blues in the "Roundabout mandatory" sign light up for that feature map as well as part of the background sky for the "Pedestrians" sign.