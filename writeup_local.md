# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/sample_train_images.png "Sample Training Images"
[image2]: ./writeup_images/sample_train_gray_images.png "Sample Training Images (Grayscale)"
[image3]: ./writeup_images/number_of_traffic_signs_per_label.png "Number of Traffic Signs per Label"
[image4]: ./writeup_images/sample_train_rotate_images.png "Sample Training Images (Rotate)"
[image5]: ./writeup_images/sample_train_shift_images.png "Sample Training Images (Shift)"
[image6]: ./writeup_images/sample_train_shear_images.png "Sample Training Images (Shear)"

---
### Writeup / README

The purpose of this project is to build a model to classify German traffic signs. The model is a convolutional neural network based on LeNet.  The image data will be augmented and an additional enhanced grayscale layer used as the input.  This will produce an acceptable model but we will use an ensemble model to improve the accuracy even further.

Here is a link to my [project code](https://github.com/ekkus93/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

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

The range for the random rotation between -12.5 and 12.5 degrees. Some of the traffic signs such as "Turn right ahead" and "Turn left ahead" depend on direction so we will limit the rotation to that range.
![][image4]

2. Shift 

The images can be shifted up or down by up to 3 pixels and left or right by up to 3 pixels.
![][image5]

3. Shear

The images can be sheared from -0.2 to 2.0 radians.
![][image6]

##### Grayscale layer

In addition to the RBG color layers, an additional grayscale layer was added to each of the images.  The grayscale layer was sharpened using histogram equalization. The color infotmation is important for distinguishing different traffic signs. The sharpened grayscale layer is important for different signs with text such as speed limit signs.
![][image2]

##### Normalization

For all of the images, each of the 4 layers are normalized around 0.0.

#### 2. The Model
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
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
 
There are 3 convolutional/max pooling layers.  Leaky ReLU's were used instead of regular ReLU's to try to avoid the “dying ReLU” problem.  Dropout was added to the fully connected layers to help prevent overfitting.

#### 3. Training
##### Simple Model
The model was trained for 60 epochs with a batch size of 125.  An Adam Optimizer with a learning rate of 0.001 was used.  In additional to this, L2 regularization was added with a beta of 0.0001 to help prevent overfitting.  For the 30 epochs, the model with the best validation accuracy was saved as the final model.

##### Ensemble Model
For the ensemble model, the simple model was used with the same training parameters but with 10 versions of that model.  The only change was the training data.  90% of the training data was randomly chosen to train each model.  The models vote on the best answer.  The most voted label for an image is the predicted label.

The reason for using slightly different datasets is create randomly different models.  One model might be better at predicting certain types of labels compared to other models.  The models should have some overlapping specializations.  Voting should help these overlapping models predict the correct label.  Multiple models also helps prevent overfitting.


#### 4. Solution and Validation

##### Simple Model

* training set accuracy of 1.000
* validation set accuracy of 0.981
* test set accuracy of 0.968

Initially, I tried using only RGB color layers for the input.  The test accuracy was a little below 93% for 30 epochs.  I tried training longer with lower training rates.  The model would overfit but the test validation would still stay below 93%.  Dropout and L2 regulization was added to deal with the overfitting.  The test validation accuracy increased over 93% but only slightly. From evaluating the misclassified traffic signs in the validation set, I could see that it was having the most trouble with different speed limit signs. The numbers were blurry. To deal with this problem, I added the histogram equalized grayscale layer.  This increased the test accuracy to around 97%. 

##### Ensemble Model

An accuracy of 97% was good but I thought that it could be better.  An ensemble model seemed like an easy way to improve the accuracy without doing a lot of changes to the model.  Only a slight modification was needed to be done to the training data.  The individual models all had around a 95-96% test accuracy but when ensembled together, the test accuracy increased to around 98%.

| Model | Train Accuracy | Validation Accuracy | Test Accuracy |
|:------:|:---------------------------------------------:| 
| 0 | 0.999
| 1 | 0.999
| 2 | 1.000
| 3 | 1.000
| 4 | 1.000
| 5 | 1.000
| 6 | 0.999
| 7 | 0.999
| 8 | 1.000
| 9 | 0.999

My final model results were:
* training set accuracy of 0.98
* validation set accuracy of ? 
* test set accuracy of ?

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


