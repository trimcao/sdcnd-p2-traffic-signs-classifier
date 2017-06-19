## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

To meet specifications, the project will require submitting three files:
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file


The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[original]: ./fig/original_image.png "Original Image"
[augmented]: ./fig/augmented_images.png "Augmented Images"
[test_image1]: ./test/originals/images-01.jpeg "Traffic Sign 1"
[test_image2]: ./test/originals/images-03.jpeg "Traffic Sign 2"
[test_image3]: ./test/originals/images-05.jpeg "Traffic Sign 3"
[test_image4]: ./test/originals/images-07.jpg "Traffic Sign 4"
[test_image5]: ./test/originals/images-11.jpeg "Traffic Sign 5"
[test_image6]: ./test/originals/images-12.jpg "Traffic Sign 6"
[softmax_1]: ./test/softmax/image-1-softmax.png "Softmax 1"
[softmax_2]: ./test/softmax/softmax_2.png "Softmax 2"
[softmax_3]: ./test/softmax/softmax_3.png "Softmax 3"
[softmax_4]: ./test/softmax/softmax_4.png "Softmax 4"
[softmax_5]: ./test/softmax/softmax_5.png "Softmax 5"
[softmax_6]: ./test/softmax/softmax_6.png "Softmax 6"
[feature_map]: ./fig/feature_map.png "Feature Maps, First Conv Layer"
[sample_hist]: ./fig/sample_hist.png "Sample Histogram"
[precision_test]: ./fig/precision_test_set.png "Precision Test Set"
[recall_test]: ./fig/recall_test_set.png "Recall Test Set"

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used simple numpy functions to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is a bar chart showing the number of samples for different traffic signs.

![alt_text][sample_hist]

It is clear that many sign types do not have many samples. This shows a need for augmenting images.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique.

Sermanet and LeCun (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) shows that using grayscale images increases the accuracy of the network. Hence, the first preprocessing step is to converting RGB-colorspace images to YUV-colorspace, then use only the Y channel.
Y-channel value is a linear combination of RGB values. The equation is:
![equation](https://latex.codecogs.com/svg.latex?\text{Y-channel}&space;=&space;0.299&space;\times&space;R&space;&plus;&space;0.587&space;\times&space;G&space;&plus;&space;0.114&space;\times&space;B)

The second step is to normalize the pixel values so we have zero mean and unit variance. For each sample set (training, validation, test), I find the mean and variance among all pixel values, then normalize each pixel value:
![equation](https://latex.codecogs.com/svg.latex?\text{pixel}&space;=&space;\frac{\text{pixel}&space;-&space;mean}{stddev})

Here is an example of an original traffic sign image in color.

![alt text][original]

And the following are the image in grayscale and augmented images.

![alt text][augmented]

Probably the most important preprocessing steps is to generate additional data, or in other words, augment the images. There are many possible ways to augment the images. I suspect the more different ways to augment images the better. In this project, I use three methods of image augmentation: translate, rotate and scale (zoom).

For each image I generate three new images, one with pixels translated (perturbed) in the range [-2, 2] pixels, one with pixels rotated in range [-15, 15] degrees, and one with pixels scaled in range [0.9, 1.1] times.

After doing image augmentation, my training set has 139196 samples. It is important to note that if I want to improve my model further, the first step would be generating more images. The possible augmentation methods include: brightness changing, shearing, adding random noise, etc.

Normalization can also be improved with brightness normalization and other fancy methods.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image						|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x64 	|
| RELU					|												|
| Dropout               | 0.5                                           |
| Max pooling           | 2x2 stride,  outputs 15x15x64                 |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 12x12x128 	|
| RELU					|												|
| Dropout               | 0.5                                           |
| Max pooling	      	| 2x2 stride,  outputs 6x6x128  				|
| Flatten               | outputs 4608                                  |
| Fully connected		| outputs 1024        							|
| RELU					|												|
| Dropout               | 0.5                                           |
| Fully connected		| outputs 1024        							|
| RELU					|            									|
| Dropout               | 0.5                                           |
| Softmax				| outputs 43  									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* The optimizer used is Adam Optimizer.
* Learning rate is 0.001.
* Dropout probability is 0.5 for each layer.
* L2 Regularizer is 10e-4.
* Number of epochs is 20.
* Batch size is 128.

Other than the dropout probability, I have not tested other hyperparameters. In addition to augment more images, playing with the hyperparameters is important to improve the performance of the model. In the current state, the model is still overfitting.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.983
* test set accuracy of 0.974

An iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    * I used the LeNet-5 as the first architecture because it performed well on MNIST data set, so I wanted to see how it worked with the Traffic Sign dataset. As mentioned in one of the videos by Udacity, LeNet-5 did not performed badly. With augmented images and dropout, it got validation set accuracy of 0.94, comfortably passing the requirement threshold of the project.
* What were some problems with the initial architecture?
    * The main problem is overfitting. After 30 epochs the training set accuracy reached 0.999, but the validation set accuracy was only 0.94. This problem leads to addtional tasks: generating ugmented images, using more aggressive dropout, and adding L2 regularization.
* How was the architecture adjusted and why was it adjusted?
    * I realized LeNet-5 was too simple for the Traffic Sign dataset. LeNet-5 was created for a simpler problem similar to MNIST, and the number of hidden variables was too small. Although a bit uncertain, I thought the small number of nodes in the neural network was another factor leading to overfitting. After studying two architectures (http://florianmuellerklein.github.io/cnn_streetview/, and https://chatbotslife.com/german-sign-classification-using-deep-learning-neural-networks-98-8-solution-d05656bf51ad), I decided to increase the number of depth of each convolution layer, and increase the number of nodes in each fully-connected layer.
    * Dropout and L2 regularization were added because they could reduce overfitting.
* Which parameters were tuned? How were they adjusted and why?
    * Different dropout probabilities were tested, and I found 0.5 for every layer worked well.
    * I have not tested different values for L2 regularizer. Perhaps increasing that value would further reduce overfitting.
    * Learning rate was still set at 0.001, the default value for Adam Optimizer. It seemed that Adam Optimizer did not have any serious problem with the default learning rate. But I would investigate the learning rate more in the future.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    * Convolution layer are needed because they extract additional features used to classify the images. If we only use fully-connected layers then the only feature we use is the value of each pixel.
    * Dropout helps because intuitively, each node in the network has some information related to classification. That means we can avoid the situation where some node has all the information while another has none. Then when we extract the features for prediction (in the end), the nodes work like an ensemble, and overfitting is reduced.

#### Precision and Recall for the Test set.

![alt text][precision_test]

![alt text][recall_test]

The sign types with low precision or low recall are the ones with less samples in the training set.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I chose these six traffic signs to compose the test set.

![alt text][test_image1] ![alt text][test_image2] ![alt text][test_image3]
![alt text][test_image4] ![alt text][test_image5] ![alt text][test_image6]

The first sign (road work) might be difficult to classify because there is another white/red sign behind it that can confuse the classifier.

The fifth sign (priority road) might be a challenge because of the brightness. The colors in the image are not clear.

Other signs should be straightfordward to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Road Work     		| Beware of ice/snow							|
| No Entry     			| No Entry 										|
| Right-of-way			| Right-of-way									|
| No Passing      		| No Passing					 				|
| Priority road			| Priority road      							|
| Pedestrians   		| Pedestrians          							|


The accuracy is 5/6 = 0.833. Since I only have 6 images, the accuracy is not statistically significant.

It can be said that the model is overfitting because the old test set accuracy is 0.974. It should have been difficult to find a traffic sign that is not correctly classified.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The first and last images seem challenging to the model. For the first image, at least the model has the correct sign (road work) with the third best softmax probability.
For the last image, although making the correct prediction, the model does not appear to be very confident.

One explanation for this performance is that these signs, Road Work and Pedestrians, are among the signs that do not have a lot of samples. Since it is not easy to obtain more real samples of these signs, I can generate more augmented images.


1st image:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .082         			| Beware of ice/snow   							|
| .074     				| Roundabout mandatory 							|
| .071					| Road work										|
| .058	      			| Right-of-way					 				|
| .051				    | Children crossing      						|

![alt text][softmax_1]


2nd image:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .926         			| No entry   									|
| .024    				| No passing 									|
| .007					| Yield											|
| .005	      			| Vehicles over 3.5 metric tons prohibited		|
| .004				    | Turn right ahead      						|

![alt text][softmax_2]

3rd image:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .959         			| Right-of-way   								|
| .016     				| Beware of ice/snow							|
| .003					| Dangerous curve to the right				   |
| .003	      			| Pedestrians					 				|
| .002				    | Children crossing      						|

![alt text][softmax_3]

4th image:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .871         			| No passing   									|
| .041     				| Yield 										|
| .03					| End of no passing							    |
| .024	      			| Vehicles over 3.5 metric tons prohibited		|
| .007				    | No vehicles      					      		|

![alt text][softmax_4]

5th image:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .838         			| Priority road   								|
| .063     				| Roundabout mandatory 							|
| .013					| Turn left ahead								|
| .01	      			| Vehicles over 3.5 metric tons prohibited		|
| .009				    | No entry     						         	|

![alt text][softmax_5]

6th image:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .045         			| Pedestrians   								|
| .04     				| Right-of-way 									|
| .04					| Double curve									|
| .036	      			| Bicycles crossing				 				|
| .036				    | Go straight or right        					|

![alt text][softmax_6]

### Visualizing the Neural Network

![alt text][feature_map]

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

We can see that every feature map is not blank, and that is usually a good thing. I think it means no neuron dies (i.e. has blank feature map) while training.

The characteristics that the neural network use are:

* The overall shape of the sign (round shape)
* The shapes of the things (two cars) in the center.
* The edge of the sign.
* The outer ring of the sign (the red ring in the original image).
* The difference between the outer ring and the center of the sign.

Notice that there are multiple neurons that learn the same feature. I think that is the result of dropout.
