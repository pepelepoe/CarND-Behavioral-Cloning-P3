**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./pics/center_2018_01_07_17_26_55_840.jpg "Model Center track 1"
[image2]: ./pics/right_2018_01_07_18_15_58_837.jpg "Model Rigth reverse track 1"
[image3]: ./pics/right_2018_01_07_18_35_23_451.jpg "Model Center track 2"
[image4]: ./pics/right_2018_01_07_18_45_06_105.jpg "Model Right track 2"
[image5]: ./pics/0.jpg "Line touching 1"
[image6]: ./pics/1.jpg "Line touching 2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

For this project I used the NVIDIA Architecture to train my model. After normalizing and cropping images, I start with a
5x5 convolutional layer with 24 outputs. Followed by two other 5x5 convolutional layers, the first with 36 output filters
and the latter with 48. These first three layers contain 2x2 strides and relu activations. The following two convolutional layers are both 3x3 with 64 output filters and relu actions. I then flatten the model and four fully connected layers wituh 100, 50, 10 and 1 outputs. I then used a mean-squared-error loss function with an Adam optimizer to minimize and split the data 20% validation and use two epochs.

#### 2. Attempts to reduce overfitting in the model

The model contains normalization and cropping layers in order to reduce overfitting (model.py lines 42-43).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 69). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 68).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving in both tracks, driving clockwise and counter-clockwise on both.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

A driving_log.csv file was created after simulator is manually run by user and frames are recorded. Each run of simulator updates this file with more data. The images recorded (center, left and right) are used as the featureset while the steering measurements are used as the labelset. This featureset is used to train the network and predict the steering measurements.

The overall strategy for deriving a model architecture was to experiment with the different models provided on lectures and evaluating the losses associated with model. I started with the LENET model using a convolution (filter size of 6, kernel size of 5 and strides of 5), maxpooling and dropout layers. This sequence repeats itself one more time and finishes off with a densely connected NN layer of 1. The results were satisfactory until the first big curve where the car failed to complete it.

In order to gauge how well the model was working, I split my image and steering angle data into a training (80%) and validation (20%) set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. After trying LENET I implemented NVIDIA architecture which I have already described.

I implemented a first implementation of NVIDIA model but I still was getting overfitting. To combat the overfitting, I added cropping to images in order to include only road portions in images. I also ran laps in reverse on the same track and also ran laps on the second track in order to better generalize my model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track:

![alt text][image5]
![alt text][image6]

To improve the driving behavior in these cases, I had to add some extra code in order to convert images from RGB to BGR format in `drive.py`. This addition greatly improve driving in autonomous mode.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My final model architecture (model.py lines 40-70) consisted of the following layers:

|          Layer        |                 Description	     	           |
|:---------------------:|:--------------------------------------------:|
| Input         		    | 320x160x3 image   							             |
| Normalization     	  | x / 255 - 0.5                                |
| Cropping2D		        |	cropping 70x25                               |
| Convolution2D		      |	filter = 24, kerne = 5, stride = 5, relu     |
| Convolution2D	    	  | filter = 36, kerne = 5, stride = 5, relu     |
| Convolution2D	        | filter = 48, kerne = 5, stride = 5, relu     |
| Convolution2D		      |	filter = 64, kerne = 3, stride = 3, relu     |
| Convolution2D		      |	filter = 64, kerne = 3, stride = 3, relu     |
| Flatten	      	      | 				                                     |
| Fully connected		    | 100 outputs  									               |
| Fully connected		    | 50 outputs  								                 |
| Fully connected		    | 10 outputs  								                 |
| Fully connected		    | 1 output    								                 |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. The second one I ran in reverse.

![alt text][image1]
![alt text][image2]

I then recorded the vehicle center lane driving on the second track.

![alt text][image3]
![alt text][image4]

My approach was to gather more data in different tracks. After the collection process, I had X number of data points. I then preprocessed this data by normalizing, cropping followed by the pipeline. I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2-3 as evidenced by the fact that model accuracy did not changed by adding more.
