#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I used the nvidia model for my network. The model is built in the build_model function. It consists of lambda normalization, cropping, 5 convolution layers, four fully connected layers. 

####2. Attempts to reduce overfitting in the model

I did not have to add any dropout layers. Using flipped images and left/right camera images with correction was sufficient to teach the car to drive around track 1. I also recorded some additional training data (saved under my_training_data) to provide more data for the challenging parts of the track i.e. getting on and off the bridge and sharp turns.

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (see build_model)

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used provided sample data and recorded more for challenging parts of the track. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to piggyback on the NVIDIA model.

My first step was to use a simplest linear regression model I thought this model might be appropriate because I wanted to make sure that preprocessing data, training the model and generating the model.h5 workflow was working end-to-end

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80-20) using test_train_split.

####2. Final Model Architecture

The final model architecture (build_model) consisted of a convolution neural network with the following layers and layer sizes 
I used the nvidia model for my network. The model is built in the build_model function. It consists of lambda normalization, cropping, 5 convolution layers, four fully connected layers. 

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To help with challenging sections of the track I recorded car getting on and off the bridge and making the sharp turns that followed.




To augment the data sat, I also flipped images and angles thinking that this would this would combat counter-clockwise dirviing direction. The images are flipped in batches when running the model. See the generator() function. Specificallly,
the section that starts with augmented_images = [].

I added a conditional to load the images from the my_training_data folder in the generator.


After the collection process, I had 7104 number of data points. I then preprocessed this data by using normalization in the lambda layer of the model and then cropped the image to focus on the road.

    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(1,1))))


I randomly shuffled the data set in the generator and put 20% of the data into a validation set. 
  

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by validation loss increasing after epoch 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.
