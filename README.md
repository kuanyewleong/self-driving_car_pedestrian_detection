# Self-driving Car Pedestrian Detection
![alt text](https://github.com/kuanyewleong/self-driving_car_pedestrian_detection/blob/master/OxfordRobotCar_dataset.JPG "detection_sample")

This is a demo of pedestrian detection for a self-driving car via a mono-chrome camera sensor. 

The idea that I have worked on is to apply a pre-trained deep learning model to detect pedestrians along the drive through pathway of a vehicle. I have prepared two variants of solution to demonstrate the usability of such pre-trained model in augmenting a self-driving car.


## Setup
### Step 1. The following dependencies or packages are needed
- Tensorflow
- OpenCV
- Python 3.6 

(one can simply create an environment with the above packages through Anaconda)

### Step 2. Clone the repository
No compilation or installation is neeeded, just clone the repository.

git clone https://github.com/kuanyewleong/self-driving_car_pedestrian_detection.git


## Dataset
The Oxford RobotCar dataset was used to test the implementation, and you may download the dataset form the following link. For the demos developed here, only the stereo vision data of the centre camera were used.

http://robotcar-dataset.robots.ox.ac.uk/

An example of the data:
(Source: Oxford RobotCar) 
![alt text](https://github.com/kuanyewleong/self-driving_car_pedestrian_detection/blob/master/0258.png "sample")


## Deep Learning Model
The model being tested to achieve the best accuracy is a pre-trained Tensorflow RCNN Inception (trained with COCO dataset) downloadable from Model Zoo: http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz

After donwload the model, place it in the directory ../tensorflow_model


## How to run both the demos

The demos were specifically built for the Oxford RobotCar dataset, minor modification might be needed if you are using it for other dataset. 

For demo on pedestrian detection of the vehicle's frontal pathway:

$ python demo_frontal.py <data_directory>

For demo on pedestrian detection of the visible area of the centre camera of the vehicle:

$ python demo_visible.py <data_directory>



## Examples
I have recorded two demo videos for my work. The first video demonstrates pedestrian detection along the drive-through in general. While the other video, which is more emphasized, demonstrates a possible idea to detect both pedestrians that could be obstacles for the self-driving car, as well as the total pedestrian visible to the frontal camera (i.e. pedestrians on the side-way or walkway).

## The Underlying Principles
The implementation in demo_frontal.py is especially more practical in self-driving car due to the reason that it enables the detection of pedestrians on the vehicle's frontal pathway, meaning it doesn't simply detect all possible pedestrian visible to the centre camera, but it filters those not in the frontal pathway which is most crucial for the vehicle to move forward. 

However, this is not all, the solution also tries to detect all pedestrians around the visible region of the camera, such that extra information can be processed for an even better manoeuvre of the vehicle; for example, one can in the future incorporate such information with fusion with other sensors to give the vehicle an awareness that at a particular surrounding, there is some presence of pedestrians, and hence the travelling mode can be tuned to a different setting as if compared to another surrounding when pesdetrians are not detected. 
