# Self-driving Car Pedestrian Detection
This is a demo of pedestrian detection for a self-driving car via a camera sensor. 

The idea that I have worked on is to apply a pre-trained deep learning model to detect pedestrians along the drive through pathway of a vehicle. I prepare two variant of solutions to demonstrate the usability of such pre-trained model in augmenting a self-driving car.


## Setup
### Step 1. The following dependencies or packages are needed
- Tensorflow
- OpenCV
- Python 3.6 
(one can create an environment with the above packages through Anaconda)

### Step 2. Clone the repo
No compilation or installation is neeeded, just clone the repository.
git clone https://github.com/kuanyewleong/self-driving_car_pedestrian_detection.git


## Dataset
The Oxford RobotCar dataset was used to test the implementation, and you may download the dataset form the following link. For the demos developed here, only the stereo vision data of the centre camera were used.
http://robotcar-dataset.robots.ox.ac.uk/


## Deep Learning Model
The model being tested to be achieving the best accuracy is a pre-trained Tensorflow RCNN Inception downloadable from Model Zoo: http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz

After donwload the model, place it in the directory ../tensorflow_model


## How to run both the demos

The demos were specifically built for the Oxford RobotCar dataset, minor modification might be needed if you are using it for other dataset. 

For demo on pedestrian detection of the vehicle's frontal pathway:

$ python demo_frontal.py <data_directory>

For demo on pedestrian detection of the visible area of the centre camera of the vehicle:

$ python demo_visible.py <data_directory>



## Examples
I have recorded two demo videos for my work. The first video demonstrates pedestrian detection along the drive-through in general. While the other video, which is more emphasized, demonstrates a possible idea to detect both pedestrians that could be obstacles for the self-driving car, as well as the total pedestrian visible to the frontal camera (i.e. pedestrians on the side-way or walkway).
