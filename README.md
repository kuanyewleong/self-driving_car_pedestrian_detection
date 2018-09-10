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
git clone https://github.com/kuanyewleong/self-driving_car_pedestrian_detection.git

## Dataset
The Oxford RobotCar dataset was used to test the implementation, and you may download the dataset form the following link. For the demos developed here, only the stereo vision data of the centre camera were used.
http://robotcar-dataset.robots.ox.ac.uk/

## 

## Examples
I have recorded two demo videos for my work. The first video demonstrates pedestrian detection along the drive-through in general. While the other video, which is more emphasized, demonstrates a possible idea to detect both pedestrians that could be obstacles for the self-driving car, as well as the total pedestrian visible to the frontal camera (i.e. pedestrians on the side-way or walkway).
