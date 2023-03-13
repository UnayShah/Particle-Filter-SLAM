
# ECE 276A: Particle Filter Simulateous Localization and Mapping (SLAM)

## Introduction
This is project 2 of the course [ECE 276A: Sensing & Estimation in Robotics](https://natanaso.github.io/ece276a/) at UCSD, being taught by professor [Nikolay Atanisov](https://natanaso.github.io/).

The project is based on data collected by a differential drive robot with specifications provided in the [Documentation File](https://github.com/UnayShah/Particle-Filter-SLAM/blob/master/docs/RobotConfiguration.pdf). The robot collects Encoder Wheels, IMU, LiDAR and Kinect data.

The data from wheels and IMU are to get a dead reckoning trajectory for the robot.
![Dead Reckoning Trajectory for Dataset 20](https://github.com/UnayShah/Particle-Filter-SLAM/blob/master/plots/dead_reckoning_trajectory_dt_20.jpg)

This is then combined with LiDAR data to get a rough scan of the room.
![Dead Reckoning Scan for Dataset 20](https://github.com/UnayShah/Particle-Filter-SLAM/blob/master/plots/dead_reckoning_LiDAR_scan_dt_20.jpg)

An attempt to improve the results, is done using particle filter SLAM, introducing Gaussian Noise and simulating 100 particles at each step. An effective number of 10 particles over the iterations. An occupancy grid is finially built showing the room traversed by the robot.
![Particle Filter SLAM Results for Dataset 20](https://github.com/UnayShah/Particle-Filter-SLAM/blob/master/plots/Particle%20Filter%20SLAM%20Occupancy%20Map20.jpg)

Finally, using the trajectory obtained, the kinect RGBD data is used to project images onto the map for texture mapping.
![Texture Mapping for Dataset 20](https://github.com/UnayShah/Particle-Filter-SLAM/blob/master/plots/Texture_Map20.png)

Data for the robot's 


## Running the code
1. Download the Kinect dataset from [this Google Drive link](https://drive.google.com/drive/folders/1_SLyxhnkSIKVsb_cdXW8BEZlQEDXkuRx)
2. Place the sensor datasets in the folder 'data' and RGBD image data in a folder named 'dataRGBD' with the following file structure:

        .
        ├── data
        │   ├── Encoder20.npz
        │   ├── Encoder21.npz
        │   ├── Hokuyo20.npz
        │   ├── Hokuyo21.npz
        │   ├── Imu20.npz
        │   ├── Imu21.npz
        │   ├── Kinect20.npz
        │   └── Kinect21.npz
        └── dataRGBD
        │   ├── Kinect20.npz
        │   └── Kinect21.npz
        
3. Install the following packages (assuming numpy and matplotlib are already installed):

        pip install transforms3d

4. Run the file project_2_final.py

        python project_2_final.py
