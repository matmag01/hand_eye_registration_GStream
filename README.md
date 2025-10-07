# hand_eye_registration_GStream
This project allows you to perform an hand eye calibration of a daVinci Research Kit, without using ROS. It is based on the project (dVRK hand-eye calibration package)[https://github.com/jhu-dvrk/dvrk_camera_registration].

## USAGE 
Activate a console
```
rosrun dvrk_robot dvrk_console_json -j ~/catkin_ws/devel/share/jhu-daVinci/console-SUJFixed-ECM-PSM1-PSM2.json
```
After placing an ArUco marker on one PSM of a dVRK, 
```
python camera_registration.py -p *PSM Numeber* -m *ArUco dimension* -d *Camera ID* -c *Path for camera parameters (YAML File)*
```
If you have a ECM, add ```-e ECM``` and copy and paste teh result in ```suj-fixed.json```
You can visualize the result:
```
python vis_gripper_pose.py -p  *PSM Numeber* -H *Path for result of calibration* -y *Path for camera parameters (YAML File)*
```
## Credits
This project is based on [original repository](https://github.com/jhu-dvrk/dvrk_camera_registration)
Modifications by Matteo Magnani (2025).
