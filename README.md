# hand_eye_registration_GStream

python camera_registration.py -p PSM1 -m 0.015 -d 1 -e ECM -c ~/Desktop/dvrk_camera_stereo_calibration/left.yaml

python vis_gripper_pose.py -p PSM1 -H ~/Desktop/dvrk_camera_registration/scripts/PSM1-registration-open-cv.json -y ~/Desktop/dvrk_camera_stereo_calibration/left.yaml
