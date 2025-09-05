#!/usr/bin/env python3

# Author: Juan Antonio Barragan
# Date: 2024-04-19

# (C) Copyright 2022-2024 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

import sys
import time
import argparse
import pathlib
import json
from dataclasses import dataclass, field
import numpy
import cv2
import crtk
import tf_conversions.posemath
import dvrk_camera_registration

from camera import Camera as CameraGst
from camera import gst_to_opencv, load_calibration, make_pipeline

@dataclass
class ImageSubscriber:
    yaml_file: str
    device_number: int = 0
    current_frame: numpy.ndarray = field(default=None, init=False)
    camera_matrix: numpy.ndarray = field(default=None, init=False)

    def __post_init__(self):
        self.cam = CameraGst(device_number=self.device_number, yaml_file=self.yaml_file)
        self.camera_matrix = self.cam.proj_matrix[:, :3]
        print("Camera matrix:\n", self.camera_matrix)

    def wait_until_first_frame(self):
        print("Waiting for camera frame...")
        timeout = 10
        start = time.time()
        while self.current_frame is None:
            sample = self.cam.appsink.emit("pull-sample")
            frame = gst_to_opencv(sample)
            #frame = cv2.resize(frame, (1300, 1024), interpolation=cv2.INTER_LINEAR)
            rectified = cv2.remap(frame, self.cam.map1, self.cam.map2, cv2.INTER_LINEAR)
            if rectified is not None:
                self.current_frame = rectified
                return
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for camera frame")
            time.sleep(0.2)



@dataclass
class PoseAnnotator:
    camera_matrix: numpy.ndarray
    cam_T_base: numpy.ndarray
    dist_coeffs: numpy.ndarray = field(
        default_factory=lambda: numpy.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape((-1, 1)),
        init=False,
    )

    def __post_init__(self):
        print(self.dist_coeffs.shape)

    def draw_pose_on_img(self, img: numpy.ndarray, local_measured_cp: numpy.ndarray):
        pose = self.cam_T_base @ local_measured_cp

        tvec = pose[:3, 3]
        rvec = cv2.Rodrigues(pose[:3, :3])[0]

        points_3d = numpy.array([[[0, 0, 0]]], numpy.float32)
        points_2d, _ = cv2.projectPoints(
            points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        print(f"camera matrix: {self.camera_matrix}")
        points_2d = tuple(points_2d.astype(numpy.int32)[0, 0])

        img = cv2.circle(img, points_2d, 10, (0, 0, 255), -1)
        img = self.draw_axis(img, self.camera_matrix, self.dist_coeffs, pose, size=0.01)

        return img

    def draw_axis(
        self,
        img: numpy.ndarray,
        mtx: numpy.ndarray,
        dist: numpy.ndarray,
        pose: numpy.ndarray,
        size: int = 10,
    ):

        s = size
        thickness = 2
        R, t = pose[:3, :3], pose[:3, 3]
        K = mtx

        rotV, _ = cv2.Rodrigues(R)
        points = numpy.float32([[s, 0, 0], [0, s, 0], [0, 0, s], [0, 0, 0]]).reshape(-1, 3)
        axisPoints, _ = cv2.projectPoints(points, rotV, t, K, dist)
        axisPoints = axisPoints.astype(int)

        img = cv2.line(
            img,
            tuple(axisPoints[3].ravel()),
            tuple(axisPoints[0].ravel()),
            (255, 0, 0),
            thickness,
        )
        img = cv2.line(
            img,
            tuple(axisPoints[3].ravel()),
            tuple(axisPoints[1].ravel()),
            (0, 255, 0),
            thickness,
        )

        img = cv2.line(
            img,
            tuple(axisPoints[3].ravel()),
            tuple(axisPoints[2].ravel()),
            (0, 0, 255),
            thickness,
        )
        return img


def run_pose_visualizer(arm_handle, yaml_file: str, cam_T_robot_base: numpy.ndarray):

    img_subscriber = ImageSubscriber(yaml_file=yaml_file, device_number=0)
    img_subscriber.wait_until_first_frame()

    pose_annotator = PoseAnnotator(img_subscriber.camera_matrix, cam_T_robot_base)

    window_name = "Pose Visualizer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    while True:
        sample = img_subscriber.cam.appsink.emit("pull-sample")
        frame = gst_to_opencv(sample)
        #frame = cv2.resize(frame, (1300, 1024), interpolation=cv2.INTER_LINEAR)
        rectified = cv2.remap(frame, img_subscriber.cam.map1, img_subscriber.cam.map2, cv2.INTER_LINEAR)
        if rectified is None:
            continue

        local_measured_cp = tf_conversions.posemath.toMatrix(
            arm_handle.local.measured_cp()[0]
        )

        img = pose_annotator.draw_pose_on_img(rectified, local_measured_cp)
        cv2.imshow(window_name, img)
        k = cv2.waitKey(1)

        if k == 27 or k == ord("q"):
            break

    cv2.destroyAllWindows()



def load_hand_eye_calibration(json_file: pathlib.Path) -> numpy.ndarray:
    with open(json_file, "r") as f:
        data = json.load(f)

    cam_T_robot_base = numpy.array(data['base-frame']['transform']).reshape(4, 4)
    return cam_T_robot_base

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--psm-name",
        type=str,
        required=True,
        choices=["PSM1", "PSM2", "PSM3"],
        help="PSM name corresponding to ROS topics",
    )
    parser.add_argument(
        "-H",
        "--hand-eye-json",
        type=str,
        required=True,
        help="hand-eye calibration matrix in JSON format",
    )
    parser.add_argument(
        "-y",
        "--yaml-file",
        type=str,
        required=True,
        help="path to camera calibration YAML (es: ~/catkin_ws/src/gst_cam/calibrations/left.yaml)",
    )
    args = parser.parse_args()

    # crea handle per PSM (quello rimane uguale, usa dvrk)
    ral = crtk.ral("vis_gripper_pose")
    arm_handle = dvrk_camera_registration.ARM(ral, arm_name=args.psm_name, expected_interval=0.1)
    ral.check_connections()

    cam_T_robot_base = load_hand_eye_calibration(args.hand_eye_json)

    run_pose_visualizer(arm_handle, args.yaml_file, cam_T_robot_base)


if __name__ == "__main__":
    main()
