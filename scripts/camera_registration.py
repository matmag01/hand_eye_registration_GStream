#!/usr/bin/env python3

# Author: Brendan Burkhart
# Date: 2022-06-16

# (C) Copyright 2022-2024 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

import argparse
import sys
import time
import cv2
import json
import math
import numpy
from scipy.spatial.transform import Rotation

import crtk
from camera import Camera
from arm import ARM
import convex_hull
import vision_tracking


class CameraRegistrationApplication:
    def __init__(self, ral, psm_name, ecm_name, marker_size, expected_interval, camera):
        self.ral = ral
        self.camera = camera
        self.marker_size = marker_size
        self.expected_interval = expected_interval
        self.psm_name = psm_name
        self.psm = ARM(ral, arm_name=psm_name, expected_interval=expected_interval)
        if ecm_name is not None:
            self.ecm = ARM(ral, arm_name=ecm_name, expected_interval=expected_interval)
        else:
            self.ecm = None
        ral.check_connections()

        print(f"measured_jp {self.psm.measured_jp()}")
        print("connections checked")

    def setup(self):
        self.messages.info("Enabling {}...".format(self.psm.name))
        if not self.psm.enable(5):
            self.messages.error(
                "Failed to enable {} within 10 seconds".format(self.psm.name)
            )
            return False

        self.messages.info("Homing {}...".format(self.psm.name))
        if not self.psm.home(10):
            self.messages.error(
                "Failed to home {} within 10 seconds".format(self.psm.name)
            )
            return False

        self.messages.info("Homing complete\n")

        return True

    def determine_safe_range_of_motion(self):
        self.messages.info(
            "Release the clutch and move the PSM around to establish the area the PSM can move in.  It doesn't matter if the ArUco tag is visible!"
        )
        self.messages.info("Press enter or 'd' when done")

        def collect_points(hull_points):
            self.done = False

            while self.ok and not self.done:
                pose, _ = self.psm.measured_jp()
                position = numpy.array([pose[0], pose[1], pose[2]])

                # make list sparser by ensuring >2mm separation
                euclidean = lambda x: numpy.array(
                    [math.sin(x[0]) * x[2], math.sin(x[1]) * x[2], math.cos(x[2])]
                )
                distance = lambda a, b: numpy.linalg.norm(euclidean(a) - euclidean(b))
                if len(hull_points) == 0 or distance(position, hull_points[-1]) > 0.005:
                    hull_points.append(position)

                time.sleep(self.expected_interval)

            return hull_points

        hull_points = []

        while True:
            hull_points = collect_points(hull_points)
            if not self.ok:
                return False, None

            hull = convex_hull.convex_hull(hull_points)
            if hull is None:
                self.messages.info("Insufficient range of motion, please continue")
            else:
                break

        self.messages.info(
            "Range of motion displayed in plot, close plot window to continue"
        )
        #convex_hull.display_hull(hull)
        return self.ok, hull

    # Make sure target is visible and PSM is within range of motion
    def ensure_target_visible(self, safe_range):
        self.done = True  # run first check immeditately
        first_check = True

        while self.ok:
            time.sleep(0.25)

            if not self.done:
                continue

            jp = numpy.copy(self.psm.measured_jp()[0])
            visible = self.tracker.is_target_visible(timeout=1)
            in_rom = convex_hull.in_hull(safe_range, jp)

            if not visible:
                self.done = False
                if first_check:
                    self.messages.warn(
                        "\nPlease position psm so ArUco target is visible, facing towards camera, and roughly centered within camera's view\n"
                    )
                    first_check = False
                else:
                    self.messages.warn(
                        "Target is not visible, please re-position. Make sure target is not too close"
                    )
                self.messages.info("Press enter or 'd' when done")
            elif not in_rom:
                self.done = False
                self.messages.warn(
                    "PSM is not within user supplied range of motion, please re-position"
                )
            else:
                return True, jp

        return False, None

    # From starting position within view of camera, determine the camera's
    # field of view via exploration while staying within safe range of motion
    # Once field of view is found, collect additional pose samples
    def collect_data(self, safe_range, start_jp, edge_samples=4):
        current_jp = numpy.copy(start_jp)
        current_jp[4:6] = numpy.zeros(2)

        target_poses = []
        robot_poses = []

        def measure_pose(joint_pose):
            nonlocal target_poses
            nonlocal robot_poses

            if not convex_hull.in_hull(safe_range, joint_pose):
                self.messages.error("Safety limit reached!")
                return False

            self.psm.move_jp(joint_pose).wait()
            time.sleep(0.5)

            ok, target_pose = self.tracker.acquire_pose(timeout=4.0)
            if not ok:
                return False

            target_poses.append(target_pose)

            pose = self.psm.local.measured_cp()[0].Inverse()
            rotation_quaternion = Rotation.from_quat(pose.M.GetQuaternion())
            rotation = numpy.float64(rotation_quaternion.as_matrix())
            translation = numpy.array([pose.p[0], pose.p[1], pose.p[2]], dtype=numpy.float64)

            robot_poses.append((rotation, numpy.array(translation)))

            return True

        def bisect_camera_view(pose, ray, min_steps=4, max_steps=6):
            start_pose = numpy.copy(pose)
            current_pose = numpy.copy(pose)

            far_limit = convex_hull.intersection(safe_range, start_pose[0:3], ray)
            near_limit = 0.0

            for i in range(max_steps):
                if not self.ok:
                    break

                mid_point = 0.5 * (near_limit + far_limit)
                current_pose[0:3] = start_pose[0:3] + mid_point * ray

                ok = measure_pose(current_pose)
                if ok:
                    near_limit = mid_point
                    #self.tracker.display_point(target_poses[-1][1], (255, 0, 255))
                else:
                    far_limit = mid_point

                # Only continue past min_steps if we haven't seen target yet
                if i + 1 >= min_steps and near_limit > 0:
                    break

            end_point = start_pose[0:3] + 0.9 * near_limit * ray
            if len(target_poses) > 0:
                self.tracker.display_point(target_poses[-1][1], (255, 123, 66), size=7)

            return end_point

        def collect(poses, tool_shaft_rotation=math.pi / 8.0):
            self.messages.progress(0.0)
            for i, pose in enumerate(poses):
                if not self.ok or self.ral.is_shutdown():
                    return

                rotation_direction = 1 if i % 2 == 0 else -1
                pose[3] = pose[3] + rotation_direction * tool_shaft_rotation
                shaft_rotations = [
                    pose[3] + rotation_direction * tool_shaft_rotation,
                    pose[3] - rotation_direction * tool_shaft_rotation,
                ]

                for shaft_rotation in shaft_rotations:
                    pose[3] = shaft_rotation
                    ok = measure_pose(pose)
                    if ok:
                        self.tracker.display_point(target_poses[-1][1], (255, 255, 0))
                        break

                self.messages.progress((i + 1) / len(sample_poses))

        self.messages.line_break()
        self.messages.info("Determining limits of camera view...")
        self.messages.progress(0.0)
        limits = []

        for axis in range(3):
            ray = numpy.array([0, 0, 0])
            for direction in [1, -1]:
                if not self.ok:
                    return None

                ray[axis] = direction
                limits.append(bisect_camera_view(current_jp, ray))
                self.messages.progress(len(limits) / 6)
        self.messages.line_break()

        # Limits found above define octahedron, take samples along all 12 edges
        sample_poses = []
        for i in range(len(limits)):
            start = i + 2 if i % 2 == 0 else i + 1
            for j in range(start, len(limits)):
                for t in numpy.linspace(
                    1 / (edge_samples + 1), 1 - 1 / (edge_samples + 1), edge_samples
                ):
                    pose = numpy.copy(current_jp)
                    pose[0:3] = limits[j] + t * (limits[i] - limits[j])
                    sample_poses.append(pose)

        self.messages.info("Collecting pose data...")
        collect(sample_poses)
        self.messages.line_break()

        self.messages.info("Data collection complete\n")
        return robot_poses, target_poses

    def compute_registration(self, robot_poses, target_poses):
        error, rotation, translation = self.camera.calibrate_pose(
            robot_poses, target_poses
        )

        if error < 1e-4:
            self.messages.info(
                "Registration error ({:.3e}) is within normal range".format(error)
            )
        else:
            self.messages.warn(
                "WARNING: registration error ({:.3e}) is unusually high! Should generally be <0.00005".format(
                    error
                )
            )

        distance = numpy.linalg.norm(translation)
        self.messages.info(
            "Measured distance from RCM to camera origin: {:.3f} m\n".format(distance)
        )

        return self.ok, rotation, translation

    def save_registration(self, rotation, translation, file_name, dvrk_format):
        rotation = numpy.linalg.inv(rotation)
        translation = -numpy.matmul(rotation, translation)

        transform = numpy.eye(4)
        transform[0:3, 0:3] = rotation
        transform[0:3, 3:4] = translation

        if dvrk_format:
            to_dvrk = numpy.eye(4)
            to_dvrk[0,0] = -to_dvrk[0,0]
            to_dvrk[1,1] = -to_dvrk[1,1]
            transform = to_dvrk @ transform
            if self.ecm:
                ecm_cp, _ = self.ecm.local.measured_cp()
                ecm_transform = numpy.eye(4)
                for i in range(0, 3):
                    ecm_transform[i, 3] = ecm_cp.p[i]
                    for j in range(0, 3):
                        ecm_transform[i, j] = ecm_cp.M[i, j]
                transform = ecm_transform @ transform


        if dvrk_format:
            if self.ecm:
                self.messages.info("The dVRK calibration needs to be copy-pasted to the suj-fixed.json")
                data = {
                    "name": self.psm_name,
                    "measured_cp": transform.tolist(),
                }
                output = json.dumps(data)
            else:
                self.messages.info("The dVRK calibration needs to be copy-pasted to the console-xxx.json")
                data = {
                    "reference-frame": "/left_frame",
                    "transform": transform.tolist(),
                }
                output = '"base-frame": {}'.format(json.dumps(data))
        else:
            data = {
                "reference-frame": "/left_frame",
                "transform": transform.tolist(),
            }
            output = '{\n' + '"base-frame": {}'.format(json.dumps(data)) + '\n}'

        with open(file_name, "w") as f:
            f.write(output)
            f.write("\n")

        self.messages.info("Hand-eye calibration saved to {}".format(file_name))

    # Exit key (q/ESCAPE) handler for GUI
    def _on_quit(self):
        self.ok = False
        self.tracker.stop()
        self.messages.info("\nExiting...")

    # Enter (or 'd') handler for GUI
    def _on_enter(self):
        self.done = True

    def _init_tracking(self):
        target_type = vision_tracking.ArUcoTarget(
            self.marker_size, cv2.aruco.DICT_4X4_50, [0]
        )
        parameters = vision_tracking.VisionTracker.Parameters(4)
        self.messages = vision_tracking.MessageManager()
        self.tracker = vision_tracking.VisionTracker(
            target_type, self.messages, self.camera, parameters
        )

    def run(self):
        try:
            cv2.setNumThreads(2)
            self.ok = True

            self._init_tracking()
            self.ok = self.ok and self.tracker.start(self._on_enter, self._on_quit)
            if not self.ok:
                return

            self.ok = self.ok and self.setup()
            if not self.ok:
                return
            print("finish setup")
            ok, safe_range = self.determine_safe_range_of_motion()
            if not self.ok or not ok:
                return

            ok, start_jp = self.ensure_target_visible(safe_range)
            if not self.ok or not ok:
                return

            data = self.collect_data(safe_range, start_jp)
            if not self.ok:
                return

            if len(data[0]) <= 10:
                self.messages.error("Not enough pose data, cannot compute registration")
                self.messages.error(
                    "Please try again, with more range of motion within camera view"
                )
                return

            ok, rvec, tvec = self.compute_registration(*data)
            if not ok:
                return

            self.tracker.stop()

            self.save_registration(
                rvec, tvec, "./{}-registration-open-cv.json".format(self.psm.name), False # using OpenCV frame coordinates
            )

            self.save_registration(
                rvec, tvec, "./{}-registration-dVRK.json".format(self.psm.name), True # using dVRK frame coordinates
            )

        finally:
            self.tracker.stop()
            # self.psm.unregister()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--marker_size",
        type=float,
        required=True,
        help="ArUco marker side length in meters (in base alla calibrazione)"
    )
    parser.add_argument(
        "-c", "--camera_calibration",
        type=str,
        required=True,
        help="Path al file .yaml di calibrazione (es: ~/catkin_ws/src/gst_cam/calibrations/left.yaml)"
    )
    parser.add_argument(
        "-d", "--device_number",
        type=int,
        default=0,
        help="Indice della webcam / device GStreamer"
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=0.1,
        help="Intervallo atteso in secondi tra frame"
    )
    parser.add_argument( "-p", "--psm-name", 
                        type=str, 
                        required=True, 
                        choices=["PSM1", "PSM2", "PSM3"], 
                        help="PSM name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace", 
                        )
    parser.add_argument( "-e", "--ecm-name", 
                        type=str, choices=["ECM"], 
                        help="ECM name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace", 
                        )
    args = parser.parse_args()
    ral = crtk.ral("dvrk_camera_calibration") 
    cam = Camera( device_number=args.device_number, yaml_file=args.camera_calibration)
    app = CameraRegistrationApplication(
        ral=ral,
        marker_size=args.marker_size,
        expected_interval=args.interval,
        camera=cam,
        psm_name=args.psm_name,
        ecm_name=args.ecm_name,
    )
    app.run()


if __name__ == "__main__":
    main()
