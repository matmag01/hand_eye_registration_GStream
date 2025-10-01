import cv2
import numpy as np
import yaml
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp
import threading
import time

Gst.init(None)

def make_pipeline(device_number):
    pipeline_str = (
        f"decklinkvideosrc device-number={device_number} ! "
        "videoconvert ! "
        "videocrop left=310 right=310 top=28 bottom=28 ! "
        "video/x-raw,format=BGR ! "
        "appsink name=appsink"
    )
    pipeline = Gst.parse_launch(pipeline_str)
    appsink = pipeline.get_by_name("appsink")
    appsink.set_property("emit-signals", True)
    appsink.set_property("max-buffers", 1)
    appsink.set_property("drop", True)
    return pipeline, appsink

def gst_to_opencv(sample):
    buf = sample.get_buffer()
    caps = sample.get_caps()
    arr = np.ndarray(
        shape=(caps.get_structure(0).get_value("height"),
               caps.get_structure(0).get_value("width"),
               3),
        buffer=buf.extract_dup(0, buf.get_size()),
        dtype=np.uint8)
    return arr

def load_calibration(file_path):
    with open(file_path, 'r') as f:
        calib = yaml.safe_load(f)

    def to_np(mat):
        return np.array(mat["data"], dtype=np.float32).reshape((mat["rows"], mat["cols"]))

    camera_matrix = to_np(calib["camera_matrix"])
    dist_coeffs   = np.array(calib["distortion_coefficients"]["data"], dtype=np.float32)
    rect_matrix   = to_np(calib["rectification_matrix"])
    proj_matrix   = to_np(calib["projection_matrix"])

    return camera_matrix, dist_coeffs, rect_matrix, proj_matrix


class Camera:
    """
    GStreamer camera -> OpenCV interface
    Usa calibrazione YAML invece di ROS CameraInfo
    """

    def __init__(self, device_number, yaml_file):
        # Carica parametri di calibrazione
        self.camera_matrix, self.dist_coeffs, self.rect_matrix, self.proj_matrix = load_calibration(yaml_file)
        self.no_distortion = np.array([], dtype=np.float32)

        # Avvia pipeline
        self.pipeline, self.appsink = make_pipeline(device_number)
        self.pipeline.set_state(Gst.State.PLAYING)

        # h, w = frame.shape[:2]
        h, w = 1024, 1300
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs,
            self.rect_matrix, self.proj_matrix,
            (w, h), cv2.CV_16SC2
        )

        self.image_callback = None
        self.running = False
        self.thread = None

    def set_callback(self, image_callback):
        """Imposta la callback che riceve immagini rettificate"""
        self.image_callback = image_callback
        if image_callback is not None and not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
        elif image_callback is None:
            self.running = False

    def _capture_loop(self):
        while self.running:
            sample = self.appsink.emit("pull-sample")
            if sample is None:
                continue
            frame = gst_to_opencv(sample)
            #frame_res = cv2.resize(frame, (1300, 1024), interpolation=cv2.INTER_LINEAR)
            rectified = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

            if self.image_callback:
                self.image_callback(rectified)

            time.sleep(0.01)

    def project_points(self, object_points, rodrigues_rotation, translation_vector):
        image_points, _ = cv2.projectPoints(
            object_points,
            rodrigues_rotation,
            translation_vector,
            self.proj_matrix[:, :3],
            self.no_distortion,
        )
        return image_points.reshape((-1, 2))

    def get_pose(self, object_points, image_points):
        ok, rotation, translation = cv2.solvePnP(
            object_points, image_points, self.proj_matrix[:, :3], self.no_distortion
        )
        if not ok:
            return ok, 0.0, rotation, translation

        projected_points = self.project_points(object_points, rotation, translation)
        reprojection_error = np.mean(
            np.linalg.norm(image_points - projected_points, axis=1)
        )
        return ok, reprojection_error, rotation, translation
    def calibrate_pose(self, robot_poses, target_poses):
        robot_poses_r = np.array([p[0] for p in robot_poses], dtype=np.float64)
        robot_poses_t = np.array([p[1] for p in robot_poses], dtype=np.float64)
        target_poses_r = np.array([p[0] for p in target_poses], dtype=np.float64)
        target_poses_t = np.array([p[1] for p in target_poses], dtype=np.float64)

        rotation, translation = cv2.calibrateHandEye(
            robot_poses_r,
            robot_poses_t,
            target_poses_r,
            target_poses_t,
            method=cv2.CALIB_HAND_EYE_HORAUD,
        )

        def to_homogenous(rotation, translation):
            X = np.eye(4)
            X[0:3, 0:3] = rotation
            X[0:3, 3] = translation.reshape((3,))
            return X

        robot_transforms = [to_homogenous(r, t) for r, t in robot_poses]
        target_transforms = [to_homogenous(r, t) for r, t in target_poses]
        camera_transform = to_homogenous(rotation, translation)

        transforms = []
        for r, t in zip(robot_transforms, target_transforms):
            a = np.matmul(np.matmul(r, camera_transform), t)
            transforms.append(np.linalg.norm(a, ord="fro"))

        transforms = np.array(transforms)

        error = np.std(transforms - np.mean(transforms))
        
        """
        robot_poses: lista di (R, t) per base→ee
        target_poses: lista di (R, t) per cam→marker
        """

        X = to_homogenous(rotation, translation)

        errors_rot = []
        errors_trans = []

        # Calcola errori sui movimenti relativi
        for i in range(len(robot_poses) - 1):
            # Robot relative motion A_i
            R1, t1 = robot_poses[i]
            R2, t2 = robot_poses[i + 1]
            A = np.linalg.inv(to_homogenous(R1, t1)) @ to_homogenous(R2, t2)

            # Target relative motion B_i
            R1, t1 = target_poses[i]
            R2, t2 = target_poses[i + 1]
            B = np.linalg.inv(to_homogenous(R1, t1)) @ to_homogenous(R2, t2)

            # Verifica A*X ≈ X*B
            left = A @ X
            right = X @ B

            # Errore rotazionale (in gradi)
            R_err = left[:3, :3] @ right[:3, :3].T
            angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
            errors_rot.append(np.degrees(angle))

            # Errore traslazionale (in metri se i tuoi dati sono in m)
            t_err = np.linalg.norm(left[:3, 3] - right[:3, 3])
            errors_trans.append(t_err)

        mean_rot_error = np.mean(errors_rot)
        mean_trans_error = np.mean(errors_trans)
        print('Reprojection error (trans): ', mean_trans_error)
        print('Reprojection error (rot): ', mean_rot_error)

        return error, rotation, translation
    
