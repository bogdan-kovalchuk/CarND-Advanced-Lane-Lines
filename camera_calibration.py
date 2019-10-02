import cv2
import glob
import pickle
import numpy as np


class CameraCalibration:
    def __init__(self):
        pass

    @staticmethod
    def prepare_undist_params(calibration_images, n, m, out_pickle):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((n*m, 3), np.float32)
        objp[:, :2] = np.mgrid[0:m, 0:n].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        img_size = None
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(calibration_images):
            img = cv2.imread(fname)
            img_size = (img.shape[1], img.shape[0])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (m, n), None)

            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = dict()
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open(out_pickle, "wb"))

    @staticmethod
    def load_pickle_with_undist_params(pickle_path):
        # Read in the saved camera matrix and distortion coefficients
        dist_pickle = pickle.load(open(pickle_path, "rb"))
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
        return mtx, dist


if __name__ == "__main__":
    calibration_images = glob.glob('camera_cal/calibration*.jpg')
    n, m = 6, 9  # image size
    out_pickle = "camera_cal/wide_dist_pickle.p"

    calibration = CameraCalibration()
    calibration.prepare_undist_params(calibration_images, n, m, out_pickle)




