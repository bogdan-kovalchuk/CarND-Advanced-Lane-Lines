import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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

    @staticmethod
    def undist_image(in_image, out_image, undist_pickle):
        """
        Prepare image for writeup
        """
        mtx, dist = CameraCalibration.load_pickle_with_undist_params(undist_pickle)
        img = cv2.imread(in_image)

        undistorted = cv2.undistort(img, mtx, dist, None, mtx)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(np.array(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)))
        ax2.set_title('Undistorted Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(out_image)


if __name__ == "__main__":
    calibration_images = glob.glob('camera_cal/calibration*.jpg')
    n, m = 6, 9  # image size
    out_pickle = "camera_cal/wide_dist_pickle.p"

    calibration = CameraCalibration()
    calibration.prepare_undist_params(calibration_images, n, m, out_pickle)

    # Chess bord undistortion
    in_image = "camera_cal/calibration1.jpg"
    out_image = "writeup_images/dist_correction_image.png"
    CameraCalibration.undist_image(in_image, out_image, out_pickle)

    # Test image undistortion
    in_image = "test_images/straight_lines1.jpg"
    out_image = "writeup_images/test_image.jpg"
    CameraCalibration.undist_image(in_image, out_image, out_pickle)
