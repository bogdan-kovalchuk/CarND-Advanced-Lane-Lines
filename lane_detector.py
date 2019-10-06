import os
import cv2
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from thresholds import Thresholds
from perspective_transform import PerspectiveTransform
from fit_polynomial import FitPolynomial
from camera_calibration import CameraCalibration


class LaneDetector:
    def __init__(self, thresh, undis_param, ksize, debug=False):
        self.s_thresh, self.sx_thresh, self.sy_thresh, self.m_thresh, self.d_thresh = thresh
        self.ksize = ksize
        self.undis_param = undis_param
        self._l_fit = None
        self._r_fit = None
        self.debug = debug

    def process_set_of_images(self, in_folder, out_folder):
        """
        Main function to detect lanes on the set of images
        """
        list_of_images = [f for f in os.listdir(in_folder) if f.endswith('.jpg')]

        for i, image_file_name in enumerate(list_of_images):
            image = mpimg.imread(in_folder + image_file_name)
            out_image, _, _, _, _ = self._detect_lane(image)

            # Save images to file
            out_file = out_folder + image_file_name
            mpimg.imsave(out_file, out_image)

    def process_video(self, test_video, out_video):
        """
        Main function to detect lanes on the video file
        """
        clip = VideoFileClip(test_video)
        processed_clip = clip.fl_image(self._process_frame)
        processed_clip.write_videofile(out_video, audio=False)

    def _process_frame(self, image):
        out_image,  self._l_fit, self._r_fit, curvature, offset = self._detect_lane(image, self._l_fit, self._r_fit)
        curvature_text = f"Radius of curvature = {int(curvature)}(m)"
        cv2.putText(out_image,  curvature_text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 255, 255), 2, cv2.LINE_AA)

        side = ('right' if offset < 0 else 'left')
        offset_text = f"Vehicle is {abs(round(offset, 2))}m {side} of center"
        cv2.putText(out_image, offset_text, (10, 90), cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 255, 255), 2, cv2.LINE_AA)

        return out_image

    def _detect_lane(self, image, left_fit_prev=None, right_fit_prev=None):
        # Undistort image
        mtx, dist = self.undis_param
        undist = cv2.undistort(image, mtx, dist, None, mtx)

        # Apply threshold
        thresh = Thresholds(undist, self.debug)
        undist_binary = thresh.apply_thresholds(ksize=self.ksize, s_thresh=self.s_thresh, sx_thresh=self.sx_thresh,
                                                sy_thresh=self.sy_thresh, m_thresh=self.m_thresh, d_thresh=self.d_thresh)
        # Warp binary image
        transform = PerspectiveTransform(undist_binary, image, self.debug)
        binary_warped = transform.warp()

        # Find left and right lane markers polynomial
        polyfit = FitPolynomial(binary_warped, self.debug)
        left_fit, right_fit = polyfit.fit_polynomial(left_fit_prev, right_fit_prev)

        # Warp image with lanes back
        out_image = transform.warp_back(undist, polyfit)

        # Curvature and offset
        curvature = polyfit.curvature()
        offset = polyfit.offset()

        return out_image, left_fit, right_fit, curvature, offset


if __name__ == "__main__":
    # Undistortion parameters
    undis_param = CameraCalibration.load_pickle_with_undist_params("camera_cal/wide_dist_pickle.p")

    # Threshold parameters
    ksize = 3  # Sobel kernel size
    s_thresh_img = (170, 255)
    s_thresh_vid = (80, 200)
    sx_thresh = (20, 200)
    sy_thresh = (20, 200)
    m_thresh = (30, 120)
    d_thresh = (0.7, 1.3)

    # Input/output folders of test images
    in_folder = os.getcwd() + "/test_images/"
    out_folder = os.getcwd() + "/output_images/"

    # Detect lanes on test images and write to out folder
    debug_img = False
    thresh_img = s_thresh_img, sx_thresh, sy_thresh, m_thresh, d_thresh
    lane_detector_img = LaneDetector(thresh_img, undis_param, ksize, debug_img)
    lane_detector_img.process_set_of_images(in_folder, out_folder)

    # Input/output folders of test video
    test_video = os.getcwd() + "/test_videos/project_video.mp4"
    out_video = os.getcwd() + "/output_video/out_vid.mp4"

    # Detect lanes on test video and write to out folder
    thresh_vid = s_thresh_vid, sx_thresh, sy_thresh, m_thresh, d_thresh
    lane_detector_vid = LaneDetector(thresh_vid, undis_param, ksize)
    lane_detector_vid.process_video(test_video, out_video)
