import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from thresholds import Thresholds
from perspective_transform import PerspectiveTransform
from fit_polynomial import FitPolynomial
from camera_calibration import CameraCalibration


class LaneDetector:

    def __init__(self, thresh, undis_param, ksize):
        self.s_thresh, self.sx_thresh, self.sy_thresh, self.m_thresh, self.d_thresh = thresh
        self.ksize = ksize
        self.undis_param = undis_param
        self._l_fit = None
        self._r_fit = None

    def process_set_of_images(self, in_folder, out_folder):
        list_of_images = [f for f in os.listdir(in_folder) if f.endswith('.jpg')]

        in_folder = '/Users/bogdankovalchuk/Documents/Study/Autonomous_Driving/Udacity_Self_Driving_Car_Engineer_nd013/CarND-Advanced-Lane-Lines/out_test_video_frames/'
        list_of_images = []
        for i in range(510, 570):
            list_of_images.append(str(i)+'.jpg')

        for i, image_file_name in enumerate(list_of_images):
            image = mpimg.imread(in_folder + image_file_name)
            out_image, left_fit, right_fit = self._detect_lane(image)

            # Save images to file
            out_file = out_folder + image_file_name
            mpimg.imsave(out_file, out_image)

    def process_video(self, test_video, out_video):
        clip = VideoFileClip(test_video)
        processed_clip = clip.fl_image(self._process_frame)
        processed_clip.write_videofile(out_video, audio=False)

    def process_image_frames(self):
        import cv2
        test_video = "test_videos/project_video.mp4"
        out_folder = os.getcwd() + "/out_test_video_frames_res/"

        vidcap = cv2.VideoCapture(test_video)
        success = True
        count = 0
        min_, max_ = 510, 2000
        while success:
            if min_ <= count <= max_:
                success, image = vidcap.read()
                out_image = self._process_frame(image)
                print(count)
                # Save images to file
                out_file = out_folder + str(count) + '.jpg'
                # mpimg.imsave(out_file, out_image)
                cv2.imwrite(out_file, out_image)  # save frame as JPEG file
            success, image = vidcap.read()
            if count > max_:
                break

            count += 1

    def _process_frame(self, image):
        out_image,  self._l_fit, self._r_fit = self._detect_lane(image, left_fit=self._l_fit, right_fit=self._r_fit)
        return out_image

    def _detect_lane(self, image, left_fit=None, right_fit=None):
        # Undistort image
        mtx, dist = self.undis_param
        undist = cv2.undistort(image, mtx, dist, None, mtx)

        # Apply threshold
        thresh = Thresholds(undist)
        undist_binary = thresh.apply_thresholds(ksize=self.ksize, s_thresh=self.s_thresh, sx_thresh=self.sx_thresh,
                                                sy_thresh=self.sy_thresh, m_thresh=self.m_thresh, d_thresh=self.d_thresh)
        # Warp binary image
        transform = PerspectiveTransform(undist_binary)
        binary_warped = transform.warp()

        # Find pixels of left and right lane markers
        polyfit = FitPolynomial(binary_warped)
        left_fitx, right_fitx, left_fit, right_fit, ploty, detected = polyfit.fit_polynomial(left_fit, right_fit)


        # curvature = FitPolynomial.measure_real_curvature(ploty, left_fitx, right_fitx)
        # offset_ = FitPolynomial.offset(ploty, image.shape, left_fit, right_fit)
        # Warp image with lanes back
        # print(curvature, offset_)
        out_image = transform.warp_back(undist, ploty, left_fitx, right_fitx)

        return out_image, left_fit, right_fit


if __name__ == "__main__":
    # Undistortion parameters
    undis_param = CameraCalibration.load_pickle_with_undist_params("camera_cal/wide_dist_pickle.p")

    # Threshold parameters
    ksize = 3  # Sobel kernel size
    s_thresh = (170, 255)
    sx_thresh = (20, 100)
    sy_thresh = (20, 255)
    m_thresh = (30, 100)
    d_thresh = (0.7, 1.3)
    thresh = s_thresh, sx_thresh, sy_thresh, m_thresh, d_thresh

    # # Input/output folders of test images
    # in_folder = os.getcwd() + "/test_images/"
    # out_folder = os.getcwd() + "/output_images/"
    #
    # # Detect lanes on test images and write to out folder
    # lane_detector = LaneDetector(thresh, undis_param, ksize)
    # lane_detector.process_set_of_images(in_folder, out_folder)

    test_video = os.getcwd() + "/test_videos/project_video.mp4"
    out_video = os.getcwd() + "/output_images/out_vid.mp4"
    lane_detector = LaneDetector(thresh, undis_param, ksize)
    # lane_detector.process_video(test_video, out_video)

    lane_detector.process_image_frames()

    # in_folder = 'out_test_video_frames/'
    # out_folder = os.getcwd() + "/out_test_video_frames_res/"
    # list_of_images = list()
    # for i in range(200):
    #     list_of_images.append(str(i) + '.jpg')
    #
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    # f.tight_layout()
    # ax1.imshow(binary_warped, cmap='gray')
    # ax1.set_title('Original Image', fontsize=50)
    # ax2.imshow(undist_binary, cmap='gray')
    # ax2.set_title('Thresholded S', fontsize=50)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # plt.show()
