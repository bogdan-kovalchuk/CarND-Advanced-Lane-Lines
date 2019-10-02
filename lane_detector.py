import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from thresholds import apply_thresholds
from perspective_transform import warp, warp_back
from fit_polynomial import find_lane_pixels, search_around_poly, fit_polynomial, measure_curvature_real


class LaneDetector:
    def __init__(self, undis_param, ksize, thresh):
        pass

    def process_image(self, image):
        # Undistort image
        mtx, dist = undis_param
        undist = cv2.undistort(image, mtx, dist, None, mtx)

        # Apply threshold
        s_thresh, sx_thresh, sy_thresh, m_thresh, d_thresh = thresh
        undist_binary = apply_thresholds(image=undist, ksize=ksize, s_thresh=s_thresh, sx_thresh=sx_thresh,
                                         sy_thresh=sy_thresh, m_thresh=m_thresh, d_thresh=d_thresh)
        # Warp binary image
        binary_warped, Minv, _, _ = warp(undist_binary)
        img_shape = binary_warped.shape

        # Find pixels of left and right lane markers
        leftx, lefty, rightx, righty, _ = find_lane_pixels(binary_warped)

        # Fit left and right lane lines with polynomial
        left_fitx, right_fitx, left_fit, right_fit, ploty = fit_polynomial(img_shape, leftx, lefty, rightx, righty)

        # Warp image with lanes back
        result = warp_back(undist, binary_warped, Minv, ploty, left_fitx, right_fitx)

        return result

    def process_next_frame_image(self, image, undis_param, ksize, thresh, left_fit=None, right_fit=None):
        mtx, dist = undis_param
        undist = cv2.undistort(image, mtx, dist, None, mtx)

        s_thresh, sx_thresh, sy_thresh, m_thresh, d_thresh = thresh
        undist_binary = apply_thresholds(image=undist, ksize=ksize, s_thresh=s_thresh, sx_thresh=sx_thresh,
                                         sy_thresh=sy_thresh, m_thresh=m_thresh, d_thresh=d_thresh)

        binary_warped, Minv, _, _ = warp(undist_binary)
        img_shape = binary_warped.shape

        leftx, lefty, rightx, righty = search_around_poly(binary_warped, left_fit, right_fit)

        left_fitx, right_fitx, left_fit, right_fit, ploty = fit_polynomial(img_shape, leftx, lefty, rightx, righty)

        result = warp_back(undist, binary_warped, Minv, ploty, left_fitx, right_fitx)

        return result, left_fit, right_fit

    def process_video(self, video):
        pass


if __name__ == "__main__":

    dist_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    undis_param = mtx, dist

    ksize = 3  # Sobel kernel size
    s_thresh = (170, 255)
    sx_thresh = (20, 100)
    sy_thresh = (20, 255)
    m_thresh = (30, 100)
    d_thresh = (0.7, 1.3)

    thresh = s_thresh, sx_thresh, sy_thresh, m_thresh, d_thresh

    # in_folder = 'test_images/'
    # out_folder = os.getcwd() + "/output_images/"
    # list_of_images = [f for f in os.listdir(in_folder) if f.endswith('.jpg')]

    in_folder = 'out_test_video_frames/'
    out_folder = os.getcwd() + "/out_test_video_frames_res/"
    list_of_images = list()
    for i in range(200):
        list_of_images.append(str(i) + '.jpg')

    left_fit = None
    right_fit = None
    for i, image_file_name in enumerate(list_of_images):
        image = mpimg.imread(in_folder + image_file_name)
        out_image, left_fit, right_fit = process_image(image, undis_param, ksize, thresh, i, left_fit, right_fit)

        # Save images to file
        out_file = out_folder + image_file_name
        mpimg.imsave(out_file, out_image)
