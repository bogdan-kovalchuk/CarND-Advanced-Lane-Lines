import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from thresholds import apply_thresholds
from perspective_transform import warp, warp_back
from fit_polynomial import measure_curvature_real, fit_polynomial


def process_image(img, undis_param, ksize, thresh):
    mtx, dist = undis_param
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    s_thresh, sx_thresh, sy_thresh, m_thresh, d_thresh = thresh
    undist_binary = apply_thresholds(image=undist, ksize=ksize, s_thresh=s_thresh, sx_thresh=sx_thresh,
                                     sy_thresh=sy_thresh, m_thresh=m_thresh, d_thresh=d_thresh)

    warped, Minv, _, _ = warp(undist_binary)

    ploty, left_fitx, right_fitx = fit_polynomial(warped, ym_per_pix=30 / 720, xm_per_pix=3.7 / 700)

    result = warp_back(undist, warped, Minv, ploty, left_fitx, right_fitx)

    return result


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

    in_folder = 'test_images/'
    out_folder = os.getcwd() + "/output_images/"
    list_of_images = [f for f in os.listdir(in_folder) if f.endswith('.jpg')]

    for image_file_name in list_of_images:
        image = mpimg.imread(in_folder + image_file_name)
        precessed_image = process_image(image, undis_param, ksize, thresh)
        # Save images to file
        out_file = out_folder + image_file_name
        mpimg.imsave(out_file, precessed_image)
        # Plot results
        # plt.figure(figsize=(8,8))
        # plt.imshow(image_with_lines)



    # plt.imshow(result)

    # # plt.imshow(fit_poly)
    # # Plot the result
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    # f.tight_layout()
    # ax1.imshow(undist)
    # ax1.set_title('Original Image', fontsize=50)
    # ax2.imshow(warped, cmap='gray')
    # ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # plt.show()
