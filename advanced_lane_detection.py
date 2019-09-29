import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from thresholds import apply_thresholds
from perspective_transform import warp
from fit_polynomial import measure_curvature_real, fit_polynomial

img = mpimg.imread('test_images/straight_lines1.jpg')

dist_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
undist = cv2.undistort(img, mtx, dist, None, mtx)

ksize = 3  # Sobel kernel size
s_thresh = (170, 255)
sx_thresh = (20, 100)
sy_thresh = (20, 255)
m_thresh = (30, 100)
d_thresh = (0.7, 1.3)

undist_binary = apply_thresholds(image=undist, ksize=ksize, s_thresh=s_thresh, sx_thresh=sx_thresh,
                                 sy_thresh=sy_thresh, m_thresh=m_thresh, d_thresh=d_thresh)

warped, Minv, _, _ = warp(undist_binary)


# left_curverad, right_curverad = measure_curvature_real(warped)
ploty, left_fitx, right_fitx = fit_polynomial(warped, ym_per_pix=30 / 720, xm_per_pix=3.7 / 700)
# Create an image to draw the lines on
warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
# Combine the result with the original image
result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
plt.imshow(result)
#
# # plt.imshow(fit_poly)
# # Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(undist)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(warped, cmap='gray')
# ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
