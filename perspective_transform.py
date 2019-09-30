import numpy as np
import cv2


def warp(undist, src=None, dst=None):

    img_size = (undist.shape[1], undist.shape[0])

    # define dst 4 pints
    upper_left = [0.3*img_size[0], 0]
    upper_right = [0.7*img_size[0], 0]
    lower_right = [0.7*img_size[0], img_size[1]]
    lower_left = [0.3*img_size[0], img_size[1]]
    dst = np.float32([upper_left, upper_right, lower_right, lower_left])

    # define src 4 point
    horizon_bottom_shift = 200
    horizon_top_shift = 600
    vertical_down_shift = 0.62*img_size[1]
    upper_left = [horizon_top_shift, vertical_down_shift]
    upper_right = [img_size[0] - horizon_top_shift, vertical_down_shift]
    lower_right = [img_size[0] - horizon_bottom_shift, img_size[1]]
    lower_left = [horizon_bottom_shift, img_size[1]]
    src = np.float32([upper_left, upper_right, lower_right, lower_left])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, Minv, src, dst


def warp_back(undist, warped, Minv, ploty, left_fitx, right_fitx):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result