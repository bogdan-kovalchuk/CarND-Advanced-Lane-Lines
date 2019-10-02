import numpy as np
import cv2


class PerspectiveTransform:
    def __init__(self, undist_binary):
        self.undist_binary = undist_binary
        self.img_size = (self.undist_binary.shape[1], self.undist_binary.shape[0])
        self.warped = None
        self.M = None
        self.Minv = None

    def warp(self):

        dst = self._define_dst()
        src = self._define_src()

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        self.warped = cv2.warpPerspective(self.undist_binary, self.M, self.img_size, flags=cv2.INTER_LINEAR)

        return self.warped

    def warp_back(self, undist, ploty, left_fitx, right_fitx):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(self.warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (undist.shape[1], undist.shape[0]))

        # Return the combined result with the original image
        return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    def _define_dst(self):
        # define dst 4 pints
        upper_left = [0.3 * self.img_size[0], 0]
        upper_right = [0.7 * self.img_size[0], 0]
        lower_right = [0.7 * self.img_size[0], self.img_size[1]]
        lower_left = [0.3 * self.img_size[0], self.img_size[1]]
        return np.float32([upper_left, upper_right, lower_right, lower_left])

    def _define_src(self):
        # define src 4 point
        horizon_bottom_shift = 200
        horizon_top_shift = 600
        vertical_down_shift = 0.62*self.img_size[1]
        upper_left = [horizon_top_shift, vertical_down_shift]
        upper_right = [self.img_size[0] - horizon_top_shift, vertical_down_shift]
        lower_right = [self.img_size[0] - horizon_bottom_shift, self.img_size[1]]
        lower_left = [horizon_bottom_shift, self.img_size[1]]
        return np.float32([upper_left, upper_right, lower_right, lower_left])
