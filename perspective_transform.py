import numpy as np
import cv2
import matplotlib.pyplot as plt


class PerspectiveTransform:
    def __init__(self, undist_binary, image, debug=False):
        self.undist_binary = undist_binary
        self.img_size = (self.undist_binary.shape[1], self.undist_binary.shape[0])
        self.warped = None
        self.M = None
        self.Minv = None
        self.dst = None
        self.src = None
        self.debug = debug
        if self.debug:
            self.image_for_debug = image

    def warp(self):
        """
        Warp undistort binary to lines plane
        """
        self.dst = self._define_dst()
        self.src = self._define_src()

        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)
        self.warped = cv2.warpPerspective(self.undist_binary, self.M, self.img_size, flags=cv2.INTER_LINEAR)

        if self.debug:
            image_with_src = self.image_for_debug.copy()
            cv2.polylines(image_with_src, np.int32([self.get_src()]), isClosed=True, color=(255, 0, 0), thickness=3)

            image_with_dst = cv2.warpPerspective(self.image_for_debug.copy(), self.M, self.img_size, flags=cv2.INTER_LINEAR)
            cv2.polylines(image_with_dst, np.int32([self.get_dst()]), isClosed=True, color=(255, 0, 0), thickness=3)

            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(image_with_src)
            ax1.set_title('Undistorted image with src point', fontsize=50)
            ax2.imshow(image_with_dst)
            ax2.set_title('Warped results with dst points', fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.savefig("writeup_images/warped_straight_lines.png")

        return self.warped

    def warp_back(self, undist, polyfit):
        """
        Warp back to normal view
        """
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(self.warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        ploty, left_fitx, right_fitx = polyfit.get_ploty()

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (undist.shape[1], undist.shape[0]))

        # Return the combined result with the original image
        combined = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        if self.debug:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(newwarp)
            ax1.set_title('Lane area', fontsize=50)
            ax2.imshow(combined)
            ax2.set_title('Lane area on test image', fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.savefig("writeup_images/lane_area.png")

        return combined

    def _define_dst(self):
        # define dst 4 pints
        upper_left = [0.05 * self.img_size[0], 100]
        upper_right = [0.95 * self.img_size[0], 100]
        lower_right = [0.95 * self.img_size[0], self.img_size[1]]
        lower_left = [0.05 * self.img_size[0], self.img_size[1]]
        return np.float32([upper_left, upper_right, lower_right, lower_left])

    def _define_src(self):
        # define src 4 point
        horizon_bottom_shift = 50
        horizon_top_shift = 530
        vertical_up_shift = 30
        vertical_down_shift = 0.65*self.img_size[1]
        upper_left = [horizon_top_shift, vertical_down_shift]
        upper_right = [self.img_size[0] - horizon_top_shift, vertical_down_shift]
        lower_right = [self.img_size[0] - horizon_bottom_shift, self.img_size[1] - vertical_up_shift]
        lower_left = [horizon_bottom_shift, self.img_size[1] - vertical_up_shift]
        return np.float32([upper_left, upper_right, lower_right, lower_left])

    def get_dst(self):
        return self.dst

    def get_src(self):
        return self.src
