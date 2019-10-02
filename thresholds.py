import numpy as np
import cv2


class Thresholds:
    def __init__(self, image):
        self.image = np.copy(image)

    def apply_thresholds(self, ksize, s_thresh, sx_thresh, sy_thresh, m_thresh, d_thresh):

        # Sobel threshold
        sxbinary = self._abs_sobel_thresh(orient='x', sobel_kernel=ksize, thresh=sx_thresh)
        sybinary = self._abs_sobel_thresh(orient='y', sobel_kernel=ksize, thresh=sy_thresh)

        # Magnitude of the gradient threshold
        mag_binary = self._mag_thresh(sobel_kernel=ksize, thresh=m_thresh)

        # Direction of the gradient threshold
        dir_binary = self._dir_thresh(sobel_kernel=ksize, thresh=d_thresh)

        # S color channel threshold
        s_binary = self._hls_threshold(thresh=s_thresh)

        # Combined binary
        combined = np.zeros_like(dir_binary)
        combined[((sxbinary == 1) & (sybinary == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1

        return combined

    def _abs_sobel_thresh(self, orient='x', sobel_kernel=3, thresh=(0, 255)):
        thresh_min, thresh_max = thresh
        # 1) Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        # 2)Take the the absolute value of derivative in 'x' or 'y' given orient
        abs_sobel = None
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

        # 3) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # 4) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # 5) Return this mask as your binary_output image
        return binary

    def _mag_thresh(self, sobel_kernel=3, thresh=(0, 255)):
        thresh_min, thresh_max = thresh
        # 1) Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3) Calculate the magnitude
        abs_sobelxy = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))

        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))

        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # 6) Return this mask as your binary_output image
        return binary

    def _dir_thresh(self, sobel_kernel=3, thresh=(0, np.pi / 2)):
        thresh_min, thresh_max = thresh

        # 1) Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3) Calculate the grad direction
        grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

        binary = np.zeros_like(grad_dir)
        binary[(grad_dir >= thresh_min) & (grad_dir <= thresh_max)] = 1

        # 6) Return this mask as your binary_output image
        return binary

    def _hls_threshold(self, thresh=(0, 255)):
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(self.image, cv2.COLOR_RGB2HLS)
        S = hls[:, :, 2]

        # 2) Apply a threshold to the S channel
        binary = np.zeros_like(S)
        binary[(S > thresh[0]) & (S <= thresh[1])] = 1

        # 3) Return a binary image of threshold result
        return binary


