import numpy as np
import cv2
import matplotlib.pyplot as plt


class Thresholds:
    def __init__(self, image, debug=False):
        self.image = np.copy(image)
        hls = cv2.cvtColor(self.image, cv2.COLOR_RGB2HLS)
        self.l_channel = hls[:, :, 1]
        self.s_channel = hls[:, :, 2]
        self.debug = debug

    def apply_thresholds(self, ksize, s_thresh, sx_thresh, sy_thresh, m_thresh, d_thresh):
        """
        Apply thresholds to undistorted image
        """
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

        if self.debug:
            def save_2figs(im1, title1, im2, title2, out_file):
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
                f.tight_layout()
                ax1.imshow(im1, cmap='gray')
                ax1.set_title(title1, fontsize=50)
                ax2.imshow(im2, cmap='gray')
                ax2.set_title(title2, fontsize=50)
                plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
                plt.savefig(out_file)

            save_2figs(sxbinary, "1. Sobel threshold X", sybinary, "2. Sobel threshold Y", "writeup_images/sobelXY.png")
            save_2figs(mag_binary, "3. Magnitude gradient", dir_binary, "4. Direction gradient", "writeup_images/magn_grad.png")
            save_2figs(s_binary, "5. S color channel", combined, "6. Combined binary", "writeup_images/s_col_comb.png")

        return combined

    def _abs_sobel_thresh(self, orient='x', sobel_kernel=3, thresh=(0, 255)):
        thresh_min, thresh_max = thresh

        # Take the the absolute value of derivative in 'x' or 'y' given orient
        abs_sobel = None
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(self.l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(self.l_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Return this mask as your binary_output image
        return binary

    def _mag_thresh(self, sobel_kernel=3, thresh=(0, 255)):
        thresh_min, thresh_max = thresh
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        # Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # Calculate the magnitude
        abs_sobelxy = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))

        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))

        # Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Return this mask as your binary_output image
        return binary

    def _dir_thresh(self, sobel_kernel=3, thresh=(0, np.pi / 2)):
        thresh_min, thresh_max = thresh

        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        # Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # Calculate the grad direction
        grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

        binary = np.zeros_like(grad_dir)
        binary[(grad_dir >= thresh_min) & (grad_dir <= thresh_max)] = 1

        # Return this mask as your binary_output image
        return binary

    def _hls_threshold(self, thresh=(0, 255)):
        # Apply a threshold to the S channel
        binary = np.zeros_like(self.s_channel)
        binary[(self.s_channel > thresh[0]) & (self.s_channel <= thresh[1])] = 1
        # Return a binary image of threshold result
        return binary


