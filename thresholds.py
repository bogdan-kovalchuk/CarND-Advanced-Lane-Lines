import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    thresh_min, thresh_max = thresh
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

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


def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    thresh_min, thresh_max = thresh
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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


def dir_thresh(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    thresh_min, thresh_max = thresh

    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the grad direction
    grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    binary = np.zeros_like(grad_dir)
    binary[(grad_dir >= thresh_min) & (grad_dir <= thresh_max)] = 1

    # 6) Return this mask as your binary_output image
    return binary


def hls_threshold(image, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2]

    # 2) Apply a threshold to the S channel
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1

    # 3) Return a binary image of threshold result
    return binary


def pipeline(image, ksize, s_thresh, sx_thresh, sy_thresh, m_thresh, d_thresh):
    image = np.copy(image)

    # Sobel threshold
    sxbinary = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=sx_thresh)
    sybinary = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=sy_thresh)

    # Magnitude of the gradient threshold
    mag_binary = mag_thresh(image, sobel_kernel=ksize, thresh=m_thresh)

    # Direction of the gradient threshold
    dir_binary = dir_thresh(image, sobel_kernel=ksize, thresh=d_thresh)

    # S color channel threshold
    s_binary = hls_threshold(img, thresh=s_thresh)

    # Combined binary
    combined = np.zeros_like(dir_binary)
    combined[((sxbinary == 1) & (sybinary == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1

    return combined


if __name__ == "__main__":
    img = mpimg.imread('bridge_shadow.jpg')

    # Choose a Sobel kernel size
    ksize = 3  # Choose a larger odd number to smooth gradient measurements
    s_thresh = (170, 255)
    sx_thresh = (20, 100)
    sy_thresh = (20, 255)
    m_thresh = (30, 100)
    d_thresh = (0.7, 1.3)

    binary = pipeline(image=img,
                      ksize=ksize,
                      s_thresh=s_thresh,
                      sx_thresh=sx_thresh,
                      sy_thresh=sy_thresh,
                      m_thresh=m_thresh,
                      d_thresh=d_thresh)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(binary, cmap='gray')
    ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
