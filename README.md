## Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_images/dist_correction_image.png "Undistorted"
[image2]: ./writeup_images/test_image.jpg "Road Transformed"
[image3_1]: ./writeup_images/sobelXY.png "Binary Example"
[image3_2]: ./writeup_images/magn_grad.png "Binary Example"
[image3_3]: ./writeup_images/s_col_comb.png "Binary Example"
[image4]: ./writeup_images/warped_straight_lines.png "Warp Example"
[image5]: ./writeup_images/color_fit_lines.png "Fit Visual"
[image6]: ./writeup_images/lane_area.png "Output"
[video1]: test_videos/project_video.mp4 "Video"

#### [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the lines 12 - 42 of the file called `camera_calibration.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

The main function of finding lane lines on images/video is `lane_detection.py`. Function `_detect_lane` (lines 54 - 78) consists steps of lane lines finding, this steps will be described below as pipeline:

#### 1. Provide an example of a distortion-corrected image.

To undistort the image I used the distortion correction function `cv2.undistort(image, mtx, dist, None, mtx)` with camera matrix and distortion coefficients that I got on the step of camera calibration. The example of undistortion of the one test images:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholds code is in `thresholds.py`):
- Sobel threshold at X and Y direction (lines 52 through 71 of function `_abs_sobel_thresh()`)
- Magnitude of the gradient threshold (lines 73 through 94 of function `_mag_thresh()`)
- Direction of the gradient threshold (lines 96 through 113 of function `_dir_thresh()`)
- S color channel threshold (lines 115 through 120 of function `_hls_threshold()`)

Applying and combining of thresholds are in lines 14 through 50 of function `apply_thresholds()` 

Thresholds maximum/minimum values are defined in lines 85 through 92 of `lane_detector.py`

Here's an example of my output for this step:

![alt text][image3_1]
![alt text][image3_2]
![alt text][image3_3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 19 through 46 in the file `perspective_transform.py`.  The `warp()` function uses the source (`self.src`) and destination (`self.dst`) points to warp image of road lane separators plane.  
I chose the hardcode the source and destination points in the following manner:

```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
|  530,  468    |   64, 100     | 
|  750,  468    | 1216, 100     |
| 1230,  690    | 1216, 720     |
|   50,  690    |   64, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

There are steps that were performed to find lane lines polynomial (the code is in `fit_polynomial.py`):
- For the binary warped image a histogram of the bottom half of the image was taken to determine left and right base of left and right lines (lines 84 - 92 in `_find_lane_pixels()`)
- With sliding windows step by step lines indices were found and non zero lines pixel were taken (lines 94 - 165 in `_find_lane_pixels()`)
- Using `np.polyfit` with non zero line indices, determined above, coefficients of 2nd order polynomial for left and right lines were computed (lines 28 - 29 in `fit_polynomial()`)

The example of sliding windows, lines indices and lane lines: 

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature of the lane and the position of the vehicle with respect to center in `fit_polynomial.py`:
- radius of curvature in lines 47 through 64 of function `curvature`
- position of the vehicle in lines 66 through 77 of function `offset`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 48 through 82 in my code in `perspective_transform.py` in the function `warp_back()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

The detection of road lane on the video are the same as for image considering avery frame as separate image. Implementation is in `lane_detector.py`. 
There are 3 main functions: `process_video`, `_process_frame` and `_detect_lane` (ines 35 through 78 of the file called `lane_detector.py`). 
To speed up detection the function `_search_around_poly` (ines 167 through 193 of the file `fit_polinomial.py`) was implemented. This function uses the previous polynomial to skip the sliding window.

Here's a [link to my video result](https://github.com/bogdan-kovalchuk/CarND-Advanced-Lane-Lines/blob/master/output_video/out_vid.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- The pipeline doesn't include reset step. So if we have situation with frames where difficult of impossible find lines it will fail.
- Sanity check wasn't perform so some time on the video frame we can note bends and line offsets (The problem is that tuning can go on forever :)).
- The algorithm to slow. It take long time to process video.
