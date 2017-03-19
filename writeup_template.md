##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/test1.jpg "Road Transformed"
[image3]: ./output_images/00_Orginal.jpg "Orginal Image"
[image4]: ./output_images/01_Orginal_Undist.jpg "Undistorted"
[image5]: ./output_images/02_Conversion2Binary.jpg "Converted to binary image"
[image6]: ./output_images/03_TransformationBinary.jpg "Transformed image"
[image7]: ./output_images/04_TransformationOrginal.jpg "Transformed orginal image"
[image8]: ./output_images/06_FindingLanesOverlay.jpg "Transformed orginal image with lane detection"
[image9]: ./output_images/07_FinalLanesOverlay.jpg "Final output with lane detection"
[image10]: ./output_images/Combimded2_HLS_Dir_vs_mag_Dir.jpg "Combind HLS & Direction vs. Magnidute & Direction"
[image11]: ./output_images/Combimded3_final.jpg "Binary detection"
[image12]: ./output_images/Combimded_X1_Y0.jpg "Binary detection gradients x==1 & gradients y ==0 "
[image13]: ./output_images/hls_binary.jpg "HLS binary detection"
[image14]: ./output_images/ThresholdedDirection.jpg "Direction binary detection"
[image15]: ./output_images/ThresholdedGradientx.jpg "Gradient x binary detection"
[image16]: ./output_images/ThresholdedGradienty.jpg "Gradient y binary detection"
[image17]: ./output_images/ThresholdedMagnidute.jpg "Thresholded Magnidute binary detection"
[image18]: ./output_images/hls_binary.jpg "HLS binary detection"
[image19]: ./output_images/hls_binary.jpg "HLS binary detection"


[video1]: ./output_images/output.avi "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
The camera calibration is done within the module CAM_calibration.py.
All chessboard images in the folder camera_cal are readin as JPG.

Every image is process via the method calibrateOnFrame(img).
- First there is the conversion to a grayscale img
- Chessboard coners are detected by using the OpenCV2 methods (findChessboardCorners)
- The object and image points for all images are stored in two arrays
- After processing all images the camera matrix are calculated via the opencv method cv2.calibrateCamera
- Thoese values are stored as dictionary in a pickle file to reload on the acutal LineFinding process.

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
For isolating and converting the lines to a binary image of the road I used a combination of

- Gradient detection X axis versus Y axis  
![alt text][image15]
![alt text][image16]
- Gradient detection combinded (x==1 and y==0) results in:  
![alt text][image12]

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
