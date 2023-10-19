import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import imutils
import os
'''
Adapted from:
https://learnopencv.com/object-tracking-using-opencv-cpp-python/
https://www.geeksforgeeks.org/perspective-transformation-python-opencv/

Camera calibration code is kept separate. Will need camera matrix and distortion coefficients. 

This code seeks to track an object bounded by a user-defined region of interest. 
Users should setup the test fixture and camera. The checkerboard from the camera calibration should be placed in the video frame; this is used
to calculate the real-world distance the object is travelling--more on that later.

Alongside camera distortion, what is known as the Keystone effect must also be correct. The Keystone effect is when an image is projected
onto an angled surface. This causes a square to appear as a trapezoid and will negatively effect object tracking.
In the first frame of the video, users will be prompted to select the top left, top right, bottom left, and bottom right (in that order) of the experiment fixture
frame. This will help to fix any keystone distortion. The second frame that is shown will show what the frames look like with the keystone effect 
corrected for and should show the checkerboard with red dots on its corners--this is done to ensure the code is detecting the corners. If the checkerboard
is not in the frame, either redo the keystone correction or reposition the checkerboard in the experiment setup and rerun. 

After Keystone is corrected for, a region of interest (ROI) must be selected. This most likely will be the mass hanging from the spring. Click and drag starting at the top left of 
the mass and cover the mass in the reectangle that is drawn.

User inputs: there is some tweaking that must be done to the video frames to ensure that the object tracking, position tracking, and 
camera distortion correction are all in agreement with the frame size. So, all frames displayed are made to be the same size; this is where variable 
'global_width' comes in. By default it is set to 600. If the frames displayed are too big/small, tweak this.

video_path1: this is the video where object tracking will occur. Ensure that the object is not moving at the beginning of the video. 
Trim the beginning of the video to ensure that the object begins to move a few frames after the video begins--the object idling at the start for too long can
mess with the object tracking.

Directory: choose the directory where position data should be sent
filename: choose the name that the position data should have. If the filename is not changed, but the code is ran twice, the file will be 
overwritten--so make sure to change this!

nx, ny: as in the camera calibration code, this is the number of inner corners of the checkerboard
known_width: the checkerboard is being used to calculate a conversion factor between the number of pixels travelled and the real-world distance. 
Users should measure the distance between the checkerboard squares and input this into known_width

camera_framerate: the camera framerate should be input here. This is used to compute the elapsed experiment time. This number can be found in most 
video software or, if using a webcam, should be readily available in the camera's manual

Examples of videos used by the authors can be found on the GitHub, titled "aml_dry_video.mp4" and "aml_wet_video.mp4"
Examples of selecting the four corners for keystone correction and the ROI can be seen in "aml_keystone_corner_selection.png" and "aml_roi_selection.png" on the GitHub

Order of operations: 
    1. Calibrate camera. Get camera matrix and distortion coefficient (this should be done in the separate camera code)
    2. Setup camera in front of test fixture in orientation it will be in during testing with checkerboard aligned in front of fixture
    3. Fix keystone, code should analyze checkerboard to get pixel-distance conversion ratio
    4. Complete test     
'''
# --- User Inputs: ---

# The values in camera_matrix and dist_coeffs were used by the authors and here as place holders; they should be changed by the user to what was found from the camera 
# calibration code.
camera_matrix = [[1.13746417e+03, 0.00000000e+00, 3.44167772e+02],
                 [0.00000000e+00, 1.14769267e+03, 2.68738206e+02],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
dist_coeffs = [ 5.32604142e-01, -1.26503446e+01,  7.85040394e-03,  5.21706755e-03, 1.22513186e+02]

global_width = 600  # width used for frames that are displayed to user. Users might need to change this based on users' computer screen size and orientation of 
                    # camera during testing, so change this value if window displayed does not look correct

# video path for keystone calibration
video_path1 = r'C:\Users\...' 

#directory and filename that will be created. Make sure to change filename to not overwrite previous data
directory = r"C:\Users\..."  
filename = "###.txt"

# Define the number of INNER CORNERS along the x and y axes in the calibration checkerboard
nx = 8
ny = 6

# Define the known width of the checkerboard block in real-world units (e.g., millimeters)
known_width = 25.4  # Assuming the checkerboard block is 25.4 mm wide

# Input camera frame rate. Used to compute time elapsed
camera_framerate = 30

# --- --- --- ---

video_path2 = cv2.VideoCapture(video_path1) 

#---

#Keystone correction
# This section involves correcting the Keystone effect. 
# Once the code is ran, the first frame will run, and users should click on the four corners of the test fixture. If the corners are 
# obstructed, users should clear their workspace and submit a new video. Only make four clicks--more or less will lead to code not running. Press any key
# when done to move on. The pixel locations of the distorted corners should be printed
# Click order: top left, top right, bottom left, bottom right of test fixture

points = []  # Global variables to store the pixel locations of the four points

def get_points(event, x, y, flags, param):
    # handle mouse events
    if event == cv2.EVENT_LBUTTONDOWN:
        # Append the clicked coordinates to the points list
        points.append((x, y))

        # Draw a circle to mark the selected point on the image
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Video Frame", frame)

        # Check if four points have been selected, then stop the GUI
        if len(points) == 4:
            # cv2.destroyAllWindows()
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def fix_keystone(distorted_corners, image, display_images):
    # Get the dimensions of the input image
    height, width = image.shape[:2]

    # Calculate the desired output corners based on the aspect ratio
    aspect_ratio = width / height
    desired_width = global_width
    # desired_height = int(desired_width / aspect_ratio)
    desired_height = 800
    desired_corners = np.float32([[0, 0], [desired_width, 0], [0, desired_height], [desired_width, desired_height]])

    # Calculate the perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(distorted_corners, desired_corners)

    # Apply the perspective transform to the image
    corrected_image = cv2.warpPerspective(image, transform_matrix, (desired_width, desired_height))
    if display_images == True:
        # Display the original and corrected images
        # cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
        cv2.imshow("Original Image", image)
        # cv2.namedWindow('Corrected Image', cv2.WINDOW_NORMAL)
        cv2.imshow("Corrected Image", corrected_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return corrected_image


# Open the video file
# video_path1 = r'C:\Users\choga\OneDrive - University of Iowa\IIHR WORK\Fluids Low Price Lab\logcam2_cal_test2.mp4'
video = cv2.VideoCapture(video_path1)

if not video.isOpened():
    print("Error: Unable to open video.")
    exit()

# Read the first frame from the video
ret, frame = video.read()
height, width = frame.shape[:2]
print( height, 'width:', width)
if not ret:
    print("Error: Unable to read video frame.")
    video.release()
    exit()

frame =  imutils.resize(frame, width=global_width) 

# Create a window to display the video frame
# cv2.namedWindow("Video Frame", cv2.WINDOW_NORMAL)
cv2.imshow("Video Frame", frame)

# Set the mouse callback function
cv2.setMouseCallback("Video Frame", get_points)

# Wait for the user to click four points on the video frame
print("Click four points on the video frame to save their pixel locations.")
cv2.waitKey(0)

# Convert the points to np.float32 and save to a variable
distorted_corners = np.float32(points)

# Now you can directly use 'distorted_corners' in your fix_keystone() function
print("Distorted corners:")
print(distorted_corners)

# ---

# Camera calibration
# This section involves establishing the known variables for the calibrated camera (which is obtained in a separate code) 
# as well as getting the distance ratio so that engineering-units can be used when the object is tracked
# The camera and test fixture should be in the same orientation as the previous section 
# First, place the checkerboard used when calibrating the camera in the test fixture. Placing it such that it is in line with the fixture's legs
# is recommended. Variables 'nx', 'ny', 'camera_matrix', and 'dist_coeffs' as well as the video path should be known and input by the users below.
# The target frame should also be considered via variable 'target_frame' This will most likely be the first frame, so input of 1 will work.

# camera_matrix = [[1.13746417e+03, 0.00000000e+00, 3.44167772e+02],
#                  [0.00000000e+00, 1.14769267e+03, 2.68738206e+02],
#                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
# dist_coeffs = [ 5.32604142e-01, -1.26503446e+01,  7.85040394e-03,  5.21706755e-03, 1.22513186e+02]

camera_matrix = np.array(camera_matrix) 
dist_coeffs = np.array(dist_coeffs)

# Path to the video file
# video_path2 = r"C:\Users\choga\OneDrive - University of Iowa\IIHR WORK\Fluids Low Price Lab\logcam2_cal_test2.mp4"
# Open the video file
cap = cv2.VideoCapture(video_path1)

target_frame = 1
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)  # -1 to account for 0-based indexing
# Read the first frame
ret, camcal_image = cap.read()
height, width = camcal_image.shape[:2]
print('cam cal height: ', height, width)

camcal_image =  imutils.resize(frame, width=global_width) 
cap.release()
height, width = camcal_image.shape[:2]

# Calculate the optimal camera matrix
new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))

# Undistort the image
undistorted_image = cv2.undistort(camcal_image, camera_matrix, dist_coeffs, None, new_camera_matrix) 

# Fix keystoning 
undistorted_image = fix_keystone(distorted_corners, undistorted_image, False)

cv2.imshow('Camera calibration, undistorted image w/ keystone fixed. This is what conv factor is looking at: ',undistorted_image)
# Find the corners of the checkerboard
gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# Draw circles around the corners
corners = np.int_(corners)
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(undistorted_image, (x, y), 3, (0, 0, 255), -1)

# Ensure you have at least two corners to calculate the distance
if len(corners) >= 2:
    # Convert the corners to integers
    corners = np.int_(corners)

    # Initialize a list to store the differences between adjacent corners
    corner_differences = []

    # Loop through the corners and calculate differences between adjacent corners within each square
    for i in range(len(corners)):
        if (i + 1) % nx != 0:  # If i = rightmost corner, will skip to next row
            corner1 = corners[i]
            corner2 = corners[i + 1]

            # Calculate the Euclidean distance in pixels
            pixel_distance = np.linalg.norm(corner1 - corner2)
            
            corner_differences.append(pixel_distance)
    # Calculate conversion factor from average of all corner distances and known width input by user
    print('Mean corner distance ', np.mean(corner_differences))
    conversion_factor = known_width / np.mean(corner_differences)

    # if user want to see value of the differences between adjacent corners 
    # print("Differences between adjacent corners (in pixels):")
    # for i, diff in enumerate(corner_differences, start=1):
    #     print(f"Segment {i}: {diff} pixels")
    # else:
    #     print("Not enough corners found to calculate differences.")

print('conversion factor: ', conversion_factor)
# Create a resizable window
# cv2.namedWindow('Corners should be highlighted', cv2.WINDOW_NORMAL)
# Display the frame
cv2.imshow("Corners should be highlighted, press spacebar to continue", undistorted_image)
cv2.waitKey(0)

# Release the video capture object
cap.release()
# Destroy any remaining OpenCV windows
cv2.destroyAllWindows()

# ---

# Object tracking

from cv2 import __version__
__version__
 
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
# Set up tracker.

tracker_types = ['MIL','KCF', 'CSRT']
tracker_type = tracker_types[0]

if int(minor_ver) < 3:
    tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

# Read video
# video = cv2.VideoCapture(r"C:\Users\choga\OneDrive - University of Iowa\IIHR WORK\Fluids Low Price Lab\logcam2_track_test2.mp4")
video = video_path2

# Exit if video not opened.
if not video.isOpened():
    print ("Could not open video")
    sys.exit()

# Read first frame.
ok, frame = video.read()
if not ok:
    print ('Cannot read video file')
    sys.exit()

height, width = frame.shape[:2]
print('tracking height: ', height, width)

frame =  imutils.resize(frame, width=global_width) 

height, width = frame.shape[:2]
print('distorted tracking vid width: ', width, 'height: ', height)

# Calculate the optimal camera matrix
new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))
# Undistort the image
frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)  

# Fix keystoning 
frame = fix_keystone(distorted_corners, frame, False)
height, width = frame.shape[:2]

# user-defined bounding box
bbox = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

#x and y centroid coordinates
cx = []
cy = []

while True:
    # Read a new frame
    ok, frame2 = video.read()
    if not ok:
        break
    # Start timer
    timer = cv2.getTickCount()

    height, width = frame2.shape[:2]
    # print('tracking loop height ', height, width)
    #---------------------------
    frame2 =  imutils.resize(frame2, width=global_width)

    height, width = frame2.shape[:2]
    # print('width of video frame:', width, ' height: ', height)
    # Calculate the optimal camera matrix
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))
    # Undistort the image
    undistorted_image = cv2.undistort(frame2, camera_matrix, dist_coeffs, None, new_camera_matrix) 
    # Fix keystoning
    frame2 = fix_keystone(distorted_corners, frame2, False)
    #---------------------------
    # print('width after cam cal:', width, ' height: ', height)
    # Update tracker
    ok, bbox = tracker.update(frame2)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok:
        # print('tracking detected')
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        #centroid calculation
        centerx = int(bbox[0]+(bbox[2]/2))
        centery = int(bbox[1]+(bbox[3]/2))

        cx.append(centerx)
        cy.append(height - centery) #y-axis of video is at top of frame and positive y is down by default, so difference between height 
                                    #of frame and y coord must be found to follow the convention of y axis starting at bottom and positive y being up
        
        cv2.rectangle(frame2, p1, p2, (255,0,0), 2, 1)
        cv2.circle(frame2, (centerx, centery), 10, (255,0,0), -1)    #circle at centroid of box

    else :
        # Tracking failure
        cv2.putText(frame2, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display tracker type on frame
    cv2.putText(frame2, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
    
    # Display FPS on frame
    cv2.putText(frame2, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

    # Display coordinates on frame
    cv2.putText(frame2, "(x, y) = " + '(' + str(centerx) + ',' + str(centery) + ')', (200,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

    # Display result
    cv2.imshow("Tracking", frame2)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break


# use conversion factor to convert pixels to real-world unit
if conversion_factor:
    cx_engr = [i * conversion_factor for i in cx]
    cy_engr = [i * conversion_factor for i in cy]
    print('true')
else:
    print('false')
    cx_engr = cx
    cy_engr=cy

#plotting data points in pixel coords and engineering units
# important: cx is x position of tracked object and is being plotted on y-axis with time on x-axis, it is NOT what should be plotted on x-axis 
plt.plot(cx)
plt.plot(cy, 'ro', cy, 'k')
plt.title("X and Y Position vs Time, original")
plt.ylabel('Position')
plt.legend(['X Position', 'Y Position'])
plt.show()

plt.plot(cx_engr)
plt.plot(cy_engr, 'ro', cy_engr, 'k')
plt.title("X and Y Position vs Time, engineering")
plt.ylabel('Position')
plt.legend(['X Position', 'Y Position'])
plt.show()

# ------------------- data saving


# directory = r"C:\Users\"
# filename = ""

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

full_path = os.path.join(directory, filename)

with open(full_path, 'w') as file:
    file.write("time(s)\tdisplacement\n")
    
    for i in range(len(cy_engr)):
        file.write(f"{i/camera_framerate:.3f}\t{cy_engr[i]:.3f}\n")

print("Data saved successfully.")



