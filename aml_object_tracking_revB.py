import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import imutils
import os
from tkinter import *
from tkinter import Label, Toplevel, Entry, Button
from tkinter.filedialog import askopenfilename

'''
Adapted from:
https://learnopencv.com/object-tracking-using-opencv-cpp-python/
https://www.geeksforgeeks.org/perspective-transformation-python-opencv/

Camera calibration code is kept separate. Will need to copy the camera matrix and distortion coefficients into this code.

This code seeks to track an object bounded by a user-defined region of interest. 
Users should setup the test fixture and camera. The checkerboard from the camera calibration should be placed in the video frame; this is used
to calculate the real-world distance the object is travelling--more on that later.

Alongside camera distortion, what is known as the Keystone effect must also be correct. The Keystone effect is when an image is projected
onto an angled surface. This causes a square to appear as a trapezoid and will negatively effect object tracking.
In the first frame of the video, users will be prompted to select the top left, top right, bottom left, and bottom right (in that order) of the experiment fixture
frame. This will help to fix any keystone distortion. The second frame that is shown will show what the frames look like with the keystone effect 
corrected for and should show the checkerboard with red dots on its corners--this is done to ensure the code is detecting the corners. If the checkerboard
is not in the frame, either redo the keystone correction or reposition the checkerboard in the experiment setup and rerun. 

After Keystone is corrected for, a region of interest (ROI) must be selected. This should be something that is hanging from the spring, but not the spring itself. 
Click and drag starting at the top left of the mass and cover the mass in the rectangle that is drawn.

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
Users should measure the distance of the checkerboard squares and input this into known_width

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

global camera_matrix
global dist_coeffs

root = Tk()
root.withdraw()  # Hide the main window
matrixchoice = False
def close_window():
    input_dialog.destroy()

def no_button():
    global matrixchoice
    matrixchoice = False
    input_dialog.destroy()

def yes_button():
    global matrixchoice
    matrixchoice = True
    input_dialog.destroy()

input_dialog = Toplevel(root)

Label(input_dialog, text="Object tracking Calibration", font=("Arial", 16)).grid(row=0)
Label(input_dialog, text="This code is intended to create position data of the oscillating object from the experiment.").grid(row=1, sticky=W)
Label(input_dialog, text="Users will be prompted to pick the .txt file with their camera's camera matrix and distortion coefficients, which should have been found during the camera calibration").grid(row=2, sticky=W)
Label(input_dialog, text="code. The authors of this code used the Logitech C920s Pro HD Webcam, shooting at 1080p, 30 FPS; if users are also using this, they will be prompted after this ").grid(row=3, sticky=W)
Label(input_dialog, text="window to say so and the matrix and coefficients will be automatically input.").grid(row=4, sticky=W)
Label(input_dialog, text="The position data will be put into a .txt file. Users will be prompted to pick a directory and file name to send this to.").grid(row=5, sticky=W)

ok_button = Button(input_dialog, text="OK", command=close_window)
ok_button.grid(row=7, pady=10)

input_dialog.wait_window(input_dialog)

input_dialog = Toplevel(root) #ask if user using camera used by authors
Label(input_dialog, text="Are you using the Logitech C920s at 1080p, 30 FPS? If not, you will be prompted to select the .txt file made from the camera calibration.").grid(column = 1, row=0, sticky=W)
nobutton = Button(input_dialog, text="No", command=no_button)
nobutton.grid(column = 0,row=1, pady=10)

yesbutton = Button(input_dialog, text="Yes", command=yes_button)
yesbutton.grid(column = 2, row=1, pady=10)
input_dialog.wait_window(input_dialog)

if matrixchoice == True:
    #camera matrix and distortion coefficients of Logitech C920s
    camera_matrix = [[1.13746417e+03, 0.00000000e+00, 3.44167772e+02],
            [0.00000000e+00, 1.14769267e+03, 2.68738206e+02],
             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    dist_coeffs = [ 5.32604142e-01, -1.26503446e+01,  7.85040394e-03,  5.21706755e-03, 1.22513186e+02]
else:
    full_path = askopenfilename(filetypes=[("Text Files", "*.txt")],title='Pick text file with camera matrix and distortion coefficients') # using explorer to open window
    loaded_data = np.loadtxt(full_path, delimiter=',')
    camera_matrix = loaded_data[:9].reshape(3, 3)
    dist_coeffs = loaded_data[9:]

# print(camera_matrix)
# # print(camera_matrix.shape)
# print(dist_coeffs)
# print(dist_coeffs.shape)

input_dialog = Toplevel(root)

Label(input_dialog, text="Users will now be prompted to select the video taken during the experiment.").grid(row=0, sticky=W)
ok_button = Button(input_dialog, text="OK", command=close_window)
ok_button.grid(row=2, pady=10)
input_dialog.wait_window(input_dialog)

# video path for keystone calibration
video_path1 = askopenfilename(title='Pick video from experiment') # using explorer to open window

from tkinter import filedialog
input_dialog = Toplevel(root)
Label(input_dialog, text="Next, users will pick the file location and name for the .txt file that will hold the position data.").grid(row=0, sticky=W)
ok_button = Button(input_dialog, text="OK", command=close_window)
ok_button.grid(row=2, pady=10)
input_dialog.wait_window(input_dialog)

full_path2 = filedialog.asksaveasfilename(filetypes=[("Text Files", "*.txt")], initialfile="", defaultextension=".txt", title='Pick directory and input file name for text file that will contain position data')
print(full_path2)

# #directory and filename that will be created. Make sure to change filename to not overwrite previous data
# directory = r"C:\Users\..."  
# filename = "###.txt"

input_dialog = Toplevel(root)  # Create a new Toplevel window for input

Label(input_dialog, text="Lastly, users will input the following parameters. See the procedure PowerPoint for an explanation on the number of inner corners. For the known width, ").grid(row=0, sticky=W)
Label(input_dialog, text="this is the width of the checkerboard squares in millimeters.").grid(row=1, sticky=W)
Label(input_dialog, text="The global width parameter is the pixel width of the video frames that will be displayed and is used to ensure that all frames of camera tracking are in agreement with the frame size.").grid(row=2, sticky=W)
Label(input_dialog, text="By default, this is 600 but may need to be tweaked depending on the user's screen size.").grid(row=3, sticky=W)
# Label(input_dialog, text="Number of inner corners along the x-axis").grid(row=4, sticky=W)
# Label(input_dialog, text="Number of inner corners along the y-axis").grid(row=5, sticky=W)
Label(input_dialog, text="Camera framerate").grid(row=4, sticky=W)
Label(input_dialog, text="Global width (defaulted to 600, but users should change if window sizes are too small/large)").grid(row=5, sticky=W)

# nx = Entry(input_dialog)
# ny = Entry(input_dialog)
known_width = Entry(input_dialog) 
camera_framerate = Entry(input_dialog) 
if matrixchoice:
    camera_framerate.insert(0,'30')
global_width = Entry(input_dialog) ; global_width.insert(0, '600')

# nx.grid(row=4, column=1)
# ny.grid(row=5, column=1)
camera_framerate.grid(row=4, column=1)
global_width.grid(row=5, column=1)

global params
params=[]
def getInput():
    # a = int(nx.get())
    # b = int(ny.get())
    # c = float(known_width.get())
    d = int(camera_framerate.get())
    e = int(global_width.get())
    input_dialog.destroy()  # Close the input dialog
    global params
    # params = [a, b, d, e]
    params = [d, e]

submit_button =Button(input_dialog, text="submit", command=getInput).grid(sticky=W)
# submit_button.grid(row = 9, pady = 10)
input_dialog.wait_window(input_dialog)
# nx, ny, camera_framerate, global_width = params
camera_framerate, global_width = params

# global_width = 600  # width used for frames that are displayed to user. Users might need to change this based on users' computer screen size and orientation of 
#                     # camera during testing, so change this value if window displayed does not look correct

# # Define the number of INNER CORNERS along the x and y axes in the calibration checkerboard
# nx = 8
# ny = 6

# # Define the known width of the checkerboard block in real-world units (e.g., millimeters)
# known_width = 25.4  # Assuming the checkerboard block is 25.4 mm wide

# # Input camera frame rate. Used to compute time elapsed
# camera_framerate = 30

input_dialog = Toplevel(root)
Label(input_dialog, text="Now, camera tracking will begin. For the first window that pops up, users should select the four corners of their test set up in the ").grid(row=0, sticky=W)
Label(input_dialog, text="following order: top left, top right, bottom left, bottom right. This is done to fix the Keystone effect. See step 5.4 in the procedure PowerPoint for").grid(row=1, sticky=W)
Label(input_dialog, text="further explanation. When the four corners have been properly selected, press the space bar to continue.").grid(row=2, sticky=W)
ok_button = Button(input_dialog, text="OK", command=close_window)
ok_button.grid(row=4, pady=10)
input_dialog.wait_window(input_dialog)

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


# p1, p2 are are top left and top right points users selected for keystone correction, respectively
p1 = np.array(distorted_corners[0])
p2 = np.array(distorted_corners[1])


pixel_distance = np.linalg.norm(p2 - p1)

# users will now input the real-world distance of the top left and top right points selected

input_dialog = Toplevel(root)

Label(input_dialog, text="What is the real-world distance (in millimeters) of the first two points selected for keystone correction?").grid(row=0, sticky=W)
Label(input_dialog, text="Real-world distance (mm): ").grid(row=1, sticky=W)
Label(input_dialog, text="IMPORTANT: to stop the program while the position is being tracked, exit out of the pop-up window.").grid(row=2, sticky=W)

known_width = Entry(input_dialog)
known_width.grid(row=1, column=1)

cv2.imshow("Corners should be highlighted.", undistorted_image)

def getInput():
    global known_width
    known_width = float(known_width.get())
    input_dialog.destroy()  # Close the input dialog

    # Release the video capture object
    cap.release()
    # Destroy any remaining OpenCV windows
    cv2.destroyAllWindows() 

Button(input_dialog, text="submit", command=getInput).grid(sticky=W)

input_dialog.wait_window(input_dialog)  # Wait for the input dialog to close

print('known width:', known_width)
conversion_factor = known_width / pixel_distance

print('conversion factor: ', conversion_factor)
# Create a resizable window
# cv2.namedWindow('Corners should be highlighted', cv2.WINDOW_NORMAL)
# Display the frame

# ---

# Object tracking
roi_ok = False
while not roi_ok:
    input_dialog = Toplevel(root)
    Label(input_dialog, text="Next, the region of interest (ROI) will be selected. The ROI is the area in the frame that users want to track, which will").grid(row=0, sticky=W)
    Label(input_dialog, text="most likely be the weight attached to the spring. Click and drag from the top left to the bottom right to select the ROI.").grid(row=1, sticky=W)
    ok_button = Button(input_dialog, text="OK", command=close_window)
    ok_button.grid(row=3, pady=10)
    input_dialog.wait_window(input_dialog)

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
    roi_ok = True
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
        # cv2.putText(frame2, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
        
        # Display FPS on frame
        cv2.putText(frame2, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        # Display coordinates on frame
        cv2.putText(frame2, "(x, y) = " + '(' + str(centerx) + ',' + str(centery) + ')', (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        # Display result
        cv2.imshow("Tracking", frame2)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : 
            roi_ok = False
            cap.release()
            cv2.destroyAllWindows() 
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break
            


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
# plt.plot(cx)
# plt.plot(cy, 'ro', cy, 'k')
# plt.title("X and Y Position vs Time, original")
# plt.ylabel('Position')
# plt.legend(['X Position', 'Y Position'])
# plt.show()

plt.plot(cx_engr)
plt.plot(cy_engr, 'ro', cy_engr, 'k')
plt.title("X and Y Position vs Time")
plt.xlabel('time (s)')
plt.ylabel('Position (mm)')
plt.legend(['X Position', 'Y Position'])
plt.show()

# ------------------- data saving


# directory = r"C:\Users\"
# filename = ""

# # Create the directory if it doesn't exist
# if not os.path.exists(full_path2):
#     os.makedirs(full_path2)

# full_path = os.path.join(directory, filename)

with open(full_path2, 'w') as file:
    file.write("time(s)\tdisplacement\n")
    
    for i in range(len(cy_engr)):
        file.write(f"{i/camera_framerate:.3f}\t{cy_engr[i]:.3f}\n")

print("Data saved successfully.")



