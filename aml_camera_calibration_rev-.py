import numpy as np
import cv2

'''
Code adapted from:
https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

In order to ensure the picture created for the test is not significantly warped, the user's camera must be calibrated. This code serves to do that
Users should print out a checkerboard on a piece of paper and lay the paper flat on a table. Take a video using the camera that will be used during
testing. Ensure that during the video, the camera is positioned such that the paper moves towards all four corners of the video frame. 
During the video, users should try and keep the camera's line of sight as normal to the piece of paper as possible--as in, do not rotate or turn the camera.
An example video of the camera calibration video used by the authors, camera-calibration-video.wmv, can be found on the GitHub 

There are three primary user inputs: the number of inner corners should go into nx and ny; the path to the video in 'video_path;' the framerate of the camera used 
Important: it is the number of INNER corners for nx and ny.

The code will return a 'camera matrix' and 'distortion coefficients.' These are used in the object tracking code to undistort the video frame, so save
these values for later use.

The checkerboard .jpg used by the author is found in the link before
https://2.bp.blogspot.com/-XDs6jTqJmlk/VvqhAYNotLI/AAAAAAAABXs/NWcRlFHGYDYdtJ6EfDLVV2eFexDu1-1sQ/s1600/Checkerboard%2B8x6%2B%2528180px%2529.JPG 

Towards the end of the code there is an optional section that allows users to undistort a frame to check if results are reasonable.
Uncomment the section and run the code to see the undistorted frame. By default, the first frame from camera calibration video will be used. 
If a different frame is desired, change value of 'target_frame' in 'User Inputs' section 
'''

# User Inputs:

# Define the number of INNER corners along the x and y axes in the calibration checkerboard
nx = 6    # number of inner corners along x-axis
ny = 8    # number of inner corners along y-axis

# Path to the video file
video_path = r"C:\Users\...."

camera_framerate = 30 # frame rate of camera

target_frame = 1
# ---

# Create arrays to store object points (3D) and image points (2D)
object_points = []  # 3D points in real-world space
image_points = []   # 2D points in the image plane

# Get coordinates for the corners of the checkerboard
objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError(f"Failed to open video file at {video_path}")

# Iterate through calibration frames
frame_count = 0
processed_frame_count = 0
while True:
    # Read a frame
    ret, frame = cap.read()
    
    # If frame reading was unsuccessful, exit the loop
    if not ret:
        break

    # Increment the frame counter
    frame_count += 1

    # Process every nth frame
    if frame_count % camera_framerate == 0: 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If corners are found, add object points and image points
        if ret:
            object_points.append(objp)
            image_points.append(corners)

        # # Print the result of corner detection for each frame. This is used to test if code has found the checkerboard corners
        # print(f"Frame {frame_count}: Corners found = {ret}")

        # Draw circles around the corners
        corners = np.int_(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # Create a resizable window
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        # Display the frame
        cv2.imshow("Video", frame)

        # Increment the processed frame counter
        processed_frame_count += 1

    # Wait for a key press to exit
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video capture object
cap.release()
# Destroy any remaining OpenCV windows
cv2.destroyAllWindows()

# Print the number of frames processed
print(f"Processed frames: {processed_frame_count}")

# Calibrate the camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, gray.shape[::-1], None, None
)
print('The camera matrix and distortion coefficients will be printed. Save these values!')
# Print the camera matrix and distortion coefficients
print("Camera matrix:\n", camera_matrix)
print(type(camera_matrix))
print("Distortion coefficients:\n", dist_coeffs)
print(type(dist_coeffs))

# --------

# # Optional section. Uncomment these lines to check how video frame is undistorted; a quick way to uncomment is to highlight the section 
# # and press both 'ctrl' and '/' 

# # Open the video file
# cap = cv2.VideoCapture(video_path)

# # target_frame = 
# cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)  # -1 to account for 0-based indexing

# # Read the first frame
# ret, image = cap.read()

# cap.release()

# height, width = image.shape[:2]

# # Calculate the optimal camera matrix
# new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))

# # Undistort the image
# undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix) 

# # Find the corners of the checkerboard
# gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
# ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# # Create a resizable window
# cv2.namedWindow('Resizable Window', cv2.WINDOW_NORMAL)

# # Display the undistorted image
# cv2.imshow("Resizable Window", undistorted_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()