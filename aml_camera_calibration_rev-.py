import numpy as np
import cv2
from tkinter import *
from tkinter import Label, Toplevel, Entry, Button
from tkinter.filedialog import askopenfilename

'''
save matrices 
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

# inputs: 
    # 1. video path (use askopenfilename())
    # 2. user types: number of inner corners, camera framerate
    # will also need to ask user if they want to see underistorted image. Use this at the end as separate window. Ask user if they want and get target_frame if they do


# User Inputs:

root = Tk()
root.withdraw()  # Hide the main window

def close_window():
    input_dialog.destroy()

input_dialog = Toplevel(root)

Label(input_dialog, text="Camera Calibration", font=("Arial", 16)).grid(row=0)
Label(input_dialog, text="This code is intended to calibrate the user's camera. Users should take a video of their printed checkerboard using the camera that will be used during testing.").grid(row=1, sticky=W)
Label(input_dialog, text="Users will then be prompted to select the video file from the file explorer. Then, users will have to input the number of inner corners along the x and y axis of the checkerboard (see the PowerPoint for further explanation) as well as the framerate of the camera used.").grid(row=2, sticky=W)
Label(input_dialog, text="The code will output a camera and distortion coefficient matrix. These values will need to be saved for later to undistort the camera.").grid(row=3, sticky=W)
Label(input_dialog, text="The authors of this code used the Logitech C920s Pro HD Webcam, shooting at 1080p, 30 FPS. If users are using this same camera, then this portion of the code can be skipped as the matrices for this camera are saved in the object tracking code.").grid(row=4, sticky=W)
Label(input_dialog, text="At the end, users have the option to see an undistorted frame using the coefficients calculated. If this is desired, enter the frame from the calibration video you would like to see undistorted.").grid(row=5, sticky=W)

ok_button = Button(input_dialog, text="OK", command=close_window)
ok_button.grid(row=6, pady=10)

input_dialog.wait_window(input_dialog)

# video_path = "r"+f"{askopenfilename()}" # using explorer to open window
video_path = askopenfilename() # using explorer to open window
# print(video_path)

input_dialog = Toplevel(root)  # Create a new Toplevel window for input

Label(input_dialog, text="Number of inner corners along the x-axis").grid(row=0, sticky=W)
Label(input_dialog, text="Number of inner corners along the y-axis").grid(row=1, sticky=W)
Label(input_dialog, text="Camera framerate").grid(row=2, sticky=W)

nx = Entry(input_dialog)
ny = Entry(input_dialog)
camera_framerate = Entry(input_dialog)
nx.grid(row=0, column=1)
ny.grid(row=1, column=1)
camera_framerate.grid(row=2, column=1)

def getInput():
    a = int(nx.get())
    b = int(ny.get())
    c = float(camera_framerate.get())
    input_dialog.destroy()  # Close the input dialog
    global params
    params = [a, b, c]

Button(input_dialog, text="submit", command=getInput).grid(sticky=W)

input_dialog.wait_window(input_dialog)  # Wait for the input dialog to close
nx, ny , camera_framerate = params
# print('params: ', params)

# # Define the number of INNER corners along the x and y axes in the calibration checkerboard
# nx = 6    # number of inner corners along x-axis
# ny = 8    # number of inner corners along y-axis

# Path to the video file
# video_path = r"C:\Users\...."

# camera_framerate = 30 # frame rate of camera

# target_frame = 1
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
    if frame_count % (camera_framerate // 2) == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        found = True
        # If corners are found, add object points and image points
        if ret:
            object_points.append(objp)
            image_points.append(corners)

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
        else:
            processed_frame_count += 1
####################################
        # # Draw circles around the corners
        # corners = np.int_(corners)
        # for corner in corners:
        #     x, y = corner.ravel()
        #     cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # # Create a resizable window
        # cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        # # Display the frame
        # cv2.imshow("Video", frame)

        # # Increment the processed frame counter
        # processed_frame_count += 1

    # Wait for a key press to exit
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video capture object
cap.release()
# Destroy any remaining OpenCV windows
cv2.destroyAllWindows()

# # Print the number of frames processed
# print(f"Processed frames: {processed_frame_count}")

# Calibrate the camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, gray.shape[::-1], None, None
)
# print('The camera matrix and distortion coefficients will be printed. Save these values!')
# # Print the camera matrix and distortion coefficients
# print("Camera matrix:\n", camera_matrix)
# print(type(camera_matrix))
# print("Distortion coefficients:\n", dist_coeffs)
# print(type(dist_coeffs))

from tkinter import filedialog

def close_window():
    input_dialog.destroy()

input_dialog = Toplevel(root)
Label(input_dialog, text="Next, users will pick the file location and name for the .txt file that will hold the camera's matrix and distortion coefficients.").grid(row=0, sticky=W)
ok_button = Button(input_dialog, text="OK", command=close_window)
ok_button.grid(row=2, pady=10)
input_dialog.wait_window(input_dialog)

full_path = filedialog.asksaveasfilename(initialfile="", defaultextension=".txt", title='Pick directory and input file name for text file that will contain camera matrices')
print(full_path)

# send camera matrix and distortion coefficients to text file
import numpy as np

# Combine arrays into a single 1D array
# combined_array = np.concatenate([camera_matrix.flatten(), dist_coeffs])
combined_array = np.concatenate([camera_matrix.flatten(), dist_coeffs.flatten()])


# Save the combined array to the text file
np.savetxt(full_path, combined_array, delimiter=',', fmt='%0.8f')

# # Load the array back for verification
# loaded_data = np.loadtxt(full_path, delimiter=',')
# loaded_camera_matrix = loaded_data[:9].reshape(3, 3)
# loaded_dist_coeffs = loaded_data[9:]

# # Print loaded arrays for verification
# print("Loaded Camera Matrix:\n", loaded_camera_matrix)
# print("Loaded Distortion Coefficients:", loaded_dist_coeffs)


# --------

# Declare the global variable
global target_frame
global see_undistorted
target_frame = 1

see_undistorted = True

def on_no():
    global see_undistorted
    see_undistorted = False
    step3.destroy()  # Close the window

def get_number():
    global number
    number = int(number_entry.get())
    # print(f"You entered the number: {number}")
    step3.destroy()  # Close the window

# Create the main window
step3 = Toplevel(root)
step3.title("See undistorted frame decision and desired frame")

step3.geometry("600x300")

Label(step3, text="Click 'No' if you do not want to see the undistorted frame:").grid(row=2, column=0, columnspan=2, pady=10)
no_button = Button(step3, text="No", command=on_no)
no_button.grid(row=3)

Label(step3, text="Else, enter the desired frame from the calibration video:").grid(row=4, column=0, columnspan=2, pady=10)
number_entry = Entry(step3)
number_entry.insert(0, '1')
number_entry.grid(row=5, column=0, columnspan=2, pady=10)
number_button = Button(step3, text="Submit Number and Close", command=get_number)
number_button.grid(row=6, column=0, columnspan=2, pady=10)

# Start the main event loop
step3.wait_window(step3)

# Print the value of the global variable after the window is closed
# print('decision: ', see_undistorted)


# Optional section. 
if see_undistorted == True:
    # print(number)
    # target_frame = 1

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # target_frame = 
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)  # -1 to account for 0-based indexing

    # Read the first frame
    ret, image = cap.read()

    cap.release()

    height, width = image.shape[:2]

    # Calculate the optimal camera matrix
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))

    # Undistort the image
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix) 

    # Find the corners of the checkerboard
    gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # Create a resizable window
    cv2.namedWindow('Undistorted Image. The corners should be especially altered from the original image.', cv2.WINDOW_NORMAL)

    # Display the undistorted image
    cv2.imshow("Undistorted Image. The corners should be especially altered from the original image.", undistorted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()