import serial
import time
from tkinter import *
from tkinter import Label, Toplevel, Entry, Button

# pick COM port number

root = Tk()
root.withdraw()  # Hide the main window

def close_window():
    input_dialog.destroy()

input_dialog = Toplevel(root)

Label(input_dialog, text="This code will interface with the Arduino/ultrasonic sensor to print the time and position data to a .txt file.").grid(row=0, sticky=W)
Label(input_dialog, text="In this window, users will input the COM port number that corresponds to their Arduino.").grid(row=1, sticky=W)
Label(input_dialog, text="This can be found by going into the Arduino app, then, at the top click").grid(row=1, sticky=W)
Label(input_dialog, text="Tools -> Port (also see the procedure PowerPoint step 3)").grid(row=2, sticky=W)
Label(input_dialog, text="The COM# should be given. Please input it below.").grid(row=3, sticky=W)
Label(input_dialog, text="COM#: ").grid(row=4, sticky=W)
Label(input_dialog, text="IMPORTANT: to stop the program while the position is being tracked, exit out of the pop-up window.").grid(row=5, sticky=W)

nx = Entry(input_dialog)
nx.grid(row=4, column=1)

def getInput():
    global COM
    COM = str(nx.get())
    input_dialog.destroy()  # Close the input dialog

Button(input_dialog, text="submit", command=getInput).grid(sticky=W)

input_dialog.wait_window(input_dialog)  # Wait for the input dialog to close
COM = 'COM'+ COM

# selecting directory for .txt file
from tkinter import filedialog
input_dialog = Toplevel(root)
Label(input_dialog, text="Next, users will pick the file location and name for the .txt file that will hold the position data.").grid(row=0, sticky=W)
ok_button = Button(input_dialog, text="OK", command=close_window)
ok_button.grid(row=2, pady=10)
input_dialog.wait_window(input_dialog)

full_path = filedialog.asksaveasfilename(initialfile="", defaultextension=".txt", title='Pick directory and input file name for text file that will contain position data')
print(full_path)

input_dialog = Toplevel(root)  # Create a new Toplevel window for input

ser = serial.Serial(COM, 9600, timeout=1)  # Change 'COMX' to your Arduino's port

# Wait for the serial connection to initialize
time.sleep(2)

filename = full_path

with open(filename, 'w') as file:
    # file.write("time(s) distance(mm)\n")
    try:
        while True:
            line = ser.readline().decode().strip()  # Read a line from serial
            if line and "Out of range" not in line:
                file.write(line + '\n')  # Write data to file if it's not "Out of range"
                file.flush()  # Ensure data is written to the file immediately
                print(line)  # Optionally, print data to console
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        # Ensure the serial connection is properly closed
        ser.close()




