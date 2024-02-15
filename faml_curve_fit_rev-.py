import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import Label, Toplevel, Entry, Button
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
# pyinstaller --onefile faml_curve_fit_rev-.py
'''
Fitting data to harmonic curve. The equation to be fit can be seen in 'damped_oscillation()' 
For the 'data' variable, users should input the directory location of the .txt file where the data is located. 

If the curve fit is not working properly, one workaround may be to change the initial values. 

This code will output the values fitted from the damped oscillation equations. A1 and A3 will be used in data reduction to compute the added mass, so save these for later.
'''

root = Tk()
root.withdraw()  # Hide the main window

def close_window():
    input_dialog.destroy()

input_dialog = Toplevel(root)

Label(input_dialog, text="Position Curve Fitting", font=("Arial", 16)).grid(row=0)
Label(input_dialog, text="This program will fit the position curve to a damped harmonic curve. Users will be prompted to select the .txt file that has the position data and then").grid(row=1, sticky=W)
Label(input_dialog, text="to pick a directory to create a .txt file to store the coefficients of the fitted curve. The R^2 value will also be given. A value close to 1 indicates a closely fit curve.").grid(row=2, sticky=W)
Label(input_dialog, text="The initial coefficient guesses below are used to help fit the curve. Some values are defaulted in, but you may have to change the values").grid(row=3, sticky=W)
Label(input_dialog, text="if the fitted curve is not accurate.").grid(row=4, sticky=W)
Label(input_dialog, text="A0").grid(row=5, sticky=W)
Label(input_dialog, text="A1").grid(row=6, sticky=W)
Label(input_dialog, text="A2").grid(row=7, sticky=W)
Label(input_dialog, text="A3").grid(row=8, sticky=W)
Label(input_dialog, text="A4").grid(row=9, sticky=W)

A0 = Entry(input_dialog); A0.insert(0,'14')
A1 = Entry(input_dialog); A1.insert(0,'.05')
A2 = Entry(input_dialog); A2.insert(0,'-28')
A3 = Entry(input_dialog); A3.insert(0,'12')
A4 = Entry(input_dialog); A4.insert(0,'-1')

A0.grid(row=5, column=1)
A1.grid(row=6, column=1)
A2.grid(row=7, column=1)
A3.grid(row=8, column=1)
A4.grid(row=9, column=1)

params=[]
def getInput():
    a = float(A0.get()) # sending initial guesses for coefficients to float 
    b = float(A1.get()); c = float(A2.get()); d = float(A3.get()); e = float(A4.get())
    input_dialog.destroy()  # Close the input dialog
    global params
    params = [a, b, c, d, e]

Button(input_dialog, text="submit", command=getInput).grid(sticky=W)

# input_dialog.wait_window(input_dialog)  # Wait for the input dialog to close

input_dialog.wait_window(input_dialog)

A0_init, A1_init, A2_init, A3_init, A4_init = params
print(A0_init, A1_init, A2_init, A3_init, A4_init)

# prompt user to pick .txt file with position data
full_path = askopenfilename(filetypes=[("Text Files", "*.txt")],title='Pick text file with position data.') # using explorer to open window

#prompt user to pick directory and file name to send coefficients from curve fitting
input_dialog = Toplevel(root)
Label(input_dialog, text="Next, users will pick the file location and name for the .txt file that will contain the fitted curve's coefficients. Do no add .txt in the file name or pick a file type as this will be done automatically.").grid(row=0, sticky=W)
ok_button = Button(input_dialog, text="OK", command=close_window)
ok_button.grid(row=2, pady=10)
input_dialog.wait_window(input_dialog)

full_path2 = filedialog.asksaveasfilename(initialfile="", defaultextension=".txt", title="Pick directory and input file name for text file that will contain fitted curve's coefficients.")

# User input:

# experimental data, assuming a text file with columns time and position 
data = np.loadtxt(full_path, skiprows=1)  # skiprows used assuming the first row contains headers

# # Initial values
# A0_init = 14
# A1_init = .05
# A2_init = -28
# A3_init = 12
# A4_init = -1

# ---

def damped_oscillation(t, A0, A1, A2, A3, A4):

    # A0 = initial displacement or offset from equilibrium position
    # A1 = damping, determines rate of decay
    # A2 = amplitude of oscillations
    # A3 = damped natural frequency, w_n
    # A4 = phase shift

    return A0 + np.exp(-A1 * t) * A2 * np.cos(A3 * t + A4)

def r_squared(f, xdata, ydata):
    # The curve fit function from scipy used does not have an r-squared readily available. This function serves to do that.
    # Adapted from https://stackoverflow.com/a/37899817
    popt, pcov = curve_fit(f, xdata, ydata, p0=initial_guess)
    residuals = ydata- f(xdata, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared

# -----------------------------

#prompt user to pick directory and file name to send coefficients from curve fitting
input_dialog = Toplevel(root)
Label(input_dialog, text="Next, users will be shown the imported data. It is important that the position points at the start where the shape is idling").grid(row=0, sticky=W)
Label(input_dialog, text="are not kept as this will lead to a poor fit. Users should click the point where this idling ends and the oscillation begins so that all points prior can be removed.").grid(row=1, sticky=W)
ok_button = Button(input_dialog, text="OK", command=close_window)
ok_button.grid(row=3, pady=10)
input_dialog.wait_window(input_dialog)

x = data[:, 0]
y = data[:,1]
fig,ax = plt.subplots()
plt.scatter(x, y)
points = list(zip(x,y))
point = 0
circle = None
def distance(a,b): # calculate distance between two points
    return(sum([(k[0]-k[1])**2 for k in zip(a,b)])**0.5)
def update_circle(position): # draw circle by where user clicks
    global circle
    if circle is not None:
        circle.remove()
    circle = plt.Circle(position, .25, color='red', fill=True)
    ax.add_patch(circle)
    plt.draw()
def onclick(event):
    global point
    dists = [distance([event.xdata, event.ydata],k) for k in points]
    closest_point = points[dists.index(min(dists))]
    update_circle(closest_point)
    point = closest_point

    print(point)
fig.canvas.mpl_connect('button_press_event', onclick)

plt.title('Select the point where the shape stopped idling and began its oscillation, then exit out of the window.')
plt.show()

if point == 0:
    time_data = x
    displacement_data = y
else:
    pair = list(zip(x, y))
    index = pair.index(point)
    xpoint = point[0]
    time_data = x[index+1:] 
    time_data = [i - xpoint for i in time_data]
    time_data = np.array(time_data)
    displacement_data = y[index+1:]

# ------------------------------
# time_data = data[:, 0]
# displacement_data = data[:,1]

initial_guess = [A0_init, A1_init, A2_init, A3_init, A4_init]
fit_params, _ = curve_fit(damped_oscillation, time_data, displacement_data, p0=initial_guess)

A0_fit, A1_fit, A2_fit, A3_fit, A4_fit = fit_params

i=0
for j in fit_params:
    print(f'A{i} = {fit_params[i]}')
    i+=1

# Generate time points for plotting the function
plot_time = np.linspace(min(time_data), max(time_data), 1000)

# Calculate the fitted displacement values using the damped_oscillation function
fitted_displacement = damped_oscillation(plot_time, A0_fit, A1_fit, A2_fit, A3_fit, A4_fit)

print('R^2 from function', r_squared(damped_oscillation, time_data, displacement_data))

# plot data and fitted curve
plt.figure(figsize=(10, 6))
plt.scatter(time_data, displacement_data, label='Data')
plt.plot(plot_time, fitted_displacement, label='Fitted Function', color='red')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Fitted Damped Oscillation')
plt.legend()
plt.grid()
plt.show()

with open(full_path2, 'w') as file:
    file.write(f"R^2 value: {r_squared(damped_oscillation, time_data, displacement_data):.8f}\n")

    for i in range(len(fit_params)):
        file.write(f"A{i}: {fit_params[i]:.8f}\n")

print("Data saved successfully.")