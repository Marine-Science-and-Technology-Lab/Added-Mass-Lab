import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

'''
Fitting data to harmonic curve. The equation to be fit can be seen in 'damped_oscillation()' 
For the 'data' variable, users should input the directory location of the .txt file where the data is located. 

If the curve fit is not working properly, one workaround may be to change the initial values. 

This code will output the values fitted from the damped oscillation equations. A1 and A3 will be used in data reduction to compute the added mass, so save these for later.
'''

# User input:

# experimental data, assuming a text file with columns time and position 
data = np.loadtxt(r"C:\Users\...", skiprows=1)  # skiprows used assuming the first row contains headers

# Initial values
A0_init = 14
A1_init = .05
A2_init = -28
A3_init = 12
A4_init = -1

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

time_data = data[:, 0]
displacement_data = data[:,1]

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