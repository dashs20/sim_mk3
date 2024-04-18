from simMk3 import rigidbody
import numpy as np
import matplotlib.pyplot as plt

"""
Example Rigid Body Simulation
"""
# ~~~~~~~~~~~~~~~~~~~~
# definition of object
# ~~~~~~~~~~~~~~~~~~~~

# physical properties
mass = 1
inertia = np.array([4, 2, 6])
# kinematic variables
dcm = np.identity(3)
pos = np.array([0, 0, 0])
# dynamic variables
vel = np.array([0, 0, 0])
ang_vel = np.array([5, 0.01, 0.01])

# define rigid body
prism = rigidbody(mass, inertia, dcm, pos, vel, ang_vel)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# define simulation parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# define some forces and torques
inertial_force = np.array([0, 0, 0])  # gravity
body_force = np.array([0, 0, 0])  # thruster, maybe
inertial_torque = np.array([0, 0, 0])  # no inertial torques
body_torque = np.array([0, 0, 0])  # reaction wheel, maybe

# define max sim time
tmax = 50
dt = 1 / 500  # dt is 1/500 of a second

# open visualization window to render 3D prism
prism.setDarkMode(False)
prism.openWindow()

# ~~~~~~~~~
# Simulate!
# ~~~~~~~~~

# angular velocity
steps = int(tmax / dt)

ang_vel_th = np.empty([3, steps])
tvec = np.empty([steps])

t = 0
for i in range(int(tmax / dt)):
    prism.step(dt, inertial_force, body_force, inertial_torque, body_torque)
    ang_vel_th[:, i] = prism.ang_vel
    tvec[i] = t
    prism.updateVisuals(dt, t)
    t = t + dt

# ~~~~~~~~~
# Plot
# ~~~~~~~~~

plt.plot(tvec, ang_vel_th[0, :], label="w1")
plt.plot(tvec, ang_vel_th[1, :], label="w2")
plt.plot(tvec, ang_vel_th[2, :], label="w3")
plt.legend()
plt.show()
