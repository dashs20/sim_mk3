from simMk3_1 import simMk3
import numpy as np
import matplotlib.pyplot as plt

# rigid body parameters
mass = 10
inertia = np.array([10, 11, 12])
dcm = np.identity(3)
pos = np.empty([1, 3])
vel = np.empty([1, 3])
ang_vel = np.array([1, 2, 3])

# define rigid body
rb1 = simMk3.rigidbody(mass, inertia, dcm, pos, vel, ang_vel)

# define torques and forces
Ntorque = np.array([0, 0, 0])
Btorque = np.array([0, 0, 0])
Nforce = np.array([0, 0, 0])
Bforce = np.array([0, 1, 0])

# simulate rigid body for 500 time steps
dt = 1 / 500
steps = 20000

# save angular velocity data
ang_vel_th = np.empty([steps, 3])
tvec = np.empty([steps, 1])

# simulate
t = 0
for i in range(steps):
    # step
    rb1.step(dt, Nforce, Bforce, Ntorque, Btorque)

    # save data
    ang_vel_th[i, :] = rb1.ang_vel
    t = t + dt
    tvec[i] = t

# plot angular velocities
plt.plot(tvec, ang_vel_th[:, 0], label="w1")
plt.plot(tvec, ang_vel_th[:, 1], label="w2")
plt.plot(tvec, ang_vel_th[:, 2], label="w3")
plt.legend()
plt.show()
