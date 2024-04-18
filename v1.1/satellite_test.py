from simMk3_1 import simMk3
import numpy as np
import matplotlib.pyplot as plt

# satellite body parameters
mass = 10
inertia = np.array([10, 11, 12])
dcm = np.identity(3)
pos = np.empty([1, 3])
vel = np.empty([1, 3])
ang_vel = np.array([1, 3, 2])
hw = np.array([0, 0, 0])

# rxn wheel stuff
L = np.identity(3)  # orthogonal reaction wheels
Iw = np.identity(3) * 0.5  # all reaction wheels have an inertia of 0.5

# define satellite
rb1 = simMk3.satellite(mass, inertia, dcm, pos, vel, ang_vel, hw)

# define torques and forces
Ntorque = np.array([0, 0, 0])
Btorque = np.array([0, 1, 0])
Nforce = np.array([0, 0, 0])
Bforce = np.array([0, 0, 0])

Tc_rxn = np.array([0, 1, 0])

# simulate rigid body
dt = 1 / 250
steps = 10000

# save angular velocity data of body and wheels
ang_vel_th = np.empty([steps, 3])
ang_vel_rxn_th = np.empty([steps, 3])
tvec = np.empty([steps, 1])
error_track = np.empty([steps, 1])

# simulate
t = 0
for i in range(steps):
    # step
    rb1.step(dt, Nforce, Bforce, Ntorque, Btorque, Tc_rxn)

    # save data
    ang_vel_th[i, :] = rb1.ang_vel
    ang_vel_rxn_th[i, :] = rb1.getWheelVel(L, Iw)
    t = t + dt
    tvec[i] = t
    error_track[i] = np.matmul(rb1.dcm, np.transpose(rb1.dcm))[0, 0]

# plot angular velocities
plt.plot(tvec, ang_vel_th[:, 0], label="w1")
plt.plot(tvec, ang_vel_th[:, 1], label="w2")
plt.plot(tvec, ang_vel_th[:, 2], label="w3")
plt.legend()

# plot reaction wheel angular velocities
plt.figure()
plt.plot(tvec, ang_vel_rxn_th[:, 0], label="w1 rxn")
plt.plot(tvec, ang_vel_rxn_th[:, 1], label="w2 rxn")
plt.plot(tvec, ang_vel_rxn_th[:, 2], label="w3 rxn")
plt.legend()

# error tracker
plt.figure()
plt.plot(tvec, error_track)

plt.show()
