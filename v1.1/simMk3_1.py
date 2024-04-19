import numpy as np


class simMk3:

    # rigid body class. simulation of any object without internal moving parts.
    class rigidbody:
        # rigid body constructor
        def __init__(self, mass, inertia, dcm, pos, vel, ang_vel):
            # properties, dynamic and kinematic variables
            self.mass = mass  # kg
            self.inertia = inertia  # principal inertia vector (diagonal) kg-m^2 (3x1)
            self.dcm = dcm  # unitless, COLUMN (3x3), transforms from frame N to B
            self.pos = pos  # m (1x3)
            self.vel = vel  # m/s (1x3)
            self.ang_vel = ang_vel  # rad/s

        # rigid body dynamics model
        def step(self, stepsize, NForce, BForce, NTorque, BTorque):
            # calculate net force in N, and net torque in B
            fnetN = np.add(np.matmul(BForce, np.transpose(self.dcm)), NForce)
            tnetB = np.add(np.matmul(NTorque, self.dcm), BTorque)

            # dstate sun-function. Calculates the derivative of every state.
            # returns list of
            """
            0: d/dt[x,y,z] (1x3 vector)
            1: d/dt[vx,vy,vz] (1x3 vector)
            2: d/dt[nCb] (3x3 matrix)
            3: d/dt[w1,w2,w3] (1x3 vector)

            Therefore, the state vector is
            [position,velocity,dcm,angular velocity]
            """

            def dstate(state_list, fnetN, tnetB, mass, inertia):
                r_dot = state_list[1]
                v_dot = np.matmul(fnetN, (1 / mass * np.identity(3)))
                poisson_matrix = np.array(
                    [
                        [0, -state_list[3][2], state_list[3][1]],
                        [state_list[3][2], 0, -state_list[3][0]],
                        [-state_list[3][1], state_list[3][0], 0],
                    ]
                )
                dcm_dot = np.matmul(state_list[2], poisson_matrix)

                w1_dot = (
                    state_list[3][1] * state_list[3][2] * (inertia[1] - inertia[2])
                    + tnetB[0]
                ) / inertia[0]
                w2_dot = (
                    state_list[3][0] * state_list[3][2] * (inertia[2] - inertia[0])
                    + tnetB[1]
                ) / inertia[1]
                w3_dot = (
                    state_list[3][0] * state_list[3][1] * (inertia[0] - inertia[1])
                    + tnetB[2]
                ) / inertia[2]
                return [r_dot, v_dot, dcm_dot, np.array([w1_dot, w2_dot, w3_dot])]

            # apply RK4 algorithm to calculate state at next time step
            # unpack state list
            state = [self.pos, self.vel, self.dcm, self.ang_vel]

            # k1
            k1 = dstate(state, fnetN, tnetB, self.mass, self.inertia)

            # k2
            b_state = [0, 0, 0, 0]
            for i in range(4):
                b_state[i] = state[i] + stepsize / 2 * k1[i]
            k2 = dstate(b_state, fnetN, tnetB, self.mass, self.inertia)

            # k3
            c_state = [0, 0, 0, 0]
            for i in range(4):
                c_state[i] = state[i] + stepsize / 2 * k2[i]
            k3 = dstate(c_state, fnetN, tnetB, self.mass, self.inertia)

            # k4
            d_state = [0, 0, 0, 0]
            for i in range(4):
                d_state[i] = state[i] + stepsize * k3[i]
            k4 = dstate(d_state, fnetN, tnetB, self.mass, self.inertia)

            # finally, obtain iteration of state
            self.pos = self.pos + stepsize / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            self.vel = self.vel + stepsize / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            self.dcm = self.dcm + stepsize / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            self.ang_vel = self.ang_vel + stepsize / 6 * (
                k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]
            )

    # satellite class. very similar to rigid body, with reaction wheels!
    class satellite:
        # satellite constructor
        def __init__(self, mass, inertia, dcm, pos, vel, ang_vel, hw):
            # rigid body stuff
            self.mass = mass  # kg
            self.inertia = inertia  # principal inertia vector (diagonal) kg-m^2 (3x1)
            self.dcm = dcm  # unitless, COLUMN (3x3), transforms from frame N to B
            self.pos = pos  # m (1x3)
            self.vel = vel  # m/s (1x3)
            self.ang_vel = ang_vel  # rad/s (1x3)

            # reaction wheel stuff
            self.hw = hw  # reaction wheel angular momentum

        # satellite dynamics model
        def step(self, stepsize, NForce, BForce, NTorque, BTorque, Tc_rxn):
            # calculate net force in N, and net torque in B
            fnetN = np.add(np.matmul(BForce, np.transpose(self.dcm)), NForce)
            tnetB = np.add(np.matmul(NTorque, self.dcm), BTorque)

            # dstate sun-function. Calculates the derivative of every state.
            # returns list of
            """
            0: d/dt[x,y,z] (1x3 vector)
            1: d/dt[vx,vy,vz] (1x3 vector)
            2: d/dt[nCb] (3x3 matrix)
            3: d/dt[w1,w2,w3] (1x3 vector)

            Therefore, the state vector is
            [position,velocity,dcm,ang_vel,hw]
            """

            def dstate(
                state_list,
                fnetN,
                tnetB,
                mass,
                inertia,
                Tc_rxn,
            ):
                # unpack state vector
                vel = state_list[1]
                dcm = state_list[2]
                ang_vel = state_list[3]
                hw = state_list[4]

                # calculate derivatives

                # dpos
                r_dot = vel

                # dvel
                v_dot = np.matmul(fnetN, (1 / mass * np.identity(3)))

                # ddcm
                poisson_matrix = np.array(
                    [
                        [0, -ang_vel[2], ang_vel[1]],
                        [ang_vel[2], 0, -ang_vel[0]],
                        [-ang_vel[1], ang_vel[0], 0],
                    ]
                )
                dcm_dot = np.matmul(dcm, poisson_matrix)

                inertia_mat = np.array(
                    [[inertia[0], 0, 0], [0, inertia[1], 0], [0, 0, inertia[2]]]
                )

                # dw
                w_dot = np.matmul(
                    np.linalg.inv(inertia_mat),
                    (
                        np.cross(-ang_vel, (np.matmul(inertia_mat, ang_vel) + hw))
                        + tnetB
                        + Tc_rxn
                    ),
                )

                # dhw
                d_hw = -Tc_rxn
                return [r_dot, v_dot, dcm_dot, w_dot, d_hw]

            # apply RK4 algorithm to calculate state at next time step
            # unpack state list
            state = [self.pos, self.vel, self.dcm, self.ang_vel, self.hw]

            # k1
            k1 = dstate(state, fnetN, tnetB, self.mass, self.inertia, Tc_rxn)

            # k2
            b_state = [0, 0, 0, 0, 0]
            for i in range(5):
                b_state[i] = state[i] + stepsize / 2 * k1[i]
            k2 = dstate(b_state, fnetN, tnetB, self.mass, self.inertia, Tc_rxn)

            # k3
            c_state = [0, 0, 0, 0, 0]
            for i in range(5):
                c_state[i] = state[i] + stepsize / 2 * k2[i]
            k3 = dstate(c_state, fnetN, tnetB, self.mass, self.inertia, Tc_rxn)

            # k4
            d_state = [0, 0, 0, 0, 0]
            for i in range(5):
                d_state[i] = state[i] + stepsize * k3[i]
            k4 = dstate(d_state, fnetN, tnetB, self.mass, self.inertia, Tc_rxn)

            # finally, obtain iteration of state
            self.pos = self.pos + stepsize / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            self.vel = self.vel + stepsize / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            self.dcm = self.dcm + stepsize / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            self.ang_vel = self.ang_vel + stepsize / 6 * (
                k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]
            )
            self.hw = self.hw + stepsize / 6 * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])

        # This get method returns the angular velocity of the reaction wheels at any
        # given time.
        # L: configuration matrix (3 x n) describes how the wheels are oriented
        # Iw: inertia of wheels (n x n)
        def getWheelVel(self, L, Iw):
            return np.matmul(np.linalg.inv(np.matmul(L, Iw)), self.hw)

    class PID:
        def __init__(self, kp, ki, kd):
            self.kp = kp
            self.ki = ki
            self.kd = kd

            self.integral = 0
            self.prev_er = 0

        def getSignal(self, err, dt):
            # compute signal
            signal = (
                self.kp * err + self.ki * self.integral + self.kd * (err - self.prev_er)
            )

            # update integral and previous error
            self.integral = self.integral + err * dt
            self.prev_er = err

            # return signal
            return signal
