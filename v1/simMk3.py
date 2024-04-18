import numpy as np
import math
import pygame

"""
####################################################################################
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ External Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
####################################################################################
"""


def calcVert(dcm, lengths):
    xLength = lengths[0]
    yLength = lengths[1]
    zLength = lengths[2]

    vertices = np.array(
        [
            [-xLength, -yLength, zLength],
            [xLength, -yLength, zLength],
            [xLength, yLength, zLength],
            [-xLength, yLength, zLength],
            [-xLength, -yLength, -zLength],
            [xLength, -yLength, -zLength],
            [xLength, yLength, -zLength],
            [-xLength, yLength, -zLength],
        ],
    )

    # rotate vertices
    for i in range(8):
        vertices[i, :] = np.matmul(vertices[i, :], dcm)
    return vertices


"""
####################################################################################
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class Definition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
####################################################################################
"""

class rigidbody:

    def __init__(self, mass, inertia, dcm, pos, vel, ang_vel):
        # properties, dynamic and kinematic variables
        self.mass = mass  # kg
        self.inertia = inertia  # principal inertia vector (diagonal) kg-m^2 (3x1)
        self.dcm = dcm  # unitless, COLUMN (3x3), transforms from frame N to B
        self.pos = pos  # m (1x3)
        self.vel = vel  # m/s (1x3)
        self.ang_vel = ang_vel  # rad/s
        self.verbose = True
        self.darkMode = True

        # visualization variables
        self.geometric_scale = (
            np.max(self.inertia) * 50
        )  # default scale factor, can be changed
        self.lengths = (
            np.array(
                [1 / self.inertia[0], 1 / self.inertia[1], 1 / self.inertia[2]]
            )
            * self.geometric_scale
        )
        self.vertices = calcVert(self.dcm, self.lengths)

    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dynamics Code ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

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

        # update location of prism vertices
        self.vertices = calcVert(self.dcm, self.lengths)

    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Set Methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    def setScale(self, val):
        self.geometric_scale = val
        self.lengths = (
            np.array(
                [1 / self.inertia[0], 1 / self.inertia[1], 1 / self.inertia[2]]
            )
            * val
        )

    def setVerbose(self, val):
        self.verbose = val

    def setDarkMode(self, val):
        self.darkMode = val

    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Get Methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    def getEuler(self):
        dcm = self.dcm

        def angleCheck(angles):
            i = 0
            for angle in angles:
                if angle < 0:
                    angles[i] = 2 * np.pi + angles[i]
                i = i + 1
            return angles

        # calculate nutation. However, precession and spin are a bit trickier
        # define guess arrays
        precession_1 = [0] * 2
        precession_2 = [0] * 2
        spin_1 = [0] * 2
        spin_2 = [0] * 2
        # calculate nutation
        nutation = np.arccos(dcm[2, 2])
        # calculate precession guesses
        precession_1[0] = np.arcsin(dcm[0, 2] / np.sin(nutation))
        precession_1[1] = np.pi - precession_1[0]
        precession_2[0] = np.arccos(dcm[1, 2] / (-np.sin(nutation)))
        precession_2[1] = 2 * np.pi - precession_2[0]
        # calculate spin guesses
        spin_1[0] = np.arcsin(dcm[2, 0] / np.sin(nutation))
        spin_1[1] = np.pi - spin_1[0]
        spin_2[0] = np.arccos(dcm[2, 1] / np.sin(nutation))
        spin_2[1] = 2 * np.pi - spin_2[0]
        # if any angles are less than 0, correct them
        precession_1 = angleCheck(precession_1)
        precession_2 = angleCheck(precession_2)
        spin_1 = angleCheck(spin_1)
        spin_2 = angleCheck(spin_2)

        # check for singularity (common with Euler angles)
        # if no singularity, apply quadrant checking algorithm
        if (
            math.isnan(precession_1[0])
            or math.isnan(precession_2[0])
            or math.isnan(precession_1[1])
            or math.isnan(precession_2[1])
        ):
            precession = 0
        else:
            guessvec = [
                abs(precession_1[0] - precession_2[0]),
                abs(precession_1[0] - precession_2[1]),
                abs(precession_1[1] - precession_2[0]),
                abs(precession_1[1] - precession_2[1]),
            ]
            ind = np.argmin(guessvec)
            if ind == 0 or ind == 1:
                precession = precession_1[0]
            if ind == 2 or ind == 3:
                precession = precession_1[1]
        if (
            math.isnan(spin_1[0])
            or math.isnan(spin_2[0])
            or math.isnan(spin_1[1])
            or math.isnan(spin_2[1])
        ):
            spin = 0
        else:
            guessvec = [
                abs(spin_1[0] - spin_2[0]),
                abs(spin_1[0] - spin_2[1]),
                abs(spin_1[1] - spin_2[0]),
                abs(spin_1[1] - spin_2[1]),
            ]
            ind = np.argmin(guessvec)
            if ind == 0 or ind == 1:
                spin = spin_1[0]
            if ind == 2 or ind == 3:
                spin = spin_1[1]
        return np.array([precession, nutation, spin])

    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Visualization Code ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    def openWindow(self):
        pygame.init()
        if self.darkMode:
            self.col_a = (0, 0, 0)
            self.col_b = (255, 255, 255)
        else:
            self.col_a = (255, 255, 255)
            self.col_b = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        WIDTH, HEIGHT = 800, 600
        pygame.display.set_caption("Sim MK3 Visualizer")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        # Text
        self.font = pygame.font.SysFont("Arial", 20)

    def updateVisuals(self, dt, t):
        # regulate visualization rate
        self.clock.tick(int(1 / dt))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()

        # connect points subfunction - draws prism vertices
        def connect_points(i, j, points):
            pygame.draw.line(
                self.screen,
                self.col_b,
                (points[i, 0], points[i, 1]),
                (points[j, 0], points[j, 1]),
            )

        # project points, and move them to center of screen
        projection_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])

        # define camera rotation matrix
        cam_dcm = np.array(
            [
                [-0.70711, -0.35355, 0.61237],
                [0.70711, -0.35355, 0.61237],
                [0, 0.86603, 0.5],
            ]
        )

        # calculate coordinates of projected vertices
        projected_vertices = np.empty([8, 3])
        for i in range(8):
            projected_vertices[i, :] = np.matmul(
                np.matmul(self.vertices[i, :], cam_dcm), projection_matrix
            ) + np.array([400, 300, 0])

        # fill background with col_b
        self.screen.fill(self.col_a)

        # draw XYZ axes
        xyzAxis = np.identity(3) * 300
        xyzProjected = np.empty([3, 3])
        for i in range(3):
            xyzProjected[i, :] = np.matmul(
                np.matmul(xyzAxis[i, :], cam_dcm), projection_matrix
            )

        pygame.draw.line(
            self.screen,
            self.RED,
            (400, 300),
            (400 + xyzProjected[0, 0], 300 + xyzProjected[0, 1]),
        )
        pygame.draw.line(
            self.screen,
            self.GREEN,
            (400, 300),
            (400 + xyzProjected[1, 0], 300 + xyzProjected[1, 1]),
        )
        pygame.draw.line(
            self.screen,
            self.BLUE,
            (400, 300),
            (400 + xyzProjected[2, 0], 300 + xyzProjected[2, 1]),
        )

        # draw prism edges in col_b
        for p in range(4):
            connect_points(p, (p + 1) % 4, projected_vertices)
            connect_points(p + 4, ((p + 1) % 4) + 4, projected_vertices)
            connect_points(p, (p + 4), projected_vertices)

        # text
        if self.verbose:
            euler_ang = self.getEuler()

            disp_strings = [
                "Translational Info:",
                "x: " + str(round(self.pos[0], 1)) + " m",
                "y: " + str(round(self.pos[1], 1)) + " m",
                "z: " + str(round(self.pos[2], 1)) + " m",
                "vx: " + str(round(self.vel[0], 1)) + " m/s",
                "vy: " + str(round(self.vel[1], 1)) + " m/s",
                "vz: " + str(round(self.vel[2], 1)) + " m/s",
                "",
                "Attitude Info:",
                "α: " + str(round(euler_ang[0] * 180 / np.pi, 1)) + " deg",
                "β: " + str(round(euler_ang[1] * 180 / np.pi, 1)) + " deg",
                "γ: " + str(round(euler_ang[2] * 180 / np.pi, 1)) + " deg",
                "ωx: " + str(round(self.ang_vel[0] * 180 / np.pi, 1)) + " deg/s",
                "ωy: " + str(round(self.ang_vel[1] * 180 / np.pi, 1)) + " deg/s",
                "ωz: " + str(round(self.ang_vel[2] * 180 / np.pi, 1)) + " deg/s",
                "",
                "Time: " + str(round(t, 1)) + " s",
            ]

            counter = 0
            for i in disp_strings:
                tmp = self.font.render(i, True, self.col_b)
                self.screen.blit(tmp, (10, 10 + 25 * counter))
                counter = counter + 1

        pygame.display.update()
