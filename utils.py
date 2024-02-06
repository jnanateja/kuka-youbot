import numpy as np
import modern_robotics as mr

def writeCSV(data, name):
    np.savetxt(name, data, delimiter=',')

#Gets the Jacobian Base of the wheels
def JacobianBase(configuration, Tse, r, l, w):
    phi, x, y = configuration[0:3]
    Tsb = np.array([[np.cos(phi), -np.sin(phi), 0, x],
                    [np.sin(phi), np.cos(phi), 0, y],
                    [0, 0, 1, 0.0963],
                    [0, 0, 0, 1]])
    Tes = mr.TransInv(Tse)
    Teb = np.dot(Tes, Tsb)
    adj = mr.Adjoint(Teb)
    F = r/4 * np.array([[-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
                        [1, 1, 1, 1],
                        [-1, 1, -1, 1]])
    F6 = np.insert(F, (0,0,3), 0, axis = 0)
    return np.dot(adj, F6)

#Gets the speed of the parts from the jacobian and the twist
def Speeds(J, V):
    J_inv = np.linalg.pinv(J, rcond = 0.01)
    speeds = np.dot(J_inv, V.T)
    return speeds

def TestJointLimits(past_angles, angles):
    limits = np.array([[-2.932, 2.932],
                       [-1.117, 1.553],
                       [-2.61, 2.53],
                       [-1.78, 1.78],
                       [-2.89, 2.89]])
    for i, angle in enumerate(angles):
        if limits[i][0] > (angle + past_angles[i]):
            angles[i] = 0
        elif  (angle + past_angles[i]) > limits[i][1]:
            angles[i] = 0
        else: 
            angles[i] = 1
    return angles

#Converts the vector configuration to matrix form
def XfromConfiguration(configuration, Tb0, T0e):
    phi, x, y = configuration[0:3]
    Tsb = np.array([[np.cos(phi), -np.sin(phi), 0, x],
                    [np.sin(phi), np.cos(phi), 0, y],
                    [0, 0, 1, 0.0963],
                    [0, 0, 0, 1]]) 
    Tse = np.dot(Tsb, np.dot(Tb0, T0e))
    return Tse

#measures the distance between two points
def Distance(T1, T2):
    return np.linalg.norm(T2[:,3] - T1[:,3])

#Converts a translation to vector-form
def TtoVector(T, gripper_state):
    vector = T[0:3,0:3].flatten()
    vector = np.append(vector, T[:3,3])
    vector = np.append(vector, gripper_state)
    return vector

#Converts a vector to a translation
def VectorToT(vector):
    T = vector[:9].reshape((3,3))
    T = np.append(T, np.expand_dims(vector[9:12], axis = 1), axis=1)
    T = np.append(T, [[0,0,0,1]],axis=0)
    return T