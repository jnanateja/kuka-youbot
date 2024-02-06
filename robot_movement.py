import numpy as np
import math
import scipy.integrate as integrate
import modern_robotics as mr
from matplotlib import pyplot as plt
import utils

#inputs
########################################################################
dt = 0.01

w = 0.15
l = 0.235
r = 0.0475
limits = np.array([-100,100])
Tse_initial = np.array([[0,0,1,0],  #Initial config. end-effector
                        [0,1,0,0],
                        [-1,0,0,0.5],
                        [0,0,0,1]])

Tsc_initial = np.array([[1,0,0,1],  #Initial config. cube
                        [0,1,0,0],
                        [0,0,1,0.025],
                        [0,0,0,1]])

Tsc_final = np.array([[np.cos(-np.pi/2),-np.sin(-np.pi/2),0,0],  #Final config. 
                                                                 #cube
                      [np.sin(-np.pi/2),np.cos(-np.pi/2),0,-1],
                      [0,0,1,0.025],
                      [0,0,0,1]])
#Initial end-effecto relative to cube while grasping
Tce_grasp = np.array([[np.cos(3*np.pi/4),0,np.sin(3*np.pi/4),0],  
                        [0,1,0,0],
                        [-np.sin(3*np.pi/4),0,np.cos(3*np.pi/4),0],
                        [0,0,0,1]])

Tce_standoff = np.array([[0,0,1,0],  #Config standoff above the cube
                        [0,1,0,0],
                        [-1,0,0,0.3],
                        [0,0,0,1]])
k = 1

Tb0 = np.array([[1,0,0,0.1662],
                [0,1,0,0],
                [0,0,1,0.0026],
                [0,0,0,1]])

M0e = np.array([[1,0,0,0.033],
                [0,1,0,0],
                [0,0,1,0.6546],
                [0,0,0,1]])

Blist = np.array([[0,0,1,0,0.033, 0],
                  [0,-1,0,-0.5076,0,0],
                  [0,-1,0,-0.3526,0,0],
                  [0,-1,0,-0.2176,0,0],
                  [0,0,1,0,0,0]]).T

Tsb = np.array([[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0.0963],
                [0,0,0,1]])

Kp = 10*np.eye(6)

Ki = 0.5*np.eye(6)
########################################################################

#Configuration of the robot in every step
def NextState(prev_state, speeds, restrictions, gripper_state, dt):
    #speeds (u1,u2,u3,u4,O1,O2,O3,O4,O5)
    #prev_state = (phi,x,y,O1,O2,O3,O4,O5,u1,u2,u3,u4,gs)
    
    state = prev_state
    #increment that will be added to the configuration
    for speed in speeds: #Check if the speed is within the boundaries
        if speed < restrictions[0]:
            speed = restrictions[0]
        elif speed > restrictions[1]:
            speed = restrictions[1]
            
    d_wheels = speeds[:4]*dt #Delta dist of the wheels
    d_arms = speeds[4:]*dt #Delta O of the arms
    state[3:8] += d_arms
    state[8:-1] += d_wheels
    F = r/4 * np.array([[-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
                         [1, 1, 1, 1],
                         [-1, 1, -1, 1]])
    Vb = np.dot(F, d_wheels)
    Wz = Vb[0]
    Vx = Vb[1]
    Vy = Vb[2]
    #d_qb is the change of the chassis configuration in the body frame
    if Wz != 0:
        d_qb = np.array([Wz,
                        (Vx*np.sin(Wz)+Vy*(np.cos(Wz)-1))/Wz,
                        (Vy*np.sin(Wz)+Vx*(1 - np.cos(Wz)))/Wz])
    else:
        d_qb = np.array([0,Vx,Vy])
    theta_k = prev_state[0]
    #Chassis configuration on the s frame
    dq = np.dot(np.array([[1, 0, 0],
                   [0, np.cos(theta_k), -np.sin(theta_k)],
                   [0, np.sin(theta_k), np.cos(theta_k)]]), d_qb.T)
    state[:3] += dq
    state[-1] = gripper_state
    return np.expand_dims(state, axis=0)

#Generates the reference trajectory that the robot wiill follow
def TrajectoryGenerator():
    #A little bit more that average speed for a robot gripper
    speed = 0.5
    separation = k/dt
    #Getting the different configurations for the end-effector
    standoff_init = np.dot(Tsc_initial, Tce_standoff)
    grasp_init = np.dot(Tsc_initial, Tce_grasp)
    standoff_final = np.dot(Tsc_final, Tce_standoff)
    grasp_final = np.dot(Tsc_final, Tce_grasp)
    #Initializing the gripper stage as "closed"
    gripper_state = 0
    #creating the first configuration as the initial one
    configurations = np.array(np.expand_dims(utils.TtoVector(Tse_initial,
                                                             gripper_state),\
                                                             axis = 0))
    
    #All the different stages of the gripper movement
    movement1 = (Tse_initial, standoff_init)
    movement2 = (standoff_init,grasp_init)
    movement3 = (grasp_init, None)
    movement4 = (grasp_init, standoff_init)
    movement5 = (standoff_init, standoff_final)
    movement6 = (standoff_final, grasp_final)
    movement7 = (grasp_final, None)
    movement8 = (grasp_final, standoff_final)
    movements = [movement1, movement2, movement3, movement4, movement5, \
                 movement6, movement7, movement8]
    #Appending the movement of all the stages 
    for movement in movements:
        #If the gripper is not moving (because is grasping the cube)
        #Wait 1 second
        if movement[1] is None:
            gripper_state = (gripper_state+1) % 2
            configurations = np.append(configurations, 
                                       np.expand_dims(utils.TtoVector(movement[0], 
                                                                     gripper_state),\
                                                      axis = 0), axis = 0)
            for i in range(math.floor(separation)):
                configurations = np.append(configurations,
                                           np.expand_dims(configurations[-1],\
                                                          axis = 0), axis = 0)  
           
        else:
            distance = utils.Distance(movement[0], movement[1])
            time = distance/speed
            N = time*separation
            trajectory = mr.CartesianTrajectory(movement[0], movement[1], 
                                                time, N, 3)
            next_config = np.asarray([utils.TtoVector(x, gripper_state) for x in trajectory])
            configurations = np.append(configurations, next_config, axis = 0)
    return configurations

#Creates the speed the robot will have at every step
def FeedbackControl(Xd, Xd_next, configuration, Kp, Ki, dt, past_err):
    #Getting the jacobian
    #configuration = (phi, x, y, O1, O2, O3, O4, O5, u1, u2, u3, u4, gs)
    thetalist = configuration[3:8]
    T0e = mr.FKinBody(M0e, Blist, thetalist)
    X = utils.XfromConfiguration(configuration, Tb0, T0e)
    Xd = utils.VectorToT(Xd)
    Xd_next = utils.VectorToT(Xd_next)
    J = mr.JacobianBody(Blist, thetalist)
    J = np.append(utils.JacobianBase(configuration, X, r, l, w), J, axis = 1)
    #Getting the PID
    X_inv = mr.TransInv(X)
    Xd_inv = mr.TransInv(Xd)
    X_err = mr.MatrixLog6(np.dot(X_inv, Xd))
    Vd = (1/dt) * mr.MatrixLog6(np.dot(Xd_inv, Xd_next))
    Vd_vec = mr.se3ToVec(Vd)
    D = np.dot(mr.Adjoint(np.dot(X_inv, Xd)), Vd_vec.T)
    integrand = [integrate.simps(i) for i in past_err.T]
    I = np.dot(Ki, integrand)
    P = np.dot(Kp, mr.se3ToVec(X_err).T)
    speeds = utils.Speeds(J, P+I+D)
    #Checks if the joints are between the limits, if not, it does not let those
    #joints move
    J_new = J
    counter = 0
    for i, jointLimit in enumerate(utils.TestJointLimits(thetalist, 
                                                         speeds[4:]*dt)):
        if jointLimit == 0:
            counter += 1
            J_new[:,i+4] = 0
    #Checks if changes were made
    if counter > 0:
        speeds = utils.Speeds(J_new, P+I+D)
    return mr.se3ToVec(X_err), speeds

#Outputs of the program
if __name__ == '__main__':  

    print("Generating animation csv file")
    ref_trajectories = TrajectoryGenerator()
    curr_state = np.array([[0.0,-0.3,0.0,0.0,0.0,0.0,
                            0.0,0.0,0.0,0.0,0.0,0.0,0.0]])
    X_errs = np.array([[0,0,0,0,0,0]])
    for i  in range(len(ref_trajectories)-1):
        X_err, speed = FeedbackControl(ref_trajectories[i], 
                                       ref_trajectories[i+1],\
                                curr_state[-1], Kp, Ki, dt, X_errs) 
        X_errs = np.append(X_errs, np.expand_dims(X_err, axis=0), axis=0)     
        
        curr_state = np.append(curr_state, NextState(curr_state[-1], speed,\
                                                     limits,ref_trajectories[i][-1],\
                                                     dt), axis = 0)

    
