import pinocchio as pin
import numpy as np
from scipy.optimize import minimize

class InverseKinematicSolver():
    
    def __init__(self, model, data, target_pos, target_ori, frame_names, frame_ids):
        
        self.model = model
        self.data = data
        self.target_pos = target_pos
        self.target_ori = target_ori
        self.frame_names = frame_names 
        self.frame_ids = frame_ids


    def rotation_error(self,R1, R2):
        R_diff = R1.T @ R2
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
        return angle


    def ik_cost(self, q, w_pos=1, w_ori=0.001):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        cost_pos = 0.0
        cost_ori = 0.0
        
        for name, frame_id in zip(self.frame_names, self.frame_ids):
            oMf = self.data.oMf[frame_id]
            pos = oMf.translation
            ori = oMf.rotation#model.frames[frame_id].placement.rotation 
            target_pos = self.target_pos[name]
            cost_pos += np.linalg.norm(pos - target_pos)**2
            
            if name in self.target_ori:
                
                target_ori = self.target_ori[name]
                #target_ori = pin.exp3(target_ori.numpy())
                cost_ori += self.rotation_error(ori, target_ori)
                
        return w_pos * cost_pos + w_ori * cost_ori



    def inverse_kinematics_position(self, q0):

        q_lower_limits = self.model.lowerPositionLimit
        q_upper_limits = self.model.upperPositionLimit
        bounds = []
        for i in range(self.model.nq):
            bounds.append((q_lower_limits[i], q_upper_limits[i]))

        res = minimize(self.ik_cost, q0, bounds=bounds, method='SLSQP', options={'maxiter': 1000, 'disp': True})
        
        q1 = np.array(res.x).reshape(-1)
        assert q1.shape[0] == self.model.nq
        
        return q1
    
    def rotation_cost(self, theta, q1, idx, joint_id):
        q_tmp = q1.copy()
        q_tmp[idx] = theta[0]
        pin.forwardKinematics(self.model, self.data, q_tmp)
        pin.updateFramePlacements(self.model, self.data)
        R = self.data.oMi[joint_id].rotation
        return self.rotation_error(R, self.target_ori_current)

    
    def end_effector_cost(self, q1, joint_name, target_name):

        joint_id = self.model.getJointId(joint_name)
        idx = self.model.joints[joint_id].idx_q  
        self.target_ori_current = self.target_ori[target_name]  # salviamo temporaneamente la rotazione

        def cost(theta):
            return self.rotation_cost(theta, q1, idx, joint_id)

        res = minimize(cost, [q1[idx]], method='SLSQP', options={'maxiter': 1000, 'disp': True})

        q1_new = q1.copy()
        q1_new[idx] = res.x[0]
        return q1_new


        
        