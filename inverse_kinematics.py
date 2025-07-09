import pinocchio as pin
import numpy as np
from scipy.optimize import minimize

class InverseKinematicSolver():
    
    def __init__(self, model, data, target_pos, target_ori, joint_names, joint_ids, frame_names, frame_ids):
        
        self.model = model
        self.data = data
        self.target_pos = target_pos
        self.target_ori = target_ori
        self.joint_names = joint_names
        self.joint_ids = joint_ids
        self.frame_names = frame_names 
        self.frame_ids = frame_ids


    def ik_cost(self, q, w_pos=1, w_ori=0.001):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        cost_pos = 0.0
        cost_ori = 0.0
        
        for name, frame_id in zip(self.joint_names, self.joint_ids):
            oMf = self.data.oMf[frame_id]
            pos = oMf.translation
            ori = oMf.rotation#model.frames[frame_id].placement.rotation 
            target_pos = self.target_pos[name]
            cost_pos += np.linalg.norm(pos - target_pos)**2
            
            if name in self.target_ori:
                
                target_ori = self.target_ori[name]
                #target_ori = pin.exp3(target_ori.numpy())
                #cost_ori += np.linalg.norm(ori-target_ori, ord='fro')**2
                cost_ori += self.angular_error(ori, target_ori)
        return w_pos * cost_pos + w_ori * cost_ori


    def ik_cost_ori(self, q):
        
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        cost_ori = 0.0
        
        for name, frame_id in zip(self.frame_names, self.frame_ids):
            oMf = self.data.oMi[frame_id]
            ori = oMf.rotation              #model.frames[frame_id].placement.rotation 
            target_ori = self.target_ori[name]
            cost_ori += np.linalg.norm(ori - target_ori, ord='fro')
            
        return cost_ori


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

    
    
    def angular_error(self,R1, target):
        
        v = np.array([0, 0, -1])  

        pred = R1 @ v
        pred = pred / np.linalg.norm(pred)

        print(pred)
        print(target)

        pred_proj = np.array(pred)
        target_proj = np.array(target)

        
        pred_proj /= np.linalg.norm(pred_proj)
        target_proj /= np.linalg.norm(target_proj)
  
        dot = np.dot(pred_proj, target_proj)
        angle_error = np.arccos(dot) 



        return np.abs(angle_error)

    