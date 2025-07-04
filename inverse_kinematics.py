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


    def ik_cost(self, q, w_pos=1, w_ori=0):
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
                cost_ori += self.rotation_error(ori, target_ori)
                
        return w_pos * cost_pos + w_ori * cost_ori


    def ik_cost_ori(self, q, w_pos=0, w_ori=1):
        
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        cost_pos = 0.0
        cost_ori = 0.0
        
        for name, frame_id in zip(self.frame_names, self.frame_ids):
            oMf = self.data.oMf[frame_id]
            ori = oMf.rotation #model.frames[frame_id].placement.rotation 
            target_ori = self.target_ori[name]
            cost_ori += self.rotation_error(ori, target_ori)
                
        return w_ori * cost_ori


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

    def inverse_kinematics_orientation(self, q0):

        q_lower_limits = self.model.lowerPositionLimit
        q_upper_limits = self.model.upperPositionLimit
        bounds = []
        for i in range(self.model.nq):
            bounds.append((q_lower_limits[i], q_upper_limits[i]))

        res = minimize(self.ik_cost_ori, q0, bounds=bounds, method='SLSQP', options={'maxiter': 1000, 'disp': True})
        
        q1 = np.array(res.x).reshape(-1)
        assert q1.shape[0] == self.model.nq
        
        return q1
    
    def rotation_error(self,R1, R2):
        v = np.array([0, -1, 0])  # direzione "palmo" locale

        pred = R1 @ v
        pred = pred / np.linalg.norm(pred)

        target = R2 @ v
        target = target / np.linalg.norm(target)

        pred_proj = np.array([pred[0], pred[2]])
        target_proj = np.array([target[0], target[2]])

        # Normalizziamo le proiezioni
        pred_proj /= np.linalg.norm(pred_proj)
        target_proj /= np.linalg.norm(target_proj)

        cross = pred_proj[0]*target_proj[1] - pred_proj[1]*target_proj[0]  
        dot = np.dot(pred_proj, target_proj)
        angle_error = np.arctan2(cross, dot) 

        return np.abs(angle_error)
    
    def rotation_cost(self, theta, q1, idx, joint_id):
        
        q1 = np.zeros(self.model.nq)
        q1[idx] = theta[0]
        pin.forwardKinematics(self.model, self.data, q1)
        pin.updateFramePlacements(self.model, self.data)
        R = self.data.oMi[joint_id].rotation

        v = np.array([0, -1, 0])  # direzione "palmo" locale

        pred = R @ v
        pred = pred / np.linalg.norm(pred)

        target = self.target_ori_current @ v
        target = target / np.linalg.norm(target)

        # Proiettiamo pred e target nel piano XZ (piano di rotazione attorno a Y)
        pred_proj = np.array([pred[0], pred[2]])
        target_proj = np.array([target[0], target[2]])

        # Normalizziamo le proiezioni
        pred_proj /= np.linalg.norm(pred_proj)
        target_proj /= np.linalg.norm(target_proj)

        # Calcoliamo l’angolo firmato tra i due vettori proiettati
        cross = pred_proj[0]*target_proj[1] - pred_proj[1]*target_proj[0]  # componente "z" del cross 2D
        dot = np.dot(pred_proj, target_proj)
        angle_error = np.arctan2(cross, dot)  # angolo con segno

        # Minimizziamo il valore assoluto dell’errore angolare
        return np.abs(angle_error)

    
    def end_effector_cost(self, q1, joint_name, target_name):

        joint_id = self.model.getJointId(joint_name)
        idx = self.model.joints[joint_id].idx_q  
        self.target_ori_current = self.target_ori[target_name]  
        def cost(theta):
            return self.rotation_cost(theta, q1, idx, joint_id)

        res = minimize(cost, [q1[idx]], method='SLSQP', options={'maxiter': 1000, 'disp': True})

        q1_new = q1.copy()

        q1_new[idx] = res.x[0]
        return q1_new


        
        