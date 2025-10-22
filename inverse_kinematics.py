import pinocchio as pin
import numpy as np
from scipy.optimize import minimize
import torch.nn as nn

class InverseKinematicSolver():
    
    def __init__(self, model, data):
        
        self.model = model
        self.data = data
        self.target_pos = None
        self.target_ori = None
        self.joint_names = None
        self.joint_ids = None
        self.frame_names = None 
        self.frame_ids = None

    def update(self, model, data, target_pos, target_ori, joint_names, joint_ids, frame_names, frame_ids):

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
            ori = oMf.rotation #model.frames[frame_id].placement.rotation 
            target_pos = self.target_pos[name]
            cost_pos += np.linalg.norm(pos - target_pos)**2
            
            if name in self.target_ori:
                
                target_ori = self.target_ori[name]
                cost_ori += self.angular_error(ori, target_ori)
        return w_pos * cost_pos + w_ori * cost_ori



    def inverse_kinematics(self, q0):

        q_lower_limits = self.model.lowerPositionLimit
        q_upper_limits = self.model.upperPositionLimit
        bounds = []
        for i in range(self.model.nq):
            bounds.append((q_lower_limits[i], q_upper_limits[i]))

        res = minimize(self.ik_cost, q0, bounds=bounds, method='SLSQP', options={'maxiter': 1000, 'disp': False})
        
        q1 = np.array(res.x).reshape(-1)
        assert q1.shape[0] == self.model.nq
        
        return q1

    
    
    def angular_error(self,R1, target):
        
        v = np.array(target[1])
        
        pred = R1 @ v
        pred = pred / np.linalg.norm(pred)


        pred_proj = np.array(pred)
        target_proj = np.array(target[0])

        target_proj /= np.linalg.norm(target_proj)
  
        dot = np.clip(np.dot(pred_proj, target_proj), -1.0, 1.0)
        angle_error = np.arccos(dot) 

        return np.abs(angle_error)










class EvolvedInverseKinematicSolver():
    def __init__(self, model, data):
        
        self.model = model
        self.data = data
        self.target_pos = None
        self.target_ori = None
        self.joint_names = None
        self.joint_ids = None
        self.frame_names = None 
        self.frame_ids = None

    def update(self, model, data, target_pos, target_ori, joint_names, joint_ids, frame_names, frame_ids):

        self.model = model
        self.data = data
        self.target_pos = target_pos
        self.target_ori = target_ori
        self.joint_names = joint_names
        self.joint_ids = joint_ids
        self.frame_names = frame_names 
        self.frame_ids = frame_ids

    def ik_cost_numpy(self, q_numpy, pose_weights, ori_weights):
        pin.forwardKinematics(self.model, self.data, q_numpy)
        pin.updateFramePlacements(self.model, self.data)
        j = 0
        cost = 0.0
        for i, (name, frame_id) in enumerate(zip(self.joint_names, self.joint_ids)):
            oMf = self.data.oMf[frame_id]
            pos = oMf.translation
            target = self.target_pos[name]
            cost += pose_weights[i].item() * np.sum((pos - target) ** 2)

           
            if name in self.target_ori:
                ori = oMf.rotation 
                target_ori = self.target_ori[name]
                cost += ori_weights[j]* self.angular_error(ori, target_ori)
                j +=1

        return cost

    def inverse_kinematics(self, q0, pose_weights, ori_weights):

        q_lower_limits = self.model.lowerPositionLimit
        q_upper_limits = self.model.upperPositionLimit
        bounds = []
        for i in range(self.model.nq):
            bounds.append((q_lower_limits[i], q_upper_limits[i]))

        res = minimize(lambda q: self.ik_cost_numpy(q, pose_weights, ori_weights),
                       q0, bounds=bounds, method='SLSQP', options={'maxiter': 500, 'disp': False})
        return np.array(res.x), res.fun
    
    def angular_error(self,R1, target):
        
        v = np.array(target[1])
        
        pred = R1 @ v
        pred = pred / np.linalg.norm(pred)


        pred_proj = np.array(pred)
        target_proj = np.array(target[0])

        target_proj /= np.linalg.norm(target_proj)
  
        dot = np.clip(np.dot(pred_proj, target_proj), -1.0, 1.0)
        angle_error = np.arccos(dot) 

        return np.abs(angle_error)

"""
import torch 

class NeuralIK(torch.nn.Module):
    def __init__(self, model, data, q0, joint_names, joint_ids, frame_names, frame_ids):
        super().__init__()
        self.weights_pos = torch.nn.Parameter(torch.ones(len(joint_names)) * 0.1)
        self.weights_ori = torch.nn.Parameter(torch.ones(len(frame_names)) * 0.1)
        self.ik_solver = EvolvedInverseKinematicSolver(model, data)
        self.q0 = q0
        self.joint_names = joint_names
        self.joint_ids = joint_ids

    def forward(self, target_pos, target_ori):
    # Aggiorna solver (solo per avere accesso ai target)
        self.ik_solver.update(
            self.ik_solver.model, 
            self.ik_solver.data, 
            target_pos, 
            target_ori, 
            self.joint_names, 
            self.joint_ids
        )

        # Risolve IK con SciPy (detach per evitare gradiente sui pesi)
        q_sol = self.ik_solver.inverse_kinematics(
            self.q0, 
            self.weights_pos.detach(), 
            self.weights_ori.detach()
        )

        # Calcola forward kinematics
        pin.forwardKinematics(self.ik_solver.model, self.ik_solver.data, q_sol)
        pin.updateFramePlacements(self.ik_solver.model, self.ik_solver.data)

        q_sol = torch.tensor(q_sol, requires_grad=True)
        # Loss totale
        loss = torch.tensor(0.0, dtype=self.weights_pos.dtype, device=self.weights_pos.device)

        for i, (name, frame_id) in enumerate(zip(self.joint_names, self.joint_ids)):
            oMf = self.ik_solver.data.oMf[frame_id]

            # --- Posizione ---
            pos = torch.tensor(oMf.translation, dtype=self.weights_pos.dtype, device=self.weights_pos.device)
            target = torch.tensor(target_pos[name], dtype=self.weights_pos.dtype, device=self.weights_pos.device)
            loss += torch.sum((pos - target)**2)

            # --- Orientamento ---
            if name in self.ik_solver.target_ori:
                R_pred = torch.tensor(oMf.rotation, dtype=self.weights_ori.dtype, device=self.weights_ori.device)
                target_dir, ref_axis = self.ik_solver.target_ori[name]

                v_ref = torch.tensor(ref_axis, dtype=self.weights_ori.dtype, device=self.weights_ori.device)
                v_target = torch.tensor(target_dir, dtype=self.weights_ori.dtype, device=self.weights_ori.device)

                v_pred = R_pred @ v_ref
                v_pred = v_pred / (torch.norm(v_pred)+1e-8)
                v_target = v_target / (torch.norm(v_target)+1e-8)

                dot = torch.clamp(torch.dot(v_pred, v_target), -0.999999, 0.999999)
                angle_error = torch.acos(dot)
                loss += angle_error

        return loss


BELLA IDEA CHE PUÒ FUNZIONARE SOLO CON PYTORCH, QUINDI NON VA USATO PINOCCHIO O SCIPY. DA INVESTIGARE IN FUTURO     

"""