import pinocchio as pin
import numpy as np
import torch
from typing import Dict, List, Tuple

class URDFRetargetingExtractor:
    """
    Estrae informazioni essenziali da URDF usando Pinocchio per motion retargeting
    """
    
    def __init__(self, urdf_path: str, package_dirs: List[str] = None):
        """
        Args:
            urdf_path: Percorso al file URDF
            package_dirs: Liste di directory per i pacchetti ROS (se necessari)
        """
        self.urdf_path = urdf_path
        
        self.model = pin.RobotWrapper.BuildFromURDF(
            filename=urdf_path,
            package_dirs=["."] if package_dirs is None else package_dirs,
            root_joint=None,
    )
            
        self.data = self.model.createData()
        
        # Estrai tutte le informazioni necessarie
        self.extract_retargeting_info()
    
    def extract_retargeting_info(self) -> Dict:
        """Estrae tutte le informazioni necessarie per il retargeting"""
        
        info = {
            'joint_names': self.get_joint_names(),
            'link_names': self.get_link_names(),
            'parent_indices': self.get_parent_indices(),
            'local_translations': self.get_local_translations(),
            'local_rotations': self.get_local_rotations(),
            'rotation_axes': self.get_rotation_axes(),
            'joint_limits': self.get_joint_limits(),
            'joint_types': self.get_joint_types(),
            'masses': self.get_link_masses(),
            'inertias': self.get_link_inertias(),
        }
        
        self.retargeting_info = info
        return info
    
    def get_joint_names(self) -> List[str]:
        """Ottieni nomi delle articolazioni (escludendo universe e root)"""
        return [self.model.names[i] for i in range(1, self.model.njoints)]
    
    def get_link_names(self) -> List[str]:
        """Ottieni nomi dei link/frame"""
        return [self.model.names[i] for i in range(self.model.nframes)]
    
    def get_parent_indices(self) -> np.ndarray:
        """Ottieni indici dei genitori per ogni joint"""
        # Pinocchio usa 0-based indexing, ma il primo joint (universe) ha parent -1
        parents = []
        for i in range(1, self.model.njoints):  # Skip universe joint
            parent_id = self.model.parents[i]
            parents.append(parent_id - 1 if parent_id > 0 else -1)
        return np.array(parents, dtype=np.int32)
    
    def get_local_translations(self) -> np.ndarray:
        """Ottieni offset di traslazione locale per ogni joint"""
        translations = []
        for i in range(1, self.model.njoints):
            # Joint placement contiene la trasformazione dal parent al joint
            placement = self.model.jointPlacements[i]
            translations.append(placement.translation)
        return np.array(translations, dtype=np.float32)
    
    def get_local_rotations(self) -> np.ndarray:
        """Ottieni rotazioni locali come quaternioni [w, x, y, z]"""
        rotations = []
        for i in range(1, self.model.njoints):
            placement = self.model.jointPlacements[i]
            # Pinocchio usa Eigen quaternions (x, y, z, w), convertiamo a (w, x, y, z)
            quat = placement.rotation
            quat_wxyz = np.array([quat.w, quat.x, quat.y, quat.z])
            rotations.append(quat_wxyz)
        return np.array(rotations, dtype=np.float32)
    
    def get_rotation_axes(self) -> np.ndarray:
        """Ottieni assi di rotazione per ogni joint"""
        axes = []
        for i in range(1, self.model.njoints):
            joint = self.model.joints[i]
            
            # Diversi tipi di joint hanno assi diversi
            if hasattr(joint, 'axis'):
                # Per joint revolute/prismatic
                axis = joint.axis
                axes.append(axis)
            elif joint.shortname() == 'JointModelRX':
                axes.append([1, 0, 0])  # Rotazione attorno X
            elif joint.shortname() == 'JointModelRY':
                axes.append([0, 1, 0])  # Rotazione attorno Y
            elif joint.shortname() == 'JointModelRZ':
                axes.append([0, 0, 1])  # Rotazione attorno Z
            elif joint.shortname() == 'JointModelFreeFlyer':
                # Free-floating joint, usa orientazione di default
                axes.append([0, 0, 1])
            else:
                # Default per joint sconosciuti
                axes.append([0, 0, 1])
                
        return np.array(axes, dtype=np.float32)
    
    def get_joint_limits(self) -> np.ndarray:
        """Ottieni limiti articolari [min, max]"""
        limits = []
        for i in range(self.model.nq):
            lower = self.model.lowerPositionLimit[i]
            upper = self.model.upperPositionLimit[i]
            limits.append([lower, upper])
        return np.array(limits, dtype=np.float32)
    
    def get_joint_types(self) -> List[str]:
        """Ottieni tipi di joint"""
        types = []
        for i in range(1, self.model.njoints):
            joint_type = self.model.joints[i].shortname()
            types.append(joint_type)
        return types
    
    def get_link_masses(self) -> np.ndarray:
        """Ottieni masse dei link"""
        masses = []
        for i in range(1, self.model.nbodies):  # Skip universe
            mass = self.model.inertias[i].mass
            masses.append(mass)
        return np.array(masses, dtype=np.float32)
    
    def get_link_inertias(self) -> np.ndarray:
        """Ottieni matrici di inerzia dei link"""
        inertias = []
        for i in range(1, self.model.nbodies):  # Skip universe
            inertia_matrix = self.model.inertias[i].inertia
            inertias.append(inertia_matrix)
        return np.array(inertias, dtype=np.float32)
    
    def get_keypoint_frames(self, keypoint_names: List[str]) -> Dict[str, int]:
        """
        Mappa nomi di keypoint a frame IDs in Pinocchio
        
        Args:
            keypoint_names: Lista di nomi di keypoint (es. ['left_shoulder', 'right_elbow'])
        
        Returns:
            Dizionario {keypoint_name: frame_id}
        """
        keypoint_mapping = {}
        
        for keypoint in keypoint_names:
            # Cerca frame che contengono il nome del keypoint
            for frame_id, frame_name in enumerate(self.model.names):
                if keypoint.lower() in frame_name.lower():
                    keypoint_mapping[keypoint] = frame_id
                    break
            else:
                print(f"Warning: Keypoint '{keypoint}' non trovato nei frame")
                
        return keypoint_mapping
    
    def compute_forward_kinematics(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calcola cinematica diretta per una configurazione articolare
        
        Args:
            q: Configurazione articolare (shape: [nq])
            
        Returns:
            Dizionario con posizioni e orientamenti di tutti i frame
        """
        # Assicurati che q abbia la dimensione corretta
        if len(q) != self.model.nq:
            raise ValueError(f"q deve avere dimensione {self.model.nq}, ricevuto {len(q)}")
        
        # Calcola cinematica diretta
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        positions = []
        orientations = []
        
        for i in range(self.model.nframes):
            placement = self.data.oMf[i]  # World placement of frame i
            positions.append(placement.translation)
            
            # Converti quaternione da (x,y,z,w) a (w,x,y,z)
            quat = placement.rotation
            quat_wxyz = np.array([quat.w, quat.x, quat.y, quat.z])
            orientations.append(quat_wxyz)
        
        return {
            'positions': np.array(positions),
            'orientations': np.array(orientations),
            'frame_names': [self.model.names[i] for i in range(self.model.nframes)]
        }
    
    def to_pytorch_format(self) -> Dict[str, torch.Tensor]:
        """Converte le informazioni estratte in formato PyTorch compatibile con H1mRetargetKeypoint"""
        
        return {
            'node_names': self.retargeting_info['link_names'],
            'parent_indices': torch.from_numpy(self.retargeting_info['parent_indices']),
            'local_translation': torch.from_numpy(self.retargeting_info['local_translations']),
            'local_rotation': torch.from_numpy(self.retargeting_info['local_rotations']),
            'joints_range': torch.from_numpy(self.retargeting_info['joint_limits']),
            'rotation_axes': torch.from_numpy(self.retargeting_info['rotation_axes']),
        }
    
    def print_robot_info(self):
        """Stampa informazioni utili sul robot"""
        print(f"=== Informazioni Robot da {self.urdf_path} ===")
        print(f"Numero di joint: {self.model.njoints - 1}")  # -1 per escludere universe
        print(f"Numero di DoF: {self.model.nq}")
        print(f"Numero di corpi: {self.model.nbodies - 1}")  # -1 per escludere universe
        print(f"Numero di frame: {self.model.nframes}")
        
        print(f"\nJoint names:")
        for i, name in enumerate(self.get_joint_names()):
            print(f"  {i}: {name}")
        
        print(f"\nLink/Frame names:")
        for i, name in enumerate(self.get_link_names()):
            print(f"  {i}: {name}")


# Esempio di utilizzo
if __name__ == "__main__":
    # Carica URDF
    urdf_path = "/home/rcatalini/TheCompleteRobot/URDF/nao.urdf"
    
    try:
        extractor = URDFRetargetingExtractor(urdf_path)
        
        # Stampa informazioni del robot
        extractor.print_robot_info()
        
        # Ottieni dati in formato PyTorch
        pytorch_data = extractor.to_pytorch_format()
        
        # Esempio: definisci keypoint di interesse
        keypoint_names = [
            'left_shoulder', 'left_elbow', 'left_wrist',
            'right_shoulder', 'right_elbow', 'right_wrist',
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle'
        ]
        
        keypoint_mapping = extractor.get_keypoint_frames(keypoint_names)
        print(f"\nKeypoint mapping: {keypoint_mapping}")
        
        # Esempio di cinematica diretta
        q = np.zeros(extractor.model.nq)  # Configurazione neutra
        fk_result = extractor.compute_forward_kinematics(q)
        print(f"\nPosizioni frame in configurazione neutra:")
        for i, (name, pos) in enumerate(zip(fk_result['frame_names'], fk_result['positions'])):
            print(f"  {name}: {pos}")
            
    except Exception as e:
        print(f"Errore nel caricamento URDF: {e}")
        print("Assicurati che il path sia corretto e che Pinocchio sia installato correttamente")