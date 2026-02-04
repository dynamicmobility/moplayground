import numpy as np
import yaml
from glob import glob
from pathlib import Path
import sys


class FlowSeqDumper(yaml.Dumper):
    def represent_sequence(self, tag, sequence, flow_style=None):
        # Force all sequences (lists) to use flow style
        return super().represent_sequence(tag, sequence, flow_style=True)

def open_file(path):
    # Read YAML from a file
    with open(path, "r") as f:
        gait = yaml.safe_load(f)
    
    return gait

def get_coeffs(key, gait, num_states):
    coeffs = gait[key]['coeff_jt'] = np.array(
        gait[key]['coeff_jt']
    ).reshape(-1, num_states)
    return coeffs
    
    
def add_arms(gait):
    LF = get_coeffs('LeftSS', gait, 10)
    RF = get_coeffs('RightSS', gait, 10)
    # print(LF.shape)
    # quit()
    arm_pos = np.concat(
        (np.zeros((LF.shape[0], 1)),
         np.ones((LF.shape[0], 1)),
         np.zeros((LF.shape[0], 1))),
        axis=1
    )
    arm_L = (LF, arm_pos.copy()) + (-arm_pos.copy(),)
    arm_R = (RF, arm_pos.copy()) + (-arm_pos.copy(),)
    gait['LeftSS']['coeff_jt']  = np.concat(arm_L, axis=1).flatten().tolist()
    gait['RightSS']['coeff_jt'] = np.concat(arm_R, axis=1).flatten().tolist()
    return gait
    
def write_to_file(gait, filename):
    with open(filename, "w") as f:
        yaml.dump(gait, f, Dumper=FlowSeqDumper, sort_keys=False)

GL_PATH = Path(sys.argv[1])
GL_NEW_PATH = Path(str(GL_PATH.as_posix() + '_witharms'))
new_dir = Path(GL_NEW_PATH)
new_dir.mkdir(parents=True, exist_ok=True)
bez_files = glob(GL_PATH.as_posix() + '/*.yaml')
for file in bez_files:
    file = Path(file)
    gait = open_file(file.as_posix())
    gait = add_arms(gait)
    new_path = GL_NEW_PATH / file.relative_to(GL_PATH.as_posix())
    write_to_file(gait, new_path)
