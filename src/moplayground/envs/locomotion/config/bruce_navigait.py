from dataclasses import dataclass

@dataclass
class MO2SO:
    enabled   : bool        = False
    weighting : list[int]   = [0.5, 0.5]

@dataclass
class BruceConfig:
    name            = 'hello'
    save_dir        = 'results/Bruce-arm-swing'
    description     = ''
    algorithm       = 'ppo'
    env             = 'NaviGait'
    robot           = 'BRUCE'
    backend         = 'np'
    gaitlib_path    = 'src/moplayground/envs/locomotion/bruce/gaits/BRUCE_GL_4bar_noarms_v10_witharms'
    
    mo2so           = MO2SO(
        enabled   = False,
        weighting = [0.5, 0.5]
    )

if __name__ == '__main__':
    c = BruceConfig()
    print(c.mo2so.enabled)
