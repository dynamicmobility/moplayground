from yaml import safe_load
import numpy as np
import jax
from jax import numpy as jnp
from enum import IntEnum
from scipy.special import comb, factorial
from dataclasses import dataclass, field
from utils.geometry import euler2quat, quat2euler

def bezier_basis_matrix(n):
    """
    Returns the Bézier basis matrix M for degree n such that:
        B(t) = [t^n, t^{n-1}, ..., 1] @ M @ [b_0, ..., b_n]^T
    """
    M = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            coeff = ((-1) ** (i - j)) * comb(n, i) * comb(i, j)
            M[j, i] += coeff
    return M.T

class Leg(IntEnum):
    LEFT = 0
    RIGHT = 1

"""
Implements a container for a gait (sequence of Bezier Polynomials)
"""
@jax.tree_util.register_static
@dataclass(frozen=True)
class P1Bezier:
    num_states:     int
    num_degree:     int
    COMBINATION:    jnp.ndarray | np.ndarray
    COMBO_MAT:      jnp.ndarray | np.ndarray
    COMBO_MAT_INV:  jnp.ndarray | np.ndarray
    DCOMBO_MAT:     jnp.ndarray | np.ndarray
    DCOEFF_T:       jnp.ndarray | np.ndarray
    POWER:          jnp.ndarray | np.ndarray    
    bnp:            object = field(default_factory=lambda: jnp)
    fact:           callable = field(default_factory=lambda: jax.scipy.special.factorial)
    FREE_STATES:    int = 6
    
    @classmethod 
    def setup(
        cls,
        num_states: int,
        num_degree: int,
        bnp,
        fact
    ):
        DCOEFF_T = cls.get_dcoeff_transform(num_degree, bnp)
        POWER = bnp.arange(num_degree + 1)
        M     = bnp.array(bezier_basis_matrix(num_degree))
        DM    = bnp.array(bezier_basis_matrix(num_degree - 1))
        COMBINATION = np.zeros((num_degree + 1, num_degree + 1))
        for n in range(num_degree + 1):
            for r in range(num_degree + 1):
                if r > n:
                    continue
                COMBINATION[n, r] = comb(n, r)
    
        return cls(
            num_states    = num_states,
            num_degree    = num_degree,
            bnp           = bnp,
            fact          = fact,
            COMBINATION   = bnp.array(COMBINATION),
            COMBO_MAT     = M,
            COMBO_MAT_INV = bnp.linalg.inv(M),
            DCOMBO_MAT    = DM,
            DCOEFF_T      = DCOEFF_T,
            POWER         = POWER
        )
    
    def reshape_coeff(self, coeff, num_states):
        """Reshapes the coefficients to be of the correct size"""
        return np.array(
            coeff
        ).reshape(-1, num_states).T
    
    def relabel(self, jt_coeff, ff_coeff):
        """Relabels the bezier polynomial according to the class strategy and 
        returns the corresponding ctrl_points and dcoeff"""

        jt_coeff_rlbl = self.bnp.vstack([
            jt_coeff[6:, :] * self.bnp.array([-1, -1, 1, 1, 1, -1]).reshape((-1, 1)),
            jt_coeff[:6, :] * self.bnp.array([-1, -1, 1, 1, 1, -1]).reshape((-1, 1))
        ])
        ff_coeff_rlbl = self.bnp.vstack([
            ff_coeff * self.bnp.array([1, -1, 1, -1, 1, -1]).reshape((-1, 1))
        ])
        jt_coeff = self.bnp.array([
            self.bnp.copy(jt_coeff), jt_coeff_rlbl
        ])
        ff_coeff = self.bnp.array([
            self.bnp.copy(ff_coeff), ff_coeff_rlbl
        ])
        return jt_coeff, ff_coeff

    @classmethod
    def get_dcoeff_transform(cls, num_degree, bnp):
        M = num_degree
        dcoeff_transform = np.zeros((M, M+1))

        for i in range(M):
            dcoeff_transform[i, i] = -(M - i) * comb(M,i) / comb(M-1,i)
            dcoeff_transform[i,i+1] = (i+1) * comb(M,i+1) / comb(M-1,i)
        dcoeff_transform[M-1, M] = M * comb(M, M)
        return dcoeff_transform
    
    def compute_dcoeff(self, coeff):
        """Computes the derivative coefficients for the bezier polynomial and
        returns the corresponding dcoeff for jt and ff in a tuple"""
        dcoeff = self.bnp.array([
            coeff[0] @ self.DCOEFF_T.T,
            coeff[1] @ self.DCOEFF_T.T
        ])
        return dcoeff
    
    def get_binomial_coeffs(self, n, z):
        """Returns the binomial coefficients for a given n"""
        # assert(z != 0)
        # assert(z != 1)
        # z = np.clip(z, 0.01, 0.99)
        coeff_part = self.bnp.array(
            self.COMBINATION[n, :] * self.bnp.power(z, n - self.POWER) * self.bnp.power(1 - z, self.POWER)
        )
        return coeff_part
    
    def get_z_transform(self, z):
        z_transform = self.bnp.vstack([
            self.get_binomial_coeffs(n, z) for n in range(self.num_degree + 1)
        ])

        return z_transform.T
        
    
    def evaluate(self, s, coeff, dcoeff, period):
        """Evaluates the bezier polynomial described by coeffs for all states at
        parameter s. s is clipped to [0,1] and is ideally monotonic."""
        s = self.bnp.clip(s, 0, 1)
        ts = self.bnp.power(s, self.POWER).reshape(1, -1)
        pos = ts @ self.COMBO_MAT @ coeff.T
        pos = pos[0]

        vel = ts[:, :-1] @ self.DCOMBO_MAT @ dcoeff.T
        vel = 1 / period * vel[0]

        return pos, vel
    
    def split(self, coeffs, z, z_transform):
        """Returns a 'split' Bezier coefficients at z, where z in [0, 1] is the
        point in the Bezier phase you wish to split at."""
        Z = z_transform
        M = self.COMBO_MAT
        new_coeffs = self.COMBO_MAT_INV @ Z @ M @ coeffs.transpose(0, 2, 1)
        return new_coeffs.transpose(0, 2, 1)
    
    def interpolate(self, start_coeffs, end_coeffs, deg=3):
        """Returns Bezier coefficients that start at start_coeff and end at
        end_coeff."""
        # print(start_coeffs.shape)
        # print(start_coeffs[0, 0, :])
        # print('---')
        # print(end_coeffs[0, 0, :])
        new_jt_coeffs = self.bnp.concatenate([
            start_coeffs[:, :, :deg],
            0.5 * (start_coeffs[:, :, deg:-deg] + end_coeffs[:, :, deg:-deg]),
            end_coeffs[:, :, -deg:]
        ], axis=2)
        # print(new_jt_coeffs[0, 0, :])
        # quit()
        return new_jt_coeffs

    def to_quat(self, ff_pos):
        """Converts a flying frame position from RPY to quaternion for mujoco
        """
        ff_pos = self.bnp.hstack([ff_pos[:3], euler2quat(self.bnp, ff_pos[3:])])
        return ff_pos
    
    def tree_flatten(self):
        return (self.num_states, 
                self.num_degree, 
                self.COMBO_MAT, 
                self.DCOMBO_MAT, 
                self.DCOEFF_T, 
                self.POWER), None
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    

if __name__ == '__main__':
    # TODO: Write tests/graphs for bezier!!!
    bezier = P1Bezier.setup(
        num_states=12,
        num_degree=7,
        bnp=np,
        fact=factorial
    )
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import time
    BEZ1 = 'control/gaits/2D_library_v4/v_7_2.yaml'
    BEZ2 = 'control/gaits/2D_library_v4/v_7_4.yaml'
    
    # BEZ1 = 'control/gaits/2D_library_v5/v_7_1.yaml'
    # BEZ2 = 'control/gaits/2D_library_v5/v_10_1.yaml'
    with open(BEZ1, 'r') as f:
        bez1_contents = safe_load(f)
    with open(BEZ2, 'r') as f:
        bez2_contents = safe_load(f)
    
    bez1_coeff = bezier.reshape_coeff(
        bezier.bnp.array(bez1_contents['RightSS']['coeff_b']),
        6
    )
    bez1_coeff[1, :] += 0.1
    # bez1_coeff[1, -1] += 0.1
    bez1_coeff = np.array([bez1_coeff.copy(), bez1_coeff.copy()])
    bez1_dcoeff = bezier.compute_dcoeff(bez1_coeff)
    bez1_period = bez1_contents['RightSS']['step_dur']
    print(bez1_coeff.shape)
    print(bez1_dcoeff.shape)
    
    bez2_coeff = bezier.reshape_coeff(
        bezier.bnp.array(bez2_contents['LeftSS']['coeff_b']),
        6
    )
    bez2_coeff = np.array([bez2_coeff.copy(), bez2_coeff.copy()])
    bez2_dcoeff = bezier.compute_dcoeff(bez2_coeff)
    bez2_period = bez2_contents['RightSS']['step_dur']

    print('bez1 vel', bez1_contents['vd_x'], bez1_contents['vd_y'])
    print('bez2 vel', bez2_contents['vd_x'], bez2_contents['vd_y'])
    N = 1000
    SPLIT1 = 0.5
    SPLIT2 = 0.5
    t = np.linspace(0, 1, N)
    bez1_pos = []
    bez2_pos = []
    
    bez1_vel = []
    bez2_vel = []
    
    # Simulate both beziers for [0, 1]
    start = time.time()
    for i in tqdm(t, disable=True):
        b1p, b1v = bezier.evaluate(
            s=i,
            coeff=bez1_coeff[0],
            dcoeff=bez1_dcoeff[0],
            period=bez1_period
        )
        bez1_pos.append(b1p)
        bez1_vel.append(b1v)
        
        b2p, b2v = bezier.evaluate(
            s=i,
            coeff=bez2_coeff[0],
            dcoeff=bez2_dcoeff[0],
            period=bez2_period
        )
        bez2_pos.append(b2p)
        bez2_vel.append(b2v)
    
    # z_transform = bezier.get_z_transform(SPLIT1)
    # # Split at SPLIT, interpolate...
    # bez3_coeff = bezier.split(
    #     bez1_coeff[0].copy(),
    #     SPLIT1,
    #     z_transform
    # )
    # bez2_remaining_period = bez2_period - bez1_period * SPLIT1
    # split1_prime = 1 - bez2_remaining_period / bez2_period
    # print(f'{split1_prime=} vs {SPLIT1=}')
    # z_transform = bezier.get_z_transform(split1_prime)
    # print(f'{split1_prime=}')
    # bez2split_coeff = bezier.split(
    #     bez2_coeff[0].copy(),
    #     split1_prime,
    #     z_transform
    # )
    # bez3_coeff = bezier.interpolate(
    #     bez3_coeff,
    #     bez2split_coeff,
    #     deg=3
    # )
    # bez3_coeff = np.array([bez3_coeff.copy(), bez3_coeff.copy()])
    # # bez3_dcoeff = 1 / (SPLIT1) * bezier.compute_dcoeff(bez3_coeff)
    # bez3_dcoeff = bezier.compute_dcoeff(bez3_coeff)
    
    # bez3_pos = []
    # bez3_vel = []
    # t_split = np.linspace(SPLIT1, 1, N)
    # # b3period = (t_split[-1] - t_split[0])
    # bez3_period = (1 - SPLIT1) * bez2_period 
    # print('B3 Period', bez3_period)
    # print('B3 Gait period', (1 - SPLIT1))
    # print()

    
    # # Now simulate this guy
    # for i in tqdm(t_split, disable=True):
    #     b3p, b3v = bezier.evaluate(
    #         s=(i - t_split[0]) / (t_split[-1] - t_split[0]),
    #         coeff=bez3_coeff[0],
    #         dcoeff=bez3_dcoeff[0],
    #         period=bez3_period
    #     )
    #     bez3_pos.append(b3p)
    #     bez3_vel.append(b3v)
        
    # z_transform = bezier.get_z_transform(SPLIT2)
    # # Split at SPLIT, interpolate...
    # bez4_coeff = bezier.split(
    #     bez3_coeff[0].copy(),
    #     SPLIT2,
    #     z_transform
    # )
    # bez1_remaining_period = bez1_period - bez3_period * (1 - SPLIT1)
    # split2_prime = bez1_remaining_period / bez1_period
    # z_transform = bezier.get_z_transform(split2_prime)
    # print(f'{split2_prime=} vs {SPLIT2=}')
    # bez1_split_coeff = bezier.split(
    #     bez1_coeff[0].copy(),
    #     split2_prime,
    #     z_transform
    # )
    # bez4_coeff = bezier.interpolate(
    #     bez4_coeff,
    #     bez1_split_coeff,
    #     deg=3
    # )
    # bez4_coeff = np.array([bez4_coeff.copy(), bez4_coeff.copy()])
    # # bez3_dcoeff = 1 / (SPLIT) * bezier.compute_dcoeff(bez3_coeff)
    # bez4_dcoeff = bezier.compute_dcoeff(bez4_coeff)
    
    # bez4_pos = []
    # bez4_vel = []
    # start_s = (1 - SPLIT1) * SPLIT2 + SPLIT1
    # print(start_s)
    # t_split2 = np.linspace(start_s, 1, N)
    # # b4period = (t_split2[-1] - t_split2[0])
    # bez4_period = (1 - SPLIT2) * bez3_period
    # # print('B4 Period', bez4_period)
    # # print('B4 Gait period', b3period - start_s) # WRONG
    # # print('B4 Gait period', 1 - ((b3period) * SPLIT2 + (1 - b3period)))
    # print('Calculated B4 gait period =', bez4_period)
    # print('True B4 gait period =', t_split2[-1] - t_split2[0])
    # # print('B4 Gait period', (1 - SPLIT2) * bez3_period)
    # # 1 - b3period * split2 - 1 + b3period
    # # (1 - split2) * b3period
    
    # # Now simulate this guy
    # for i in tqdm(t_split2, disable=True):
    #     b4p, b4v = bezier.evaluate(
    #         s=(i - t_split2[0]) / (t_split2[-1] - t_split2[0]),
    #         coeff=bez4_coeff[0],
    #         dcoeff=bez4_dcoeff[0],
    #         period=bez4_period
    #     )
    #     bez4_pos.append(b4p)
    #     bez4_vel.append(b4v)
    
    bez1_pos = np.array(bez1_pos)
    bez2_pos = np.array(bez2_pos)
    bez1_vel = np.array(bez1_vel)
    bez2_vel = np.array(bez2_vel)
    # bez3_pos = np.array(bez3_pos)
    # bez3_vel = np.array(bez3_vel)
    # bez4_pos = np.array(bez4_pos)
    # bez4_vel = np.array(bez4_vel)
    
    print(f'{N} iters done in {time.time() - start:.2f}s')
    print('plotting...')
    K = 4
    fig, axs = plt.subplots(nrows=K, ncols=1)
    axs = axs.flatten()
    for i in range(K):
        axs[i].plot(t, bez1_pos[:, i], c='blue', ls='--', label='v = 0.1')
        axs[i].plot(t, bez2_pos[:, i], c='red', ls='--', label='v = 0.3')
        # axs[i].plot(t_split, bez3_pos[:, i], c='green', label='interpolated1')
        # axs[i].plot(t_split2, bez4_pos[:, i], c='purple', label='interpolated2')
        
        # axs[i + K].plot(t, bez1_vel[:, i], c='blue', ls='--', label='v = 0.1')
        # axs[i + K].plot(t, bez2_vel[:, i], c='red', ls='--', label='v = 0.3')
        # axs[i + K].plot(t_split, bez3_vel[:, i], c='green', label='interpolated1')
        # axs[i + K].plot(t_split2, bez4_vel[:, i], c='purple', label='interpolated2')
        # axs[i].scatter(
        #     t_split[0], bez3[0, i], c='green', s=50
        # )
        axs[i].set_title(f'Joint {i}')

    axs[0].set_ylabel('Position')
    # axs[K].set_ylabel('Velocity')
    fig.set_size_inches((5, 15))
    fig.tight_layout()
    
    print('saving...')
    plt.savefig('control/gait_plots', dpi=500)

    
