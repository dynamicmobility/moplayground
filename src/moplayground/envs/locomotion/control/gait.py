from moplayground.envs.locomotion.control.bezier import P1Bezier, Leg
import numpy as np
from enum import IntEnum
from scipy.spatial import KDTree
import jax
from jax import numpy as jnp
from dataclasses import dataclass, field
from scipy.special import factorial
from glob import glob
from yaml import safe_load
from pathlib import Path

MIN_SWING_PHASE = 0.7

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GaitLibrary:
    num_gaits:    int
    num_states:   int
    num_degree:   int
    num_vx_gaits: int
    bezier:       P1Bezier
    jt_coeffs:    jnp.ndarray | np.ndarray
    jt_dcoeffs:   jnp.ndarray | np.ndarray
    ff_coeffs:    jnp.ndarray | np.ndarray
    ff_dcoeffs:   jnp.ndarray | np.ndarray
    curr_jt:      jnp.ndarray | np.ndarray
    curr_djt:     jnp.ndarray | np.ndarray
    curr_ff:      jnp.ndarray | np.ndarray
    curr_dff:     jnp.ndarray | np.ndarray
    curr_period:  float
    step_period:  float
    curr_vdes:    jnp.ndarray | np.ndarray
    periods:      jnp.ndarray | np.ndarray
    vel_xs:       jnp.ndarray | np.ndarray
    vel_ys:       jnp.ndarray | np.ndarray
    swing_leg:    int
    t0:           float
    t0_step:      float
    blend:        float
    offset_pos:   jnp.ndarray | np.ndarray
    offset_vel:   jnp.ndarray | np.ndarray
    gnp:          object   = field(default_factory=lambda: jnp)
    side_offset:  jnp.ndarray | np.ndarray = field(default_factory=lambda: jnp.zeros(6, dtype=jnp.float32))
    
    @classmethod
    def from_directory(
        cls, 
        path, 
        v0,
        num_states,
        num_degree,
        gnp: object = field(default_factory=lambda: jnp),
        fact: callable = field(default_factory=lambda: jax.scipy.special.factorial),
        swing_leg=Leg.LEFT, 
        t0=0.0, 
        blend=0.3,
        gait_type='P1'
    ):
        """Loads a gait library from a directory of yaml files. It should setup
        jt_coeffs and ff_coeffs to hold a list of coefficients for each gait in 
        the directory."""
        def open_yaml(file):
            with open(file, 'r') as f:
                contents = safe_load(f)
            return dict(contents)
        
        bez_files = glob(path + '/*.yaml')
        if bez_files == []:
            raise ValueError(f'No yaml files found in {path}. Please check the path.')
        
        def get_gait(x):
            gait = open_yaml(x)
            return [float(gait['vd_x']), float(gait['vd_y'])], gait
        
        vels = []
        for file in bez_files:
            vdes, gait = get_gait(file)
            vels.append(vdes)


        num_vx_gaits = len(np.unique(np.array(vels)[:,0]))
        
        # print(f'{num_vx_gaits} vx gaits found in {path}')
        
        bez_files = glob(path + '/*.yaml')
        def sort_key(x):
            x = Path(x).stem
            nums = x.split('_')
            return int(nums[1]) * num_vx_gaits + int(nums[2])

        bez_files.sort(key=sort_key)


        bezier = P1Bezier.setup(
            num_states = num_states,
            num_degree = num_degree,
            bnp        = gnp,
            fact       = fact
        )
        
        jt_coeffs = []
        jt_dcoeffs = []
        ff_coeffs = []
        ff_dcoeffs = []
        vel_xs = []
        vel_ys = []
        periods = []
        vel_lims = np.array([[np.inf, -np.inf], [np.inf, -np.inf]])
        for bez in bez_files:
            gait = open_yaml(bez)
            if gait_type == 'P1':
                # Reshape the bezier coefficients
                # contents['RightSS'] = contents
                jt_coeff = bezier.reshape_coeff(
                    coeff=gait['RightSS']['coeff_jt'],
                    num_states=num_states
                )
                ff_coeff = bezier.reshape_coeff(
                    coeff=gait['RightSS']['coeff_b'],
                    num_states=P1Bezier.FREE_STATES
                )
                num_degree = jt_coeff.shape[1] - 1

                # Compute relabeling...
                jt_coeff, ff_coeff = bezier.relabel(
                    jt_coeff=jt_coeff,
                    ff_coeff=ff_coeff
                )
                periods.append([gait['RightSS']['step_dur']])
            elif gait_type == 'P2':
                jt_coeff1 = bezier.reshape_coeff(
                    coeff=gait['RightSS']['coeff_jt'],
                    num_states=num_states
                )
                ff_coeff1 = bezier.reshape_coeff(
                    coeff=gait['RightSS']['coeff_b'],
                    num_states=P1Bezier.FREE_STATES
                )
                num_degree = jt_coeff1.shape[1] - 1
                
                # if contents['vd_y'] < 0:
                #     ff_coeff1[1, :] -= contents['LeftSS']['step_len'][1]
                
                # Reshape the bezier coefficients
                jt_coeff2 = bezier.reshape_coeff(
                    coeff=gait['LeftSS']['coeff_jt'],
                    num_states=num_states
                )
                ff_coeff2 = bezier.reshape_coeff(
                    coeff=gait['LeftSS']['coeff_b'],
                    num_states=P1Bezier.FREE_STATES
                )
                
                # if contents['vd_y'] < 0:
                #     ff_coeff2[1, :] -= contents['LeftSS']['step_len'][1]

                jt_coeff = gnp.array([jt_coeff1, jt_coeff2])
                ff_coeff = gnp.array([ff_coeff1, ff_coeff2])
                periods.append([gait['RightSS']['step_dur']])
            else:
                raise ValueError(f'Gait type {gait_type} not supported')

            # Compute derivatives...
            jt_dcoeff = bezier.compute_dcoeff(jt_coeff)
            ff_dcoeff = bezier.compute_dcoeff(ff_coeff)
            
            jt_coeffs.append(jt_coeff)
            jt_dcoeffs.append(jt_dcoeff)
            ff_coeffs.append(ff_coeff)
            ff_dcoeffs.append(ff_dcoeff)

            vel_lims[0, 0] = min(vel_lims[0, 0], gait['vd_x'])
            vel_lims[0, 1] = max(vel_lims[0, 1], gait['vd_x'])
            vel_lims[1, 0] = min(vel_lims[1, 0], gait['vd_y'])
            vel_lims[1, 1] = max(vel_lims[1, 1], gait['vd_y'])
        
        num_vy_gaits = len(bez_files) // num_vx_gaits
        vel_xs = np.linspace(*vel_lims[0, :], num=num_vx_gaits)
        vel_ys = np.linspace(*vel_lims[1, :], num=num_vy_gaits)
        # print(num_vx_gaits, num_vy_gaits)
        # quit()
        
        jt_coeffs  = gnp.array(jt_coeffs)
        jt_dcoeffs = gnp.array(jt_dcoeffs)
        ff_coeffs  = gnp.array(ff_coeffs)
        ff_dcoeffs = gnp.array(ff_dcoeffs)
        periods    = gnp.array(periods) 
        vel_xs     = gnp.array(vel_xs)
        vel_ys     = gnp.array(vel_ys)
        
        side_offset = np.zeros(6)
        
        # print("Variable Names and Shapes:")
        # print(f"jt_coeffs: {jt_coeffs.shape}")
        # print(f"jt_dcoeffs: {jt_dcoeffs.shape}")
        # print(f"ff_coeffs: {ff_coeffs.shape}")
        # print(f"ff_dcoeffs: {ff_dcoeffs.shape}")
        # print(f"periods: {periods.shape}")
        # print(f"vel_xs: {vel_xs.shape}")
        # print(f"vel_ys: {vel_ys.shape}")
        # quit()
        
        gaitlib = cls(
            num_gaits    = jt_coeffs.shape[0],
            num_states   = num_states,
            num_degree   = num_degree,
            num_vx_gaits = num_vx_gaits,
            bezier       = bezier,
            jt_coeffs    = jt_coeffs,
            jt_dcoeffs   = jt_dcoeffs,
            ff_coeffs    = ff_coeffs,
            ff_dcoeffs   = ff_dcoeffs,
            curr_jt      = jt_coeffs[0], # place holders...
            curr_djt     = jt_dcoeffs[0],
            curr_ff      = ff_coeffs[0],
            curr_dff     = ff_dcoeffs[0],
            curr_period  = periods[0, 0],
            step_period  = periods[0, 0],
            curr_vdes    = v0,
            periods      = periods,
            vel_xs       = vel_xs,
            vel_ys       = vel_ys,
            swing_leg    = swing_leg,
            t0           = 0.0,
            t0_step      = 0.0,
            blend        = blend,
            offset_pos   = gnp.zeros(num_states, dtype=gnp.float32),
            offset_vel   = gnp.zeros(num_states, dtype=gnp.float32),
            gnp          = gnp,
            side_offset  = side_offset,
        )

        neighbors, weights = gaitlib.find_nearest(v0)
        jt_coeff, ff_coeff, jt_dcoeff, ff_dcoeff, period = gaitlib.combine_gaits(
            neighbors,
            weights
        )
        return gaitlib.set_new_coeffs(
            jt_coeff=jt_coeff,
            jt_dcoeff=jt_dcoeff,
            ff_coeff=ff_coeff,
            ff_dcoeff=ff_dcoeff,
            period=period,
            step_period=period,
            t0=t0,
            t0_step=t0,
            vdes=v0,
            swing_leg=swing_leg
        )

    def get_1D_idx(self, vx_idx, vy_idx):
        num_vy_gaits = self.num_gaits // self.num_vx_gaits
        return vx_idx * num_vy_gaits + vy_idx
    
    def find_nearest(self, vdes):
        """Finds the nearest gaits in the gait library with respect to the\
            desired velocity. The output will be FOUR gaits.
            
            vdes: (np.ndarray) desired velocity (vx, vy)

            Returns: (vx_gaits, vy_gaits)
                vx_gaits: (alpha1, alpha2, weight_alpha)
                vy_gaits: (beta1, beta2, weight_beta)
            """
        # print(f'vdes {vdes}')
        num_vy_gaits = self.num_gaits // self.num_vx_gaits
        x1_idx = self.gnp.searchsorted(self.vel_xs, vdes[0])
        x1_idx = self.gnp.clip(x1_idx, 0, self.num_vx_gaits - 1)
        x2_idx = self.gnp.clip(x1_idx - 1, 0, self.num_vx_gaits - 1)
        y1_idx = self.gnp.searchsorted(self.vel_ys, vdes[1])
        y1_idx = self.gnp.clip(y1_idx, 0, num_vy_gaits - 1)
        y2_idx = self.gnp.clip(y1_idx - 1, 0, num_vy_gaits - 1)
        # print(f"x1_idx: {x1_idx}, x2_idx: {x2_idx}, y1_idx: {y1_idx}, y2_idx: {y2_idx}")
        # print(f"num_gaits: {self.num_gaits}")
        # print()
        
        neighbor_idxs = self.gnp.array([
            self.get_1D_idx(x1_idx, y1_idx),
            self.get_1D_idx(x2_idx, y1_idx),
            self.get_1D_idx(x1_idx, y2_idx)
        ])
        # print(f"neighbor_idxs: {neighbor_idxs}")
        # print()
        
        neighbors = self.gnp.array([idx for idx in neighbor_idxs])

        vx1, vy1 = self.vel_xs[x1_idx], self.vel_ys[y1_idx]
        vx2, vy2 = self.vel_xs[x2_idx], self.vel_ys[y1_idx]
        vx3, vy3 = self.vel_xs[x1_idx], self.vel_ys[y2_idx]
        
        def solve_convex_comb(args):
            """Solves the convex combination of two points"""
            v1, v2, vd = args
            return (vd - v2) / (v1 - v2)
        
        w1_zero = self.gnp.array(self.gnp.abs(vx1 - vx2) < 1e-2).astype(self.gnp.int32)
        w1 = self.gnp.array([
            solve_convex_comb([vx1, vx2 + 1 * w1_zero, vdes[0]]),
            vx1,
        ])[w1_zero]

        w2_zero = self.gnp.array(self.gnp.abs(vy1 - vy3) < 1e-2).astype(self.gnp.int32)
        w2 = self.gnp.array([
            solve_convex_comb([vy1, vy3 + 1 * w2_zero, vdes[1]]) ,
            vy1,
        ])[w2_zero]

        # print('v', vdes, neighbor_idxs, np.array([w1, w2]), x1_idx * w1 + x2_idx * (1 - w1))

        return neighbors, self.gnp.array([w1, w2])
        

    def combine_gaits(
            self,
            neighbors,
            weights
    ):
        """Combines gaits.
        
        Returns a single convex combination of the four gaits. Size will be
        (2, nstates, ndegree + 1) as it includes the relabeling.
        """
        jt_ns = self.jt_coeffs[neighbors] # joint neighbors
        jt_coeff = jt_ns[0] * weights[0] + jt_ns[1] * (1 - weights[0])
        jt_coeff = jt_coeff * weights[1] + jt_ns[2] * (1 - weights[1]) # add this back in for 2D lib
        
        ff_ns = self.ff_coeffs[neighbors] # flying frame neighbors
        ff_coeff = ff_ns[0] * weights[0] + ff_ns[1] * (1 - weights[0])
        ff_coeff = ff_coeff * weights[1] + ff_ns[2] * (1 - weights[1]) # add this back in for 2D lib
        
        jt_dcoeff = self.bezier.compute_dcoeff(jt_coeff)
        ff_dcoeff = self.bezier.compute_dcoeff(ff_coeff)
        
        period_ns = self.periods[neighbors]
        period = period_ns[0] * weights[0] + period_ns[1] * (1 - weights[0])
        period = period * weights[1] + period_ns[2] * (1 - weights[1])
        
        return jt_coeff, ff_coeff, jt_dcoeff, ff_dcoeff, period[0]
    
    def interpolate_gait(self, end_coeff_jt, end_coeff_ff, end_period, s, t):
        """Interpolates a gait between the current gait and the input gait.
        This method should be iteratively run to ensure smoothness.
        """
        z_transform = self.bezier.get_z_transform(s)
        
        split_coeff_jt = self.bezier.split(self.curr_jt, s, z_transform)
        split_coeff_ff = self.bezier.split(self.curr_ff, s, z_transform)

        end_remaining_period = self.curr_period - (self.curr_period * s)
        sprime = 1 - end_remaining_period / end_period
        # sprime = self.gnp.clip(sprime, 0.01, 0.99)

        z_transform = self.bezier.get_z_transform(sprime)
        split_end_coeff_jt = self.bezier.split(end_coeff_jt, sprime, z_transform)
        split_end_coeff_ff = self.bezier.split(end_coeff_ff, sprime, z_transform)
        
        
        interp_coeff_jt = self.bezier.interpolate(
            split_coeff_jt,
            split_end_coeff_jt,
            deg=3
        )
        interp_coeff_ff = self.bezier.interpolate(
            split_coeff_ff,
            split_end_coeff_ff,
            deg=3
        )
        # interp_period = self.curr_period - (t - self.t0)
        interp_period = end_remaining_period 
        # interp_coeff_jt, interp_coeff_ff = self.bezier.relabel(
        #     jt_coeff=interp_coeff_jt,
        #     ff_coeff=interp_coeff_ff,
        # )

        interp_jt_dcoeff = self.bezier.compute_dcoeff(interp_coeff_jt)
        interp_ff_dcoeff = self.bezier.compute_dcoeff(interp_coeff_ff)
        return (interp_coeff_jt, interp_jt_dcoeff, 
                interp_coeff_ff, interp_ff_dcoeff, 
                interp_period)
        
    def _set_gait(self, vdes, t):
        cond = ((vdes[1] < 0.0) & (self.swing_leg == Leg.LEFT)) | ((vdes[1] > 0.0) & (self.swing_leg == Leg.RIGHT))
        ret = jax.lax.cond(
            cond,
            lambda _: self,
            lambda _: self._set_gait(vdes, t),
            None
        )
        # if cond:
        #     ret = self
        # else:
        #     ret = self._set_gait(vdes, t)
        return ret
    
    def set_gait(self, vdes, t):
        """Defines an interpolated gait between the current gait, and the input
        gait. This method should be iteratively run to ensure smoothness.

        rho_desired: (np.ndarray) gait to interpolate to, shape includes relabeling (2, nstates, ndegree + 1)

        Returns rho_predicted: (np.ndarray) interpolated gait, shape includes relabeling (2, nstates, ndegree + 1)
        """
        neighbors, weights = self.find_nearest(vdes)
        # print(neighbors, vdes, weights)
        jt_coeff, ff_coeff, jt_dcoeff, ff_dcoeff, step_period = self.combine_gaits(
            neighbors,
            weights
        )
        jt_coeff, jt_dcoeff, ff_coeff, ff_dcoeff, period = self.interpolate_gait(
            jt_coeff, ff_coeff, step_period, self.get_phase(t), t
        )
        
        # Update the current gait
        return self.set_new_coeffs(
            jt_coeff=jt_coeff.copy(),
            jt_dcoeff=jt_dcoeff.copy(),
            ff_coeff=ff_coeff.copy(),
            ff_dcoeff=ff_dcoeff.copy(),
            period=self.gnp.copy(period),
            t0=self.gnp.copy(t),
            t0_step=self.t0_step,
            vdes=self.gnp.copy(vdes),
            step_period=self.gnp.copy(step_period),
            swing_leg=self.swing_leg
        )
    
    def reset_gait(self, vdes, t, swing_leg):
        neighbors, weights = self.find_nearest(vdes)
        jt_coeff, ff_coeff, jt_dcoeff, ff_dcoeff, step_period = self.combine_gaits(
            neighbors,
            weights
        )
        
        # Update the current gait
        return self.set_new_coeffs(
            jt_coeff=jt_coeff.copy(),
            jt_dcoeff=jt_dcoeff.copy(),
            ff_coeff=ff_coeff.copy(),
            ff_dcoeff=ff_dcoeff.copy(),
            period=self.gnp.copy(step_period),
            t0=self.gnp.copy(t),
            t0_step=self.gnp.copy(t),
            vdes=self.gnp.copy(vdes),
            step_period=self.gnp.copy(step_period),
            swing_leg=swing_leg
        )
    
    def set_new_coeffs(self, jt_coeff, jt_dcoeff, ff_coeff, ff_dcoeff, period, t0, vdes, step_period, t0_step, swing_leg):
        return GaitLibrary(
            num_gaits    = self.gnp.copy(self.num_gaits),
            num_states   = self.gnp.copy(self.num_states),
            num_degree   = self.gnp.copy(self.num_degree),
            num_vx_gaits = self.gnp.copy(self.num_vx_gaits),
            bezier       = self.bezier,
            jt_coeffs    = self.gnp.copy(self.jt_coeffs),
            jt_dcoeffs   = self.gnp.copy(self.jt_dcoeffs),
            ff_coeffs    = self.gnp.copy(self.ff_coeffs),
            ff_dcoeffs   = self.gnp.copy(self.ff_dcoeffs),
            curr_jt      = self.gnp.copy(jt_coeff),
            curr_djt     = self.gnp.copy(jt_dcoeff),
            curr_ff      = self.gnp.copy(ff_coeff),
            curr_dff     = self.gnp.copy(ff_dcoeff),
            curr_period  = self.gnp.copy(period),
            step_period  = self.gnp.copy(step_period),
            curr_vdes    = self.gnp.copy(vdes),
            periods      = self.gnp.copy(self.periods),
            vel_xs       = self.gnp.copy(self.vel_xs),
            vel_ys       = self.gnp.copy(self.vel_ys),
            swing_leg    = self.gnp.copy(swing_leg),
            t0           = self.gnp.copy(t0),
            t0_step      = self.gnp.copy(t0_step),
            blend        = self.gnp.copy(self.blend),
            offset_pos   = self.gnp.copy(self.offset_pos),
            offset_vel   = self.gnp.copy(self.offset_pos),
            gnp          = self.gnp,
            side_offset  = self.gnp.copy(self.side_offset)
        )
            
    
    def get_phase(self, t: float):
        phase = (t - self.t0) / self.curr_period
        return phase
    
    def get_step_phase(self, t: float):
        phase = (t - self.t0_step) / self.step_period
        return phase

    def evaluate(self, s: float, leg: int):
        des_pos, des_vel = self.bezier.evaluate(
            s          = s,
            coeff      = self.curr_jt[leg],
            dcoeff     = self.curr_djt[leg],
            period     = self.curr_period
        )
        
        return self.gnp.hstack((des_pos, des_vel))
    
    def evaluate2(self, s: float, leg: int, coeff, dcoeff, period):
        des_pos, des_vel = self.bezier.evaluate(
            s          = s,
            coeff      = coeff[leg],
            dcoeff     = dcoeff[leg],
            period     = period
        )
        return self.gnp.hstack((des_pos, des_vel))
    
    def blend_gait(self, s, pos, vel):
        # Blending
        lambda_comb = self.gnp.clip(s / self.blend, 0, 1)
        des_pos_blend = (1-lambda_comb) * self.offset_pos + pos
        des_vel_blend = (1-lambda_comb) * self.offset_vel + vel
        return self.gnp.hstack((des_pos_blend, des_vel_blend))

    def __call__(self, s: int):
        state = self.evaluate(s, self.swing_leg)        
        return state
    
    def ff_evaluate(self, s: float):
        des_pos, des_vel = self.bezier.evaluate(
            s          = s,
            coeff      = self.curr_ff[self.swing_leg],
            dcoeff     = self.curr_dff[self.swing_leg],
            period     = self.curr_period
        )
        des_pos = self.bezier.to_quat(des_pos)
        # des_pos[1] -= self.side_offset[1]
        # des_pos[1] -= 0.22
        return self.gnp.hstack((des_pos, des_vel))
    
    
    def impact_reset(self, t: float, cond=jax.lax.cond):
        """Returns a new GaitLibrary object with the swing leg switched and
        blending applied
        """
        neighbors, weights = self.find_nearest(self.curr_vdes)
        jt_coeff, ff_coeff, jt_dcoeff, ff_dcoeff, period = self.combine_gaits(
            neighbors,
            weights
        )
        # print(period)
        new_swing_leg = cond(
            self.swing_leg == Leg.RIGHT,
            lambda _: Leg.LEFT,
            lambda _: Leg.RIGHT,
            None
        )
        

        new_t0 = self.gnp.copy(t)
        return GaitLibrary(
            num_gaits    = self.num_gaits,
            num_states   = self.num_states,
            num_degree   = self.num_degree,
            num_vx_gaits = self.num_vx_gaits,
            bezier       = self.bezier,
            jt_coeffs    = self.jt_coeffs,
            jt_dcoeffs   = self.jt_dcoeffs,
            ff_coeffs    = self.ff_coeffs,
            ff_dcoeffs   = self.ff_dcoeffs,
            curr_jt      = jt_coeff,
            curr_djt     = jt_dcoeff,
            curr_ff      = ff_coeff,
            curr_dff     = ff_dcoeff,
            curr_period  = period,
            step_period  = period,
            periods      = self.periods,
            curr_vdes    = self.curr_vdes,
            vel_xs       = self.vel_xs,
            vel_ys       = self.vel_ys,
            swing_leg    = new_swing_leg,
            t0           = new_t0 - 0.005,
            t0_step      = new_t0 - 0.005,
            blend        = self.blend,
            offset_pos   = self.offset_pos,
            offset_vel   = self.offset_vel,
            gnp          = self.gnp,
            side_offset  = self.side_offset
        )
    
    def tree_flatten(self):
        return (self.num_gaits,
                self.num_states,
                self.num_degree,
                self.num_vx_gaits,
                self.bezier,
                self.jt_coeffs,
                self.jt_dcoeffs,
                self.ff_coeffs,
                self.ff_dcoeffs,
                self.curr_jt,
                self.curr_djt,
                self.curr_ff,
                self.curr_dff,
                self.curr_period,
                self.step_period,
                self.curr_vdes,
                self.periods,
                self.vel_xs,
                self.vel_ys,
                self.swing_leg,
                self.t0,
                self.t0_step,
                self.blend,
                self.offset_pos,
                self.offset_vel), ()
    
    @classmethod
    def tree_unflatten(self, aux_data, children):
        return GaitLibrary(*children)
    

if __name__ == '__main__':
    # Example usage
    import matplotlib.pyplot as plt
    from utils.plotting import get_subplot_grid
    path = 'control/gaits/2D_library_v6'
    # path = 'control/gaits/Bruce_singleton_v1'1
    v0 = np.array([0.0, 0.0])
    num_states = 12
    num_degree = 7
    gaitlib = GaitLibrary.from_directory(
        path=path,
        v0=v0,
        num_states=num_states,
        num_degree=num_degree,
        gait_type='P2',
        gnp=np,
        fact=factorial,
    )
    
    nrows, ncols = get_subplot_grid(gaitlib.num_states)  
    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * nrows, 2 * ncols))
    axs = axs.flatten()
    
    def get_states(gaitlib, low=0, high=1):
        states = []
        phase = np.linspace(low, high, 100)
        for s in phase:
            state = gaitlib.evaluate(s, Leg.RIGHT)
            states.append(state)
        return np.array(states)

    phase = np.linspace(0, 1, 100)
    old_states = get_states(gaitlib)
    
    # T_SPLIT = 0.45
    # phase_split = gaitlib.get_phase(T_SPLIT)
    # gaitlib = gaitlib.set_gait(vdes=jnp.array([-0.3, 0.0]), t=T_SPLIT)
    # new_states = get_states(gaitlib)
    # global_phase = np.linspace(phase_split, 1, 100)

    # gaitlib = gaitlib.reset_gait(vdes=jnp.array([-0.3, 0.0]), t=0.0, swing_leg=Leg.LEFT)
    # phase = np.linspace(0, 1, 100)
    # next_states = get_states(gaitlib)
        
    for i in range(gaitlib.num_states):
        axs[i].plot(phase, old_states[:, num_states + i], c='r')
        # axs[i].plot(global_phase, new_states[:, i], c='b')
        # axs[i].plot(phase, next_states[:, i], c='g')
        axs[i].set_title(f'Joint {i+1}')
        # axs[i].set_xlabel('Normalized Time (s)')
        # axs[i].set_ylabel('State Value')
        # axs[i].set_xticks([])
        # axs[i].set_yticks([])
        # axs[i].set_xlabel('')
        # axs[i].set_ylabel('')

    path = 'visualization/gaits.png'
    print(f'Saving to {path}')
    fig.tight_layout()
    plt.savefig(path, dpi=300)