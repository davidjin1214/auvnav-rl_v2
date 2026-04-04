"""
REMUS 100 AUV -- 6-DOF dynamic model.

Provides the full nonlinear equations of motion for a REMUS-100-class autonomous
underwater vehicle, including rigid-body dynamics, added mass, cross-flow drag,
lift/drag, propulsion, control surfaces, and restoring forces.  Also includes
actuator dynamics, a fourth-order Runge-Kutta integrator, and several guidance /
control utilities (LOS, SMC heading, depth PID).

State vector (12 elements):
    x = [u, v, w, p, q, r, x_n, y_e, z_d, phi, theta, psi]
         -----body vel-----  ---body rate---  --NED pos--  --Euler angles--

Control input (3 elements):
    u = [delta_r, delta_s, n_rpm]
         rudder   elevator  propeller RPM
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Vec3 = NDArray[np.floating]
Vec6 = NDArray[np.floating]
Mat6 = NDArray[np.floating]
Array = NDArray[np.floating]


# ---------------------------------------------------------------------------
# State-vector index constants
# ---------------------------------------------------------------------------
class S:
    """Named indices into the 12-element state vector."""

    U, V, W, P, Q, R = 0, 1, 2, 3, 4, 5
    XN, YE, ZD, PHI, THETA, PSI = 6, 7, 8, 9, 10, 11
    N = 12

    BODY_VEL = slice(0, 3)
    BODY_RATE = slice(3, 6)
    NED_POS = slice(6, 9)
    EULER = slice(9, 12)

    # Channels that flip sign under port/starboard mirror symmetry
    LATERAL = [V, P, R, YE, PHI, PSI]  # indices 1, 3, 5, 7, 9, 11


# ---------------------------------------------------------------------------
# Basic math utilities
# ---------------------------------------------------------------------------

def ssa(angle: float) -> float:
    """Smallest signed angle -- maps *angle* into [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def sat(value: float, limit: float) -> float:
    """Symmetric saturation to [-limit, +limit]."""
    return float(np.clip(value, -limit, limit))


def gravity(latitude_rad: float) -> float:
    """Normal gravity on the WGS-84 ellipsoid (Somigliana formula)."""
    g_e = 9.7803253359
    k = 0.00193185265241
    e = 0.0818191908426
    sin_mu = np.sin(latitude_rad)
    return float(g_e * (1.0 + k * sin_mu**2) / np.sqrt(1.0 - e**2 * sin_mu**2))


def smtrx(a: Vec3) -> Array:
    """Skew-symmetric (cross-product) matrix: S(a) b = a x b."""
    return np.array(
        [[0.0, -a[2], a[1]],
         [a[2], 0.0, -a[0]],
         [-a[1], a[0], 0.0]],
        dtype=float,
    )


def r_zyx(phi: float, theta: float, psi: float) -> Array:
    """ZYX Euler-angle rotation matrix (body -> NED)."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array([
        [cpsi * cth, -spsi * cphi + cpsi * sth * sphi,  spsi * sphi + cpsi * cphi * sth],
        [spsi * cth,  cpsi * cphi + spsi * sth * sphi, -cpsi * sphi + sth * spsi * cphi],
        [-sth,        cth * sphi,                        cth * cphi],
    ], dtype=float)


def t_zyx(phi: float, theta: float, singular_tol: float = 1e-6) -> Array:
    """Euler-rate transform: omega_body -> [phi_dot, theta_dot, psi_dot]."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    if abs(cth) < singular_tol:
        raise ValueError("Euler-angle kinematics singular near theta = ±90 deg.")
    return np.array([
        [1.0, sphi * sth / cth, cphi * sth / cth],
        [0.0, cphi,             -sphi],
        [0.0, sphi / cth,        cphi / cth],
    ], dtype=float)


def m2c(M: Mat6, nu: Vec6) -> Mat6:
    """Coriolis-centripetal matrix from a 6x6 inertia matrix and velocity."""
    M_sym = 0.5 * (M + M.T)
    M11, M12 = M_sym[:3, :3], M_sym[:3, 3:]
    M22 = M_sym[3:, 3:]
    nu1, nu2 = nu[:3], nu[3:]
    p1 = M11 @ nu1 + M12 @ nu2
    p2 = M12.T @ nu1 + M22 @ nu2
    C = np.zeros((6, 6), dtype=float)
    C[:3, 3:] = -smtrx(p1)
    C[3:, :3] = -smtrx(p1)
    C[3:, 3:] = -smtrx(p2)
    return C


# ---------------------------------------------------------------------------
# Vehicle configuration data-classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class VehicleParams:
    """Physical parameters of the AUV hull, propulsion, and fins."""

    length: float = 1.6
    diameter: float = 0.19
    rho: float = 1026.0
    latitude_rad: float = np.deg2rad(63.4468)

    # Propeller
    d_prop: float = 0.14
    t_prop: float = 0.10
    kt_0: float = 0.4566
    kq_0: float = 0.0700
    kt_max: float = 0.1798
    kq_max: float = 0.0312
    ja_max: float = 0.6632

    # Control fins
    s_fin: float = 0.00665
    cl_delta_r: float = 0.5
    cl_delta_s: float = 0.7

    # Actuator limits
    delta_r_max: float = np.deg2rad(15.0)
    delta_s_max: float = np.deg2rad(15.0)
    n_rpm_max: float = 1525.0
    n_rpm_min: float = 200.0

    # Time constants and damping ratios
    t_surge: float = 20.0
    t_sway: float = 20.0
    t_yaw: float = 1.0
    zeta_roll: float = 0.3
    zeta_pitch: float = 0.8

    # CG/CB offsets and added-mass ratio
    cg_z: float = 0.02
    cb_z: float = 0.00
    r44: float = 0.3

    # Hull drag
    body_drag_cd: float = 0.42

    # Ellipsoidal correction factor for semi-axis computation
    geometry_scale: float = 1.0096


@dataclass(slots=True)
class ActuatorConfig:
    """First-order actuator dynamics parameters."""

    delta_tau: float = 0.25
    rpm_tau: float = 0.80
    delta_rate_limit: float = np.deg2rad(35.0)
    rpm_rate_limit: float = 400.0


@dataclass(slots=True)
class ActuatorState:
    """Current actuator output (rudder, elevator, RPM)."""

    delta_r: float = 0.0
    delta_s: float = 0.0
    n_rpm: float = 0.0

    def as_array(self) -> Array:
        return np.array([self.delta_r, self.delta_s, self.n_rpm], dtype=float)


@dataclass(slots=True)
class RelativeFlow:
    """Body-frame relative-flow quantities (after subtracting current)."""

    nu_r: Array        # 6-DOF relative velocity
    nu_c: Array        # 6-DOF current velocity in body frame
    u_r: float
    v_r: float
    w_r: float
    U_r: float         # relative speed magnitude
    alpha: float       # angle of attack
    beta: float        # sideslip angle
    alpha_abs: float
    beta_abs: float


@dataclass(slots=True)
class ValidityReport:
    """Result of a numerical sanity check on the state."""

    ok: bool
    reason: str


# ---------------------------------------------------------------------------
# Hydrodynamic coefficient data (DNV cylinder Cd vs Reynolds number)
# ---------------------------------------------------------------------------

_CD_DATA_DNV = np.array([
    [1.0211e4, 1.20769], [1.5544e4, 1.20369], [2.4435e4, 1.20823],
    [3.5720e4, 1.21060], [6.0881e4, 1.21093], [8.6175e4, 1.20901],
    [1.1809e5, 1.21134], [1.4926e5, 1.21148], [2.1471e5, 1.21171],
    [2.3094e5, 1.20109], [2.7158e5, 1.16918], [2.9710e5, 1.11163],
    [3.2525e5, 1.00926], [3.6194e5, 0.89411], [4.0980e5, 0.70855],
    [4.7530e5, 0.53155], [5.2473e5, 0.40785], [5.7892e5, 0.32683],
    [6.3829e5, 0.28422], [7.4405e5, 0.29711], [8.5324e5, 0.32280],
    [1.0687e6, 0.38909], [1.3934e6, 0.47033], [1.8319e6, 0.53878],
    [2.2584e6, 0.58372], [2.8990e6, 0.62868], [3.6633e6, 0.64803],
    [4.9373e6, 0.67808],
], dtype=float)

_KAPPA_SUBCRITICAL = np.array(
    [[2, 0.58], [5, 0.62], [10, 0.68], [20, 0.74],
     [40, 0.82], [50, 0.87], [100, 0.98]], dtype=float)

_KAPPA_SUPERCRITICAL = np.array(
    [[2, 0.80], [5, 0.80], [10, 0.82], [20, 0.90],
     [40, 0.98], [50, 0.99], [100, 1.00]], dtype=float)


def _cylinder_drag_coeff(length: float, diameter: float, nu_r: Vec6) -> float:
    """2-D cylinder drag coefficient adjusted for finite aspect ratio."""
    u_cross = float(np.hypot(nu_r[1], nu_r[2]))
    reynolds = u_cross * length * 1e6
    cd = float(np.interp(reynolds, _CD_DATA_DNV[:, 0], _CD_DATA_DNV[:, 1]))
    aspect_ratio = length / diameter
    table = _KAPPA_SUBCRITICAL if reynolds < 2e5 else _KAPPA_SUPERCRITICAL
    kappa = float(np.interp(aspect_ratio, table[:, 0], table[:, 1]))
    return cd * kappa


def _cross_flow_drag_out(length: float, diameter: float, nu_r: Vec6,
                         rho: float, out: Vec6, n_strips: int = 20) -> None:
    """Strip-theory cross-flow drag (vectorised over hull strips)."""
    dx = length / n_strips
    cd_2d = _cylinder_drag_coeff(length, diameter, nu_r)
    v_r = float(nu_r[1])
    w_r = float(nu_r[2])
    q = float(nu_r[4])
    r_rate = float(nu_r[5])

    x_l = np.linspace(-length / 2.0 + dx / 2.0, length / 2.0 - dx / 2.0, n_strips)

    v_local = v_r + x_l * r_rate
    w_local = w_r + x_l * q

    uh = np.abs(v_local) * v_local
    uv = np.abs(w_local) * w_local

    coeff = -0.5 * rho * diameter * cd_2d * dx
    out[0] = 0.0
    out[1] = coeff * np.sum(uh)
    out[2] = coeff * np.sum(uv)
    out[3] = 0.0
    out[4] = coeff * np.dot(x_l, uv)
    out[5] = coeff * np.dot(x_l, uh)


def _lift_drag_force_out(span: float, area: float, cd_0: float,
                         alpha: float, u_r: float, rho: float, out: Vec6) -> None:
    """Axial lift and induced drag on the hull (thin-wing model)."""
    if area <= 0.0:
        out.fill(0.0)
        return

    e_oswald = 0.3
    ar = span**2 / area
    cl_alpha = np.pi * ar / (1.0 + np.sqrt(1.0 + (ar / 2.0)**2))
    cl = cl_alpha * alpha
    cd = cd_0 + cl**2 / (np.pi * e_oswald * ar)

    q_dyn = 0.5 * rho * u_r**2 * area
    f_drag = q_dyn * cd
    f_lift = q_dyn * cl

    ca, sa = np.cos(alpha), np.sin(alpha)
    out[0] = ca * (-f_drag) - sa * (-f_lift)
    out[1] = 0.0
    out[2] = sa * (-f_drag) + ca * (-f_lift)
    out[3] = 0.0
    out[4] = 0.0
    out[5] = 0.0


# ---------------------------------------------------------------------------
# Main vehicle class
# ---------------------------------------------------------------------------


def smtrx_out(a: Array, out: Array) -> None:
    """Skew-symmetric (cross-product) matrix: S(a) b = a x b."""
    out[0, 0] = 0.0
    out[0, 1] = -a[2]
    out[0, 2] = a[1]
    out[1, 0] = a[2]
    out[1, 1] = 0.0
    out[1, 2] = -a[0]
    out[2, 0] = -a[1]
    out[2, 1] = a[0]
    out[2, 2] = 0.0

class Remus100:
    """REMUS 100 six-degree-of-freedom dynamic model."""

    def __init__(self,
                 params: Optional[VehicleParams] = None,
                 actuator: Optional[ActuatorConfig] = None) -> None:
        self.params = params or VehicleParams()
        self.actuator = actuator or ActuatorConfig()
        self.g_mu = gravity(self.params.latitude_rad)
        self._refresh_derived()

    def clone_with(self, **kwargs: float) -> Remus100:
        """Return a copy with selected VehicleParams fields overridden."""
        return Remus100(params=replace(self.params, **kwargs),
                        actuator=self.actuator)

    # -- derived geometry ---------------------------------------------------

    def _refresh_derived(self) -> None:
        p = self.params
        self.a = p.geometry_scale * p.length / 2.0       # ellipsoid semi-major
        self.b = p.geometry_scale * p.diameter / 2.0      # ellipsoid semi-minor
        self.S_ref = 0.7 * p.length * p.diameter          # reference area
        self.CD_0 = p.body_drag_cd * np.pi * self.b**2 / max(self.S_ref, 1e-8)
        self.x_fin = -self.a                              # fin position (stern)
        self.A_rudder = 2.0 * p.s_fin
        self.A_stern = 2.0 * p.s_fin
        self.r_bG = np.array([0.0, 0.0, p.cg_z], dtype=float)
        self.r_bB = np.array([0.0, 0.0, p.cb_z], dtype=float)

        m = (4.0 / 3.0) * np.pi * p.rho * self.a * self.b**2
        self.mass = m
        Ix = (2.0 / 5.0) * m * self.b**2
        Iy = (1.0 / 5.0) * m * (self.a**2 + self.b**2)
        self.Ig = np.array([Ix, Iy, Iy], dtype=float)
        
        MRB_CG = np.diag([m, m, m, Ix, Iy, Iy]).astype(float)
        
        self.H = np.block([
            [np.eye(3), smtrx(self.r_bG).T],
            [np.zeros((3, 3)), np.eye(3)],
        ])
        self.HT = self.H.T.copy()
        self.MRB = self.HT @ MRB_CG @ self.H

        MA_44 = p.r44 * Ix
        e = np.sqrt(max(1.0 - (self.b / self.a)**2, 1e-8))
        alpha_0 = (2.0 * (1.0 - e**2) / e**3) * (
            0.5 * np.log((1.0 + e) / (1.0 - e)) - e)
        beta_0 = (1.0 / e**2
                   - (1.0 - e**2) / (2.0 * e**3)
                   * np.log((1.0 + e) / (1.0 - e)))
        k1 = alpha_0 / (2.0 - alpha_0)
        k2 = beta_0 / (2.0 - beta_0)
        k_prime = (e**4 * (beta_0 - alpha_0)
                   / ((2.0 - e**2)
                      * (2.0 * e**2 - (2.0 - e**2) * (beta_0 - alpha_0))))
        self.MA_diag = np.array([
            m * k1, m * k2, m * k2,
            MA_44, k_prime * Iy, k_prime * Iy,
        ], dtype=float)
        self.MA = np.diag(self.MA_diag).astype(float)

        self.M = self.MRB + self.MA
        
        self._nu_c = np.zeros(6, dtype=float)
        self._nu_r = np.zeros(6, dtype=float)
        self._Dnu_c = np.zeros(6, dtype=float)
        self._CRB_CG = np.zeros((6, 6), dtype=float)
        self._CRB_tmp = np.zeros((6, 6), dtype=float)
        self._CRB = np.zeros((6, 6), dtype=float)
        self._CA = np.zeros((6, 6), dtype=float)
        self._C = np.zeros((6, 6), dtype=float)
        self._D = np.zeros((6, 6), dtype=float)
        self._tau_ctrl = np.zeros(6, dtype=float)
        self._tau_ld = np.zeros(6, dtype=float)
        self._tau_cf = np.zeros(6, dtype=float)
        self._g_vec = np.zeros(6, dtype=float)
        self._rhs = np.zeros(6, dtype=float)
        self._eta_dot = np.zeros(6, dtype=float)
        self._x_dot = np.zeros(12, dtype=float)
        
        self.W = self.MRB[0, 0] * gravity(p.latitude_rad)
        self.B = self.W

    # -- relative flow ------------------------------------------------------

    def compute_relative_flow(self, x: Array,
                              Vc: float = 0.0, beta_Vc: float = 0.0,
                              w_c: float = 0.0,
                              min_forward_speed: float = 0.05) -> RelativeFlow:
        """Compute body-frame relative-flow quantities."""
        nu = x[:6]
        psi = float(x[S.PSI])
        self._nu_c[0] = Vc * np.cos(beta_Vc - psi)
        self._nu_c[1] = Vc * np.sin(beta_Vc - psi)
        self._nu_c[2] = w_c
        self._nu_r[:] = nu - self._nu_c
        
        u_r, v_r, w_r = float(self._nu_r[0]), float(self._nu_r[1]), float(self._nu_r[2])
        U_r = float(np.linalg.norm(self._nu_r[:3]))

        safe_u = u_r if abs(u_r) >= min_forward_speed else (
            min_forward_speed if u_r >= 0.0 else -min_forward_speed)
        alpha = float(np.arctan2(w_r, safe_u))
        beta = float(np.arctan2(v_r, safe_u))

        return RelativeFlow(
            nu_r=self._nu_r, nu_c=self._nu_c,
            u_r=u_r, v_r=v_r, w_r=w_r, U_r=U_r,
            alpha=alpha, beta=beta,
            alpha_abs=abs(alpha), beta_abs=abs(beta),
        )

    # -- rigid-body and added mass matrices (combined) ----------------------

    def _compute_C_matrices(self, omega: Vec3, nu_r: Vec6) -> tuple[Mat6, Mat6]:
        """Compute CRB and CA in-place and return them."""
        smtrx_out(omega, self._CRB_CG[0:3, 0:3])
        self._CRB_CG[0:3, 0:3] *= self.mass
        
        Ig_omega = self.Ig * omega
        smtrx_out(Ig_omega, self._CRB_CG[3:6, 3:6])
        self._CRB_CG[3:6, 3:6] *= -1.0
        
        np.dot(self.HT, np.dot(self._CRB_CG, self.H, out=self._CRB_tmp), out=self._CRB)
        
        p1 = self.MA_diag[:3] * nu_r[:3]
        p2 = self.MA_diag[3:] * nu_r[3:]
        
        smtrx_out(p1, self._CA[0:3, 3:6])
        self._CA[0:3, 3:6] *= -1.0
        self._CA[3:6, 0:3] = self._CA[0:3, 3:6]
        
        smtrx_out(p2, self._CA[3:6, 3:6])
        self._CA[3:6, 3:6] *= -1.0
        
        self._CA[0, 4] = self._CA[0, 5] = self._CA[1, 5] = self._CA[2, 4] = 0.0
        self._CA[4, 0] = self._CA[4, 2] = self._CA[5, 0] = self._CA[5, 1] = 0.0
        
        return self._CRB, self._CA

    # -- restoring forces ---------------------------------------------------

    def _compute_restoring_forces(self, R: Array) -> Vec6:
        """Gravity and buoyancy restoring force/moment vector."""
        dW = self.W - self.B
        c30, c31, c32 = R[2, 0], R[2, 1], R[2, 2]
        self._g_vec[0] = -dW * c30
        self._g_vec[1] = -dW * c31
        self._g_vec[2] = -dW * c32
        
        W, B, rG, rB = self.W, self.B, self.r_bG, self.r_bB
        self._g_vec[3] = -(rG[1] * W - rB[1] * B) * c32 + (rG[2] * W - rB[2] * B) * c31
        self._g_vec[4] = -(rG[2] * W - rB[2] * B) * c30 + (rG[0] * W - rB[0] * B) * c32
        self._g_vec[5] = -(rG[0] * W - rB[0] * B) * c31 + (rG[1] * W - rB[1] * B) * c30
        return self._g_vec

    # -- damping matrix -----------------------------------------------------

    def _compute_damping_matrix(self, U_r: float) -> Mat6:
        """Linear + speed-dependent damping."""
        p, M = self.params, self.M
        dz = max(self.r_bG[2] - self.r_bB[2], 1e-6)
        w4 = np.sqrt(max(self.W * dz / max(M[3, 3], 1e-8), 1e-8))
        w5 = np.sqrt(max(self.W * dz / max(M[4, 4], 1e-8), 1e-8))

        self._D[0, 0] = M[0, 0] / p.t_surge
        self._D[1, 1] = M[1, 1] / p.t_sway
        self._D[2, 2] = M[2, 2] / p.t_sway
        self._D[3, 3] = M[3, 3] * 2.0 * p.zeta_roll * w4
        self._D[4, 4] = M[4, 4] * 2.0 * p.zeta_pitch * w5
        self._D[5, 5] = M[5, 5] / p.t_yaw

        exp_factor = np.exp(-3.0 * U_r)
        self._D[0, 0] *= exp_factor
        self._D[1, 1] *= exp_factor
        return self._D

    # -- propulsion and control surfaces ------------------------------------

    def _compute_propulsion_and_control(self, flow: RelativeFlow,
                                        U: float, delta_r: float,
                                        delta_s: float, n_rpm: float) -> Vec6:
        """Propeller thrust/torque and fin forces."""
        p = self.params
        n_rps = n_rpm / 60.0
        Va = 0.944 * U
        Ja = Va / (n_rps * p.d_prop) if abs(n_rps) > 1e-8 else 0.0

        if abs(n_rps) > 1e-8:
            KT = p.kt_0 + (p.kt_max - p.kt_0) / p.ja_max * Ja
            KQ = p.kq_0 + (p.kq_max - p.kq_0) / p.ja_max * Ja
        else:
            KT, KQ = p.kt_0, p.kq_0

        X_prop = p.rho * p.d_prop**4 * KT * abs(n_rps) * n_rps
        K_prop = p.rho * p.d_prop**5 * KQ * abs(n_rps) * n_rps

        U_rh = float(np.hypot(flow.u_r, flow.v_r))
        U_rv = float(np.hypot(flow.u_r, flow.w_r))

        X_r = -0.5 * p.rho * U_rh**2 * self.A_rudder * p.cl_delta_r * delta_r**2
        X_s = -0.5 * p.rho * U_rv**2 * self.A_stern * p.cl_delta_s * delta_s**2
        Y_r = -0.5 * p.rho * U_rh**2 * self.A_rudder * p.cl_delta_r * delta_r
        Z_s = -0.5 * p.rho * U_rv**2 * self.A_stern * p.cl_delta_s * delta_s

        self._tau_ctrl[0] = (1.0 - p.t_prop) * X_prop + X_r + X_s
        self._tau_ctrl[1] = Y_r
        self._tau_ctrl[2] = Z_s
        self._tau_ctrl[3] = K_prop / 10.0
        self._tau_ctrl[4] = -self.x_fin * Z_s
        self._tau_ctrl[5] = self.x_fin * Y_r
        return self._tau_ctrl

    # -- control helpers ----------------------------------------------------

    def saturate_control(self, cmd: Array) -> Array:
        """Clip a command vector to actuator limits."""
        p = self.params
        return np.array([
            np.clip(cmd[0], -p.delta_r_max, p.delta_r_max),
            np.clip(cmd[1], -p.delta_s_max, p.delta_s_max),
            np.clip(cmd[2], -p.n_rpm_max, p.n_rpm_max),
        ], dtype=float)

    def step_actuators(self, cmd: Array,
                       state: Optional[ActuatorState],
                       dt: float) -> ActuatorState:
        """Advance actuator states by one time step (first-order + rate limit)."""
        if state is None:
            state = ActuatorState()
        cmd_sat = self.saturate_control(cmd)
        ac = self.actuator

        def _first_order(current: float, target: float,
                         tau: float, rate_limit: float) -> float:
            rate = (target - current) / max(tau, 1e-6)
            rate = float(np.clip(rate, -rate_limit, rate_limit))
            return current + dt * rate

        p = self.params
        return ActuatorState(
            delta_r=float(np.clip(
                _first_order(state.delta_r, cmd_sat[0],
                             ac.delta_tau, ac.delta_rate_limit),
                -p.delta_r_max, p.delta_r_max)),
            delta_s=float(np.clip(
                _first_order(state.delta_s, cmd_sat[1],
                             ac.delta_tau, ac.delta_rate_limit),
                -p.delta_s_max, p.delta_s_max)),
            n_rpm=float(np.clip(
                _first_order(state.n_rpm, cmd_sat[2],
                             ac.rpm_tau, ac.rpm_rate_limit),
                -p.n_rpm_max, p.n_rpm_max)),
        )

    # -- equations of motion ------------------------------------------------

    def dynamics(self, x: Array, ui: Array,
                 Vc: float = 0.0, beta_Vc: float = 0.0,
                 w_c: float = 0.0) -> tuple[Array, float, Mat6]:
        """
        Evaluate 6-DOF equations of motion.
        """
        p = self.params
        nu, eta = x[:6], x[6:]
        phi, theta, psi = float(eta[3]), float(eta[4]), float(eta[5])
        
        cmd_sat = self.saturate_control(ui)
        delta_r, delta_s, n_rpm = cmd_sat[0], cmd_sat[1], cmd_sat[2]

        flow = self.compute_relative_flow(x, Vc=Vc, beta_Vc=beta_Vc, w_c=w_c)
        U = float(np.linalg.norm(nu[:3]))
        u_c, v_c = flow.nu_c[0], flow.nu_c[1]

        self._Dnu_c[0] = nu[5] * v_c
        self._Dnu_c[1] = -nu[5] * u_c
        
        CRB, CA = self._compute_C_matrices(nu[3:], flow.nu_r)
        self._C[:] = CRB + CA

        R = r_zyx(phi, theta, psi)
        g_vec = self._compute_restoring_forces(R)

        D = self._compute_damping_matrix(flow.U_r)

        _lift_drag_force_out(p.diameter, self.S_ref, self.CD_0,
                             flow.alpha, flow.U_r, p.rho, self._tau_ld)
        _cross_flow_drag_out(p.length, p.diameter, flow.nu_r, p.rho, self._tau_cf)
        tau_ctrl = self._compute_propulsion_and_control(
            flow, U, delta_r, delta_s, n_rpm)

        self._rhs[:] = tau_ctrl
        self._rhs += self._tau_ld
        self._rhs += self._tau_cf
        self._rhs -= self._C @ flow.nu_r
        self._rhs -= D @ flow.nu_r
        self._rhs -= g_vec
        
        nu_dot = self._Dnu_c + np.linalg.solve(self.M, self._rhs)

        T = t_zyx(phi, theta)
        self._eta_dot[:3] = R @ nu[:3]
        self._eta_dot[3:] = T @ nu[3:]

        self._x_dot[:6] = nu_dot
        self._x_dot[6:] = self._eta_dot
        
        return self._x_dot.copy(), U, self.M

    # -- validity check -----------------------------------------------------

    def state_validity_check(self, x: Array,
                             x_dot: Optional[Array] = None,
                             flow: Optional[RelativeFlow] = None,
                             max_speed: float = 3.5,
                             max_angle_deg: float = 85.0,
                             max_rate: float = 5.0,
                             max_derivative: float = 250.0) -> ValidityReport:
        """Quick numerical sanity check on the state vector."""
        if not np.all(np.isfinite(x)):
            return ValidityReport(False, "nonfinite_state")
        nu, eta = x[:6], x[6:]
        if float(np.linalg.norm(nu[:3])) > max_speed:
            return ValidityReport(False, "speed_limit")
        if np.max(np.abs(np.rad2deg(eta[3:5]))) > max_angle_deg:
            return ValidityReport(False, "attitude_limit")
        if np.max(np.abs(nu[3:])) > max_rate:
            return ValidityReport(False, "angular_rate_limit")
        if flow is not None and (not np.isfinite(flow.U_r)
                                  or flow.U_r > max_speed + 1.0):
            return ValidityReport(False, "relative_speed_limit")
        if x_dot is not None:
            if not np.all(np.isfinite(x_dot)):
                return ValidityReport(False, "nonfinite_derivative")
            if np.max(np.abs(x_dot[:6])) > max_derivative:
                return ValidityReport(False, "derivative_limit")
        return ValidityReport(True, "ok")


# ---------------------------------------------------------------------------
# Fourth-order Runge-Kutta integrator
# ---------------------------------------------------------------------------

def rk4_step(vehicle: Remus100, x: Array, ui: Array, h: float,
             Vc: float = 0.0, beta_Vc: float = 0.0,
             w_c: float = 0.0) -> Array:
    """Single RK4 integration step of the vehicle dynamics."""
    k1, _, _ = vehicle.dynamics(x, ui, Vc, beta_Vc, w_c)
    k2, _, _ = vehicle.dynamics(x + 0.5 * h * k1, ui, Vc, beta_Vc, w_c)
    k3, _, _ = vehicle.dynamics(x + 0.5 * h * k2, ui, Vc, beta_Vc, w_c)
    k4, _, _ = vehicle.dynamics(x + h * k3, ui, Vc, beta_Vc, w_c)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# Guidance and control utilities
# ---------------------------------------------------------------------------

class LowPassFilter:
    """First-order discrete low-pass filter."""

    def __init__(self, wn: float, h: float, initial_value: float = 0.0) -> None:
        self.alpha = float(np.exp(-wn * h))
        self.y = float(initial_value)

    def update(self, u: float) -> float:
        self.y = self.alpha * self.y + (1.0 - self.alpha) * u
        return self.y


class ALOS2D:
    """Adaptive Line-Of-Sight guidance for 2-D waypoint tracking."""

    def __init__(self, R_switch: float, Delta_h: float,
                 gamma_h: float, h: float) -> None:
        self.R_switch = R_switch
        self.Delta_h = Delta_h
        self.gamma_h = gamma_h
        self.h = h
        self.k = 0
        self.beta_hat = 0.0
        self.xk: Optional[float] = None
        self.yk: Optional[float] = None

    def update(self, x: float, y: float,
               wpt_x: Array, wpt_y: Array) -> tuple[float, float, int]:
        if self.xk is None or self.yk is None:
            self.xk = float(wpt_x[self.k])
            self.yk = float(wpt_y[self.k])

        n = len(wpt_x)
        if self.k < n - 1:
            xk_next = float(wpt_x[self.k + 1])
            yk_next = float(wpt_y[self.k + 1])
        else:
            bearing = np.arctan2(wpt_y[-1] - wpt_y[-2],
                                 wpt_x[-1] - wpt_x[-2])
            xk_next = float(wpt_x[-1] + 1e5 * np.cos(bearing))
            yk_next = float(wpt_y[-1] + 1e5 * np.sin(bearing))

        pi_h = float(np.arctan2(yk_next - self.yk, xk_next - self.xk))
        y_e = float(-(x - self.xk) * np.sin(pi_h)
                     + (y - self.yk) * np.cos(pi_h))

        d_seg = float(np.hypot(xk_next - self.xk, yk_next - self.yk))
        x_proj = float((x - self.xk) * np.cos(pi_h)
                       + (y - self.yk) * np.sin(pi_h))

        if (d_seg - x_proj < self.R_switch) and (self.k < n - 1):
            self.k += 1
            self.xk, self.yk = xk_next, yk_next

        psi_ref = pi_h - self.beta_hat - np.arctan2(y_e, self.Delta_h)
        self.beta_hat += (self.h * self.gamma_h * self.Delta_h * y_e
                          / np.sqrt(self.Delta_h**2 + y_e**2))
        return float(psi_ref), y_e, self.k


class IntegralSMCHeading:
    """Integral sliding-mode heading controller (Nomoto model)."""

    def __init__(self, h: float, K_d: float, K_sigma: float,
                 lambda_p: float, phi_b: float,
                 K_nomoto: float, T_nomoto: float) -> None:
        self.h = h
        self.K_d = K_d
        self.K_sigma = K_sigma
        self.lambda_p = lambda_p
        self.phi_b = phi_b
        self.K_nomoto = K_nomoto
        self.T_nomoto = T_nomoto
        self.psi_int = 0.0

    def update(self, psi: float, r: float,
               psi_d: float, r_d: float,
               a_d: float = 0.0) -> float:
        lam = self.lambda_p
        e_psi = ssa(psi - psi_d)
        e_r = r - r_d
        r_r = r_d - 2.0 * lam * e_psi - lam**2 * self.psi_int
        r_r_dot = a_d - 2.0 * lam * e_r - lam**2 * e_psi
        sigma = r - r_r

        if abs(sigma / self.phi_b) > 1.0:
            sw = self.K_sigma * np.sign(sigma)
        else:
            sw = self.K_sigma * sigma / self.phi_b

        delta = ((self.T_nomoto * r_r_dot + r_r
                   - self.K_d * sigma - sw) / self.K_nomoto)
        self.psi_int += self.h * e_psi
        return float(delta)


class DepthController:
    """Cascaded depth-pitch controller with gradient-based reference."""

    THETA_MAX = np.deg2rad(30.0)

    def __init__(self, h: float, Kp_z: float, T_z: float,
                 k_grad: float, Kp_theta: float,
                 Kd_theta: float, Ki_theta: float) -> None:
        self.h = h
        self.Kp_z = Kp_z
        self.T_z = T_z
        self.k_grad = k_grad
        self.Kp_theta = Kp_theta
        self.Kd_theta = Kd_theta
        self.Ki_theta = Ki_theta
        self.z_int = 0.0
        self.theta_int = 0.0
        self.theta_d = 0.0

    def update(self, z_ref: float, z_now: float,
               theta: float, q: float,
               u: float, w: float) -> tuple[float, float]:
        sigma_z = (-u * np.sin(theta) + w * np.cos(theta)
                   + self.Kp_z * ((z_now - z_ref) + self.z_int / self.T_z))

        if abs(u) > 1e-6:
            grad_J = (-(u * np.cos(theta) + w * np.sin(theta))
                      if abs(w / u) > 0.176 else -np.sign(u))
        else:
            grad_J = (-(u * np.cos(theta) + w * np.sin(theta))
                      if abs(w) > 0.1 else -np.sign(u))

        self.theta_d -= self.h * self.k_grad * grad_J * sigma_z
        self.theta_d = sat(self.theta_d, self.THETA_MAX)

        e_theta = ssa(theta - self.theta_d)
        delta_s = (-self.Kp_theta * e_theta
                   - self.Kd_theta * q
                   - self.Ki_theta * self.theta_int)

        self.z_int += self.h * (z_now - z_ref)
        self.theta_int += self.h * e_theta
        return float(delta_s), float(self.theta_d)


class LOSObserver:
    """Second-order LOS reference filter / observer."""

    def __init__(self, h: float, K_f: float) -> None:
        self.h = h
        self.K_f = K_f
        self.T_f = 1.0 / (K_f + 2.0 * np.sqrt(K_f) + 1.0)
        self.xi = 0.0

    def update(self, psi_d: float, r_d: float,
               psi_ref: float) -> tuple[float, float]:
        psi_d_next = psi_d + self.h * (r_d + self.K_f * ssa(psi_ref - psi_d))
        phi_coeff = np.exp(-self.h / self.T_f)
        self.xi = phi_coeff * self.xi + (1.0 - phi_coeff) * psi_d_next
        r_d_next = psi_d_next - self.xi
        return float(psi_d_next), float(r_d_next)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    vehicle = Remus100()
    x0 = np.zeros(S.N, dtype=float)
    x0[S.U] = 1.5
    ui = np.array([0.0, 0.0, 1200.0], dtype=float)
    x_dot, U, M = vehicle.dynamics(x0, ui)
    print("x_dot[:6] =", np.round(x_dot[:6], 6))
    print("U         =", round(U, 4))
    print("diag(M)   =", np.round(np.diag(M), 4))
