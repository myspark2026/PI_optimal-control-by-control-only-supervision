"""SDRE trajectory generators; plotting removed and CARE failures are fatal.
V_all=x^T P(x)x and DV_all=2P(x)x are SDRE proxies, not optimal-value labels.
The learning methods train on control labels, not these value proxies.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import scipy.io as sio
from tqdm import tqdm

class CuckerDataGenerator:

    def __init__(self, Na=20, T=10.0, dt=0.01):
        self.Na = Na
        self.state_dim = 2 * Na
        self.control_dim = Na
        self.T = T
        self.dt = dt
        self.time_points = np.arange(0, T + dt, dt)
        self.ONa = np.zeros((Na, Na))
        self.INa = np.eye(Na)
        self.Q = 1.0 / Na * np.eye(self.state_dim)
        self.R = 1.0 / Na * np.eye(self.control_dim)
        self.R_inv = Na * np.eye(self.control_dim)

    def compute_A_matrix(self, y):
        A = np.zeros((self.Na, self.Na))
        y_i = y[:, np.newaxis]
        y_j = y[np.newaxis, :]
        P = 1.0 / (1.0 + (y_i - y_j) ** 2)
        diag_mask = np.eye(self.Na, dtype=bool)
        off_diag_sum = np.sum(P * ~diag_mask, axis=1)
        np.fill_diagonal(A, -(1.0 / self.Na) * off_diag_sum)
        A = A + 1.0 / self.Na * P * ~diag_mask
        return A

    def compute_B_matrix(self):
        B_top = np.zeros((self.Na, self.Na))
        B_bottom = np.eye(self.Na)
        B = np.vstack([B_top, B_bottom])
        return B

    def compute_system_matrices(self, x):
        y = x[:self.Na]
        A_small = self.compute_A_matrix(y)
        A_top = np.hstack([self.ONa, self.INa])
        A_bottom = np.hstack([self.ONa, A_small])
        A = np.vstack([A_top, A_bottom])
        B = self.compute_B_matrix()
        return (A, B)

    def solve_sdre_are(self, x):
        (A, B) = self.compute_system_matrices(x)
        try:
            P = solve_continuous_are(A, B, self.Q, self.R)
            K = self.R_inv @ B.T @ P
            u_opt = -K @ x
            V_opt = x.T @ P @ x
            grad_V_opt = 2 * P @ x
            return (u_opt, V_opt, grad_V_opt, P)
        except Exception as e:
            raise RuntimeError('SDRE Riccati solve failed; no zero-control fallback was used.') from e

    def compute_optimal_control(self, x):
        return self.solve_sdre_are(x)[:3]

    def system_dynamics(self, t, x, u=None):
        if u is None:
            u = np.zeros(self.control_dim)
        y = x[:self.Na]
        A_small = self.compute_A_matrix(y)
        M_top = np.hstack([self.ONa, self.INa])
        M_bottom = np.hstack([self.ONa, A_small])
        M = np.vstack([M_top, M_bottom])
        B_top = np.zeros((self.Na, self.Na))
        B_bottom = np.eye(self.Na)
        B = np.vstack([B_top, B_bottom])
        dxdt = M @ x + B @ u
        return dxdt

    def generate_optimal_trajectory(self, x0):
        n_steps = len(self.time_points)
        x_traj = np.zeros((self.state_dim, n_steps))
        u_traj = np.zeros((self.control_dim, n_steps))
        V_traj = np.zeros(n_steps)
        gradV_traj = np.zeros((self.state_dim, n_steps))
        x_traj[:, 0] = x0
        (u_opt, V_opt, grad_V_opt) = self.compute_optimal_control(x0)
        u_traj[:, 0] = u_opt
        V_traj[0] = V_opt
        gradV_traj[:, 0] = grad_V_opt
        for k in range(n_steps - 1):
            t_current = self.time_points[k]
            x_current = x_traj[:, k]
            (u_opt, V_opt, grad_V_opt) = self.compute_optimal_control(x_current)
            u_traj[:, k] = u_opt

            def ode_func(t, x):
                return self.system_dynamics(t, x, u_opt)
            sol = solve_ivp(ode_func, [t_current, t_current + self.dt], x_current, t_eval=[t_current + self.dt], method='RK45')
            x_next = sol.y[:, -1]
            x_traj[:, k + 1] = x_next
            (u_next, V_next, grad_V_next) = self.compute_optimal_control(x_next)
            V_traj[k + 1] = V_next
            gradV_traj[:, k + 1] = grad_V_next
        (u_final, _, _) = self.compute_optimal_control(x_traj[:, -1])
        u_traj[:, -1] = u_final
        return (x_traj, u_traj, V_traj, gradV_traj)

    def generate_training_data(self, n_trajectories=100, n_samples_per_traj=100, pos_range=(-2, 2), vel_range=(-1, 1)):
        print(f'生成训练数据: {n_trajectories}条轨迹，每条轨迹{n_samples_per_traj}个采样点')
        total_samples = n_trajectories * n_samples_per_traj
        X_all = np.zeros((total_samples, self.state_dim))
        V_all = np.zeros((total_samples, 1))
        DV_all = np.zeros((total_samples, self.state_dim))
        U_all = np.zeros((total_samples, self.control_dim))
        sample_idx = 0
        for traj_idx in tqdm(range(n_trajectories), desc='生成轨迹'):
            x0 = np.zeros(self.state_dim)
            x0[:self.Na] = np.random.uniform(pos_range[0], pos_range[1], self.Na)
            x0[self.Na:] = np.random.uniform(vel_range[0], vel_range[1], self.Na)
            (x_traj, u_traj, V_traj, gradV_traj) = self.generate_optimal_trajectory(x0)
            traj_length = len(self.time_points)
            if traj_length < n_samples_per_traj:
                sample_indices = np.arange(traj_length)
            else:
                sample_indices = np.linspace(0, traj_length - 1, n_samples_per_traj, dtype=int)
            for idx in sample_indices:
                X_all[sample_idx] = x_traj[:, idx]
                V_all[sample_idx] = V_traj[idx]
                DV_all[sample_idx] = gradV_traj[:, idx]
                U_all[sample_idx] = u_traj[:, idx]
                sample_idx += 1
        X_all = X_all[:sample_idx]
        V_all = V_all[:sample_idx]
        DV_all = DV_all[:sample_idx]
        U_all = U_all[:sample_idx]
        print(f'生成完成！共{sample_idx}个样本')
        print(f'状态X形状: {X_all.shape}')
        print(f'价值V形状: {V_all.shape}')
        print(f'梯度DV形状: {DV_all.shape}')
        print(f'控制U形状: {U_all.shape}')
        return (X_all, V_all, DV_all, U_all)

    def save_data(self, filename, X, V, DV, U):
        data_dict = {'X': X, 'V_all': V, 'DV_all': DV.T, 'U_all': U.T, 'Na': self.Na, 'state_dim': self.state_dim, 'control_dim': self.control_dim}
        sio.savemat(filename, data_dict)
        print(f'数据已保存到: {filename}')

class AllenCahnDataGenerator:

    def __init__(self, d=51, T=2.0, dt=0.01):
        self.state_dim = d
        self.control_dim = 1
        self.T = T
        self.dt = dt
        self.time_points = np.arange(0, T + dt, dt)
        self.sigma = 0.1
        self.xi_range = [0, 1]
        self.omega_left = 0.6
        self.omega_right = 0.9
        (self.A_base, self.xi, self.h) = self._build_fd_A_base()
        self.B = self._build_fd_B()
        self.Q = np.eye(self.state_dim) * 0.02
        self.R = 0.1 * np.eye(self.control_dim)
        self.R_inv = 1.0 / 0.1 * np.eye(self.control_dim)

    def _build_fd_A_base(self):
        d = self.state_dim
        h = (self.xi_range[1] - self.xi_range[0]) / (d - 1)
        xi = np.linspace(self.xi_range[0], self.xi_range[1], d).reshape(-1, 1)
        A_base = np.zeros((d, d))
        coeff = self.sigma / h ** 2
        A_base[0, 0] = -2 * coeff
        A_base[0, 1] = 2 * coeff
        for i in range(1, d - 1):
            A_base[i, i - 1] = coeff
            A_base[i, i] = -2 * coeff
            A_base[i, i + 1] = coeff
        A_base[d - 1, d - 2] = 2 * coeff
        A_base[d - 1, d - 1] = -2 * coeff
        return (A_base, xi, h)

    def _build_fd_B(self):
        B = np.zeros((self.state_dim, 1))
        omega_idx = np.where((self.xi >= self.omega_left) & (self.xi <= self.omega_right))[0]
        B[omega_idx, 0] = 1.0
        return B

    def compute_system_matrices(self, x):
        x_flat = x.flatten()
        A_x = self.A_base + np.diag(1.0 - x_flat ** 2)
        return (A_x, self.B)

    def solve_sdre_are(self, x):
        (A_x, B) = self.compute_system_matrices(x)
        x_col = x.reshape(-1, 1)
        try:
            P = solve_continuous_are(A_x, B, self.Q, self.R)
            K = self.R_inv @ B.T @ P
            u_opt = -K @ x_col
            u_opt_flat = u_opt.flatten()
            V_opt = (x_col.T @ P @ x_col).item()
            grad_V_opt = (2 * P @ x_col).flatten()
            return (u_opt_flat, V_opt, grad_V_opt, P)
        except Exception as e:
            raise RuntimeError('SDRE Riccati solve failed; no zero-control fallback was used.') from e

    def compute_optimal_control(self, x):
        return self.solve_sdre_are(x)[:3]

    def system_dynamics(self, t, x, u=None):
        if u is None:
            u = np.zeros(self.control_dim)
        x_flat = x.flatten()
        dxdt = self.A_base @ x_flat + x_flat * (1.0 - x_flat ** 2) + self.B.flatten() * u[0]
        return dxdt

    def generate_optimal_trajectory(self, x0):
        n_steps = len(self.time_points)
        x_traj = np.zeros((self.state_dim, n_steps))
        u_traj = np.zeros((self.control_dim, n_steps))
        V_traj = np.zeros(n_steps)
        gradV_traj = np.zeros((self.state_dim, n_steps))
        x_traj[:, 0] = x0
        (u_opt, V_opt, grad_V_opt) = self.compute_optimal_control(x0)
        u_traj[:, 0] = u_opt
        V_traj[0] = V_opt
        gradV_traj[:, 0] = grad_V_opt
        for k in range(n_steps - 1):
            t_current = self.time_points[k]
            x_current = x_traj[:, k]
            (u_opt, V_opt, grad_V_opt) = self.compute_optimal_control(x_current)
            u_traj[:, k] = u_opt

            def ode_func(t, x):
                return self.system_dynamics(t, x, u_opt)
            sol = solve_ivp(ode_func, [t_current, t_current + self.dt], x_current, t_eval=[t_current + self.dt], method='RK45')
            x_next = sol.y[:, -1]
            x_traj[:, k + 1] = x_next
            (u_next, V_next, grad_V_next) = self.compute_optimal_control(x_next)
            V_traj[k + 1] = V_next
            gradV_traj[:, k + 1] = grad_V_next
        (u_final, _, _) = self.compute_optimal_control(x_traj[:, -1])
        u_traj[:, -1] = u_final
        return (x_traj, u_traj, V_traj, gradV_traj)

    def generate_smooth_state(self, d, max_amp, num_harmonics=5):
        xi = np.linspace(0, 1, d)
        x = np.zeros(d)
        for k in range(1, num_harmonics + 1):
            amp = np.random.uniform(-1 / k, 1.0 / k)
            phase = np.random.uniform(0, 2 * np.pi)
            x += amp * np.cos(2 * np.pi * k * xi + phase)
        scaling = np.random.uniform(0.5, max_amp)
        x = scaling * x
        baseline_shift = np.random.uniform(-max_amp / 2, max_amp / 2)
        x += baseline_shift
        return x

    def generate_training_data(self, n_trajectories=50, n_samples_per_traj=40, state_range=(-4.0, 4.0)):
        print(f'生成训练数据: {n_trajectories}条轨迹，每条轨迹{n_samples_per_traj}个采样点')
        max_amplitude = max(abs(state_range[0]), abs(state_range[1]))
        total_samples = n_trajectories * n_samples_per_traj
        X_all = np.zeros((total_samples, self.state_dim))
        V_all = np.zeros((total_samples, 1))
        DV_all = np.zeros((total_samples, self.state_dim))
        U_all = np.zeros((total_samples, self.control_dim))
        sample_idx = 0
        for traj_idx in tqdm(range(n_trajectories), desc='生成轨迹'):
            x0 = self.generate_smooth_state(self.state_dim, max_amp=max_amplitude)
            (x_traj, u_traj, V_traj, gradV_traj) = self.generate_optimal_trajectory(x0)
            traj_length = len(self.time_points)
            if traj_length < n_samples_per_traj:
                sample_indices = np.arange(traj_length)
            else:
                sample_indices = np.linspace(0, traj_length - 1, n_samples_per_traj, dtype=int)
            for idx in sample_indices:
                X_all[sample_idx] = x_traj[:, idx]
                V_all[sample_idx] = V_traj[idx]
                DV_all[sample_idx] = gradV_traj[:, idx]
                U_all[sample_idx] = u_traj[:, idx]
                sample_idx += 1
        X_all = X_all[:sample_idx]
        V_all = V_all[:sample_idx]
        DV_all = DV_all[:sample_idx]
        U_all = U_all[:sample_idx]
        print(f'生成完成！共{sample_idx}个样本')
        return (X_all, V_all, DV_all, U_all)

    def save_data(self, filename, X, V, DV, U):
        data_dict = {'X': X, 'V_all': V, 'DV_all': DV.T, 'U_all': U.T, 'd': self.state_dim, 'state_dim': self.state_dim, 'control_dim': self.control_dim}
        sio.savemat(filename, data_dict)
        print(f'数据已保存到: {filename}')
