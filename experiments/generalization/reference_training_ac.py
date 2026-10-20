"""Allen--Cahn training losses and dynamics."""
import torch
state_dim = 51
control_dim = 1
sigma = 0.1

def get_Q_matrix():
    """获取系统中的Q矩阵 (51×51)，PDE中为单位矩阵"""
    Q = torch.eye(state_dim, dtype=torch.float32)*0.02
    return Q


def get_R_matrix():
    """获取系统中的R矩阵 (1×1)"""
    R = torch.tensor([[0.1]], dtype=torch.float32)
    return R


def get_R_inv():
    """获取R的逆矩阵 (1×1)"""
    return torch.tensor([[10.0]], dtype=torch.float32)


def get_B_matrix(device):
    """获取系统中的固定B矩阵 (51×1)，控制作用在 xi in [0.6, 0.9]"""
    xi = torch.linspace(0, 1, state_dim, device=device)
    B = ((xi >= 0.6) & (xi <= 0.9)).float().unsqueeze(1) # (51, 1)
    return B


def compute_A_base(device):
    """计算有限差分的基础矩阵 A_base (Neumann边界)"""
    h = 1.0 / (state_dim - 1)
    coeff = sigma / (h**2)
    A = torch.zeros((state_dim, state_dim), dtype=torch.float32, device=device)
    
    A[0, 0] = -2 * coeff
    A[0, 1] = 2 * coeff
    for i in range(1, state_dim - 1):
        A[i, i-1] = coeff
        A[i, i] = -2 * coeff
        A[i, i+1] = coeff
    A[-1, -2] = 2 * coeff
    A[-1, -1] = -2 * coeff
    
    return A


def compute_f_x(x: torch.Tensor, device) -> torch.Tensor:
    """计算PDE非线性系统动力学 f(x) = A_base * x + x * (1 - x^2)"""
    A_base = compute_A_base(device)
    # x: (batch, 51), A_base: (51, 51)
    # 线性项 A_base * x
    linear_term = torch.matmul(x, A_base.T) 
    # 非线性项 x * (1 - x^2)
    nonlinear_term = x * (1.0 - x**2)
    
    f = linear_term + nonlinear_term
    return f


def compute_HJB_H(model, x: torch.Tensor, device) -> torch.Tensor:
    """计算HJB哈密顿函数H，返回H, ∇V, V"""
    if not x.requires_grad:
        x = x.requires_grad_(True)
    
    # 1. 计算x^T Q x (因为Q是单位矩阵，直接求平方和)
    x_T_Q_x = torch.sum(x**2, dim=1, keepdim=True)*0.02  # (batch, 1)
    
    # 2. 计算V(x)和∇V(x)
    V = model(x)  # (batch, 1)
    
    grad_V = torch.autograd.grad(
        outputs=V,
        inputs=x,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True
    )[0]  # (batch, 51)
    
    # 3. 计算∇V^T f(x)
    f = compute_f_x(x, device)
    grad_V_T_f = torch.sum(grad_V * f, dim=1, keepdim=True)  # (batch, 1)
    
    # 4. 计算控制项 (1/4) ∇V^T (B R⁻¹ B^T) ∇V
    B = get_B_matrix(device)
    R_inv = get_R_inv().to(device)
    
    # grad_V @ B -> (batch, 51) @ (51, 1) -> (batch, 1)
    grad_V_B = torch.matmul(grad_V, B) 
    control_term = 0.25 * R_inv * (grad_V_B ** 2)  # (batch, 1)
    
    # 5. 总哈密顿函数H
    H = x_T_Q_x + grad_V_T_f - control_term
    
    return H, grad_V, V


def compute_u_from_gradV(grad_V: torch.Tensor, device) -> torch.Tensor:
    """从梯度计算控制量u(x) = -0.5 R^{-1} B^T ∇V"""
    B = get_B_matrix(device)
    R_inv = get_R_inv().to(device)
    
    # B^T * ∇V
    grad_V_B = torch.matmul(grad_V, B)  # (batch, 1)
    u = -0.5 * R_inv * grad_V_B         # (batch, 1)
    return u


