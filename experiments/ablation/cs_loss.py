"""Cucker--Smale dynamics and HJB/control-loss definitions."""
import torch
Na = 20
state_dim = 40
control_dim = 20

def get_Q_matrix():
    """获取SDRE系统中的Q矩阵 (40×40)"""
    Q = (1.0 / Na) * torch.eye(state_dim, dtype=torch.float32)
    return Q


def get_R_matrix():
    """获取SDRE系统中的R矩阵 (20×20)"""
    R = (1.0 / Na) * torch.eye(control_dim, dtype=torch.float32)
    return R


def get_R_inv():
    """获取R的逆矩阵"""
    return Na * torch.eye(control_dim, dtype=torch.float32)


def get_B_matrix(device):
    """获取SDRE系统中的固定B矩阵 (40×20)"""
    ONa = torch.zeros((Na, Na), dtype=torch.float32, device=device)
    INa = torch.eye(Na, dtype=torch.float32, device=device)
    
    B_top = ONa
    B_bottom = INa
    B = torch.vstack([B_top, B_bottom])  # (40, 20)
    return B


def compute_A_matrix(y: torch.Tensor, device) -> torch.Tensor:
    """计算状态相关的A(y)矩阵 (batch, 20, 20)"""
    batch_size = y.shape[0]
    A = torch.zeros((batch_size, Na, Na), dtype=torch.float32, device=device)
    
    y_expand_i = y.unsqueeze(2)  # (batch, 20, 1)
    y_expand_j = y.unsqueeze(1)  # (batch, 1, 20)
    P_matrix = 1.0 / (1.0 + (y_expand_i - y_expand_j) ** 2)  # (batch, 20, 20)
    
    diag_mask = torch.eye(Na, dtype=torch.bool, device=device)
    off_diag_sum = (P_matrix * (~diag_mask)).sum(dim=2)  # (batch, 20)
    A.diagonal(dim1=1, dim2=2)[:] = - (1.0 / Na) * off_diag_sum
    
    A = A + (1.0 / Na) * P_matrix * (~diag_mask)
    return A


def compute_f_x(x: torch.Tensor, device) -> torch.Tensor:
    """计算SDRE系统动力学 f(x) = M(x) @ x"""
    batch_size = x.shape[0]
    y = x[:, :Na]  
    
    A = compute_A_matrix(y, device)
    
    ONa = torch.zeros((batch_size, Na, Na), dtype=torch.float32, device=device)
    INa = torch.eye(Na, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    M_top = torch.cat([ONa, INa], dim=2)  
    M_bottom = torch.cat([ONa, A], dim=2) 
    M = torch.cat([M_top, M_bottom], dim=1)  
    
    f = torch.bmm(M, x.unsqueeze(2)).squeeze(2)  
    return f


def compute_HJB_H(model, x: torch.Tensor, device) -> torch.Tensor:
    """计算HJB哈密顿函数H，返回H和∇V"""
    if not x.requires_grad:
        x = x.requires_grad_(True)
    
    Q = get_Q_matrix().to(device)
    x_T_Q_x = torch.sum(x @ Q * x, dim=1, keepdim=True) 
    
    V = model(x) 
    
    grad_V = torch.autograd.grad(
        outputs=V,
        inputs=x,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True
    )[0] 
    
    f = compute_f_x(x, device)
    grad_V_T_f = torch.sum(grad_V * f, dim=1, keepdim=True) 
    
    B = get_B_matrix(device)
    R_inv = get_R_inv().to(device)
    B_Rinv_BT = B @ R_inv @ B.T
    
    grad_V_T_BRBt = grad_V @ B_Rinv_BT
    grad_V_T_BRBt_grad_V = torch.sum(grad_V_T_BRBt * grad_V, dim=1, keepdim=True)
    control_term = 0.25 * grad_V_T_BRBt_grad_V
    
    H = x_T_Q_x + grad_V_T_f - control_term
    
    return H, grad_V, V


def compute_u_from_gradV(grad_V: torch.Tensor, device) -> torch.Tensor:
    """从梯度计算控制量u(x)"""
    B = get_B_matrix(device)
    R_inv = get_R_inv().to(device)
    
    B_T_grad_V = torch.matmul(grad_V, B)  
    u = -0.5 * torch.matmul(B_T_grad_V, R_inv) 
    return u


