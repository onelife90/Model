import torch

dtype = torch.float
device = torch.device("cuda:0")

# N: batch size
# D_in: Input Dimension
# H: Hidden node
# D_out: Output Dimension
#1. Network set
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)        # (64, 1000)
y = torch.randn(N, D_out, device=device, dtype=dtype)       # (64, 10)

w1 = torch.randn(D_in, H, device=device, dtype=dtype)       # (1000, 100)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)      # (100, 10)

learning_rate = 1e-6

for t in range(500):
    h = x.mm(w1)                # (64, 1000) * (1000, 100) = (1000, 100)
    h_relu = h.clamp(min=0)     # clamp == relu
    y_pred = h_relu.mm(w2)      # (1000, 100) * (100, 10) = (100, 10)

    loss = (y_pred - y).pow(2).sum().item()

    if t % 100 == 0:
        print(t, loss)

    grad_y_pred = 2.0 * (y_pred - y)    # partial derivatives of the loss
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2