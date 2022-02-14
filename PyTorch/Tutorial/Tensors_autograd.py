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

# Back-Propagation, w1 & w2 changed values calculating
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)       # (1000, 100)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)      # (100, 10)

learning_rate = 1e-6

for t in range(500):
    # Back-Propagation
    y_pred = x.mm(w1).clamp(min=0).mm(w2)      # (64, 1000) * (1000, 100) * (100, 10) = (100, 10)

    loss = (y_pred - y).pow(2).sum()

    if t % 100 == 99:
        print(t, loss.item())

    # using autograd
    loss.backward()

    # using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # After weight update, set value 0
        w1.grad.zero_()
        w2.grad.zero_()