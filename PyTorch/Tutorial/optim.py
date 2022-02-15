import torch

# N: batch size
# D_in: Input Dimension
# H: Hidden node
# D_out: Output Dimension
#1. Network set
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Module -> Sequential
model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                            torch.nn.ReLU(),
                            torch.nn.Linear(H, D_out),)

# nn package includes loss function
loss_fn = torch.nn.MSELoss(reduction="sum")

learning_rate = 1e-4

# using optim package: for update weight
# Adam first factor: which Tensor needs to be updated
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    # Forward-Propagation: model input(x) predict(y)
    y_pred = model(x)

    loss = loss_fn(y_pred, y)

    if t % 100 == 99:
        print(t, loss.item())

    # Before Backward-Propagation, set value 0
    # every time backward() is called, without overwritting in the buffer
    optimizer.zero_grad()

    # Backward-Propagation: calculating all learnable parameters
    loss.backward()

    # parameter update
    optimizer.step()