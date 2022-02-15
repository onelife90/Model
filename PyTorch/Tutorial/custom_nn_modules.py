import torch

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        instance two nn.Linear modules, assign them as member variables
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        Input Tensor -> Output Tensor
        """        
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        
        return y_pred


# N: batch size
# D_in: Input Dimension
# H: Hidden node
# D_out: Output Dimension
#1. Network set
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Module instance
model = TwoLayerNet(D_in, H, D_out)

# nn package includes loss function
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for t in range(500):
    # Forward-Propagation: model input(x) predict(y)
    y_pred = model(x)

    loss = criterion(y_pred, y)

    if t % 100 == 99:
        print(t, loss.item())

    # Before Backward-Propagation, set value 0
    # every time backward() is called, without overwritting in the buffer
    optimizer.zero_grad()

    # Backward-Propagation: calculating all learnable parameters
    loss.backward()

    # parameter update
    optimizer.step()