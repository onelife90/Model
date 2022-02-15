import random
import torch

class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        initialize three nn.Linear instances using Forward-Propagation
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        Forward-Propagation: reuse middle_linear for calculating hidden layer
        computational graph: perfectly safe to reuse the same module multiple times
        """        
        h_relu = self.input_linear(x).clamp(min=0)
        
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        
        y_pred = self.output_linear(h_relu)
                
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
model = DynamicNet(D_in, H, D_out)

# nn package includes loss function
criterion = torch.nn.MSELoss(reduction="sum")

# difficult SGD for learning, using momentum
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

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