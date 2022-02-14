import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose


#1. download train & test data
training_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())

# initializing dataloader
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

#2. Modeling
# set GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# defines a class in which nn.Module is inherited.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(nn.Linear(28*28, 512), nn.ReLU(),
                                               nn.Linear(512, 512), nn.ReLU(),
                                               nn.Linear(512, 10))
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

#3. Optimizing model parameters.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def tutorial_train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # calculating loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss:\t{loss:>7f}\t[{current:>5d}/{size:>5d}]")

def tutorial_test(dataloader, model, loss_fn):
    size = len(dataloader)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error:\n Accuracy:\t{(100 * correct):>0.1f}%, Avg loss:\t{test_loss:>8f}\n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------------------")
    tutorial_train(train_dataloader, model, loss_fn, optimizer)
    tutorial_test(test_dataloader, model, loss_fn)
print("Done!")

#4. Model save
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

#5. Model load
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

#6. Predict
classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f"Predicted:\t{predicted}", f"Actual:\t{actual}")