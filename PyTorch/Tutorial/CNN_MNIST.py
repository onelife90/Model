from pickle import TRUE
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# load dataset
train_dataset = pd.read_csv("./PyTorch/data/MNIST/train.csv")
test_dataset = pd.read_csv("./PyTorch/data/MNIST/test.csv")

#chenck train set
print(train_dataset.head())
# [5 rows x 785 columns]

print(test_dataset.head())
# [5 rows x 785 columns]


# split to image & label
train_images = (train_dataset.iloc[:, 1:].values).astype("float32")
train_labels = train_dataset["label"].values
test_images = (test_dataset.iloc[:, 1:].values).astype("float32")
# test_images = (test_dataset.values).astype("float32")

# check train data image
print(f"\ntrain_images:\n{train_images}")
# check train data label
print(f"\ntrain_labels:\n{train_labels}")

# split into train, valid dataset
# :param stratify -> using class labels split
train_images, valid_images, train_labels, valid_labels = train_test_split(train_images,
                                                                          train_labels,
                                                                          stratify=train_labels,
                                                                          random_state=42,
                                                                          test_size=0.2)

# check train, valid, test images shape
print(f"\nShape of train images: {train_images.shape}\n")
print(f"Shape of valid images: {valid_images.shape}\n")
print(f"Shape of test images: {test_images.shape}\n")

# check train, valid label shape
print(f"Shape of train labels: {train_labels.shape}\n")
print(f"Shape of valid labels: {valid_labels.shape}\n")

# reshape image size to check for ours
train_images = train_images.reshape(-1, 28, 28)
valid_images = valid_images.reshape(-1, 28, 28)
test_images = test_images.reshape(-1, 28, 28)

# check train, valid, test images shape after reshape
print(f"\nShape of train images: {train_images.shape}\n")
print(f"Shape of valid images: {valid_images.shape}\n")
print(f"Shape of test images: {test_images.shape}\n")

# make dataloader to feed on MLP model
# change train_images: ndarray -> tensor
train_images_tensor = torch.tensor(train_images)
train_labels_tensor = torch.tensor(train_labels)

train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)
train_loader = DataLoader(train_tensor, batch_size=64, num_workers=0, shuffle=True)

valid_images_tensor = torch.tensor(valid_images)
valid_labels_tensor = torch.tensor(valid_labels)

valid_tensor = TensorDataset(valid_images_tensor, valid_labels_tensor)
valid_loader = DataLoader(valid_tensor, batch_size=64, num_workers=0, shuffle=True)

test_images_tensor = torch.tensor(test_images)

# create CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x)
        
        return x
    
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

model = CNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"Model:\t{model}")
print(f"Device:\t{DEVICE}")

# definite train & evaluate
def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        # optimizer initalize -> output -> loss -> Back-Propagation -> optimizer step
        optimizer.zero_grad()
        output = model(data)
        
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format\
                  (epoch, batch_idx * len(data), len(train_loader.dataset),\
                  100. * batch_idx / len(train_loader), loss.item()))

def evaluate(model, valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            
            valid_loss += F.cross_entropy(output, target, reduction="sum").item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()
            
    valid_loss /= len(valid_loader.dataset)
    valid_accuracy = 100. * correct / len(valid_loader.dataset)
    
    return valid_loss, valid_accuracy


"""Training"""
EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer)
    valid_loss, valid_accuracy = evaluate(model, valid_loader)
    
    print(f"[EPOCH: {epoch}],\t\
          validation loss: {valid_loss:.4f},\t\
          validation Accuracy:\t{valid_accuracy:.2f}%\n")


# predict test dataset
def testset_prediction(model, test_images_tensor):
    model.eval()
    result = []
    
    with torch.no_grad():
        for data in test_images_tensor:
            data = data.to(DEVICE)
            output = model(data)
            prediction = output.max(1, keepdim=True)[1]
            result.append(prediction.tolist())
            
    return result

test_predict_result = testset_prediction(model, test_images_tensor)
print(test_predict_result[:5])