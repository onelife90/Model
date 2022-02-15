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
train_images_tensor = torch.Tensor(train_images)
train_labels_tensor = torch.Tensor(train_labels)

train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)
train_loader = DataLoader(train_tensor, batch_size=64, num_workers=0, shuffle=True)

valid_images_tensor = torch.Tensor(valid_images)
valid_labels_tensor = torch.Tensor(valid_labels)

valid_tensor = TensorDataset(valid_images_tensor, valid_labels_tensor)
valid_loader = DataLoader(valid_tensor, batch_size=64, num_workers=0, shuffle=True)

test_images_tensor = torch.Tensor(test_images)

# create MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(28*28, 128)
        self.hidden_layer = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=1)
        
        return x
    
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

model = MLP().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"Model:\t{model}")
print(f"Device:\t{DEVICE}")

