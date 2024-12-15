# Step 1: Library Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# Step 2: Set random seed for reproducibility
# Instructions: "...use torch.manual_seed(1693) at the very beginning when you import the libraries"
torch.manual_seed(1693)

# Step 3: Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Step 4: Import the MNIST Dataset
# Instructions: "The dataset can be directly downloaded from the library, ..."
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)


# Steps 5-10: Implementing the CNN model
# Instructions: "For this problem, based on the examples explained in class, you will design your own convolutional neural network that will train on the MNIST data set, 
#                and it will achieve at least 99% accuracy on the Test." 

# Step 5: create data loaders for training and testing
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Step 6: Define the CNN model class
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # convolutional layers
        self.layer1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.layer3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        #Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)     # output size is 10 for the 10 digits in the MNIST dataset (0 to 9)
        self.dropout = nn.Dropout(0.5)    # cut half the dendrites to prevent overfitting

    def forward(self, x):
        x = self.pool(F.relu(self.layer1(x)))    # first layer + pool
        x = self.pool(F.relu(self.layer2(x)))    # second layer + pool
        x = self.pool(F.relu(self.layer3(x)))    #third layer + pool
        x = x.view(-1, 128 * 3 * 3)              
        x = self.dropout(F.relu(self.fc1(x)))     # fully connected layer with dropout
        x = self.fc2(x)                          # output layer
        return F.log_softmax(x, dim=1)

# Step 7: Initialize the model, loss function, and optimizer
model = MNIST_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Steo 8: Define a training function
def train(model, train_loader, optimizer, criterion, epochs=10):
    model.train()  
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for data, target in train_loader:
            optimizer.zero_grad()            
            output = model(data)               # do forward pass
            loss = criterion(output, target)   # compute the loss
            loss.backward()                   # do backward pass
            optimizer.step()                   # update weights
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Step 9: Define an evaluation function
def evaluate(model, test_loader):
    model.eval()  
    test_loss = 0
    correct = 0
    with torch.no_grad():  
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest Set: Average Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n")
    return accuracy

# Step 10: Train and eval the model
train(model, train_loader, optimizer, criterion, epochs=10)
final_accuracy = evaluate(model, test_loader)

# Print final result
print(f"Final Test Accuracy: {final_accuracy:.2f}%")