import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from skimage.filters import gaussian
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Custom Gaussian Blur Layer
class GaussianBlur(nn.Module):
    def __init__(self, sigma=1.0):
        super(GaussianBlur, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        # Convert input tensor to numpy array
        x_np = x.detach().cpu().numpy()
        
        # Apply Gaussian filter
        x_blurred_np = np.array([gaussian(x_i.squeeze(), sigma=self.sigma) for x_i in x_np])
        
        # Convert numpy array back to tensor
        x_blurred = torch.tensor(x_blurred_np, dtype=x.dtype, device=x.device)
        return x_blurred

# Define the LSTM Model with Gaussian Blur
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, blur_sigma=1.0):
        super(LSTMModel, self).__init__()
        self.gaussian_blur = GaussianBlur(sigma=blur_sigma)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.gaussian_blur(x)
        h_out, _ = self.lstm(x)
        out = self.fc(h_out[:, -1, :])  # Take the last time step
        return out

# Generate synthetic data for demonstration
np.random.seed(0)
data_length = 100
look_back = 5

# Create synthetic data
X = pd.DataFrame(np.random.randn(data_length, 1), columns=['feature'])
Y = pd.DataFrame(np.random.randn(data_length, 1), columns=['target'])

# Normalize the data
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

# Apply Gaussian Blur
def apply_gaussian_blur(data, sigma=1):
    return gaussian(data, sigma=sigma, mode='reflect')

X_blurred = apply_gaussian_blur(X_scaled, sigma=1)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_blurred, dtype=torch.float32)
Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)

# Create dataset
dataset = TensorDataset(X_tensor, Y_tensor)

# Split dataset into training and testing
total_size = len(dataset)
test_size = int(total_size * 0.25)
train_size = total_size - test_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoader for both training and testing
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model parameters
input_size = X_blurred.shape[1]
hidden_size = 50
output_size = 1
blur_sigma = 1.0  # Adjust sigma as needed

# Initialize model, criterion, and optimizer
model = LSTMModel(input_size, hidden_size, output_size, blur_sigma)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Evaluate the model
model.eval()
with torch.no_grad():
    total_loss = 0
    for batch_X, batch_Y in test_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        total_loss += loss.item()
    
    average_loss = total_loss / len(test_loader)
    print(f'Test loss: {average_loss}')
