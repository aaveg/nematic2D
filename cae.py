import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from architectures.conv_ae import ConvAutoencoder


def train_autoencoder(model, train_loader, criterion, optimizer, epochs, device):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            # print(batch.shape)
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)  # Compare output with original first channel
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")
    
def evaluate(model, testdata):
    model.eval()  # Set to evaluation mode
    total_loss = 0
    with torch.no_grad():  # No gradients needed
        batch = testdata.to(device)
        output = model(batch)
        loss = criterion(output, batch)
        total_loss += loss.item()
        print(f'Loss on test data: {total_loss:.6f}')
    print(f"Test Accuracy: {total_loss:.6f}")
    return total_loss

def animate_comparison(original, reconstructed):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    img1 = axes[0, 0].imshow(original[0, 0], cmap='gray', vmin=original.min(), vmax=original.max())
    img2 = axes[1, 0].imshow(original[0, 1], cmap='gray', vmin=original.min(), vmax=original.max())
    img3 = axes[0, 1].imshow(reconstructed[0, 0], cmap='gray', vmin=reconstructed.min(), vmax=reconstructed.max())
    img4 = axes[1, 1].imshow(reconstructed[0, 0], cmap='gray', vmin=reconstructed.min(), vmax=reconstructed.max())
    
    axes[0, 0].set_title("Original Channel 1")
    axes[1, 0].set_title("Original Channel 2")
    axes[0, 1].set_title("Reconstructed Channel 1")
    axes[1, 1].set_title("Reconstructed Channel 2")
    
    def update(frame):
        img1.set_array(original[frame, 0])
        img2.set_array(original[frame, 1])
        img3.set_array(reconstructed[frame, 0])
        img4.set_array(reconstructed[frame, 1])
        return img1, img2, img3, img4
    
    ani = animation.FuncAnimation(fig, update, frames=original.shape[0], interval=800, blit=False)
    plt.show()

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def train_test_split(data, frac):
    # Define split sizes (80% train, 20% test)
    train_size = int(frac * len(data))
    test_size = len(data) - train_size
    # Split dataset
    return random_split(data, [train_size, test_size])


learning_rate = 5e-4
num_epochs = 300
save_loc = "models/cae.pt"
set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")

data = np.load('data/data_a_05.npy').astype(np.float32)
data = data[100:,]
print(data.shape,data.min(),data.max(),data.dtype)

# Create DataLoaders
train_dataset, test_dataset = train_test_split(data,frac = 0.75)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = ConvAutoencoder(latent_dim=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
train_autoencoder(model, train_loader, criterion, optimizer, epochs=num_epochs, device=device)

torch.save(model.state_dict(), save_loc)
print(f"Model saved as {save_loc}")


testdata = torch.from_numpy(test_dataset.dataset[test_dataset.indices])
acc = evaluate(model, testdata)
print(testdata.shape)

# input_tensor = torch.randn(50, 2, 64, 64).to(device) #
input_tensor = testdata.to(device)  
output = model(input_tensor)
animate_comparison(input_tensor.detach().cpu(),output.detach().cpu())
