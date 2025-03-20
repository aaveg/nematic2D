import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from architectures.conv_ae import ConvAutoencoder
from torch.utils.data import DataLoader
import wandb
import os
import matplotlib.animation as animation
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    num_epochs: int = 100
    datafile_path: str = "data/data_a_05.npy"  # Replace with your actual file path
    autoencoder_path: str = "models/cae.pt"  # Replace with your actual file path
    latent_dim: int = 256





# Load your true dynamics data from .npy file
def load_data(file_path):
    data = np.load(file_path)[1000:1300,]  # Shape: (time_steps, input_dim)
    t = torch.linspace(0, 1, data.shape[0])  # Assuming normalized time [0,1]
    x_true = torch.tensor(data, dtype=torch.float32)  # Convert to PyTorch tensor
    return t, x_true


# Define ODE Function in Latent Space
class LatentODEFunc(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512):
        super(LatentODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, t, z):
        return self.net(z)  # Compute dz/dt in latent space


# Define Neural ODE in Latent Space
class LatentNeuralODE(nn.Module):
    def __init__(self, ode_func):
        super(LatentNeuralODE, self).__init__()
        self.ode_func = ode_func

    def forward(self, z0, t):
        return odeint(self.ode_func, z0, t, method='rk4')  # Runge-Kutta solver

# Training Function
def train_latent_neural_ode(latent_ode, decoder, z_true, x_true, t, num_epochs=2000, lr=0.001):
    optimizer = optim.Adam(list(latent_ode.parameters()), lr=lr)
    loss_fn = nn.MSELoss()

    z0 = z_true[0]#.unsqueeze(0)  # Initial condition in latent space

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Predict latent trajectory
        z_pred = latent_ode(z0, t)

        # Decode back to original space
        # x_pred = decoder(z_pred)
        # print(z_pred.shape)
        # Compute loss between reconstructed and true data
        loss = loss_fn(z_pred, z_true)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# Visualization Function (Projecting first latent dimension)
def visualize_results(t, z_true, z_pred):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex = True, sharey=True)

    # True Latent Dynamics Heatmap (Time on y-axis, Latent Features on x-axis)
    im1 = axes[0].imshow(z_true.detach().numpy(), aspect="auto", cmap="viridis", extent=[0, 256, t[-1], t[0]])
    axes[0].set_xlabel("Latent Features")
    axes[0].set_ylabel("Time")
    axes[0].set_title("True Latent Dynamics")
    fig.colorbar(im1, ax=axes[0])

    # Predicted Latent Dynamics Heatmap (Time on y-axis, Latent Features on x-axis)
    im2 = axes[1].imshow(z_pred.detach().numpy(), aspect="auto", cmap="viridis", extent=[0, 256, t[-1], t[0]])
    axes[1].set_xlabel("Latent Features")
    axes[1].set_ylabel("Time")
    axes[1].set_title("Predicted Latent Dynamics")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()

def animate_comparison(original, reconstructed):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    img1 = axes[0, 0].imshow(original[0, 0], cmap='gray', vmin=original.min(), vmax=original.max())
    img2 = axes[1, 0].imshow(original[0, 1], cmap='gray', vmin=original.min(), vmax=original.max())
    img3 = axes[0, 1].imshow(reconstructed[0, 0], cmap='gray', vmin=reconstructed.min(), vmax=reconstructed.max())
    img4 = axes[1, 1].imshow(reconstructed[0, 1], cmap='gray', vmin=reconstructed.min(), vmax=reconstructed.max())
    errx = np.abs(original[0, 0] - reconstructed[0, 0])
    erry = np.abs(original[0, 1] - reconstructed[0, 1])
    img5 = axes[0, 2].imshow(errx, cmap='gray', vmin=errx.min(), vmax=errx.max() )
    img6 = axes[1, 2].imshow(erry, cmap='gray', vmin=erry.min(), vmax=erry.max() )
    
    axes[0, 0].set_title("Original Channel 1")
    axes[1, 0].set_title("Original Channel 2")
    axes[0, 1].set_title("Reconstructed Channel 1")
    axes[1, 1].set_title("Reconstructed Channel 2")
    axes[0, 2].set_title("Abs error Channel 1")
    axes[1, 2].set_title("Abs error Channel 2")
    
    
    def update(frame):
        img1.set_array(original[frame, 0])
        img2.set_array(original[frame, 1])
        img3.set_array(reconstructed[frame, 0])
        img4.set_array(reconstructed[frame, 1])
        img5.set_array( np.abs(original[frame, 0] - reconstructed[frame, 0]) )
        img6.set_array( np.abs(original[frame, 0] - reconstructed[frame, 0]) )
        return img1, img2, img3, img4, img5, img6
        
    ani = animation.FuncAnimation(fig, update, frames=original.shape[0], interval=50, blit=False)
    plt.show()

def encode_in_batches(encoder, data, batch_size=256):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    encoded_batches = []
    with torch.no_grad():  # No need to track gradients
        for batch in dataloader:
            encoded_batches.append(encoder(batch))  # Encode batch-wise
    
    return torch.cat(encoded_batches, dim=0)  # Concatenate results

def train_latent_neural_ode_in_batches(latent_ode, encoder, x_true, t, lr=0.001, batch_size=50, include_encoder=False):
    optimizer = optim.Adam(list(latent_ode.parameters()) + (list(encoder.parameters()) if include_encoder else []), lr=lr)
    loss_fn = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(t, x_true)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    epoch_loss = 0
    for batch_t, batch_x_true in dataloader:
        optimizer.zero_grad()
        with torch.set_grad_enabled(include_encoder):
            batch_z_true = encoder(batch_x_true)
        z0 = batch_z_true[0]  # Initial condition in latent space for the batch
        z_pred = latent_ode(z0, batch_t)  # Predict latent trajectory
        # Compute loss between predicted and true latent trajectories
        loss = loss_fn(z_pred, batch_z_true)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


# Function to create a new experiment folder
def create_experiment_folder(base_path="experiments"):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    experiment_id = len(os.listdir(base_path)) + 1
    experiment_folder = os.path.join(base_path, f"experiment_{experiment_id}")
    os.makedirs(experiment_folder)
    return experiment_folder

# MAIN: Load data, encoder, and train Neural ODE
args = tyro.cli(Args)
t, x_true = load_data(args.datafile_path)
print(f"x_true shape: {x_true.shape}, type: {type(x_true)}")

ae = ConvAutoencoder(latent_dim=args.latent_dim)
ae.load_state_dict(torch.load(args.autoencoder_path, weights_only=True))

latent_ode_func = LatentODEFunc(latent_dim=args.latent_dim)
latent_ode = LatentNeuralODE(latent_ode_func)

# Create a new experiment folder
experiment_folder = create_experiment_folder()

# Train Neural ODE
save_interval = args.num_epochs // 10

wandb.init(project="latent_neural_ode_project_m16", 
           name= f"experiment_{experiment_folder.split('_')[-1]}")

for epoch in range(args.num_epochs):
    loss = train_latent_neural_ode_in_batches(latent_ode, ae.encoder, x_true, t, lr=1e-4, batch_size=50)
    wandb.log({"epoch": epoch, "loss": loss})
    
    if (epoch + 1) % save_interval == 0:
        model_save_path = os.path.join(experiment_folder, f"latent_neural_ode_epoch_{epoch + 1}.pt")
        torch.save(latent_ode.state_dict(), model_save_path)
        print(f"Model saved successfully at '{model_save_path}'.")

# Save the final trained model
final_model_save_path = os.path.join(experiment_folder, "latent_neural_ode_final.pt")
torch.save(latent_ode.state_dict(), final_model_save_path)
print(f"Final model saved successfully at '{final_model_save_path}'.")


# Predict and visualize latent dynamics
z0 = ae.encoder(x_true[0:1])[0] #z_true[0]  # Initial condition in latent space
z_pred = latent_ode(z0, t)  # Predict in latent space
# visualize_results(t, z_true, z_pred)

x_pred = encode_in_batches(ae.decoder, z_pred, batch_size=256)
animate_comparison(x_true.detach().numpy(), x_pred.detach().numpy())

