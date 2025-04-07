from RL.custom_env import cuPSS_env
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import tyro

from dataclasses import dataclass


@dataclass
class Args:
    size: tuple = (128,128)
    sname: str = '/home/aaveg/projects/nematic2D/RL/nematic2D_pbc.cu'
    save: str = 'data_a_02.pt'
    action_amp: float = 0.2
    rmode: str = "human"
    seed: int = 42
    cuda: bool = True


if __name__ == "__main__": 
    args = tyro.cli(Args)
    env = cuPSS_env(env_config= {"sname": args.sname, 'size': args.size, 'rmode': args.rmode })
    # device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # print('Model is using the device: ', device)

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, 100)  # Adjust as needed
    ax.set_ylim(-1, 1)  # Adjust as needed
    x_data, y_data = [], []

    def update_plot(data_point):
        x_data.append(len(x_data))  # Incremental x-axis
        y_data.append(data_point)
        line.set_data(x_data, y_data)
        ax.set_xlim(0, max(100, len(x_data)))  # Dynamically adjust x-axis
        ax.set_ylim(min(y_data) - 0.1, max(y_data) + 0.1)  # Adjust y-axis
        plt.draw()
        plt.pause(0.01)


    num_episodes = 2
    data = []
    for i in range(num_episodes):
        observation, info = env.reset(seed = args.seed)
        done: bool = False
        iter_count: int = 0
        flag = 1
        while not done:
            action = (i+1)*args.action_amp*np.ones_like(env.action_space.sample())
            observation, reward, terminated, truncated, info = env.step(action)
            # print(observation)
            update_plot(observation)
            data.append(observation)
            done = terminated or truncated
            if done:
                next_state = None
            iter_count+=1
    
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the plot open after the run

    env.close()
    data = np.array(data)
    print(data.shape)
    # fname_append = args.action_amp*10
    # np.save(f'data/data_a_{fname_append:02.0f}.npy', data)
    # print(f'data saved to file: data/data_a_{fname_append:02.0f}.npy')

    # t = torch.from_numpy(data)
    # print(t.shape)
    # torch.save(t, 'data/data_a_02.pt')