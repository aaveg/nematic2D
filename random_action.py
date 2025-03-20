from custom_env import cuPSS_env
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import tyro

from dataclasses import dataclass


@dataclass
class Args:
    size: tuple = (128,128)
    sname: str = '/home/aaveg/projects/conv/nematic.cu'
    save: str = 'data_a_02.pt'
    action_amp: float = 0.5
    rmode: str = "none"
    seed: int = 42
    cuda: bool = True


if __name__ == "__main__": 
    args = tyro.cli(Args)
    env = cuPSS_env(env_config= {"sname": args.sname, 'size': args.size, 'rmode': args.rmode })
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print('Model is using the device: ', device)


    num_episodes = 1
    data = []
    for i in range(num_episodes):
        observation, info = env.reset(seed = args.seed)
        done: bool = False
        iter_count: int = 0
        while not done:
            action = args.action_amp*np.ones_like(env.action_space.sample())
            observation, reward, terminated, truncated, info = env.step(action)
            data.append(observation)
            done = terminated or truncated
            if done:
                next_state = None
            iter_count+=1

    env.close()
    data = np.array(data)
    print(data.shape)
    fname_append = args.action_amp*10
    np.save(f'data/data_a_{fname_append:02.0f}.npy', data)
    print(f'data saved to file: data/data_a_{fname_append:02.0f}.npy')

    # t = torch.from_numpy(data)
    # print(t.shape)
    # torch.save(t, 'data/data_a_02.pt')