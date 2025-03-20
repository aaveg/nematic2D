import pygame
import numpy as np
import json
# from utils import *
import matplotlib as mpl
from pprint import pprint

def coarse_grain(arr,binr=128,binc=128):
    rows,columns =  arr.shape
    return arr.reshape(binr,rows//binr, binc, columns//binc).sum(3).sum(1)

def compute_n(Qxx,Qxy):
    S = 2*np.sqrt( Qxx**2 + Qxy**2)
    Qxx = Qxx/(S+0.0001)
    Qxy = Qxy/(S+0.0001)
    # Evaluate nx and ny from normalized Qxx and Qxy
    nx = np.sqrt( Qxx + 0.5 )
    ny = Qxy / (nx+0.0001) # This ensures ny>0, such that theta lies between 0 to pi
    # nx = nx * sign( Qxy ) # This gives back nx its correct sign.
    return (S, nx, ny)


def find_interface(mat, size, dcol = 1, edge_val =0):
    assert type(mat) == np.ndarray
    nx,ny = size
    mat = mat.reshape((nx,ny))
    # mat =  mat.transpose()

    # search between the 25% and 75% of the visible area
    low, high = 0.25, 0.75
    lrow,hrow = int(low*nx), int(high*nx)
    # edge = []
    interface = []
    coords = []
    for col in range(0,ny,dcol):
        temp = []
        for row in range(lrow,hrow):
            # find the edge location 
            if edge_val-0.2 < mat[row][col] < edge_val+0.2:
                temp.append(row)
        if len(temp)==0:
            temp = [lrow]

        pos = sum(temp)/len(temp)
        interface.append(pos)
        coords.append((col,pos))
    interface = np.array(interface)
    coords = np.array(coords)
    # interface_rfft = np.fft.rfft(interface, norm = 'forward')
    # print("fft interface -> ", edge_rfft)
    return interface, coords  


class Renderer():
    def __init__(self, mode, config) -> None:
 
        self.mode = mode
        self.config = config

        self._nx, self._ny = config['size']
        self._channel_order = config['channel_order']
        self._bins = config['bin_size']
        self._n_channels = len(self._channel_order.keys())

        # self.save_loc = '/home/aaveg/projects/env_config/'
        # data render options and configurations
        
        self.avail_modes =  ['human', 'cli', 'none', None]
        assert self.mode in self.avail_modes
        self.screen = None
        self.clock = None
        self.scale = 3
        self.count = 0
        

    def render(self, frame = None):
        if self.mode == 'cli':
            self._render_cli(frame)
            return 0
        
        if self.mode == 'human':
            self._render_human(frame)
            return 0
        
        if self.mode == 'none':
            return 0
        
        if self.mode is None:
            return 0

        return -1

    def _render_cli(frame):
        pass

    def _render_human(self,dataframe):
        # assert self.nx == int(len(frame)**0.5) # check if true for non-square arrays
        scaled_nx = self.scale*self._nx
        scaled_ny = self.scale*self._ny
        num_disp =2

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((scaled_nx, num_disp*scaled_ny),pygame.RESIZABLE) #set size according to the simulation
            
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # build the surfaces and set transparency levels
        surf_phi = self._array2surf(dataframe['phi'], limits = (-1,1), colormap = 'bwr')
        # print(dataframe['phi'].reshape((self._nx,self._ny)))
        surf_activity = self._array2surf(dataframe['action'], limits = (0,1), colormap = 'gray')
        surf_activity.set_alpha(255)
        surf_nematic = self._draw_nematic(dataframe['Qxx'], dataframe['Qxy'], color = 'white')

        # render surfaces on screen in desired order 
        self.screen.blit(surf_phi, (0, 0))
        self.screen.blit(surf_activity, (0, scaled_ny))
        self.screen.blit(surf_nematic, (0, 0))

        # draw the interface with a black line
        # interface_coords = self.scale*find_interface(dataframe['phi'], size = (self._nx,self._ny), dcol = self._ny//self._bins)[1]
        # interface_coords[:,1] = self.scale*self._ny - interface_coords[:,1]
        # pygame.draw.lines(self.screen,'black',closed = False, points = interface_coords, width = 2)
        # surf_nematic = pygame.transform.flip(surf_nematic,flip_x = False, flip_y = True)

        # print(interface_coords)
        # refresh surface
        self.clock.tick(self.config["fps"])
        # self.screen = pygame.transform.flip(self.screen,flip_x = True, flip_y =False)
        pygame.display.flip()
        # pygame.image.save(self.screen,'data/t{0:03d}.jpeg'.format(self.count))
        # self.count+=1

        # Handle events in pygame window. 
        pygame.event.pump()
        for event in pygame.event.get(): 
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    self.save_config(dataframe.copy())
            if event.type == pygame.QUIT: 
                pygame.quit()
                # add a kill signal for the environment to know that renderer ha exited
                exit()

    def _array2surf(self, frame, limits = 'auto', colormap = 'gray'):
        if limits == 'auto':
            limits = (frame.min(),frame.max())
        assert limits[0] <= limits[1] 

        frame = np.reshape(frame, (self._nx,self._ny)).T
        frame = (frame - limits[0])/(limits[1]-limits[0]+0.0001)
        # frame = np.rot90(frame, k=1)
        # print(frame)
        frame = 255*self.colorize(frame, colormap)

        surf = pygame.surfarray.make_surface(frame) 

        scaled_nx = self.scale*self._nx
        scaled_ny = self.scale*self._ny
        surf = pygame.transform.scale(surf,(scaled_nx,scaled_ny)) 
        

        return pygame.transform.flip(surf,flip_x = False, flip_y = True)


    def _draw_nematic(self, Qxx, Qxy, color = 'black'):
        # Qxx = np.rot90(np.reshape( Qxx, (self._nx,self._ny) ) , k=1)
        # Qxy = np.rot90(np.reshape( Qxy, (self._nx,self._ny) )  , k=1)
        Qxx = np.reshape( Qxx, (self._nx,self._ny) ).T
        Qxy = np.reshape( Qxy, (self._nx,self._ny) ).T

        scaled_nx = self.scale*self._nx
        scaled_ny = self.scale*self._ny
        surf = pygame.Surface((scaled_nx,scaled_ny),flags=pygame.SRCALPHA)
        surf.fill(pygame.Color(0,0,0,0))

        gap = 4
        for i in range(0, self._nx, gap):
            for j in range(0, self._ny, gap):
                s,nx,ny = compute_n(Qxx[i,j],Qxy[i,j])
                if s<0.2:
                    continue
                r = i*self.scale
                c = j*self.scale
                l = 1*self.scale # l is the length scaling for nematic lines and 2.5 value visually looks good when plotting (can be chosen arbitrarily). 
                pygame.draw.line(surf, color, (r - l*s*nx, c - l*s*ny),(r + l*s*nx, c + l*s*ny), width=2)

        return pygame.transform.flip(surf,flip_x = False, flip_y = True)
    

    def save_config(self, data):
        pass
        # for k in data.keys():
        #     data[k] = data[k].tolist()
        # print('Saving environment configuration...')
        # with open(self.save_loc + "env_target.json", 'w') as f:
        #     json.dump(data,f)
        # print('Configuration saved')


    def colorize(self, arr, colormap):
        """
        genrates grayscale to RGB Mapping of an numpy ndarray.
        Input: (n,n) numpy.ndarray 
        Output: (n,n,3) numpy.ndarray   
        """
        assert type(arr) == np.ndarray
        temp = mpl.colormaps[colormap](arr) # creates RGBA channels 
        # Return only RGB channels and remove the Alpha channel
        return temp[:,:,0:3] 






class DataHandler():
    def __init__(self, config ) -> None:
        self._nx, self._ny = config['size']
        self.channel_order = config['channel_order']
        self._bins = config['bin_size']

        # self.save_loc = '/home/aaveg/projects/env_config/'
        # self._target = self._load_target(fname = self.save_loc + "env_target.json")     

    # def _load_target(self,fname):
    #     with open(fname, 'r') as fp:
    #         target = json.load(fp)
    #     for field in target.keys():
    #         target[field] = np.array(target[field])
    #     return target

    def generate_init_state(self, seed = 42):
        rng = np.random.default_rng(seed = seed)

        angle_radians = np.pi/3 
        nx_ = np.cos(angle_radians)
        ny_ = np.sin(angle_radians)
        Qxx = (nx_*nx_ - 0.5) + (2*rng.random((self._nx, self._ny)) -1)
        Qxy = nx_*ny_ + (2*rng.random((self._nx, self._ny)) -1)

        vx = np.zeros((self._nx, self._ny))
        vy = np.zeros((self._nx, self._ny))
        
        phi = np.zeros((self._nx, self._ny)) 
        # phi[:,:self._ny//2] = -1.0
        phi[:self._nx//2,:] = 1.0
        phi[self._nx//2 + 1:,:] = 1#-1.0
         # phi[:,1+self._ny//2:] = 1.0 
        
        output = {'Qxx':Qxx, 'Qxy': Qxy, 'vx': vx, 'vy': vy, 'phi': phi }
        return output 
     
        
    def extract(self, rawdata):
        if len(rawdata) == 0:
            # self.reset()
            return {}

        n_channels = len(self.channel_order.keys())
        data = rawdata.split(' ')
        n = len(data[0].split('&'))
        # print("n --->  ", data)
        assert n == n_channels, "recieved data: " + rawdata
        
        data_processed = {}
        for k in range(n_channels):
            channel_name = self.channel_order[k]
            data_processed[channel_name] = []

        for i in data:
            vals = i.split('&')
            for j in range(n_channels):
                channel_name = self.channel_order[j]
                data_processed[channel_name].append(float(vals[j]))

        # print(data_processed['phi'].min(),data_processed['phi'].max())

        for k in data_processed.keys():
            data_processed[k] = np.array(data_processed[k])
    
        return data_processed

    def action_preprocess(self,action, interface, mode):
        interface = interface[:self._bins]
        rep = self._ny/self._bins
        assert int(rep) == rep, "action dimension should be divisible ny"
        rep = int(rep)
        # print("hereeeee")
        temp = np.repeat(action, rep)
        action_mat = np.tile(temp,(self._nx,1))
        if mode == "all":
            return action_mat
        elif mode == "interface":

            coord_r = np.repeat(self.denormalize_obs(interface), rep)
            coord_c = np.arange(0,self._ny)

            # print(coord_r, coord_c)
            action_to_write = np.zeros(action_mat.shape)
            for i in range(-20,5):
                action_to_write[coord_r+i, coord_c] = action_mat[coord_r, coord_c]

            # action_mat = np.transpose(action_mat)
            # action_to_write = np.transpose(action_to_write)
            action_to_write = action_to_write[::1,:]
            action_to_write = action_to_write.flatten()
            return action_to_write
        else:
            raise ValueError(f"action mode should be either 'all' or 'interface'. mode value provided is {mode}")



    # def get_obs(self, data):
    #     field = data['phi']
    #     temp = find_interface(field, size = (self._nx,self._ny), dcol = self._ny//self._bins)
    #     interface = temp[0]
    #     interface_norm = self.normalize_obs(interface)
        
    #     vx = np.reshape(data['vx'],(self._nx, self._ny))#.transpose()
    #     vy = np.reshape(data['vy'],(self._nx, self._ny))#.transpose()
        
    #     interface = interface.astype(int)
    #     interface_vx = vx[interface, np.arange(0,self._ny, self._ny//self._bins)]
    #     interface_vy = vy[interface, np.arange(0,self._ny, self._ny//self._bins)]
        
    #     obs = np.hstack((interface_norm, interface_vx, interface_vy))
    #     # print(len(obs))
    #     return obs

    def get_obs(self, data):
        Qxx = data['Qxx'].reshape(self._nx,self._ny)
        Qxy = data['Qxy'].reshape(self._nx,self._ny)
        
        obs = np.array([Qxx,Qxy])
        return obs
    
    def normalize_obs(self, obs):
        obs = obs/self._ny
        return obs
        

    def denormalize_obs(self, n_obs):
        obs = (n_obs*self._ny).astype(int)
        return obs

    # # reward in real space
    # def _calc_reward(self, obs ):
    #     target = self._load_target(fname = self.save_loc + "env_target.json")
    #     target_state = self.get_obs(target)
    #     obs_state = obs['observation'] 
    #     # print(target_state,0*obs_state+0.5)
    #     diff = target_state - obs_state
    #     mse = diff.dot(diff)

    #     b1 = 1
    #     a1 = 64
    #     reward = b1*np.exp(-a1*mse)  

    #     return reward

    # # reward in fft space
    # def _calc_reward(self, obs ):
    #     target = self._load_target(fname = self.save_loc + "env_target.json")
    #     target_state = self.get_obs(target)
    #     obs_state = obs['observation'] 

    #     target_rfft = np.fft.rfft(target_state, norm = 'forward')
    #     target_rfft_mag = np.abs(target_rfft)

    #     obs_rfft = np.fft.rfft(obs_state, norm = 'forward')
    #     obs_rfft_mag = np.abs(obs_rfft)

    #     # print(target_rfft_mag, obs_rfft_mag)

    #     diff = target_rfft_mag - obs_rfft_mag
    #     mse = diff.dot(diff)
    #     # print('mse - ', mse)

    #     b1 = 1
    #     a1 = 10000
    #     reward = b1*np.exp(-a1*mse)  

    #     return reward


    # def _calc_reward(self, obs ):
    #     target = self._load_target(fname = self.save_loc + "env_target.json")
    #     target_state = self.get_obs(target)
    #     # print(target_state)
    #     obs_state = obs['observation'] 

    #     target_rfft = np.fft.rfft(target_state, norm = 'forward')
    #     target_rfft_mag = np.abs(target_rfft)

    #     obs_rfft = np.fft.rfft(obs_state, norm = 'forward')
    #     obs_rfft_mag = np.abs(obs_rfft)

    #     # print(target_rfft_mag, obs_rfft_mag)
    #     # print(target_rfft_mag)
    #     target_rfft_mag = np.zeros_like(obs_rfft_mag)
    #     target_rfft_mag[0] = 0.5
    #     target_rfft_mag[2] = 0.02

    #     diff = target_rfft_mag - obs_rfft_mag
    #     mse = diff.dot(diff)
    #     # print('mse - ', mse)

    #     b1 = 1
    #     a1 = 10000
    #     reward = b1*np.exp(-a1*mse)  
    #     print(mse,reward)

    #     # action = obs['action_out']
    #     # reward = -action.dot(action)

    #     return reward

    # def _calc_reward(self, obs ):
    #     obs_state = obs['observation'] 
    #     obs_int = obs_state[:self._bins] 
    #     obs_vx = obs_state[self._bins:2*self._bins]

    #     reward = obs_vx.sum()
    #     # obs_rfft = np.fft.rfft(obs_state, norm = 'forward')
    #     # obs_rfft_mag = np.abs(obs_rfft)

    #     # target_rfft_mag = np.zeros_like(obs_rfft_mag)
    #     # target_rfft_mag[0] = 0.5
    #     # target_rfft_mag[2] = 0.02

    #     # diff = target_rfft_mag - obs_rfft_mag
    #     # mse = diff.dot(diff)

    #     # b1 = 1
    #     # a1 = 10000
    #     # reward = b1*np.exp(-a1*mse)  

    #     # action = obs['action_out']
    #     # reward = -action.dot(action)

    #     return reward

    def _calc_reward(self, obs ):
        obs_state = obs['observation'] 
        obs_int = obs_state[:self._bins] 

        obs_rfft = np.fft.rfft(obs_int, norm = 'forward')
        obs_rfft_mag = np.abs(obs_rfft)

        target_rfft_mag = np.zeros_like(obs_rfft_mag)
        target_rfft_mag[0] = 0.5
        target_rfft_mag[2] = 0.02

        diff = target_rfft_mag - obs_rfft_mag
        mse = diff.dot(diff)
        # print('mse - ', mse)

        b1 = 1
        a1 = 10000
        reward = b1*np.exp(-a1*mse)  
        # print(mse,reward)

        # action = obs['action_out']
        # reward = -action.dot(action)

        return reward