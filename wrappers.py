import numpy as np
import tensorflow as tf
import gym
import os


class ToDoubleTensor(gym.ObservationWrapper):
    
    def observation(self, obs):
        return tf.convert_to_tensor(obs, dtype=tf.float64)


class ToDoubleTensorFloat32(gym.ObservationWrapper):
    
    def observation(self, obs):
        return tf.convert_to_tensor(obs, dtype=tf.float32)


class CartPoleEncoding(gym.ObservationWrapper):
    
    def observation(self, obs):

        ## Scaled Encoding
        # Scale cart position (range [-4.8, 4.8]) to range [0, 2pi]
        obs[0] = ((obs[0] + 4.8) / 9.6) * 2 * np.pi
        
        # Scale pole angle (range [-0.418, 0.418]) to range [0, 2pi]
        obs[2] = ((obs[2] + 0.418) / 0.836) * 2 * np.pi
        
        ## Directional Encoding
        obs[1] = np.pi if obs[1] > 0 else 0
        obs[3] = np.pi if obs[1] > 0 else 0
        
        return obs

class BlackjackEncoding(gym.ObservationWrapper):
    
    def observation(self, obs):

        ## Scaled Encoding
        # Players current sum (range [2, 31]) to range [0, 2pi]
        curr_sum= ((obs[0] - 2) / 29) * 2 * np.pi
        
        ## Scaled Encoding
        # Dealers one showing card (range [1,10]) to range [0, 2pi]
        dealer = ((obs[1] - 1) / 9) * 2 * np.pi
        
        ## Directional Encoding
        usable = np.pi if obs[2] else 0
        
        return (curr_sum, dealer, usable)
