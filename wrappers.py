import numpy as np
import tensorflow as tf
import gym
import os


class ToDoubleTensor(gym.ObservationWrapper):
    
    def observation(self, obs):
        return tf.convert_to_tensor(obs, dtype=tf.float64)

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
        # Players current sum (range [4, 21]) to range [0, 2pi]
        curr_sum= ((obs[0] - 4) / 17) * 2 * np.pi
        
        ## Scaled Encoding
        # Dealers one showing card (range [1,10]) to range [0, 2pi]
        dealer = ((obs[1] - 1) / 9) * 2 * np.pi
        
        ## Directional Encoding
        usable = np.pi if obs[2] else 0
        
        return (curr_sum, dealer, usable)

class CartPoleEncodingMix(gym.ObservationWrapper):
    
    def observation(self, obs):

        ## Scaled Encoding
        # Scale cart position (range [-4.8, 4.8]) to range [0, 2pi]
        obs[0] = ((obs[0] + 4.8) / 9.6) * 2 * np.pi
        
        # Scale pole angle (range [-0.418, 0.418]) to range [0, 2pi]
        obs[2] = ((obs[2] + 0.418) / 0.836) * 2 * np.pi
        
        ## Continous Encoding
        obs[1] = np.arctan(obs[1])
        obs[3] = np.arctan(obs[3])
        
        return obs

class BlackjackEncodingContinous(gym.ObservationWrapper):
    
    def observation(self, obs):

        curr_sum= np.arctan(obs[0])
        
        dealer = np.arctan(obs[1])
        
        usable = np.pi if obs[2] else 0
        
        return (curr_sum, dealer, usable)

class ContinousEncoding(gym.ObservationWrapper):

    def observation(self, obs):
        return np.arctan(obs)