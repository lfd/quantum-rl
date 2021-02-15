import numpy as np
import tensorflow as tf
import gym


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