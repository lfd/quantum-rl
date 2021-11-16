from datetime import datetime
import importlib
import sys
import numpy as np

import tensorflow as tf
from dqn.algorithm import DQN

if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.exit('Usage: python train.py <path-to-config>')


    ### EXPERIMENTAL: Restrict GPU memroy for Tensorflow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)]
        )

    config_name = sys.argv[1]
    config = importlib.import_module(f'configs.{config_name}')

    update_every_start=config.update_every_start if hasattr(config, 'update_every_start') else None
    update_every_end=config.update_every_end if hasattr(config, 'update_every_end') else None
    update_every_duration=config.update_every_duration if hasattr(config, 'update_every_duration') else None
    optimizer_output=config.optimizer_output if hasattr(config, 'optimizer_output') else None
    num_steps_per_layer=config.num_steps_per_layer if hasattr(config, 'num_steps_per_layer') else None
    update_every=config.update_every if hasattr(config, 'update_every') else None
    num_val_trials=config.num_val_trials if hasattr(config, 'num_val_trials') else None
    num_val_steps=config.num_val_steps if hasattr(config, 'num_val_steps') else None
    num_val_steps_acceptance=config.num_val_steps_acceptance if hasattr(config, 'num_val_steps_acceptance') else None
    acceptance_threshold=config.acceptance_threshold if hasattr(config, 'acceptance_threshold') else None

    algorithm = DQN(
        env=config.env,
        val_env=config.val_env,
        policy_model=config.policy_model,
        target_model=config.target_model,
        replay_capacity=config.replay_capacity,
        epsilon_duration=config.epsilon_duration,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        update_every_start=update_every_start,
        update_every_end=update_every_end,
        update_every_duration=update_every_duration, 
        gamma=config.gamma,
        optimizer=config.optimizer,
        optimizer_output=optimizer_output,
        loss=config.loss
    )

    experiment = f'logs/{config_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    summary_writer = tf.summary.create_file_writer(experiment)

    # TODO log hyperparameters

    # Episode statistics
    episode = 0
    episode_rewards = []
    episode_explorations = 0

    val_returns = []
    val_steps = 0


    def on_transition(step, transition, did_explore, **kwargs):
        global config, summary_writer, episode, episode_rewards, episode_explorations

        episode_rewards.append(transition.reward)
        episode_explorations += int(did_explore)

        if transition.is_terminal:

            episode_length = len(episode_rewards)
            episode_return = sum(episode_rewards)
            exploration_freq = episode_explorations / episode_length

            with summary_writer.as_default():
                tf.summary.scalar('episode/length', episode_length, episode)
                tf.summary.scalar('episode/return', episode_return, episode)
                tf.summary.scalar('episode/exploration', exploration_freq, episode)

            episode_rewards = []
            episode_explorations = 0
            episode += 1


    def on_train(step, loss, batch, **kwargs):
        global summary_writer

        reward_strength = tf.reduce_sum(tf.abs(batch.rewards)) / len(batch.rewards)

        with summary_writer.as_default():
            tf.summary.scalar('step/loss', loss, step)
            tf.summary.scalar('step/reward_strength', reward_strength, step)


    def on_validate(epoch, val_return, grads, **kwargs):
        global summary_writer, config, num_val_steps_acceptance, val_returns, val_steps

        # TODO model checkpointing
        # TODO video rendering

        with summary_writer.as_default():

            for g, v in zip(grads, config.policy_model.trainable_variables):
                tf.summary.histogram(f'epoch/grads/{v.name}', g, epoch)
                tf.summary.histogram(f'epoch/weights/{v.name}', v, epoch)

            tf.summary.scalar('epoch/avg_return', val_return, epoch)

        if num_val_steps_acceptance and acceptance_threshold:
            val_returns.append(val_return) 
            val_steps +=1
            if num_val_steps_acceptance < val_steps and np.min(val_returns[-num_val_steps_acceptance:]) > acceptance_threshold:
                exit("Environment solved. Exit...")
    
    algorithm.train(
        num_steps=config.num_steps,
        train_after=config.train_after,
        train_every=config.train_every,
        update_every=update_every,       
        validate_every=config.validate_every,
        num_val_steps=num_val_steps,
        num_val_trials=num_val_trials,
        batch_size=config.batch_size,
        on_transition=on_transition,
        on_train=on_train,
        on_validate=on_validate,
        num_steps_per_layer=num_steps_per_layer
    )
