from datetime import datetime
import importlib
import sys
import numpy as np

if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.exit('Usage: python train.py <path-to-config>')

    config_name = sys.argv[1]

    experiment = f'logs/{config_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    if 'qiskit' in str(config_name):
        from torch.utils.tensorboard import SummaryWriter
        from PT.dqn.algorithm import DQN

        summary_writer = SummaryWriter(log_dir=experiment)
    
    else:
        import tensorflow as tf
        from TF.dqn.algorithm import DQN

        ### EXPERIMENTAL: Restrict GPU memory for Tensorflow
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)]
            )

        summary_writer = tf.summary.create_file_writer(experiment)

    
    config = importlib.import_module(f'configs.{config_name}')

    framework=config.framework if hasattr(config, 'framework') else 'tf'
    update_every_start=config.update_every_start if hasattr(config, 'update_every_start') else None
    update_every_end=config.update_every_end if hasattr(config, 'update_every_end') else None
    update_every_duration=config.update_every_duration if hasattr(config, 'update_every_duration') else None
    optimizer_output=config.optimizer_output if hasattr(config, 'optimizer_output') else None
    optimizer_input=config.optimizer_input if hasattr(config, 'optimizer_input') else None
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
        optimizer_input=optimizer_input,
        optimizer_output=optimizer_output,
        loss=config.loss
    )

    # Episode statistics
    episode = 0
    episode_rewards = []
    episode_explorations = 0

    val_returns = []
    val_step = 0


    def on_transition(transition, did_explore):
        global config, summary_writer, episode, episode_rewards, episode_explorations

        episode_rewards.append(transition.reward)
        episode_explorations += int(did_explore)

        if transition.is_terminal:

            episode_length = len(episode_rewards)
            episode_return = sum(episode_rewards)
            exploration_freq = episode_explorations / episode_length

            if framework == 'tf':
                with summary_writer.as_default():
                    tf.summary.scalar('episode/length', episode_length, episode)
                    tf.summary.scalar('episode/return', episode_return, episode)
                    tf.summary.scalar('episode/exploration', exploration_freq, episode)
            else:
                summary_writer.add_scalar('episode/length', episode_length, episode)
                summary_writer.add_scalar('episode/return', episode_return, episode)
                summary_writer.add_scalar('episode/exploration', exploration_freq, episode)

            episode_rewards = []
            episode_explorations = 0
            episode += 1


    def on_train(step, loss, batch):
        global summary_writer

        if framework == 'tf':
            reward_strength = tf.reduce_sum(tf.abs(batch.rewards)) / len(batch.rewards)

            with summary_writer.as_default():
                tf.summary.scalar('step/loss', loss, step)
                tf.summary.scalar('step/reward_strength', reward_strength, step)
        else:
            reward_strength = np.sum(batch.rewards.numpy(), axis=0) / len(batch.rewards)
            summary_writer.add_scalar('step/loss', loss, step)
            summary_writer.add_scalar('step/reward_strength', reward_strength, step)


    def on_validate(val_return, grads=None):
        global summary_writer, config, num_val_steps_acceptance, val_returns, val_step

        if framework == 'tf':
            with summary_writer.as_default():
                if grads:
                    for g, v in zip(grads, config.policy_model.trainable_variables):
                        tf.summary.histogram(f'epoch/grads/{v.name}', g, val_step)
                        tf.summary.histogram(f'epoch/weights/{v.name}', v, val_step)

                tf.summary.scalar('epoch/avg_return', val_return, val_step)
        else:
            summary_writer.add_scalar('epoch/avg_return', val_return, val_step)

        if num_val_steps_acceptance and acceptance_threshold:
            val_returns.append(val_return) 
            val_step +=1
            if num_val_steps_acceptance < val_step and np.mean(val_returns[-num_val_steps_acceptance:]) > acceptance_threshold:
                exit("Environment solved. Exit...")
    
    def on_validation_step(step, obs, q_values):
        global summary_writer, val_step

        if framework == 'tf':
            with summary_writer.as_default():
                for i, ob in enumerate(obs):
                    tf.summary.scalar(f'val_step/obs{i}', ob, val_step+step+1)

                for i, q_value in enumerate(q_values):
                    tf.summary.scalar(f'val_step/q_value{i}', q_value, val_step+step+1)
        else:
            for i, ob in enumerate(obs):
                summary_writer.add_scalar(f'val_step/obs{i}', ob, val_step+step+1)

            for i, q_value in enumerate(q_values):
                summary_writer.add_scalar(f'val_step/q_value{i}', q_value, val_step+step+1)
            
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
        on_validation_step=on_validation_step
    )
