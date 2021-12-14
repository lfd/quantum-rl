import tensorflow as tf

from TF.dqn.policies import ActionValuePolicy, EpsilonGreedyPolicy, LinearDecay, LinearIncrease
from TF.dqn.replay import ReplayMemory
from TF.dqn.utils import Sampler
from TF.dqn.validate import validate

# TODO documentation

class DQN:

    def __init__(self, env, val_env, policy_model, target_model, replay_capacity, 
        epsilon_duration, epsilon_start, epsilon_end, gamma, optimizer, loss,
        optimizer_output = None, update_every_start = None, update_every_end = None, 
        update_every_duration = None):

        self.env = env
        self.val_env = val_env

        self.policy_model = policy_model
        self.target_model= target_model
        
        self.greedy_policy = ActionValuePolicy(self.policy_model)
        self.behavior_policy = EpsilonGreedyPolicy(
            self.greedy_policy, self.env.action_space
        )

        self.memory = ReplayMemory(replay_capacity)

        self.epsilon_schedule = LinearDecay(
            policy=self.behavior_policy,
            num_steps=epsilon_duration,
            start=epsilon_start,
            end=epsilon_end
        )

        if update_every_start and update_every_end and update_every_duration:
            self.update_every_schedule = LinearIncrease(
                num_steps=update_every_duration,
                start=update_every_start,
                end=update_every_end
            )
        else: 
            self.update_every_schedule = None

        self.gamma = gamma
        self.optimizer = optimizer
        self.optimizer_output = optimizer_output
        self.loss = loss



    def train(self, num_steps, train_after, train_every, validate_every, batch_size, 
        update_every=None, on_transition=None, num_val_steps=None,
        num_val_trials=None, on_train=None, on_validate=None, 
        on_validation_step=None):

        # counts validation epochs
        epoch = 0

        sampler = Sampler(self.behavior_policy, self.env)

        if not update_every:
            update_every_current = self.update_every_schedule.current
        else:
            update_every_current = None
        
        for step in range(num_steps):

            train_step = step - train_after
            is_training = train_step >= 0

            if is_training:
                self.epsilon_schedule.step()
                if self.update_every_schedule:
                    self.update_every_schedule.step()

            transition = sampler.step()
            self.memory.store(transition)

            if on_transition:
                on_transition(
                    transition=transition, 
                    did_explore=self.behavior_policy.did_explore
                )

            if not is_training:
                continue

            if train_step % train_every == 0:
                
                # Sample a batch of transitions from the replay buffer
                batch = self.memory.sample(batch_size)

                # Convert target to correct datatype (PennyLane requires 64bit
                # floats, Cirq/TFQ works with standard 32bit)
                targets = tf.cast(batch.rewards, tf.keras.backend.floatx())

                # Check whether the batch contains next_states (the sampled
                # batch might contain terminal states only)
                if len(batch.next_states) > 0:

                    target_next_q_values = self.target_model(batch.next_states)
                    target_next_v_values = tf.reduce_max(
                        target_next_q_values, 
                        axis=-1
                    )

                    non_terminal_indices = tf.where(~batch.is_terminal)

                    targets = tf.cast(batch.rewards, target_next_v_values.dtype)
                    targets = tf.tensor_scatter_nd_add(
                        targets,
                        non_terminal_indices,
                        self.gamma * target_next_v_values
                    )

                with tf.GradientTape() as tape:
                    policy_q_values = self.policy_model(batch.states)

                    action_indices = tf.expand_dims(batch.actions, axis=-1)

                    policy_v_values = tf.gather(
                        policy_q_values, 
                        action_indices, 
                        batch_dims=1
                    )

                    policy_v_values = tf.squeeze(
                        policy_v_values,
                        axis=-1
                    )

                    loss = self.loss(targets, policy_v_values)

                grads = tape.gradient(
                    loss, 
                    self.policy_model.trainable_variables
                )

                if self.optimizer_output:

                    self.optimizer.apply_gradients(
                        zip(grads[-1:], self.policy_model.trainable_variables[-1:])
                    )
                    self.optimizer_output.apply_gradients(
                        zip(grads[:-1], self.policy_model.trainable_variables[:-1])
                    )
                else:
                    self.optimizer.apply_gradients(
                        zip(grads, self.policy_model.trainable_variables)
                    )

                if on_train:
                    on_train(
                        step=step, 
                        loss=loss, 
                        batch=batch
                    )

            if update_every and train_step % update_every == 0 or update_every_current and train_step == update_every_current:

                self.target_model.set_weights(
                    self.policy_model.get_weights()
                )

                if update_every_current:
                    update_every_current = train_step + self.update_every_schedule.current

            if train_step % validate_every == 0:
                val_return = validate(
                    self.val_env, 
                    self.greedy_policy, 
                    num_val_steps,
                    num_val_trials,
                    on_validation_step
                )

                if on_validate:
                    on_validate(
                        epoch=epoch, 
                        val_return=val_return,
                        grads=grads
                    )

                epoch += 1
