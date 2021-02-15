
import tensorflow as tf

from dqn.policies import ActionValuePolicy, EpsilonGreedyPolicy, LinearDecay
from dqn.replay import ReplayMemory
from dqn.utils import Sampler
from dqn.validate import validate

# TODO documentation

class DQN:

    def __init__(self, env, val_env, policy_model, target_model, replay_capacity, 
        epsilon_duration, epsilon_start, epsilon_end, gamma, optimizer, loss):

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

        self.gamma = gamma
        self.optimizer = optimizer
        self.loss = loss



    def train(self, num_steps, train_after, train_every, update_every, 
        validate_every, num_val_steps, batch_size, on_transition=None,
        on_train=None, on_validate=None):

        # counts validation epochs
        epoch = 0

        sampler = Sampler(self.behavior_policy, self.env)
        
        for step in range(num_steps):

            train_step = step - train_after
            is_training = train_step >= 0

            if is_training:
                self.epsilon_schedule.step()

            transition = sampler.step()
            self.memory.store(transition)

            if on_transition:
                on_transition(
                    step=step,
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

                self.optimizer.apply_gradients(
                    zip(grads, self.policy_model.trainable_variables)
                )

                if on_train:
                    on_train(
                        step=step, 
                        loss=loss, 
                        batch=batch
                    )

            if train_step % update_every == 0:
                self.target_model.set_weights(
                    self.policy_model.get_weights()
                )

            if train_step % validate_every == 0:
                val_return = validate(
                    self.val_env, 
                    self.greedy_policy, 
                    num_val_steps
                )

                if on_validate:
                    on_validate(
                        epoch=epoch, 
                        val_return=val_return,
                        # Linters might complain that grads is unbound; however,
                        # this is not a problem since all previous if-statements
                        # are guaranteed to be executed at least once before 
                        # this one (in particular, when train_step == 0)
                        grads=grads # type: ignore
                    )

                epoch += 1