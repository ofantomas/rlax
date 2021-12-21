import collections
import random
from absl import app
from absl import flags
from bsuite.environments import catch
from buffer import Trajectory
import haiku as hk
from haiku import nets
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

from buffer import Buffer

Params = collections.namedtuple("Params", "params")
ActorOutput = collections.namedtuple("ActorOutput", "actions logits")
LearnerState = collections.namedtuple("LearnerState", "params opt_state")
Data = collections.namedtuple("Data", "obs_tm1 a_tm1 r_t discount_t obs_t")

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("train_episodes", 1001, "Number of train episodes.")
flags.DEFINE_integer("batch_size", 32, "Size of the training batch")
flags.DEFINE_integer("hidden_units", 64, "Number of network hidden units.")
flags.DEFINE_float("discount_factor", 0.99, "Q-learning discount factor.")
flags.DEFINE_float("learning_rate", 0.005, "Optimizer learning rate.")
flags.DEFINE_integer("eval_episodes", 100, "Number of evaluation episodes.")
flags.DEFINE_integer("evaluate_every", 50,
                     "Number of episodes between evaluations.")


class REINFORCE:
  """A simple REINFORCE agent."""

  def __init__(self, observation_spec, action_spec,
               learning_rate, max_len, rng):
    self._observation_spec = observation_spec
    self._action_spec = action_spec
    self._buffer = Buffer(observation_spec, action_spec, max_len)
    self._rng = rng
    # Neural net and optimiser.
    self._network = build_network(action_spec.num_values)
    self._optimizer = optax.adam(learning_rate)
    self._state = self.initial_learner_state(next(rng))
    # Jitting for speed.
    # self.actor_step = jax.jit(self.actor_step)
    self.learner_step = jax.jit(self.learner_step)

  def initial_learner_state(self, key):
    sample_input = self._observation_spec.generate_value()
    sample_input = jnp.expand_dims(sample_input, 0)
    params = self._network.init(key, sample_input)
    opt_state = self._optimizer.init(params)
    return LearnerState(params, opt_state)

  def select_action(self, env_output):
    obs = env_output.observation[None, :] # add dummy batch
    logits = self._network.apply(self._state.params, obs)
    action = jax.random.categorical(next(self._rng), logits).squeeze()
    return int(action)

  def select_action_eval(self, env_output):
    obs = env_output.observation[None, :] # add dummy batch
    logits = self._network.apply(self._state.params, obs)
    action = rlax.greedy().sample(next(self._rng), logits)
    return int(action)

  def learner_update(self, env_output, action, new_env_output):
    self._buffer.append(env_output, action, new_env_output)
    if new_env_output.last() or self._buffer.full():
      trajectory = self._buffer.drain()
      self._state = self.learner_step(self._state, trajectory)

  def _compute_adv(self, trajectory):
    returns = []
    R = 0.
    for r in trajectory.rewards[::-1]:
      R = r + FLAGS.discount_factor * R
      returns.insert(0, R)
    return jnp.array(returns, dtype=jnp.float32)
  
  def _loss(self, params, trajectory):
    logits = self._network.apply(params, trajectory.observations)
    adv = self._compute_adv(trajectory)
    return rlax.policy_gradient_loss(logits[:-1], trajectory.actions, 
                                     adv, jnp.ones_like(adv, dtype=jnp.float32))

  def learner_step(self, state, trajectory):
    grads = jax.grad(self._loss)(state.params, trajectory)
    updates, new_opt_state = self._optimizer.update(grads, state.opt_state)
    new_params = optax.apply_updates(updates, state.params)
    return LearnerState(params=new_params, opt_state=new_opt_state)

def build_network(num_actions: int) -> hk.Transformed:
  """Factory for a simple MLP network for approximating Q-values."""

  def policy(obs):
    network = hk.Sequential(
        [hk.Flatten(),
         nets.MLP([FLAGS.hidden_units, num_actions])])
    return network(obs)

  return hk.without_apply_rng(hk.transform(policy))

def run_loop(
    agent, environment,
    train_episodes, evaluate_every,
    eval_episodes):
  """A simple run loop for examples of reinforcement learning with rlax."""

  print(f"Training agent for {train_episodes} episodes")
  for episode in range(train_episodes):

    # Prepare agent, environment and accumulator for a new episode.
    timestep = environment.reset()

    while not timestep.last():

      # Acting.
      action = agent.select_action(timestep)

      # Agent-environment interaction.
      new_timestep = environment.step(action)

      # Update agent if needed
      agent.learner_update(timestep, action, new_timestep)

      timestep = new_timestep


    # Evaluation.
    if not episode % evaluate_every:
      returns = 0.
      for _ in range(eval_episodes):
        timestep = environment.reset()
        while not timestep.last():
          action = agent.select_action_eval(timestep)
          timestep = environment.step(action)
          returns += timestep.reward

      avg_returns = returns / eval_episodes
      print(f"Episode {episode:4d}: Average returns: {avg_returns:.2f}")

def main(unused_arg):
  env = catch.Catch(seed=FLAGS.seed)
  agent = REINFORCE(
      observation_spec=env.observation_spec(),
      action_spec=env.action_spec(),
      learning_rate=FLAGS.learning_rate,
      max_len=1000,
      rng=hk.PRNGSequence(FLAGS.seed)
  )

  run_loop(
      agent=agent,
      environment=env,
      train_episodes=FLAGS.train_episodes,
      evaluate_every=FLAGS.evaluate_every,
      eval_episodes=FLAGS.eval_episodes,
  )


if __name__ == "__main__":
  app.run(main)