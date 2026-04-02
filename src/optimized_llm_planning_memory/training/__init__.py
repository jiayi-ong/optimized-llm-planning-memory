"""
training/
=========
RL training pipeline: Gymnasium environment + SB3 PPO + reward function.

Episode as RL environment
--------------------------
Each planning episode is framed as a Gymnasium MDP:

  State (observation) : Token IDs of the trajectory since the last compression.
                         Box(shape=(max_obs_tokens,), dtype=np.int32).
  Action              : Token IDs of the generated compressed state.
                         Box(shape=(max_action_tokens,), dtype=np.int32).
  Reward              : Multi-component shaped scalar from RewardFunction.
  Terminal            : Episode ends when the agent signals DONE or hits max_steps.

Key design decisions
--------------------
1. ``CompressionEnv`` creates a fresh ``SimulatorAdapter(seed=N)`` on every
   ``reset()``. No global state — safe for ``make_vec_env(n_envs=N)``.
2. ``CompressorPolicy`` wraps ``TrainableCompressorBase`` as an SB3 policy.
   ``evaluate_actions()`` provides token-level log-probs for the PPO ratio.
3. ``RewardFunction`` uses the same ``ConstraintSatisfactionEngine`` as
   ``DeterministicEvaluator`` — this is the critical shared primitive.
4. ``RLTrainer`` calls ``SB3 PPO.learn()`` with the custom policy and env.

Contents
--------
reward.py        — RewardFunction (multi-component shaped reward)
env.py           — CompressionEnv (gymnasium.Env)
policy.py        — CompressorPolicy (SB3 BasePolicy)
trainer.py       — RLTrainer (wires SB3 PPO + logging + checkpointing)
episode_buffer.py— EpisodeBuffer (transition storage for PPO mini-batches)
"""
