---
layout: "contents"
title: Evaluation
firstpage:
---

# Evaluation

In Metaworld, agents are to be evaluated using their **success rate** on a set of tasks and goal positions, not the episodic reward achieved during training.

Each environment computes a success flag which is available through the `info` dictionary's `"success"` key, which is `0` when the task has not yet been accomplished, and `1` when success has been achieved.

To aid in easier evaluation of agents, the benchmark provides two evaluation utility functions through the `metaworld.evaluation` package, one for multi-task reinforcement learning and one for meta-reinforcement learning respectively.

## Methodology

Specifically, success rate is measured in the following way for each benchmark:

### Multi-Task Reinforcement Learning (MT1, MT10, MT50)

The agent, trained on the set of training tasks and training goal positions from the benchmark, is evaluated for one episode per training goal position, per training task. Meaning it is in practice evaluated for 50 episodes (one for each of the 50 goals) for each training task, and these goal positions are the same ones seen during training. During this episode, the agent is considered to have succeeded if the success flag is `1` *at any point during the episode*, not just at the end of the episode.

The final metric is the average success rate achieved across all tasks and episodes.

Here is python pseudocode for this procedure:

```python
def multi_task_eval(agent, envs, num_evaluation_episodes = 50, episode_horizon = 500):
   success_rate = 0.0

   for episode in range(num_evaluation_episodes):
      envs.iterate_goal_position()
      obs = envs.reset()
      for env in envs:
         obs = env.reset()
         for step in range(episode_horizon):
            action, _ = agent.eval_action(obs)
            next_obs, _, _, _, info = env.step(action)
            obs = next_obs

            if info["success"] == 1:
               success_rate += 1
               break

   success_rate /= (num_evaluation_episodes * envs.num_envs)

   return success_rate
```

### Meta-Reinforcement Learning (ML1, ML10, ML45)

The agent, trained on the set of training tasks and training goal positions from the benchmark, is evaluated for one episode per *testing goal position*, *per testing task*. In ML1, the task is the same between training and testing, but the set of goal positions is different and not seen during training. For ML10 and ML45, there are 5 held out testing tasks which are different from the training tasks, and each of them also has 50 goal positions never seen during training.

However, since meta-RL is all about adaptation, additionally the evaluation procedure also allows the agent to adapt. Specifically, one would collect `adaptation_steps * adaptation_episodes` number of episodes per testing task, per testing goal, and give them back to the network to adapt from, before computing the final post-adaptation evaluation metric (in a single episode) for that testing goal / task, in a similar fashion to the multi-task reinforcement learning setting. In practice, for each adaptation step, `adaptation_episodes` number of episodes is collected per testing task and given back to the network to adapt from as a batch of rollouts, so the agent can iteratively adapt.

Success is measured during each episode for each tasks the same way as it is in the multi-task setting: the agent is considered to have succeeded if the success flag is `1` at any point during the episode. And the final metric is likewise still the average success rate across all testing tasks and episodes.

Here is python pseudocode for this procedure:

```python
def metalearning_eval(agent, eval_envs, adaptation_steps = 1, adaptation_episodes = 10, num_evaluation_episodes = 50, episode_horizon):
   success_rate = 0.0
   initial_obs = eval_envs.reset()

   for episode in range(num_evaluation_episodes):
      eval_envs.iterate_goal_position()

      for step in range(adaptation_steps):
         rollout_buffer = []
         for _ in range(adaptation_episodes):
            obs = eval_envs.reset()
            buffer = [obs]
            for _ in range(episode_horizon):
               action, misc_outs = agent.adapt_action(obs)
               next_obs, reward, terminated, truncated, info = eval_envs.step(action)
               buffer += [action, reward, next_obs]
               if (log_probs := misc_outs["log_probs"]):
                  buffer += [log_probs]
               if (means := misc_outs["means"]):
                  buffer += [means]
               if (stds := misc_outs["stds"]):
                  buffer += [stds]
            rollout_buffer.append(buffer)

         agent.adapt(buffer)

      success_rate += multi_task_eval(agent, eval_envs, num_evaluation_episodes=1)

   success_rate /= num_evaluation_episodes

   return success_rate
```

## Utility functions

To avoid the need to implement the evaluation procedure from scratch, implementations for both the multi-task and meta-reinforcement learning evaluation procedures can be found in the `metaworld.evaluation` package under the functions `evaluation` and `metalearning_evaluation` respectively.

### The `Agent` / `MetaLearningAgent` protocols

The evaluation utilities are agnostic to your agent's architecture and implementation framework, but in their signature they both expect an object that adheres to a given protocol as the first argument (`agent`).

Specifically, this is the protocol each evaluation function respectively expects:

```python
class Agent(Protocol):
    def eval_action(
        self, observations: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...


class MetaLearningAgent(Agent):
    def adapt_action(
        self, observations: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray]]: ...

    def adapt(self, rollouts: Rollout) -> None: ...
```

For both multi-task and meta-reinforcement learning evaluations, the agent object should have a `eval_action` method that takes in a numpy array of some observations and outputs actions. One should think of this as the action the agent takes when it is being evaluated and therefore this should probably be deterministic.

For meta-reinforcement learning, the agent should also have an `adapt_action` method that takes in a numpy array of some observations and outputs a tuple of actions and miscellaneous policy outputs that might be needed during adaptation. This latter part should be a python dictionary with string keys and numpy array values. Currently supported miscellaneous policy outputs include:

- `"log_probs"`: log probabilities of the actions taken
- `"means"`: the modes of the distributions generated for each observation
- `"stds"`: the standard deviations of the distributions generated for each observation.

Additionally, the agent should also have an `adapt` method for meta-reinforcement learning that takes in a `Rollout` named tuple with numpy arrays containing batches of rollouts for a given data modality. This is to let the agent ingest the generated adaptation data and adapt to the new task.

The `Rollout` named tuple that the agent will be given at the end of each adaptation step looks like so:
```python
class Rollout(NamedTuple):
    observations: npt.NDArray
    actions: npt.NDArray
    rewards: npt.NDArray
    dones: npt.NDArray

    # Auxiliary policy outputs
    log_probs: npt.NDArray | None = None
    means: npt.NDArray | None = None
    stds: npt.NDArray | None = None
```

### Utility outputs

The evaluation utilities output multiple items, not just the overall success rate. Specifically, they return a tuple of three items:
- `mean_success_rate`: the aforementioned success rate. This is a float scalar.
- `mean_returns`: the returns achieved during evaluation averaged across all goal positions / tasks. This is a float scalar.
- `success_rate_per_task`: the success rate achieved for each task evaluated. This is a dictionary keyed by the task name as a string and a float scalar as a value. This scalar is the success rate averaged across goal positions for the given task.

### Other assumptions

The evaluation utilities assume that you have instantiated the environments required using `gym.make`. If not, then these are the implicit assumptions for the `envs` / `eval_envs` provided into the utilities:
- The object is `SyncVectorEnv` or `AsyncVectorEnv`.
- Each sub-env has the following wrappers:
  - `metaworld.wrappers.RandomTaskSelectWrapper` or `metaworld.wrappers.PseudoRandomTaskSelectWrapper`, which have been initialised with the correct set of tasks.
  - `metaworld.wrappers.AutoTerminateOnSuccessWrapper`.
