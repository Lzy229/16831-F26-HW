# hw3
## Double DQN
### run_dqn
- define class Q_Trainer, call `run_training_loop`
- `run_training_loop` call `rl_trainer.run_training_loop`
- in the `rl_trainer.run_training_loop`
    - iteration n_iters+1 times
    - collect training trajectories
        - using the collecting policy to sample the trajectories, the policy is argmax_policy in this experiments
        - action is obtained from Argmax_policy.get_action(sample a batch of trajectories)
        - sample some paths to the replay buffer. 
    - call `train_agent()`
        - per iterations do `num_agent_train_steps_per_iter` times training
        - sample a batch, from replay buffer. 
            - randomly sample pathes
        - call `agent.train()`
- in agent `train()`
    - call `critic.update`
    - call `critic.update_target_network`

### critic update
- `qa_t_values = self.q_net(ob_no)`
    - `self.q_net` is the network predicting the q_values
    - the network is created by `create_lander_q_network`
        - the network receive the observation, output the action_number dim
        - (I guess) the q_value q(s, a), means for the input observation, and output q_values for each actions
    - based on the observation this batch, output the value of taking actions, `(batch_num, ac_num)`
- `torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)`
    - we have the `(batch_num, ac_num)` of q_values, based on the actions we take (`ac_na`), get the q_values corresponding to the actions.
    - the action is get using origin/target network?
        - `collect_policy` is DQNAgent
        - `ac_na` is obtained by `self.agent.sample`
        -  the action is cellecting the origin network
    - the size after gather is `(batch_num, 1)`, after `squeeze(1)` is `(batch_num, )`
- `qa_tp1_values`
    - compute the q_values of target_network, the target network compute the q_values at s', as the next observation
    - the output is `(batch_num, ac_num)`
- `double_q`
    - is you use the double q algorithms
    - the action is get from the origin network (`ac_na`), compute the q_values based on target network `qa_tp1_values`
    - compute the `q_tp1`
- `target`
    - currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
    - currentReward: `reward_n`
    - qValuesOfNextTimestep: `q_tp1`
    - not terminal: `1 - terminal_n`

### Policy
- the policy is `Argmax_policy`
- based on currect states of observations, use the `q_net` to get the q_values
- use the `argmax` to get the index of the highest value action for each observation

## Actor_critic
### run_actor_critic
- define class `AC_Trainer`, call `run_traning_loop`
- call `RL_trainer.run_training_loop`, call `train_agent`
- in `train_agent`
    - call `replay_buffer.sample()` to sample trajectories
    - call `agent.train()` to do training, record the loss

### Agent
#### `train()`
- update the critic, call `critic.update()` function

#### Advantage
- estimate the advantages using network


### Critic
- the network `critic_network`
    - the input_dim is `ob_dim`, input observation
    - the output_dim is 1, output the v(s)
    - call `forward_np` to complete the process
    - (batch_num, ob_dim) to (batch_num, 1)
- 