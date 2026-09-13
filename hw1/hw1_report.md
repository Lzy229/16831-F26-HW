# HW1
## Behaviour Cloning
### Code
#### `rl_trainers`
- call the function of agent class, in this case is BC_Agent
- the Agent includes the actor, which is the policy class

#### `MLP_policy`
- the MLP_policy class based on basepolicy, the MLP_policy SL is based on MLP_policy
- The MLP_Policy is to train MLP network, building the policy class we need to construct the MLP module in `pytorch_util.build_mlp` where we need to build the mlp by given `input_size, output_size, n_layers, size` 
- `get_action:` 
    - given the observation, get the action. Behaviour cloning means we only need to train the network. the same as Loaded_gaussian_policy
- `forward`:
    - whats the difference between get_action and forward
        - forward: only get the result of passing the network
- `update`:
    - train process: forward, calculate loss, backward, update optimizer

#### `replay_buffer`
- the buffer is actually a buffer stores all the observations and other information
- `random_sample`:
    - randomly choose the informoations.
    - use `np.random.permutation` to get the random indics, and return batch size


#### `build_mlp`
- the input
    - `input_size, output_size, n_layers, size`
    - `input_size: size of input layer`, determined by ob_dim(observation), get from sample process, `sample_from_data`
    - `output_size: size of output layer`, determined by ac_dim(action), get from sample process
    - `n_layer: the number of hidden layer`, determine by the cmd
    - `size: the dimension of each hidden layer`, default value of size is 64
- the output
    - an `nn.module` represents the Multi Layer Perception
- hidden layer
    - the hidden layer 

#### `sample_trajectory`
- one trajectory is one rollout
- how to reset environment?
    - `env.reset()`?
- the `min_timesteps_per_batch` is not equal to number of rollouts. A rollout could have several timesteps, the maximum timesteps is 40.

### Running the codes
#### `collect_training_trajectories`
-  TODO decide whether to load training data or use the current policy to collect more data
    - at the beginning, there is no current data to sample.
    - how to use `load_initial_expertdata` 
        - use `pickle` to load .pkl file get the init data.


### mean and standard deviation
- get the expert return
    - the vanilla bc only get `n_iter == 1`
- mean refers to `Eval_StdReturn`, standard refers to `Eval_AverageReturn` 
    - `Eval_StdReturn, Eval_AverageReturn` are report in `perform_logging`
    - first compute the summary of the Paths[rewards], mean == mean(Paths), standard == std(Paths)
- what the difference between eval and train?
    - eval is the result of bc agent, train is the result of expert data
- parameters:
    - the `eval_batch_size` must greater than `ep_len`
    - set `eval_batch_size == 5000` and `ep_len == 1000`
    - change `num_agent_train_steps_per_iter` to increase accuracy.
    - the param of 1.3 is `eval_batch_size == 5000`, `num_agent_train_steps_per_iter == 3000`, `data_size == batch_size == 1000`, `network_size == n_layers == 2`

### large training steps means greater performamce?
- my hypothesis is more training steps would lead to greater performance
- my task is hopper
- hyperparameters:
    - `num_agent_train_steps_per_iter`: on the vanilla
    - `n_layers` and `size`: decide whether each mlp has a corresponding training steps
- results:
    - `n_layers` and `size` are default:

    |  steps |  mean |  std  | 
    | 1000 | 1108.3494873046875 | 85.01180267333984 |
    | 1500 | 841.3966674804688. | 51.948974609375 |
    | 2000 | 885.49853515625 | 57.91110610961914 |
    | 2500 | 1141.0955810546875 | 48.47797775268555 |
    | 3000 | 1245.7635498046875 | 113.24816131591797 |
    | 3500 | 1258.8677978515625 | 156.43727111816406 |
    | 4000 | 1388.70263671875 | 220.0292510986328 |
    | 4500 | 1432.702880859375 | 224.86033630371094 |
    | 5000 | 1216.836181640625 | 126.21666717529297 | 
    | 5500 | 1329.0811767578125 | 280.8166809082031 | 
    | 6000 | 1311.243896484375 | 98.08900451660156 |
    | 6500 | 925.0961303710938 | 367.1213684082031 |
    | 7000 | 1836.080322265625 | 603.3098754882812 |
    | 7500 | 1303.78515625 | 522.232177734375 |
    | 8000 | 1589.4453125 | 347.8478088378906 |
    | 8500 | 1413.504638671875 | 86.26758575439453 |
    | 9000 | 1076.051025390625 | 329.76153564453125 | 
    | 9500 | 1505.958984375 | 126.43050384521484 |
    | 10000 | 1433.8330078125 | 308.71466064453125 |
    - basically, when we get more training steps, we can result in a greater performance


    

