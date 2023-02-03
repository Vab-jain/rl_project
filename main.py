import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from pathlib import Path
import numpy as np
import pickle
import time
import platform

from envs.env_project import GridWorldEnv
import model
from utils import plot_durations, plot_loss, plot_rewards

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


steps_done = 0

def select_action(state):
    '''
    samples an action given a state with epsilon-greedy policy
    '''
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)



def optimize_model():
    if len(memory) < BATCH_SIZE:    # if memory buffer is less than BATCH_SIZE, return
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = memory.Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    # print(loss.item())

    return (loss.item())


if __name__ == "__main__":

    isTraining = True
    isTesting = False
    isSaving = True
    isLogging = True

    training_log = {}
    hyperparams_log = {}

    # define the PATHS here
    # data_folder = Path("/home/vj/Link to WiSe 2022-23/Reinforcement Learning/project/Implementation")
    train_data_folder = Path("/home/vj/Link to WiSe 2022-23/Reinforcement Learning/project/Implementation/datasets/data_easy/train/task")
    val_data_folder = Path("/home/vj/Link to WiSe 2022-23/Reinforcement Learning/project/Implementation/datasets/data_easy/val/task")
    trained_model_path = Path("/home/vj/Link to WiSe 2022-23/Reinforcement Learning/project/Implementation/model.pth")

    # create the environment    
    env = GridWorldEnv(jsondata=train_data_folder/"0_task.json")

    #### HYPER-PARAMETERS ####
    BATCH_SIZE = 128    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    GAMMA = 0.99        # GAMMA is the discount factor as mentioned in the previous section
    EPS_START = 0.9     # EPS_START is the starting value of epsilon
    EPS_END = 0.05      # EPS_END is the final value of epsilon
    EPS_DECAY = 1000    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    TAU = 0.005         # TAU is the update rate of the target network
    LR = 1e-4           # LR is the learning rate of the AdamW optimizer
    H = 500             # H is the hyperpameter to control the lenght of the episode (to avoid infinite episode lenghts)
    NUM_EPISODES = 10000   # total number of episodes to train on


    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the state observations
    state, info = env.reset(jsondata=train_data_folder/"0_task.json")
    n_observations = len(state)     # lenght of state feature

    # Define the POLICY network and the TARGET network
    policy_net = model.DQN(n_observations, n_actions).to(device)
    target_net = model.DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())     # copy the state of Policy net to Target net

    if isTraining:
        # Define the optimizer
        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        # Initialize the Replay Memory
        memory = model.ReplayMemory(10000)

        
        episode_durations = []
        losses = []
        episode_rewards = []

        #### Training loop: START ####
        training_start_time = time.time()
        
        for i_episode in range(NUM_EPISODES):
            episode_loss = 0
            episode_reward = 0

            datapath = f'{train_data_folder}/{random.randrange(4000)}_task.json'
            # Initialize the environment and get it's state
            state, info = env.reset(jsondata=datapath)
            state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            for t in count():
                action = select_action(state)
                observation, reward, terminated, _ = env.step(action.item())
                reward = torch.tensor([reward])
                episode_reward += int(reward)
                done = terminated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, device=device, dtype=torch.float32).unsqueeze(0)

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                temp_loss = optimize_model()
                if temp_loss:
                    episode_loss += optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)

                if done or t>H:
                    episode_durations.append(t + 1)
                    losses.append(episode_loss)
                    episode_rewards.append(episode_reward)
                    plot_durations(episode_durations)
                    plot_loss(losses)
                    plot_rewards(episode_rewards)
                    break
        
        training_end_time = time.time()

        # calculating metrics
        avg_episode_len = np.average(np.array(episode_durations))
        avg_episode_reward = np.average(np.array(episode_rewards))

        avg_episode_len_clipped = np.average(np.array(episode_durations[-100:]))
        avg_episode_reward_clipped = np.average(np.array(episode_rewards[-100:]))

        total_training_time = training_end_time - training_start_time
        avg_training_time_per_episode = total_training_time/NUM_EPISODES


        print('Complete')
        plot_durations(episode_durations,show_result=True)
        plot_loss(losses,show_result=True)
        plot_rewards(episode_rewards,show_result=True)
        plt.ioff()
        plt.show()


        #### LOGGING THE TRAINING SPACE ####
        if isSaving:
            # save the model and logs on a local file
            torch.save(policy_net.state_dict(),trained_model_path)
            training_log['episode_durations'] = episode_durations
            training_log['losses'] = losses
            training_log['episode_rewards'] = episode_rewards
            with open('training_log.pkl', 'wb') as f:
                pickle.dump(training_log, f)
        if isLogging:
            # log the hyper-parameters
            hyperparams_log['System_info'] = platform.uname()
            if torch.cuda.is_available():
                hyperparams_log['GPU_info'] = torch.cuda.get_device_name(device=device)
            else:
                hyperparams_log['GPU_info'] = "None"
            hyperparams_log['BATCH_SIZE'] = BATCH_SIZE
            hyperparams_log['GAMMA'] = GAMMA
            hyperparams_log['EPS_START'] = EPS_START
            hyperparams_log['EPS_END'] = EPS_END
            hyperparams_log['EPS_DECAY'] = EPS_DECAY
            hyperparams_log['TAU'] = TAU
            hyperparams_log['LR'] = LR
            hyperparams_log['H'] = H
            hyperparams_log['NUM_EPISODES'] = NUM_EPISODES
            hyperparams_log['avg_episode_len'] = avg_episode_len
            hyperparams_log['avg_episode_reward'] = avg_episode_reward
            hyperparams_log['avg_episode_len_clipped'] = avg_episode_len_clipped
            hyperparams_log['avg_episode_reward_clipped'] = avg_episode_reward_clipped
            hyperparams_log['total_training_time'] = total_training_time
            hyperparams_log['avg_training_time_per_episode'] = avg_training_time_per_episode
            hyperparams_log['POLICY_NETWORK'] = policy_net
            hyperparams_log['TARGET_NETWORK'] = target_net
            # hyperparams_log['training_log'] = training_log
            with open('hyper_parameters_log.txt', 'a') as f:
                f.write('\nNEW TRAINING\n')
                for param_name,param in hyperparams_log.items():
                    f.write(f'{param_name}: {param}\n')
                f.write('\n')


        #### Training loop: END ####

    if isTesting:
        
        ####### TESTING THE TRAINED MODEL OVER NEW ENVIRONMENT #######
        # define the paths
        datapath = f'{val_data_folder}/{random.randrange(100000,100399)}_task.json'
        # outpath = f'{val_data_folder}/results/'
        # Path(outpath).mkdir(parents=True, exist_ok=True)
        # pathlist = Path(val_data_folder).glob()

        policy_net.load_state_dict(torch.load(trained_model_path))     # copy the state of Policy net from the trained model
        policy_net.eval()



        solved_list = []
        test_episode_rewards = []

        for path in val_data_folder.iterdir():
            # initialize the state for the task
            state, info = env.reset(jsondata=path)

            test_episode_reward = 0

            # env.render()    # render initial state
            # env.render(toFilePath=outpath)

            policy_net.load_state_dict(torch.load(trained_model_path))     # copy the state of Policy net to Target net

            state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)

            for t in count():
                action = select_action(state)
                observation, reward, terminated, info = env.step(action.item())
                reward = torch.tensor([reward])
                test_episode_reward += reward
                done = terminated

                # print(action)
                # env.render(toFilePath=outpath)

                if terminated:
                    next_state = None
                else:
                    # env.render()    # render the state
                    next_state = torch.tensor(observation, device=device, dtype=torch.float32).unsqueeze(0)
                
                # Move to the next state
                state = next_state
                
                if done or t>H:
                    if info['solved']:
                        print('Solved')
                    # else:
                    #     # print('Crashed')
                    solved_list.append(info['solved'])
                    test_episode_rewards.append(test_episode_reward)
                    break
        
        print(solved_list)
        print(test_episode_rewards)

        print(np.array(solved_list).sum()/len(solved_list)*100)
        ####### TESTING END #######
        