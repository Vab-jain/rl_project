import torch

import matplotlib
import matplotlib.pyplot as plt

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display



def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    # plt.subplot(211)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('No of Steps')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
    

def plot_loss(losses, show_result=False):
    losses_tensor = torch.tensor(losses,dtype=torch.float)

    plt.figure(2)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.plot(losses_tensor.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def plot_rewards(episode_rewards, show_result=False):
    episode_rewards_tensor = torch.tensor(episode_rewards,dtype=torch.float)

    plt.figure(3)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.plot(episode_rewards_tensor.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
    