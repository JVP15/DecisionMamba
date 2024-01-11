# episode evaluation taken from https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/evaluation/evaluate_episodes.py

import numpy as np
import torch


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        max_length=None
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state, info = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            max_length=max_length,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, terminated, truncated, info = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if terminated or truncated:
            break

    return episode_return, episode_length

def evaluate_episode_recurrent(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    model.eval()
    model.recurrent(batch_size=1)
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state, info = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((1, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(1, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            max_length=1,
        )
        #actions = action
        action = action.detach().cpu().numpy()

        state, reward, terminated, truncated, info = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = cur_state
        rewards[-1] = reward
        timesteps += 1

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]

        episode_return += reward
        episode_length += 1

    model.reset_cache()

    return episode_return, episode_length

def evaluate_episode_few_shot(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        dataset=None,
        num_examples=2,
    ):

    rewards = np.sum(dataset.with_format('np')['train']['rewards'], axis=-1)
    top_trajectories = np.argpartition(rewards, -num_examples)[-num_examples:]

    # get the top actions and actions
    top_actions = dataset.with_format('np')['train']['actions'][top_trajectories]
    top_states = dataset.with_format('np')['train']['observations'][top_trajectories]
    traj_len = 1000
    top_timesteps = np.arange(0, traj_len).reshape(1, -1).repeat(num_examples, axis=0)

    # reshape them from (num_examples, timesteps, dim) to (num_examples * timesteps, dim)
    top_actions = top_actions.reshape(-1, top_actions.shape[-1])
    top_states = top_states.reshape(-1, top_states.shape[-1])
    top_timesteps = top_timesteps.reshape(-1, 1)

    model.eval()
    model.recurrent(batch_size=1)
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state, info = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((1, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(1, device=device, dtype=torch.float32)

    # concat the top_states with the current state, with the top states first
    states = torch.cat([torch.from_numpy(top_states).to(device=device, dtype=torch.float32), states], dim=0)
    actions = torch.cat([torch.from_numpy(top_actions).to(device=device, dtype=torch.float32), actions], dim=0)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1,1).expand(-1, traj_len+1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    timesteps = torch.cat([torch.from_numpy(top_timesteps).to(device=device, dtype=torch.long), timesteps], dim=0)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            max_length=1,
        )
        actions = action
        action = action.detach().cpu().numpy()

        state, reward, terminated, truncated, info = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = cur_state
        rewards[-1] = reward
        timesteps += 1

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]

        episode_return += reward
        episode_length += 1

    model.reset_cache()

    return episode_return, episode_length
