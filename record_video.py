import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from evaluation.evaluate_episodes import evaluate_episode_rtg, evaluate_episode_recurrent, evaluate_episode_few_shot
from models.decision_mamba import TrainableDM, TrainableDT
from transformers import Trainer

import mujoco
import gymnasium as gym


def evaluate_episodes(num_eval_episodes, model):
    returns, lengths = [], []

    model.eval()

    if isinstance(model, TrainableDM):
        eval_fn = evaluate_episode_rtg # evaluate_episode_recurrent
    else:
        eval_fn = evaluate_episode_rtg

    #eval_fn = evaluate_episode_few_shot

    with torch.no_grad():
        for _ in tqdm(range(num_eval_episodes)):
            ret, length = eval_fn(
                env=env,
                state_dim=state_dim,
                act_dim=act_dim,
                model=model,
                scale=scale,
                state_mean=state_mean,
                state_std=state_std,
                device=device,
                target_return=TARGET_RETURN,
                #max_length=1000,
               # dataset=load_dataset("edbeeching/decision_transformer_gym_replay", "halfcheetah-medium-v2")
            )

            returns.append(ret)
            lengths.append(ret)

    return {
        f'target_{TARGET_RETURN}_return_mean': np.mean(returns),
        f'target_{TARGET_RETURN}_return_std': np.std(returns),
        f'target_{TARGET_RETURN}_length_mean': np.mean(lengths),
        f'target_{TARGET_RETURN}_length_std': np.std(lengths),
    }


if __name__ == '__main__':
    is_mamba = True

    if is_mamba:
        model = TrainableDM.from_pretrained('trained_models/dm_halfcheetah-expert-v2')
    else:
        model = TrainableDT.from_pretrained('trained_models/dt_halfcheetah-expert-v2')

    dataset = load_dataset("edbeeching/decision_transformer_gym_replay", "halfcheetah-expert-v2")


    act_dim = len(dataset['train'][0]["actions"][0])
    state_dim = len(dataset['train'][0]["observations"][0])
    # calculate dataset stats for normalization of states
    states = []
    traj_lens = []
    for obs in dataset['train']["observations"]:
        states.extend(obs)
        traj_lens.append(len(obs))
    states = np.vstack(states)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    state_mean = state_mean.astype(np.float32)
    state_std = state_std.astype(np.float32)

    device = 'cuda'


    model.to(device)

    env = gym.make("HalfCheetah-v4", render_mode='rgb_array')

    env = gym.wrappers.RecordVideo(env, 'video')
    max_ep_len = 1000
    scale = 1000.0  # normalization for rewards/returns
    TARGET_RETURN = 12000 / scale  # evaluation is conditioned on a return of 12000, scaled accordingly

    print(evaluate_episodes(3, model))


