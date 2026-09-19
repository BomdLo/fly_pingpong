import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SelfPlayPingPongEnv(gym.Env):
    """
    雙人對打 Self-Play 環境包裝器
    """

    def __init__(self, base_env, opponent_model=None):
        super().__init__()
        self.env = base_env  # 假設你的底層 MuJoCo 環境支援兩隻手臂
        self.opponent_model = opponent_model  # 對手 (Agent 2) 的 PPO 模型

        # 繼承原本單人的 Observation 與 Action 空間
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def set_opponent_model(self, model):
        """動態更新對手的模型"""
        self.opponent_model = model

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # obs 應該包含兩邊的觀察值: obs['agent_1'], obs['agent_2']
        self.latest_obs = obs
        return obs['agent_1'], info

    def step(self, action_1):
        # 1. 取得對手 (Agent 2) 的動作
        if self.opponent_model is not None:
            action_2, _ = self.opponent_model.predict(self.latest_obs['agent_2'], deterministic=True)
        else:
            # 如果還沒有對手模型，對手先站著不動或隨機打
            action_2 = self.action_space.sample() * 0.0

        # 2. 將兩邊的動作一起送入物理引擎
        combined_action = {'agent_1': action_1, 'agent_2': action_2}
        next_obs, rewards, terminated, truncated, info = self.env.step(combined_action)

        self.latest_obs = next_obs

        # 3. 只回傳 Agent 1 的狀態與獎勵給目前正在學習的 PPO
        return next_obs['agent_1'], rewards['agent_1'], terminated, truncated, info