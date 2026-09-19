import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

class SelfPlayWrapper(gym.Env):
    """
    支援角色對調 (Role Swapping) 的 Self-Play 包裝器
    讓訓練中的 Agent 有 50% 機率扮演 P1，50% 機率扮演 P2，徹底解決 P2 偏弱的問題。
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, env, opponent_model_path=None):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.opponent_model = None
        if opponent_model_path:
            self.load_opponent(opponent_model_path)

        self.latest_opponent_obs = None
        self.swap_roles = False  # 是否對調角色

    def load_opponent(self, opponent_model_path):
        print(f"🤖 [SelfPlayWrapper] 載入對手模型: {opponent_model_path}")
        self.opponent_model = PPO.load(opponent_model_path)

    def step(self, agent_action):
        # 1. 取得對手動作
        if self.opponent_model is not None and self.latest_opponent_obs is not None:
            opponent_action, _ = self.opponent_model.predict(self.latest_opponent_obs, deterministic=True)
        else:
            opponent_action = np.zeros(self.action_space.shape, dtype=np.float32)

        # 2. 根據是否對調角色，指派動作給 P1 / P2
        if self.swap_roles:
            action_1 = opponent_action
            action_2 = agent_action
        else:
            action_1 = agent_action
            action_2 = opponent_action

        # 3. 執行環境步進
        obs_1, obs_2, r1, r2, terminated, truncated, info = self.env.step_2p(action_1, action_2)

        # 4. 回傳當前 Agent 視角的觀測與獎勵
        if self.swap_roles:
            self.latest_opponent_obs = obs_1
            return obs_2, r2, terminated, truncated, info
        else:
            self.latest_opponent_obs = obs_2
            return obs_1, r1, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_1, obs_2, info = self.env.reset(seed=seed, options=options)

        # 每局有 50% 機率對調角色，確保 Agent 雙向均衡學習
        self.swap_roles = np.random.random() > 0.5

        if self.swap_roles:
            self.latest_opponent_obs = obs_1
            return obs_2, info
        else:
            self.latest_opponent_obs = obs_2
            return obs_1, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()