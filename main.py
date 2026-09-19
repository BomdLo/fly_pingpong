import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class FlyPingPongEnv(gym.Env):
    """簡化版的蒼蠅打桌球強化學習環境。"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.dt = 0.02
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.fly_pos = np.array([-1.5, 0.0, 1.0], dtype=np.float32)
        self.fly_vel = np.zeros(3, dtype=np.float32)
        self.ball_pos = np.array([1.5, np.random.uniform(-0.3, 0.3), 1.2], dtype=np.float32)
        self.ball_vel = np.array([-2.5, np.random.uniform(-0.2, 0.2), 1.5], dtype=np.float32)
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.fly_pos, self.fly_vel, self.ball_pos, self.ball_vel]).astype(np.float32)

    def step(self, action):
        self.step_count += 1
        force = action[:3] * 5.0
        self.fly_vel += force * self.dt
        self.fly_pos += self.fly_vel * self.dt
        self.ball_vel[2] -= 9.8 * self.dt
        self.ball_pos += self.ball_vel * self.dt

        dist_to_ball = np.linalg.norm(self.fly_pos - self.ball_pos)
        is_hit = dist_to_ball < 0.3 and action[3] > 0.0
        if is_hit:
            self.ball_vel[0] = abs(self.ball_vel[0]) + 2.0
            self.ball_vel[2] += 2.0

        reward = -dist_to_ball * 0.1
        if is_hit:
            reward += 50.0
        if self.ball_pos[0] > 0 and is_hit:
            reward += 100.0

        terminated = self.ball_pos[2] <= 0.0 or self.fly_pos[2] <= 0.0
        truncated = self.step_count >= 200
        return self._get_obs(), reward, terminated, truncated, {}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 開始訓練，使用運算硬體裝置: {device}")
    env = make_vec_env(FlyPingPongEnv, n_envs=8)
    model = PPO(
        "MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048,
        batch_size=64, gae_lambda=0.95, gamma=0.99,
        tensorboard_log="./fly_ppo_tensorboard/", device=device,
    )
    print("🏋️ 訓練中...")
    model.learn(total_timesteps=200_000)
    model.save("fly_pingpong_ppo_model")
    print("✅ 訓練完成，模型已儲存為 fly_pingpong_ppo_model.zip")
