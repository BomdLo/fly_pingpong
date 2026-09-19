import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import TimeLimit  # 修正：改用 gymnasium
from dm_tt_env import DeepMindTableTennisEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env():
    env = DeepMindTableTennisEnv()
    # 強制單局最多 500 步，時間到自動 reset
    return TimeLimit(env, max_episode_steps=500)

if __name__ == "__main__":
    device = "cpu"
    print(f"🚀 啟動修正後的高擬真訓練，使用裝置: {device}")

    env = make_vec_env(make_env, n_envs=8, vec_env_cls=SubprocVecEnv)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        tensorboard_log="./mujoco_tb/",
        device=device
    )

    print("🏋️ 修正邊界與時間限制，重新訓練中...")
    model.learn(total_timesteps=3_000_000)

    model.save("fly_pingpong_mujoco_model")
    print("✅ 訓練完成！模型已儲存為 fly_pingpong_mujoco_model.zip")