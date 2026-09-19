import gymnasium as gym
from stable_baselines3 import PPO
from dm_tt_env import DeepMindTableTennisEnv


def main():
    # 1. 建立環境 (如果你有把落檯大獎改成 +2000，它會在這次訓練中發揮作用！)
    env = DeepMindTableTennisEnv()

    # 2. 職人級微調超參數
    custom_objects = {
        # 調低學習率 (原本預設是 3e-4)：讓它不要對新分數反應過度，慢慢修正力道
        "learning_rate": 5e-5,

        # 絕對不要加 ent_coef！我們希望它保持現有的完美揮拍姿勢
    }

    # 3. 載入我們剛練好的 300 萬步冠軍模型
    model_path = "fly_pingpong_mujoco_model"
    print(f"正在載入冠軍模型: {model_path}.zip ...")

    model = PPO.load(
        model_path,
        env=env,
        custom_objects=custom_objects
    )

    # 4. 開始精雕訓練 (建議拉長到 300 萬步，讓它有時間摸索力道)
    print("開始精雕細琢訓練...")
    model.learn(total_timesteps=3_000_000, progress_bar=True)

    # 5. 儲存進化後的新模型
    new_model_name = "fly_pingpong_pro_model"
    model.save(new_model_name)
    print(f"微調完成！已儲存為 {new_model_name}.zip")

    env.close()


if __name__ == "__main__":
    main()