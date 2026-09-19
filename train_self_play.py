"""Train a PPO table-tennis policy by repeatedly playing frozen past versions.

The first opponent is ``fly_pingpong_pro_model``.  Each round saves the
current learner as a snapshot, and that snapshot becomes the opponent for the
next round.  This prevents both sides from changing at the same time, which is
usually unstable for PPO self-play.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from rally_env import RallyTableTennisEnv
from self_play_wrapper import SelfPlayWrapper


def make_training_env(opponent_path: str):
    """Build one independent learner-vs-frozen-opponent environment."""
    return SelfPlayWrapper(
        RallyTableTennisEnv(),
        opponent_model_path=opponent_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO self-play for fly ping-pong")
    parser.add_argument("--seed-model", default="fly_pingpong_pro_model",
                        help="initial frozen opponent and learner checkpoint (without .zip is OK)")
    parser.add_argument("--output", default="self_play_model",
                        help="base name for the final PPO model")
    parser.add_argument("--snapshot-dir", default="self_play_snapshots",
                        help="directory in which frozen opponents are retained")
    parser.add_argument("--rounds", type=int, default=12,
                        help="number of opponent-update rounds")
    parser.add_argument("--steps-per-round", type=int, default=250_000,
                        help="PPO timesteps collected before freezing a new opponent")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="parallel environments; lower this if MuJoCo uses too much CPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="deterministic matches against the original model after every round")
    return parser.parse_args()


def evaluate(model, baseline_path: str, episodes: int) -> tuple[float, float]:
    """Return learner win rate and mean legal hits against the original policy."""
    baseline = PPO.load(baseline_path, device="cpu")
    env = RallyTableTennisEnv()
    wins = 0
    total_rally_hits = 0
    try:
        for episode in range(episodes):
            learner_is_p1 = episode % 2 == 0
            obs_1, obs_2, _ = env.reset(seed=10_000 + episode)
            while True:
                if learner_is_p1:
                    action_1, _ = model.predict(obs_1, deterministic=True)
                    action_2, _ = baseline.predict(obs_2, deterministic=True)
                else:
                    action_1, _ = baseline.predict(obs_1, deterministic=True)
                    action_2, _ = model.predict(obs_2, deterministic=True)
                obs_1, obs_2, _, _, terminated, truncated, info = env.step_2p(action_1, action_2)
                if terminated or truncated:
                    winner = info.get("winner")
                    learner_player = 1 if learner_is_p1 else 2
                    wins += winner == learner_player
                    total_rally_hits += info.get("rally_hits", 0)
                    break
    finally:
        env.close()
    return wins / episodes, total_rally_hits / episodes


def main() -> None:
    args = parse_args()
    if args.rounds < 1 or args.steps_per_round < 1 or args.n_envs < 1:
        raise ValueError("rounds, steps-per-round, and n-envs must all be positive")

    seed_path = Path(args.seed_model)
    if not seed_path.exists() and not seed_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"Cannot find seed model: {args.seed_model}")

    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    opponent_path = str(seed_path)

    # PPO needs the wrapped single-agent interface; the wrapper internally
    # queries a frozen opponent for the second player's action.
    train_env = make_vec_env(
        lambda: make_training_env(opponent_path),
        n_envs=args.n_envs,
        seed=args.seed,
    )
    model = PPO.load(args.seed_model, env=train_env, device=args.device)
    model.tensorboard_log = "./self_play_tb/"

    try:
        for round_index in range(1, args.rounds + 1):
            print(
                f"\n=== Self-play round {round_index}/{args.rounds} "
                f"vs {opponent_path} ({args.steps_per_round:,} steps) ==="
            )
            model.learn(
                total_timesteps=args.steps_per_round,
                reset_num_timesteps=False,
                progress_bar=True,
            )

            # Save before replacing the opponent, so it is an immutable policy
            # for the entire next round.
            snapshot_path = snapshot_dir / f"round_{round_index:03d}"
            model.save(snapshot_path)
            opponent_path = str(snapshot_path)
            train_env.env_method("load_opponent", opponent_path)
            if args.eval_episodes:
                win_rate, mean_hits = evaluate(model, args.seed_model, args.eval_episodes)
                print(
                    f"Evaluation vs original: win rate {win_rate:.0%} | "
                    f"mean legal returns {mean_hits:.2f}"
                )

        model.save(args.output)
        print(f"\nTraining complete: {args.output}.zip")
        print(f"Latest frozen opponent: {opponent_path}.zip")
    finally:
        train_env.close()


if __name__ == "__main__":
    main()
