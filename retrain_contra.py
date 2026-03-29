"""
魂斗罗强化学习 - 继续训练脚本
加载已有模型，在现有基础上继续训练（使用相同的环境、动作空间和奖励函数）
"""

import os
import sys
import time
import glob
from typing import Callable

import stable_retro as retro
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
)

from contra_vision_env import ContraEnv, RewardConfig

# ==================== 配置参数 ====================

# 模型路径（按优先级自动查找，也可手动指定）
MODEL_PATH = None  # 设为 None 则自动查找最新模型

# 自动查找顺序:
#   1. models/contra_ppo/best_model/best_model.zip
#   2. models/contra_ppo/final_model.zip
#   3. models/contra_ppo/checkpoints/ 下最新的 checkpoint
AUTO_SEARCH_PATHS = [
    "models/contra_ppo/best_model/best_model.zip",
    "models/contra_ppo/final_model.zip",
]

# VecNormalize 统计文件路径（如有）
VEC_NORMALIZE_PATH = "models/contra_ppo/vec_normalize.pkl"

# 环境参数
FRAME_STACK = 4

# ==================== 奖励配置 ====================
# 继续训练时使用与原训练相同的奖励函数（也可微调）
REWARD_CONFIG = RewardConfig(
    progress_coef=1.0,
    progress_penalty=-0.3,
    no_progress_penalty=-0.1,
    new_max_bonus=0.5,
    score_coef=0.1,
    death_penalty=-2.0,
    survival_bonus=0.005,
)

# 并行环境数量
N_ENVS = 4

# ==================== 继续训练超参数 ====================
# 可以和原训练不同（如降低学习率做微调）
LEARNING_RATE = 1e-4       # ↓ 比初始训练低，避免遗忘已学到的策略
N_STEPS = 256
BATCH_SIZE = 256
N_EPOCHS = 4
ENT_COEF = 0.02            # 探索系数（可根据需要调整）
GAMMA = 0.98
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2

# ==================== 训练控制 ====================
RETRAIN_TIMESTEPS = 1_000_000  # 继续训练的步数
DEBUG_MODE = True              # 调试模式

# 调试模式下的参数覆盖
if DEBUG_MODE:
    RETRAIN_TIMESTEPS = 300_000

# 保存路径（继续训练的模型单独保存，不覆盖原模型）
LOG_DIR = "logs/contra_ppo_retrain"
MODEL_SAVE_DIR = "models/contra_ppo_retrain"
TENSORBOARD_LOG = "logs/tensorboard/contra_ppo_retrain"

# 回调参数
EVAL_FREQ = 10_000 if DEBUG_MODE else 100_000
N_EVAL_EPISODES = 3 if DEBUG_MODE else 2
CHECKPOINT_FREQ = 100_000

# ==================== 环境工厂函数 ====================


def make_env(
    reward_config: RewardConfig = None,
    frame_stack: int = 1,
    rank: int = 0,
    seed: int = 0,
    render_mode: str = "rgb_array",
) -> Callable:
    """创建环境的工厂函数"""

    def _init():
        env = ContraEnv(
            frame_stack=frame_stack,
            reward_config=reward_config,
            render_mode=render_mode,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


# ==================== 自动查找模型 ====================


def find_model_path() -> str:
    """自动查找最新的可用模型"""
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        return MODEL_PATH

    # 按优先级查找
    for path in AUTO_SEARCH_PATHS:
        if os.path.exists(path):
            return path

    # 查找最新 checkpoint
    checkpoint_dir = "models/contra_ppo/checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.zip")))
        if checkpoints:
            return checkpoints[-1]

    # 也查找 retrain 目录
    for path in [
        "models/contra_ppo_retrain/best_model/best_model.zip",
        "models/contra_ppo_retrain/final_model.zip",
    ]:
        if os.path.exists(path):
            return path

    retrain_ckpt = "models/contra_ppo_retrain/checkpoints"
    if os.path.exists(retrain_ckpt):
        checkpoints = sorted(glob.glob(os.path.join(retrain_ckpt, "*.zip")))
        if checkpoints:
            return checkpoints[-1]

    return None


# ==================== 实时渲染回调 ====================


class RenderCallback(BaseCallback):
    """训练过程中定期渲染游戏画面的回调"""

    def __init__(self, render_env, render_freq=5000, n_render_steps=200, verbose=0):
        super().__init__(verbose)
        self.render_env = render_env
        self.render_freq = render_freq
        self.n_render_steps = n_render_steps

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            obs = self.render_env.reset()
            for _ in range(self.n_render_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.render_env.step(action)
                self.render_env.render()
                time.sleep(1 / 30)
                if done.any():
                    obs = self.render_env.reset()
        return True


# ==================== 主训练函数 ====================


def main():
    """继续训练主函数"""
    print("=" * 60)
    print("魂斗罗强化学习 - 继续训练")
    print("=" * 60)

    # 查找模型
    model_path = find_model_path()
    if model_path is None:
        print("\n❌ 未找到可用模型！请先运行 train_contra.py 完成初始训练。")
        print("搜索路径:")
        for p in AUTO_SEARCH_PATHS:
            print(f"  - {p}")
        print(f"  - models/contra_ppo/checkpoints/*.zip")
        sys.exit(1)

    print(f"\n📦 加载模型: {model_path}")

    # 创建保存目录
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)

    print(f"\n配置信息:")
    print(f"  - 训练模式: {'调试模式' if DEBUG_MODE else '正式训练'}")
    print(f"  - 基础模型: {model_path}")
    print(f"  - 继续训练步数: {RETRAIN_TIMESTEPS:,}")
    print(f"  - 学习率: {LEARNING_RATE}")
    print(f"  - 熵系数: {ENT_COEF}")
    print(f"  - 并行环境数: {N_ENVS}")
    print(f"  - 帧堆叠数: {FRAME_STACK}")

    # 创建并行环境
    print("\n创建并行环境...")
    env = SubprocVecEnv(
        [
            make_env(
                reward_config=REWARD_CONFIG,
                frame_stack=1,
                rank=i,
                seed=42,
            )
            for i in range(N_ENVS)
        ]
    )
    env = VecFrameStack(env, n_stack=FRAME_STACK, channels_order="first")

    # 加载或创建 VecNormalize
    if os.path.exists(VEC_NORMALIZE_PATH):
        print(f"加载 VecNormalize 统计: {VEC_NORMALIZE_PATH}")
        env = VecNormalize.load(VEC_NORMALIZE_PATH, env)
        env.training = True  # 继续训练时更新统计
        env.norm_reward = True
    else:
        print("未找到 VecNormalize 统计文件，创建新的...")
        env = VecNormalize(
            env,
            norm_obs=False,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=GAMMA,
        )

    print(f"环境观察空间: {env.observation_space}")
    print(f"环境动作空间: {env.action_space}")

    # 加载已有模型并更新超参数
    print("\n加载模型并设置新超参数...")
    model = PPO.load(
        model_path,
        env=env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        ent_coef=ENT_COEF,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        tensorboard_log=TENSORBOARD_LOG,
        device="mps",
    )

    print(f"\n模型架构:")
    print(model.policy)

    # 创建回调函数
    callbacks = []

    # 1. 评估回调
    eval_env = DummyVecEnv(
        [
            make_env(
                reward_config=REWARD_CONFIG,
                frame_stack=1,
                rank=0,
                seed=100,
                render_mode="human" if DEBUG_MODE else "rgb_array",
            )
        ]
    )
    eval_env = VecFrameStack(eval_env, n_stack=FRAME_STACK, channels_order="first")
    if os.path.exists(VEC_NORMALIZE_PATH):
        eval_env = VecNormalize.load(VEC_NORMALIZE_PATH, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=False,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=GAMMA,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_SAVE_DIR, "best_model"),
        log_path=os.path.join(LOG_DIR, "eval"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=DEBUG_MODE,
        verbose=1,
    )
    callbacks.append(eval_callback)

    # 2. 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=os.path.join(MODEL_SAVE_DIR, "checkpoints"),
        name_prefix="contra_retrain",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1,
    )
    callbacks.append(checkpoint_callback)

    # 3. 实时渲染回调
    if DEBUG_MODE:
        render_callback = RenderCallback(
            eval_env, render_freq=2000, n_render_steps=300
        )
        callbacks.append(render_callback)

    callback = CallbackList(callbacks)

    # 开始继续训练
    print(f"\n开始继续训练（{RETRAIN_TIMESTEPS:,} 步）...")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=RETRAIN_TIMESTEPS,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=False,  # 不重置步数计数器，保持连续
        )
    except KeyboardInterrupt:
        print("\n训练被用户中断，保存当前模型...")

    # 保存最终模型
    final_model_path = os.path.join(MODEL_SAVE_DIR, "final_model")
    model.save(final_model_path)
    print(f"\n最终模型已保存至: {final_model_path}")

    # 保存 VecNormalize 统计信息
    vec_normalize_path = os.path.join(MODEL_SAVE_DIR, "vec_normalize.pkl")
    env.save(vec_normalize_path)
    print(f"VecNormalize 统计信息已保存至: {vec_normalize_path}")

    # 关闭环境
    env.close()
    eval_env.close()

    print("\n继续训练完成!")
    print(f"模型保存在: {MODEL_SAVE_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
