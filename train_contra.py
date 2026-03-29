"""
魂斗罗强化学习训练脚本
使用 PPO 算法训练一个基于纯视觉的智能体玩 FC 版《魂斗罗》
"""

import os
import time
from typing import Callable

import stable_retro as retro
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
)

from contra_vision_env import ContraEnv, RewardConfig

# ==================== 配置参数 ====================

# 环境参数
FRAME_STACK = 4  # 帧堆叠数量

# ==================== 奖励配置 ====================
# 可根据训练效果调整各项奖励值
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

# PPO 超参数
LEARNING_RATE = 2.5e-4
N_STEPS = 256        # ↑ 从128增加至256，收集更长经验序列，减少梯度噪声
BATCH_SIZE = 256     # 保持256，N_STEPS*N_ENVS=1024，256可整除
N_EPOCHS = 4
ENT_COEF = 0.02      # 适当提高探索力度，帮助发现跳跃等关键动作
GAMMA = 0.98         # ↓ 从0.99降低，更重视近期奖励，加快对向右移动信号的响应
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2

# ==================== 调试/正式训练模式 ====================
DEBUG_MODE = True  # True: 本地调试模式（可视化、快速验证）  False: 正式训练模式（高性能、长时间）

# 训练参数（根据模式自动调整）
TOTAL_TIMESTEPS = 500_000 if DEBUG_MODE else 10_000_000   # 调试: 50万步  正式: 1000万步
RENDER_ENABLED = DEBUG_MODE  # 是否启用实时渲染回调
RENDER_FREQ = 2000           # 渲染回调频率（仅 DEBUG_MODE 生效）
RENDER_STEPS = 300           # 每次渲染步数（仅 DEBUG_MODE 生效）

# 保存路径
LOG_DIR = "logs/contra_ppo"
MODEL_SAVE_DIR = "models/contra_ppo"
TENSORBOARD_LOG = "logs/tensorboard/contra_ppo"

# 回调参数
EVAL_FREQ = 10_000 if DEBUG_MODE else 100_000  # 评估频率  调试: 1万步  正式: 10万步
N_EVAL_EPISODES = 3 if DEBUG_MODE else 2  # 每次评估的回合数
CHECKPOINT_FREQ = 100_000  # 检查点保存频率

# ==================== 环境工厂函数 ====================


def make_env(
    reward_config: RewardConfig = None,
    frame_stack: int = 1,  # 设为1，使用 VecFrameStack 进行堆叠
    rank: int = 0,
    seed: int = 0,
    render_mode: str = "rgb_array",
) -> Callable:
    """
    创建环境的工厂函数

    Args:
        reward_config: 奖励配置
        frame_stack: 帧堆叠数（内部使用）
        rank: 环境序号（用于多进程）
        seed: 随机种子
        render_mode: 渲染模式，"rgb_array" 或 "human"

    Returns:
        环境创建函数
    """

    def _init():
        # 创建 ContraEnv 实例
        env = ContraEnv(
            frame_stack=frame_stack,  # 内部不堆叠，由 VecFrameStack 处理
            reward_config=reward_config,
            render_mode=render_mode,
        )

        # 包装 Monitor 以记录统计信息
        env = Monitor(env)

        # 设置随机种子
        env.reset(seed=seed + rank)

        return env

    return _init


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
                time.sleep(1 / 30)  # ~30fps
                if done.any():
                    obs = self.render_env.reset()
        return True


# ==================== 主训练函数 ====================


def main():
    """主训练函数"""
    print("=" * 60)
    print("魂斗罗强化学习训练")
    print("=" * 60)

    # 创建保存目录
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)

    print(f"\n配置信息:")
    print(f"  - 训练模式: {'🔧 调试模式' if DEBUG_MODE else '🚀 正式训练'}")
    print(f"  - 奖励模式: RAM 变量驱动")
    print(f"  - 并行环境数: {N_ENVS}")
    print(f"  - 总训练步数: {TOTAL_TIMESTEPS:,}")
    print(f"  - 学习率: {LEARNING_RATE}")
    print(f"  - 帧堆叠数: {FRAME_STACK}")
    print(f"\n奖励配置:")
    print(f"  - 进度系数: {REWARD_CONFIG.progress_coef}")
    print(f"  - 后退惩罚: {REWARD_CONFIG.progress_penalty}")
    print(f"  - 原地惩罚: {REWARD_CONFIG.no_progress_penalty}")
    print(f"  - 新最远奖励: {REWARD_CONFIG.new_max_bonus}")
    print(f"  - 分数系数: {REWARD_CONFIG.score_coef}")
    print(f"  - 死亡惩罚: {REWARD_CONFIG.death_penalty}")
    print(f"  - 生存奖励: {REWARD_CONFIG.survival_bonus}")

    # 创建并行环境
    print("\n创建并行环境...")
    env = SubprocVecEnv(
        [
            make_env(
                reward_config=REWARD_CONFIG,
                frame_stack=1,  # 环境内部不堆叠
                rank=i,
                seed=42,
            )
            for i in range(N_ENVS)
        ]
    )

    # 使用 VecFrameStack 堆叠帧
    # 观察空间将变为 (4, 84, 84)
    env = VecFrameStack(env, n_stack=FRAME_STACK, channels_order="first")

    # 使用 VecNormalize 对观察和奖励进行归一化
    # 这有助于训练稳定性
    env = VecNormalize(
        env,
        norm_obs=False,  # 图像观察不做归一化（CnnPolicy 要求 uint8 图像）
        norm_reward=True,  # 归一化奖励
        clip_obs=10.0,  # 观察裁剪范围
        clip_reward=10.0,  # 奖励裁剪范围
        gamma=GAMMA,  # 折扣因子
    )

    print(f"环境观察空间: {env.observation_space}")
    print(f"环境动作空间: {env.action_space}")

    # 创建 PPO 模型
    print("\n创建 PPO 模型...")
    model = PPO(
        "CnnPolicy",  # 使用 CNN 策略（适合图像输入）
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        ent_coef=ENT_COEF,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG,
        device="mps",  # 自动选择设备（优先使用 GPU）
    )

    print(f"模型架构:")
    print(model.policy)

    # 创建回调函数
    callbacks = []

    # 1. 评估回调 - 定期评估并保存最佳模型
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
    eval_env = VecNormalize(
        eval_env,
        norm_obs=False,  # 图像观察不做归一化
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
        render=DEBUG_MODE,  # 调试模式下评估时渲染画面
        verbose=1,
    )
    callbacks.append(eval_callback)

    # 2. 检查点回调 - 定期保存模型
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=os.path.join(MODEL_SAVE_DIR, "checkpoints"),
        name_prefix="contra_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1,
    )
    callbacks.append(checkpoint_callback)

    # 3. 实时渲染回调（复用 eval_env，避免同进程多模拟器实例冲突）
    if RENDER_ENABLED:
        render_callback = RenderCallback(eval_env, render_freq=RENDER_FREQ, n_render_steps=RENDER_STEPS)
        callbacks.append(render_callback)

    # 合并回调
    callback = CallbackList(callbacks)

    # 开始训练
    print("\n开始训练...")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            progress_bar=True,
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

    print("\n训练完成!")
    print("=" * 60)


def load_and_test(model_path: str, vec_normalize_path: str):
    """
    加载模型并测试

    Args:
        model_path: 模型路径
        vec_normalize_path: VecNormalize 统计文件路径
    """
    print(f"加载模型: {model_path}")

    # 创建测试环境
    env = ContraEnv(
        reward_config=REWARD_CONFIG,
        frame_stack=1,
    )
    env = Monitor(env)
    env = VecFrameStack(env, n_stack=FRAME_STACK, channels_order="first")
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False  # 测试模式下不更新统计
    env.norm_reward = False  # 测试时不归一化奖励

    # 加载模型
    model = PPO.load(model_path, env=env)

    # 运行测试
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0

    print("\n开始测试...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        if step % 100 == 0:
            print(f"Step {step}, Reward: {total_reward:.2f}")

    print(f"\n测试结束:")
    print(f"  - 总步数: {step}")
    print(f"  - 总奖励: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 测试模式
        if len(sys.argv) < 4:
            print("用法: python train_contra.py test <model_path> <vec_normalize_path>")
            sys.exit(1)
        load_and_test(sys.argv[2], sys.argv[3])
    else:
        # 训练模式
        main()