"""
魂斗罗 (Contra) 强化学习训练环境
基于 Retro RAM 变量的奖励系统（无 YOLO 依赖）
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import stable_retro as retro
from dataclasses import dataclass
from typing import Optional, Dict, Tuple


# NES 魂斗罗自定义动作集（基于 DISCRETE 模式的 combo groups）
# Combo Group 0: [无, UP, DOWN]       → 索引 0/1/2
# Combo Group 1: [无, LEFT, RIGHT]    → 索引 0/1/2
# Combo Group 2: [无, B(射), A(跳), B+A(射+跳)] → 索引 0/1/2/3
# DISCRETE action = group0_idx + 3 * group1_idx + 9 * group2_idx
CUSTOM_ACTIONS = [
    0,   # 0:  不动 (NOOP)            = (0, 0, 0)
    6,   # 1:  右走                   = (0, RIGHT, 0)
    3,   # 2:  左走                   = (0, LEFT, 0)
    18,  # 3:  跳                     = (0, 0, A)
    24,  # 4:  右跳                   = (0, RIGHT, A)
    21,  # 5:  左跳                   = (0, LEFT, A)
    9,   # 6:  射击                   = (0, 0, B)
    15,  # 7:  右走+射击              = (0, RIGHT, B)
    12,  # 8:  左走+射击              = (0, LEFT, B)
    33,  # 9:  右跳+射击              = (0, RIGHT, B+A)
    30,  # 10: 左跳+射击              = (0, LEFT, B+A)
    2,   # 11: 下蹲                   = (DOWN, 0, 0)
    11,  # 12: 下蹲+射击              = (DOWN, 0, B)
    10,  # 13: 上射                   = (UP, 0, B)
    16,  # 14: 右走+上射              = (UP, RIGHT, B)
]

# 动作名称（用于调试显示）
ACTION_NAMES = [
    "不动(NOOP)", "右走", "左走", "跳", "右跳", "左跳",
    "射击", "右走+射击", "左走+射击", "右跳+射击", "左跳+射击",
    "下蹲", "下蹲+射击", "上射", "右走+上射",
]

# 含跳跃的动作索引（用于跳跃奖励判断）
JUMP_ACTIONS = {3, 4, 5}           # 纯跳跃动作
JUMP_RIGHT_ACTIONS = {4, 9}        # 跳跃+右移动作（含/不含射击）


@dataclass
class RewardConfig:
    """奖励配置"""
    # 进度奖励（基于 xscroll）
    progress_coef: float = 1.0          # xscroll 每增加1的奖励系数
    progress_penalty: float = -0.3      # xscroll 后退的惩罚系数
    no_progress_penalty: float = -0.02  # 原地不动惩罚（降低，避免画面未滚动时错误惩罚）
    no_progress_grace: int = 120        # 开局免惩罚步数（画面需要~90步才开始滚动）
    new_max_bonus: float = 0.5          # 到达新最远位置额外奖励
    # 分数奖励（击杀敌人、拾取道具等都会增加分数）
    score_coef: float = 0.1             # 分数增加的奖励系数
    # 生存奖励
    death_penalty: float = -5.0         # 死亡惩罚（加大，让模型更怕死）
    survival_bonus: float = 0.005       # 每步存活奖励
    # 跳跃奖励（鼓励跳跃+前进的组合动作）
    jump_forward_bonus: float = 3.0     # 跳跃+右移时的额外奖励（高于单步右移的进度奖励）
    jump_bonus: float = 0.5             # 任何跳跃动作的小额奖励（帮助模型多尝试跳跃）


class ContraEnv(gym.Env):
    """
    魂斗罗强化学习环境

    观察空间：堆叠 N 帧 84x84 灰度图
    动作空间：离散动作（来自 retro 环境）
    奖励：基于 RAM 变量（关卡进度、分数变化、生存状态）
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        frame_stack: int = 4,
        reward_config: Optional[RewardConfig] = None,
        render_mode: str = "rgb_array",
    ):
        super().__init__()

        # 创建 retro 环境（使用 DISCRETE 模式，按键组合经过验证）
        self.retro_env = retro.RetroEnv(
            game="Contra-Nes",
            state="Level1",
            inttype=retro.data.Integrations.ALL,
            use_restricted_actions=retro.Actions.DISCRETE,
            render_mode=render_mode,
        )

        # 自定义动作映射表（我们的索引 → retro DISCRETE 索引）
        self._action_map = CUSTOM_ACTIONS

        # 覆盖动作空间为 Discrete(15)
        self.action_space = spaces.Discrete(len(CUSTOM_ACTIONS))

        # 观察空间：堆叠 frame_stack 帧的 84x84 灰度图
        self.frame_stack = frame_stack
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(frame_stack, 84, 84),
            dtype=np.uint8,
        )

        # 奖励配置
        self.reward_config = reward_config or RewardConfig()

        # 状态变量
        self._reset_state()

        # 原始画面尺寸
        self.raw_height, self.raw_width = self.retro_env.observation_space.shape[:2]

    def _reset_state(self):
        """重置所有状态变量"""
        self.frame_buffer = np.zeros((self.frame_stack, 84, 84), dtype=np.uint8)
        self.prev_frame_raw = None
        self.prev_action = 0
        self.current_step = 0
        self.max_episode_steps = 4000
        # RAM 状态
        self.prev_xscroll = 0
        self.prev_score = 0
        self.prev_lives = 3
        self.max_xscroll = 0

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理：转灰度 + 缩放到 84x84（retro 返回 RGB 格式）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def _stack_frames(self, new_frame: np.ndarray) -> np.ndarray:
        """更新帧堆叠缓冲区"""
        self.frame_buffer = np.roll(self.frame_buffer, shift=-1, axis=0)
        self.frame_buffer[-1] = new_frame
        return self.frame_buffer.copy()

    def _compute_rewards_from_ram(self, info: dict) -> Tuple[float, dict]:
        """
        基于 RAM 变量计算奖励

        Returns:
            (total_reward, reward_details)
        """
        config = self.reward_config
        reward = 0.0
        reward_details = {}

        # 读取当前 RAM 变量
        curr_xscroll = info.get('xscroll', 0)
        curr_score = info.get('score', 0)
        curr_lives = info.get('lives', self.prev_lives)
        player_state = info.get('player_state', 0)

        # === 1. 进度奖励（基于 xscroll 变化）===
        xscroll_delta = curr_xscroll - self.prev_xscroll

        if xscroll_delta > 0:
            # 向右移动
            progress_reward = xscroll_delta * config.progress_coef
            reward += progress_reward
            reward_details['progress'] = progress_reward

            # 到达新的最远位置额外奖励
            if curr_xscroll > self.max_xscroll:
                reward += config.new_max_bonus
                reward_details['new_max'] = config.new_max_bonus
                self.max_xscroll = curr_xscroll
        elif xscroll_delta < 0:
            # 向左移动（后退）
            penalty = xscroll_delta * abs(config.progress_penalty)
            reward += penalty
            reward_details['regress'] = penalty
        else:
            # 原地不动（开局免惩罚期内不惩罚，因为画面需要~90步才开始滚动）
            if self.current_step > config.no_progress_grace:
                reward += config.no_progress_penalty
                reward_details['no_progress'] = config.no_progress_penalty

        # === 2. 分数奖励（击杀敌人、拾取道具等）===
        score_delta = curr_score - self.prev_score
        if score_delta > 0:
            score_reward = score_delta * config.score_coef
            reward += score_reward
            reward_details['score'] = score_reward

        # === 3. 生存奖励 ===
        if self.current_step > 1 and (curr_lives < self.prev_lives or player_state == 15):
            # 死亡
            reward += config.death_penalty
            reward_details['death'] = config.death_penalty
        else:
            # 存活
            reward += config.survival_bonus
            reward_details['survival'] = config.survival_bonus

        # 更新前一帧状态
        self.prev_xscroll = curr_xscroll
        self.prev_score = curr_score
        self.prev_lives = curr_lives

        return reward, reward_details

    def reset(self, seed=None, options=None):
        """重置环境"""
        self._reset_state()

        obs_raw, info = self.retro_env.reset(seed=seed)

        # 从 info 初始化 RAM 状态
        self.prev_xscroll = info.get('xscroll', 0)
        self.prev_score = info.get('score', 0)
        self.prev_lives = info.get('lives', 3)
        self.max_xscroll = self.prev_xscroll

        # 预处理并填充帧缓冲
        obs_processed = self._preprocess_frame(obs_raw)
        for i in range(self.frame_stack):
            self.frame_buffer[i] = obs_processed

        self.prev_frame_raw = obs_raw.copy()

        return self.frame_buffer.copy(), info

    def step(self, action: int):
        """执行一步"""
        # 确保 action 是 int（model.predict 可能返回 numpy 数组）
        action = int(action)
        # 将我们的动作索引映射为 retro DISCRETE 动作索引
        retro_action = self._action_map[action]
        obs_raw, retro_reward, terminated, truncated, info = self.retro_env.step(retro_action)
        self.current_step += 1

        # 超过最大步数则截断
        if self.current_step >= self.max_episode_steps:
            truncated = True

        # 预处理 + 帧堆叠
        obs_processed = self._preprocess_frame(obs_raw)
        stacked_obs = self._stack_frames(obs_processed)

        # 基于 RAM 变量计算奖励
        reward, reward_details = self._compute_rewards_from_ram(info)

        # 跳跃奖励（基于动作索引判断，不再依赖按键数组）
        if action in JUMP_RIGHT_ACTIONS:
            # 跳跃+右移：大额奖励
            reward += self.reward_config.jump_forward_bonus
            reward_details['jump_forward'] = self.reward_config.jump_forward_bonus
        elif action in JUMP_ACTIONS:
            # 原地跳/左跳：小额奖励（帮助模型发现跳跃）
            reward += self.reward_config.jump_bonus
            reward_details['jump'] = self.reward_config.jump_bonus

        # 更新 info
        info['reward_details'] = reward_details
        info['retro_reward'] = retro_reward
        info['current_step'] = self.current_step
        info['max_xscroll'] = self.max_xscroll

        # 保存当前帧
        self.prev_frame_raw = obs_raw.copy()
        self.prev_action = action

        return stacked_obs, reward, terminated, truncated, info

    def render(self, mode=None):
        """渲染环境"""
        return self.retro_env.render()

    def close(self):
        """关闭环境"""
        try:
            self.retro_env.close()
        except AttributeError:
            pass


# 向后兼容别名
ContraVisionEnv = ContraEnv


def create_contra_env(
    reward_config: Optional[RewardConfig] = None,
    **kwargs,
) -> ContraEnv:
    """
    创建魂斗罗环境的便捷函数

    Args:
        reward_config: 奖励配置，若为 None 则使用默认配置
        **kwargs: 其他参数（frame_stack, render_mode 等）

    Returns:
        ContraEnv 实例
    """
    return ContraEnv(
        reward_config=reward_config,
        **kwargs,
    )


if __name__ == "__main__":
    # 简单测试
    reward_config = RewardConfig()

    env = ContraEnv(reward_config=reward_config)
    obs, info = env.reset()
    print(f"观察空间形状: {obs.shape}")
    print(f"动作空间: {env.action_space}")
    print(f"自定义动作数量: {len(CUSTOM_ACTIONS)}")
    print(f"初始 xscroll: {info.get('xscroll', 'N/A')}")
    print(f"初始 lives: {info.get('lives', 'N/A')}")
    print(f"初始 score: {info.get('score', 'N/A')}")

    # 打印动作列表
    action_names = [
        "不动(NOOP)", "右走", "左走", "跳", "右跳", "左跳",
        "射击", "右走+射击", "左走+射击", "右跳+射击", "左跳+射击",
        "下蹲", "下蹲+射击", "上射", "右走+上射",
    ]
    print("\n可用动作:")
    for i, name in enumerate(action_names):
        print(f"  {i:2d}: {name}  {CUSTOM_ACTIONS[i]}")

    # 运行几步测试
    print("\n开始测试 10 步:")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(
            f"Step {i}: action={action}({action_names[action]}), reward={reward:.4f}, "
            f"xscroll={info.get('xscroll', 'N/A')}, "
            f"lives={info.get('lives', 'N/A')}, "
            f"score={info.get('score', 'N/A')}, "
            f"details={info['reward_details']}"
        )
        if term or trunc:
            break

    # 验证跳跃动作是否生效
    print("\n=== 按键验证（300 步右走）===")
    env.reset()
    print("连续执行 300 步「右走」(action=1):")
    for i in range(300):
        obs, reward, term, trunc, info = env.step(1)  # 纯右走
        if i % 30 == 0:
            print(f"  Step {i:3d}: xscroll={info.get('xscroll', 0)}, lives={info.get('lives', '?')}, reward={reward:.4f}")
        if term or trunc:
            print(f"  Step {i}: 游戏结束 (terminated={term}, truncated={trunc})")
            break

    print("\n=== 按键验证（300 步右跳+射击）===")
    env.reset()
    print("连续执行 300 步「右跳+射击」(action=9):")
    for i in range(300):
        obs, reward, term, trunc, info = env.step(9)  # 右跳+射击
        if i % 30 == 0:
            print(f"  Step {i:3d}: xscroll={info.get('xscroll', 0)}, lives={info.get('lives', '?')}, reward={reward:.4f}")
        if term or trunc:
            print(f"  Step {i}: 游戏结束 (terminated={term}, truncated={trunc})")
            break

    print("\n如果 xscroll 在增加，说明按键正常工作")

    env.close()
