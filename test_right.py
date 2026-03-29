"""
简单测试脚本 - 使用固定动作让玩家向右走
"""

import os
import time
from contra_vision_env import ContraEnv

os.environ['SDL_VIDEODRIVER'] = 'dummy'

# 动作映射 (Contra-Nes Discrete 36)
# Action 2: RIGHT (向右走)
# Action 3: RIGHT + A (向右射击)
# Action 19: UP + RIGHT + A (右上射击)
# Action 10: DOWN + RIGHT (向右蹲下)
# Action 11: DOWN + RIGHT + A (向右蹲下射击)

RIGHT_ACTIONS = [2, 3, 10, 11, 18, 19]  # 向右移动的动作


def main():
    print("=" * 60)
    print("测试向右移动策略")
    print("=" * 60)

    env = ContraEnv(frame_stack=1)

    print(f"\n向右动作: {RIGHT_ACTIONS}")
    print("这些动作包含 RIGHT 按钮\n")

    obs, info = env.reset()
    prev_xscroll = info.get('xscroll', 0)

    total_reward = 0
    right_count = 0

    # 运行 100 步，始终选择向右移动
    for i in range(100):
        # 选择向右移动的动作
        action = RIGHT_ACTIONS[i % len(RIGHT_ACTIONS)]

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # 通过 xscroll 增加判断是否向右移动
        curr_xscroll = info.get('xscroll', 0)
        if curr_xscroll > prev_xscroll:
            right_count += 1
        prev_xscroll = curr_xscroll

        if i % 20 == 0:
            print(f"Step {i}: xscroll={curr_xscroll}, reward={total_reward:.2f}, "
                  f"details={info.get('reward_details', {})}")

        if done or truncated:
            print(f"\n游戏结束于 Step {i}")
            break

    final_xscroll = info.get('xscroll', 0)
    print(f"\n结果:")
    print(f"  xscroll 增加次数（向右移动）: {right_count}")
    print(f"  最终 xscroll: {final_xscroll}")
    print(f"  最远 xscroll: {info.get('max_xscroll', 0)}")
    print(f"  总奖励: {total_reward:.2f}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
