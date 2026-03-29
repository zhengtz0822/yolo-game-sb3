"""
test_jump.py - 直观验证跳跃指令是否让游戏角色跳起来
使用 stable_retro 原生环境 + cv2 显示标注画面
"""

import numpy as np
import cv2
import stable_retro as retro

# ============================================================
# 按键定义（使用 DISCRETE 模式的整数索引）
# Combo Group 0: [无, UP, DOWN]       → 0/1/2
# Combo Group 1: [无, LEFT, RIGHT]    → 0/1/2
# Combo Group 2: [无, B(射), A(跳), B+A(射+跳)] → 0/1/2/3
# action = group0 + 3*group1 + 9*group2
# ============================================================
NOOP             = 0   # (0, 0, 0)
RIGHT            = 6   # (0, RIGHT, 0)
JUMP             = 18  # (0, 0, A)
RIGHT_JUMP       = 24  # (0, RIGHT, A)
RIGHT_JUMP_SHOOT = 33  # (0, RIGHT, B+A)
RIGHT_SHOOT      = 15  # (0, RIGHT, B)

ACTION_LABELS = {
    0: "[NOOP]", 6: "[RIGHT]", 18: "[A]",
    24: "[RIGHT,A]", 33: "[RIGHT,B,A]", 15: "[RIGHT,B]",
}

# ============================================================
# 测试阶段定义: (阶段名称, 按键, 帧数)
# ============================================================
PHASES = [
    ("等待开始(NOOP)",       NOOP,             60),
    ("右跳(RIGHT_JUMP)",     RIGHT_JUMP,      120),
    ("等待(NOOP)",           NOOP,             30),
    ("原地跳(JUMP)",         JUMP,             60),
    ("等待(NOOP)",           NOOP,             30),
    ("右走(RIGHT)",          RIGHT,            60),
    ("等待(NOOP)",           NOOP,             30),
    ("右跳射(RJ_SHOOT)",     RIGHT_JUMP_SHOOT,120),
]


def get_info_str(info: dict) -> str:
    """从 info 字典提取关键 RAM 变量"""
    xscroll = info.get("xscroll", info.get("x_pos", "N/A"))
    lives   = info.get("lives",   "N/A")
    score   = info.get("score",   "N/A")
    return f"xscroll={xscroll}, lives={lives}, score={score}"


def buttons_str(action) -> str:
    return ACTION_LABELS.get(action, f"[action={action}]")


def run_test():
    print("=" * 60)
    print("Contra 跳跃测试启动")
    print("按 'q' 随时退出")
    print("=" * 60)

    env = retro.RetroEnv(
        game="Contra-Nes",
        state="Level1",
        inttype=retro.data.Integrations.ALL,
        use_restricted_actions=retro.Actions.DISCRETE,
        render_mode="human",
    )

    obs, info = env.reset()
    print(f"环境初始化完成 | obs shape: {obs.shape} | info: {info}")
    print()

    total_frame = 0
    quit_flag   = False

    for phase_name, action, num_frames in PHASES:
        if quit_flag:
            break

        print(f"{'─' * 50}")
        print(f"[开始阶段] {phase_name}  ({num_frames} 帧)  按键: {buttons_str(action)}")
        print(f"{'─' * 50}")

        for frame_idx in range(num_frames):
            obs, reward, terminated, truncated, info = env.step(action)
            total_frame += 1
            done = terminated or truncated

            # ── cv2 显示 ──────────────────────────────────────
            frame_bgr   = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            frame_large = cv2.resize(frame_bgr, (512, 480),
                                     interpolation=cv2.INTER_NEAREST)

            cv2.putText(frame_large,
                        f"Phase: {phase_name}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_large,
                        f"Frame: {total_frame}  ({frame_idx + 1}/{num_frames})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_large,
                        f"Buttons: {buttons_str(action)}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame_large,
                        get_info_str(info),
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            cv2.imshow("Contra Jump Test", frame_large)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("用户按下 'q'，退出测试。")
                quit_flag = True
                break

            # ── 每 10 帧打印一次控制台日志 ────────────────────
            if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
                print(f"  [Phase: {phase_name}] Frame {total_frame:4d}: "
                      f"{get_info_str(info)} | reward={reward:.2f} | "
                      f"buttons={buttons_str(action)}")

            if done:
                print(f"  !! 游戏结束 (terminated={terminated}, truncated={truncated})，重置环境")
                obs, info = env.reset()

        # 阶段结束汇报
        if not quit_flag:
            print(f"\n[阶段结束] {phase_name} — 最终状态: {get_info_str(info)}\n")

    # ============================================================
    # 清理
    # ============================================================
    cv2.destroyAllWindows()
    env.close()
    print("=" * 60)
    print(f"测试完毕，共执行 {total_frame} 帧")
    print("=" * 60)


if __name__ == "__main__":
    run_test()
