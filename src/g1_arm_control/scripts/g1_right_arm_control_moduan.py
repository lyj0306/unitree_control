#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G1 右臂末端位姿控制（G1 29 自由度）：根据末端位置与姿态逆解得到 7 关节角，通过 SDK 控制右臂。

使用流程：连接好机器人（上电、网络连通）→ 运行本脚本 → 输入末端位姿（一行 6 个数）→ 回车即开始控制右臂。

输入格式：一行 6 个数，空格分隔 → x y z roll pitch yaw（单位：m, rad）
  例如：0.15 -0.3 0.15 0 0 -0.3

用法：
  python g1_right_arm_ee_control.py                    # 运行后按提示输入一行，回车即运行
  python g1_right_arm_ee_control.py -i "0.15 -0.3 0.15 0 0 -0.3"
  echo "0.15 -0.3 0.15 0 0 -0.3" | python g1_right_arm_ee_control.py

运动速度：由 --duration 控制（到达目标所用时间，秒）。duration 越小运动越快，越大越慢。
  例如：--duration 0.5 较快，--duration 2 较慢。默认 1.0。

运动完一个位姿后可继续输入新位姿，右臂会从当前位置插值到新目标。输入 q 回车退出。

按 Ctrl+C 立即停止控制并释放模式。

依赖：numpy<2, pinocchio(conda-forge), casadi；unitree_sdk2_python。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# 与 test_g1_29_arm_ik_visual 一致：优先使用 conda/venv 的 pinocchio
def _strip_ros_from_path():
    ros_keywords = ("/opt/ros", "ros/humble", "ros/noetic")
    removed = [p for p in sys.path if any(kw in p for kw in ros_keywords)]
    for p in removed:
        sys.path.remove(p)

def _prefer_env_site_packages():
    _strip_ros_from_path()
    base = os.environ.get("CONDA_PREFIX") or sys.prefix
    if "/opt/ros" in base:
        return
    site = os.path.join(base, "lib", "python{}.{}".format(*sys.version_info[:2]), "site-packages")
    if os.path.isdir(site):
        if site in sys.path:
            sys.path.remove(site)
        sys.path.insert(0, site)

_prefer_env_site_packages()

# 路径：g1_arm_control/src（逆解）；workspace/src/unitree_sdk2_python（SDK）
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PKG_DIR, "src")
WS_SRC = os.path.dirname(PKG_DIR)
SDK_PATH = os.path.join(WS_SRC, "unitree_sdk2_python")
for p in (SDK_PATH, SRC_DIR):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import numpy as np


def rpy_to_matrix(rpy):
    """RPY(roll, pitch, yaw) rad -> 3x3 旋转矩阵，固定轴 x-y-z。"""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def pose_4x4(x, y, z, rpy=(0.0, 0.0, 0.0)):
    """末端位姿 4x4 齐次矩阵：位置 (m) + 姿态 RPY (rad)。"""
    R = rpy_to_matrix(np.asarray(rpy, dtype=np.float64))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [float(x), float(y), float(z)]
    return T


# 左臂固定位姿（仅控制右臂时，逆解仍需传入左臂目标，此处取一个合理默认）
DEFAULT_LEFT_EE_POSE = pose_4x4(0.3, 0.2, 0.3, rpy=(0.0, 0.0, 0.0))

# G1 29 电机数、右臂 7 关节在 SDK 中的索引（与 g1_low_level_example 一致）
G1_NUM_MOTOR = 29
RIGHT_ARM_SDK_INDICES = [22, 23, 24, 25, 26, 27, 28]  # RightShoulderPitch -> RightWristYaw
# 逆解输出 q 为 14 维：左臂 0:7，右臂 7:14
IK_RIGHT_SLICE = slice(7, 14)

Kp_arm = [40] * 7
Kd_arm = [1] * 7


def run_ik_and_get_right_arm_q(arm_ik, right_ee_pose_4x4, left_ee_pose_4x4=None):
    """
    调用逆解，返回右臂 7 个关节角 (rad)。
    left_ee_pose_4x4 不传则使用 DEFAULT_LEFT_EE_POSE。
    """
    if left_ee_pose_4x4 is None:
        left_ee_pose_4x4 = DEFAULT_LEFT_EE_POSE
    q_full, _ = arm_ik.solve_ik(left_ee_pose_4x4, right_ee_pose_4x4)
    q_full = np.asarray(q_full, dtype=np.float64).ravel()
    return q_full[IK_RIGHT_SLICE].copy()


def parse_xyz_rpy(line):
    """解析一行 6 个数：x y z roll pitch yaw。支持逗号或空格分隔。"""
    line = (line or "").strip().replace(",", " ")
    parts = line.split()
    if len(parts) != 6:
        raise ValueError(f"需要恰好 6 个数 (x y z roll pitch yaw)，当前得到 {len(parts)} 个")
    return [float(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(
        description="G1 29 自由度右臂末端位姿控制：输入一行 x y z roll pitch yaw，逆解后通过 SDK 控制右臂 7 关节。"
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help='末端位姿，6 个数空格分隔，如 "0.15 -0.3 0.15 0 0 -0.3"'
    )
    parser.add_argument("--duration", type=float, default=2.0,
                        help="到达目标所用时间 (s)，越小运动越快，如 0.5=较快 2=较慢")
    parser.add_argument("--channel-ip", type=str, default=None, help="ChannelFactoryInitialize 的 IP，不传则默认")
    args = parser.parse_args()

    # 输入：优先 --input，否则读一行 stdin，否则提示输入
    if args.input is not None:
        x, y, z, roll, pitch, yaw = parse_xyz_rpy(args.input)
    else:
        try:
            print("请输入右臂末端位姿（一行 6 个数：x y z roll pitch yaw），粘贴后按回车即运行，例如：0.15 -0.3 0.15 0 0 -0.3")
            line = input("> ").strip()
            if not line:
                line = "0.15 -0.3 0.15 0 0 -0.3"
            x, y, z, roll, pitch, yaw = parse_xyz_rpy(line)
        except (ValueError, EOFError) as e:
            print("解析失败，使用默认 0.15 -0.3 0.15 0 0 -0.3:", e)
            x, y, z, roll, pitch, yaw = 0.15, -0.3, 0.15, 0.0, 0.0, -0.3
    right_pose = pose_4x4(x, y, z, rpy=(roll, pitch, yaw))
    print(f"目标右臂末端: 位置 [{x}, {y}, {z}] m, RPY [{roll}, {pitch}, {yaw}] rad")

    # 逆解（不打开可视化，无需连接机器人）
    try:
        from g1_29_arm_ik import G1_29_ArmIK
    except ImportError as e:
        print("导入 g1_29_arm_ik 失败，请确保已安装 pinocchio(conda-forge)、numpy<2、casadi，且 PYTHONPATH 含 g1_arm_control/src。")
        raise SystemExit(1) from e

    arm_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)
    q_right_target = run_ik_and_get_right_arm_q(arm_ik, right_pose)
    print("逆解得到右臂 7 关节角 (rad):", np.round(q_right_target, 4).tolist())

    # SDK：与 g1_low_level_example 一致
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.utils.thread import RecurrentThread
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

    print("正在连接机器人并获取控制权...")
    if args.channel_ip:
        ChannelFactoryInitialize(0, args.channel_ip)
    else:
        ChannelFactoryInitialize(0)

    # 控制参数：全身 Kp/Kd（与示例一致），右臂用 Kp_arm/Kd_arm
    Kp_all = [
        60, 60, 60, 100, 40, 40,
        60, 60, 60, 100, 40, 40,
        60, 40, 40,
        40, 40, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40,
    ]
    Kd_all = [
        1, 1, 1, 2, 1, 1,
        1, 1, 1, 2, 1, 1,
        1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
    ]
    ModePR = 0

    class RightArmController:
        def __init__(self, target_q_right, duration_s, control_dt=0.002):
            self.target_q_right = np.asarray(target_q_right, dtype=np.float64)
            self.duration_s = duration_s
            self.control_dt = control_dt
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = None
            self.mode_machine_ = 0
            self.update_mode_machine_ = False
            self.crc = CRC()
            self.time_ = 0.0
            self.start_q_right_ = None  # 收到首帧 state 时填充
            self.running = True  # False 时立即停止发令，供 Ctrl+C 使用

        def set_new_target(self, new_q_right):
            """更新目标关节角，从当前实际关节角插值到新目标（可运动完一位姿后继续输入新位姿时调用）。"""
            self.target_q_right = np.asarray(new_q_right, dtype=np.float64).ravel()[:7]
            if self.low_state is not None:
                self.start_q_right_ = np.array(
                    [self.low_state.motor_state[i].q for i in RIGHT_ARM_SDK_INDICES],
                    dtype=np.float64,
                )
            else:
                self.start_q_right_ = self.target_q_right.copy()
            self.time_ = 0.0

        def Init(self):
            self.msc = MotionSwitcherClient()
            self.msc.SetTimeout(5.0)
            self.msc.Init()
            status, result = self.msc.CheckMode()
            if result is None:
                print("错误：无法连接机器人（CheckMode 失败）。请检查：")
                print("  1) 机器人已上电且与电脑在同一网段")
                print("  2) 若需指定 IP，使用 --channel-ip <机器人IP>")
                sys.exit(1)
            while result.get("name"):
                self.msc.ReleaseMode()
                time.sleep(0.5)
                status, result = self.msc.CheckMode()
                if result is None:
                    print("错误：获取控制权过程中连接异常。")
                    sys.exit(1)
            self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
            self.lowcmd_publisher_.Init()
            self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
            self.lowstate_subscriber.Init(self.LowStateHandler, 10)

        def LowStateHandler(self, msg):
            self.low_state = msg
            if not self.update_mode_machine_:
                self.mode_machine_ = self.low_state.mode_machine
                self.update_mode_machine_ = True
            if self.start_q_right_ is None:
                self.start_q_right_ = np.array(
                    [self.low_state.motor_state[i].q for i in RIGHT_ARM_SDK_INDICES],
                    dtype=np.float64,
                )

        def Start(self):
            self.lowCmdWriteThreadPtr = RecurrentThread(
                interval=self.control_dt, target=self.LowCmdWrite, name="control"
            )
            while not self.update_mode_machine_:
                time.sleep(0.1)
            self.lowCmdWriteThreadPtr.Start()

        def LowCmdWrite(self):
            if not self.running:
                return
            if self.low_state is None:
                return
            self.time_ += self.control_dt
            ratio = min(1.0, self.time_ / self.duration_s) if self.duration_s > 0 else 1.0
            ratio = ratio * (2.0 - ratio)  # smooth (ease-out)
            if self.start_q_right_ is not None:
                q_right_now = (1.0 - ratio) * self.start_q_right_ + ratio * self.target_q_right
            else:
                q_right_now = self.target_q_right

            self.low_cmd.mode_pr = ModePR
            self.low_cmd.mode_machine = self.mode_machine_
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.motor_cmd[i].mode = 1
                self.low_cmd.motor_cmd[i].tau = 0.0
                self.low_cmd.motor_cmd[i].q = self.low_state.motor_state[i].q
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = Kp_all[i]
                self.low_cmd.motor_cmd[i].kd = Kd_all[i]
            for k, idx in enumerate(RIGHT_ARM_SDK_INDICES):
                self.low_cmd.motor_cmd[idx].q = float(q_right_now[k])
                self.low_cmd.motor_cmd[idx].kp = Kp_arm[k]
                self.low_cmd.motor_cmd[idx].kd = Kd_arm[k]

            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)

    ctrl = RightArmController(q_right_target, duration_s=args.duration)
    ctrl.Init()
    ctrl.Start()
    print("右臂正在朝目标末端位姿运动。")
    print("输入下一目标位姿 (6 个数: x y z roll pitch yaw) 回车即更新目标继续运动；输入 q 回车退出。")
    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line.lower() == "q":
                break
            try:
                x, y, z, roll, pitch, yaw = parse_xyz_rpy(line)
                new_pose = pose_4x4(x, y, z, rpy=(roll, pitch, yaw))
                new_q = run_ik_and_get_right_arm_q(arm_ik, new_pose)
                ctrl.set_new_target(new_q)
                print(f"已更新目标: 位置 [{x}, {y}, {z}] m, RPY [{roll}, {pitch}, {yaw}] rad")
            except ValueError as e:
                print("解析失败，需要 6 个数 (x y z roll pitch yaw):", e)
            except Exception as e:
                print("逆解或更新失败:", e)
    except KeyboardInterrupt:
        pass
    # Ctrl+C：立即停止发令并释放模式，确保机器人停止
    ctrl.running = False
    time.sleep(0.05)
    try:
        ctrl.msc.ReleaseMode()
        print("已释放控制模式，已停止。")
    except Exception:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
