#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G1 29-DOF 双臂逆解可视化测试（参考 xr_teleoperate 遥操作/示教思路）。

- 使用 G1_29 URDF：g1_body29_hand14.urdf（默认 xr_teleoperate 路径）
- Meshcat 中显示：基坐标系(world)、左/右末端目标(L_ee_target, R_ee_target)、
  逆解后的实际末端(L_ee, R_ee)与机器人姿态，便于直观检查是否到达预期位姿。

运行（在 g1_arm_control 包根目录或 workspace 根目录）：
  PYTHONPATH=src python scripts/test_g1_29_arm_ik_visual.py
  PYTHONPATH=src python scripts/test_g1_29_arm_ik_visual.py --no-pause   # 不暂停，连续跑完

依赖：conda-forge 的 pinocchio + NumPy 1.x（与 NumPy 2 不兼容）。推荐：
  conda create -n g1_ik python=3.10 -y && conda activate g1_ik
  conda install -c conda-forge pinocchio -y
  pip install "numpy<2" meshcat casadi
运行前不要 source /opt/ros/...。若报 CXXABI/libstdc++：export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
"""

from __future__ import annotations

import argparse
import os
import sys

# 避免加载 ROS 的 pinocchio，强制优先使用当前 conda/venv 的 site-packages
def _strip_ros_from_path():
    ros_keywords = ("/opt/ros", "ros/humble", "ros/noetic")
    removed = [p for p in sys.path if any(kw in p for kw in ros_keywords)]
    for p in removed:
        sys.path.remove(p)


def _prefer_env_site_packages():
    """把当前环境（conda/venv）的 site-packages 放到 sys.path 最前，确保先于 ROS 被搜到。"""
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

# 可导入 g1_29_arm_ik（插在 conda site-packages 之后，保证 import pinocchio 仍先走 conda）
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PKG_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(1, SRC_DIR)

# 默认 URDF：优先绝对路径，否则用相对 workspace 的 xr_teleoperate 路径
def _default_urdf_paths():
    abs_path = "/home/lyj/unitree_ws_ros2/src/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
    if os.path.isfile(abs_path):
        return abs_path, os.path.dirname(abs_path)
    # 相对 g1_arm_control 包：../../xr_teleoperate/assets/g1/
    rel_urdf = os.path.join(PKG_DIR, "..", "..", "xr_teleoperate", "assets", "g1", "g1_body29_hand14.urdf")
    rel_urdf = os.path.normpath(os.path.abspath(rel_urdf))
    if os.path.isfile(rel_urdf):
        return rel_urdf, os.path.dirname(rel_urdf)
    return abs_path, os.path.dirname(abs_path)

def _get_default_urdf():
    p, d = _default_urdf_paths()
    return p
def _get_default_mesh_dir():
    p, d = _default_urdf_paths()
    return d


def main():
    parser = argparse.ArgumentParser(description="G1_29 双臂逆解可视化测试")
    parser.add_argument("--urdf", type=str, default=None, help="G1_29 URDF 文件路径（默认自动查找 xr_teleoperate）")
    parser.add_argument("--mesh-dir", type=str, default=None, help="mesh 所在目录（默认与 URDF 同目录）")
    parser.add_argument("--no-pause", action="store_true", help="每个用例后不暂停，连续执行")
    args = parser.parse_args()

    urdf_path = os.path.abspath(args.urdf) if args.urdf else _get_default_urdf()
    model_dir = os.path.abspath(args.mesh_dir) if args.mesh_dir else _get_default_mesh_dir()
    if not os.path.isfile(urdf_path):
        print(f"错误: URDF 不存在: {urdf_path}")
        sys.exit(1)
    if not os.path.isdir(model_dir):
        print(f"错误: mesh 目录不存在: {model_dir}")
        sys.exit(1)

    import numpy as np
    try:
        npy_major = int(np.__version__.split(".")[0])
    except Exception:
        npy_major = 1
    if npy_major >= 2:
        print("错误: conda-forge 的 pinocchio 与 NumPy 2.x 不兼容，会崩溃。请先降级：")
        print('  pip install "numpy<2"')
        sys.exit(1)
    try:
        import pinocchio as pin
    except (ModuleNotFoundError, AttributeError) as e:
        err = str(e)
        if "pinocchio_pywrap" in err or "ARRAY_API" in err or "NumPy" in err:
            print("错误: pinocchio 与当前环境不兼容（常见原因：用了 ROS 自带的 pinocchio 或 NumPy 2.x）。")
            print("请在当前环境中安装兼容版本：")
            print('  pip install "numpy<2" pinocchio meshcat casadi')
            print("运行本脚本前不要 source /opt/ros/...，或已通过脚本自动跳过 ROS 路径。")
        else:
            print("错误: 无法导入 pinocchio（当前环境未安装或未激活正确环境）。")
            print("新开终端后需先激活装有 pinocchio 的 conda 环境，例如：")
            print("  conda activate g1_ik    # 若之前按说明创建过 g1_ik 并安装了 pinocchio")
            print("  python test_g1_29_arm_ik_visual.py")
            print("若从未创建 g1_ik，请先：conda create -n g1_ik python=3.10 -y && conda activate g1_ik")
            print("  conda install -c conda-forge pinocchio -y && pip install \"numpy<2\" meshcat casadi")
        sys.exit(1)

    from g1_29_arm_ik import G1_29_ArmIK

    print("=" * 60)
    print("G1_29 双臂逆解可视化测试")
    print("URDF:", urdf_path)
    print("可视化说明: world=基坐标系(原点) | L_ee_target/R_ee_target=目标末端 | 机械臂上 L_ee/R_ee=实际末端")
    print("=" * 60)

    try:
        arm_ik = G1_29_ArmIK(
            Unit_Test=True,
            Visualization=True,
            urdf_path=urdf_path,
            model_dir=model_dir,
        )
    except (ImportError, OSError) as e:
        err = str(e)
        if "CXXABI" in err or "libstdc++" in err or "libicu" in err or "sqlite3" in err or "libicui18n" in err:
            print("错误: Conda/系统 libstdc++ 版本冲突（meshcat/IPython 链依赖的库与当前运行库不匹配）。")
            print("请先设置使用当前环境的 lib，再运行本脚本：")
            print('  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # Conda 环境')
            print("  python test_g1_29_arm_ik_visual.py")
            print("或：conda activate ik 后执行上述 export 再运行。")
        else:
            raise
        sys.exit(1)
    nq = arm_ik.reduced_robot.model.nq
    # G1_29 缩模 14 关节顺序：左臂 0~6，右臂 7~13（与 URDF 中未锁关节顺序一致）
    ARM_JOINT_NAMES = [
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
        "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
        "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
    ]
    print(f"双臂关节数: {nq}")
    print("关节顺序: 左臂 [0:7] → 右臂 [7:14]\n")

    # RPY(roll,pitch,yaw) -> 3x3 旋转矩阵，固定轴 x-y-z
    def _rpy_to_matrix(rpy):
        r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ], dtype=np.float64)

    # 3x3 旋转矩阵 -> RPY(roll,pitch,yaw) rad，便于打印姿态
    def _matrix_to_rpy(R):
        R = np.asarray(R, dtype=np.float64)
        sy = -R[2, 0]
        cy = np.sqrt(1 - min(1.0, sy * sy))
        if cy > 1e-8:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], cy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], cy)
            yaw = 0.0
        return np.array([roll, pitch, yaw])

    def pose(x, y, z, rpy=(0.0, 0.0, 0.0)):
        R = _rpy_to_matrix(np.array(rpy))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    # 测试用例：每个为 (名称, 左末端位姿, 右末端位姿)；位姿 = 位置(x,y,z)m + 姿态RPY(roll,pitch,yaw)rad
    test_cases = [
        ("对称前伸-零姿态", pose(0.3, 0.1, 0.3, rpy=(0.0, 0.0, 0.0)), pose(0.19984, -0.14869, 0.095317, rpy=(0.0, 0.0, 0.0))),
        ("对称略高-带俯仰", pose(0.25, 0.22, 0.25, rpy=(0.0, 0.2, 0.0)), pose(-0.25, -0.22, 0.25, rpy=(0.0, -0.2, 0.0))),
        ("两侧张开-带偏航", pose(0.30, 0.30, 0.15, rpy=(0.0, 0.0, 0.3)), pose(0.15, -0.30, 0.15, rpy=(0.0, 0.0, -0.3))),
        ("近身低位-带横滚", pose(0.15, 0.20, 0.05, rpy=(0.15, 0.0, 0.0)), pose(0.1, -0.20, 0.05, rpy=(-0.15, 0.0, 0.0))),
        ("前伸高位-复合姿态", pose(0.20, 0.18, 0.35, rpy=(0.1, 0.1, 0.0)), pose(0.20, -0.4, 0.35, rpy=(0.1, -0.1, 0.0))),
    ]

    model = arm_ik.reduced_robot.model
    data = arm_ik.reduced_robot.data
    nq = model.nq
    lid, rid = arm_ik.L_hand_id, arm_ik.R_hand_id

    def get_frame_placement_4x4(frame_id):
        """FK 后取 frame 的 4x4 位姿，兼容不同 pinocchio 版本。"""
        try:
            if hasattr(pin, "getFramePlacement"):
                M = pin.getFramePlacement(model, data, int(frame_id))
            else:
                nf = len(data.oMf) if hasattr(data.oMf, "__len__") else getattr(model, "nframes", 0)
                if frame_id < 0 or frame_id >= nf:
                    return np.eye(4)
                M = data.oMf[int(frame_id)]
            if hasattr(M, "homogeneous"):
                return np.array(M.homogeneous, dtype=np.float64)
            if hasattr(M, "toHomogeneousMatrix"):
                return np.array(M.toHomogeneousMatrix(), dtype=np.float64)
            R, p = np.array(M.rotation), np.array(M.translation)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, p
            return T
        except (IndexError, TypeError, AttributeError):
            return np.eye(4)

    results = []
    for name, L_target, R_target in test_cases:
        print(f"--- {name} ---")
        q, tau = arm_ik.solve_ik(L_target, R_target)
        q = np.asarray(q, dtype=np.float64).ravel()[:nq].copy()
        if hasattr(pin, "framesForwardKinematics"):
            pin.framesForwardKinematics(model, data, q)
        else:
            pin.forwardKinematics(model, data, q)
            if hasattr(pin, "updateFramePlacements"):
                pin.updateFramePlacements(model, data)
        L_actual = get_frame_placement_4x4(lid)
        R_actual = get_frame_placement_4x4(rid)
        err_L = np.linalg.norm(L_actual[:3, 3] - L_target[:3, 3])
        err_R = np.linalg.norm(R_actual[:3, 3] - R_target[:3, 3])
        ok = err_L < 0.01 and err_R < 0.01
        results.append((name, ok, err_L, err_R))
        print(f"  末端位置误差: 左 {err_L:.4f} m, 右 {err_R:.4f} m  {'[PASS]' if ok else '[FAIL]'}")
        # 目标位姿：位置 (m) + 姿态 RPY (rad)
        L_rpy_t = _matrix_to_rpy(L_target[:3, :3])
        R_rpy_t = _matrix_to_rpy(R_target[:3, :3])
        L_rpy_a = _matrix_to_rpy(L_actual[:3, :3])
        R_rpy_a = _matrix_to_rpy(R_actual[:3, :3])
        print(f"  左臂目标位姿: 位置 [x,y,z] = {np.round(L_target[:3, 3], 4).tolist()} m, 姿态 RPY(rad) = {np.round(L_rpy_t, 4).tolist()}")
        print(f"  左臂实际位姿: 位置 [x,y,z] = {np.round(L_actual[:3, 3], 4).tolist()} m, 姿态 RPY(rad) = {np.round(L_rpy_a, 4).tolist()}")
        print(f"  右臂目标位姿: 位置 [x,y,z] = {np.round(R_target[:3, 3], 4).tolist()} m, 姿态 RPY(rad) = {np.round(R_rpy_t, 4).tolist()}")
        print(f"  右臂实际位姿: 位置 [x,y,z] = {np.round(R_actual[:3, 3], 4).tolist()} m, 姿态 RPY(rad) = {np.round(R_rpy_a, 4).tolist()}")
        qr = np.round(q, 4)
        print(f"  左臂关节角(rad) [0:7]: {qr[:7].tolist()}")
        print(f"  右臂关节角(rad) [7:14]: {qr[7:14].tolist()}")
        if not args.no_pause:
            input("  按 Enter 继续下一用例（可在 Meshcat 中查看当前位姿）...")

    print("\n" + "=" * 60)
    for name, ok, eL, eR in results:
        print(f"  {name}: 左 err={eL:.4f} 右 err={eR:.4f}  {'PASS' if ok else 'FAIL'}")
    print("=" * 60)
    print("测试结束。关闭 Meshcat 窗口即可退出。")


if __name__ == "__main__":
    main()
