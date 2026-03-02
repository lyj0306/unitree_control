# -*- coding: utf-8 -*-
"""
G1 29-DOF 机械臂逆解（仅双臂，不关心腿部）。
仅包含 G1_29，无 H1、G1_23 等机型。
输入：左右末端位姿 4x4 齐次矩阵（世界系）；输出：双臂关节角 q 与力矩 tau。
"""

import os
import pickle
import logging
import sys

# 强制使用当前 Python 环境（conda/venv）的 pinocchio，避免加载 ROS 自带的（无 casadi 子模块）
_ros_keywords = ("/opt/ros", "ros/humble", "ros/noetic")
for _p in list(sys.path):
    if any(_k in _p for _k in _ros_keywords):
        sys.path.remove(_p)
# 仅在「非 ROS」环境下把当前 env 的 site-packages 插到最前（conda/venv 优先）
_base = os.environ.get("CONDA_PREFIX") or (sys.prefix if "/opt/ros" not in sys.prefix else None)
if _base:
    _site = os.path.join(_base, "lib", "python{}.{}".format(*sys.version_info[:2]), "site-packages")
    if os.path.isdir(_site):
        if _site in sys.path:
            sys.path.remove(_site)
        sys.path.insert(0, _site)

import casadi
import numpy as np
import pinocchio as pin
try:
    from pinocchio import casadi as cpin
except ImportError as _e:
    if "casadi" in str(_e):
        raise ImportError(
            "当前加载的 pinocchio 无 casadi 子模块（可能是 ROS 的或 PyPI 上的「假」pinocchio 0.4.3 鼻子插件）。\n"
            "请先卸载错误包并安装机器人库：\n"
            "  pip uninstall pinocchio\n"
            "  conda install -c conda-forge pinocchio\n"
            "运行前不要 source /opt/ros/...，并用 conda 的 python。"
        ) from _e
    raise
# meshcat / MeshcatVisualizer 仅在 Visualization=True 时按需导入，避免拖入 IPython/sqlite 等依赖

logger_mp = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class WeightedMovingFilter:
    def __init__(self, weights, dim):
        self.filtered_data = np.zeros(dim)

    def add_data(self, data):
        self.filtered_data = np.asarray(data).ravel()


def _get_g1_29_urdf_paths():
    """优先使用本包 include 下 URDF，否则退回 xr_teleoperate 路径。"""
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_urdf = os.path.join(pkg_dir, "include", "g1 urdf", "g1_body29_hand14.urdf")
    local_mesh = os.path.join(pkg_dir, "include", "g1 urdf")
    if os.path.isfile(local_urdf):
        return local_urdf, local_mesh
    # 退回到 xr_teleoperate（与 g1_arm_ik 原逻辑一致）
    parent3 = os.path.dirname(pkg_dir)
    xr_urdf = os.path.join(parent3, "xr_teleoperate", "assets", "g1", "g1_body29_hand14.urdf")
    xr_mesh = os.path.join(parent3, "xr_teleoperate", "assets", "g1")
    if os.path.isfile(xr_urdf):
        return xr_urdf, xr_mesh
    # Unit_Test 常用：从 src 算两级到 workspace
    alt_urdf = os.path.join(pkg_dir, "..", "..", "xr_teleoperate", "assets", "g1", "g1_body29_hand14.urdf")
    alt_urdf = os.path.normpath(os.path.abspath(alt_urdf))
    alt_mesh = os.path.dirname(alt_urdf)
    if os.path.isfile(alt_urdf):
        return alt_urdf, alt_mesh
    return local_urdf, local_mesh


class G1_29_ArmIK:
    """
    G1 29-DOF 机械臂逆解。锁住腿、腰、手部手指，仅保留双臂 14 自由度，
    根据左右末端位姿求关节角与重力补偿力矩。
    """

    # 锁住的关节（腿、腰、手指），只保留双臂
    JOINTS_TO_LOCK = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint", "left_hand_middle_1_joint",
        "left_hand_index_0_joint", "left_hand_index_1_joint",
        "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
        "right_hand_index_0_joint", "right_hand_index_1_joint",
        "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    ]

    def __init__(self, Unit_Test=False, Visualization=False, urdf_path=None, model_dir=None):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        self.Unit_Test = Unit_Test
        self.Visualization = Visualization
        self.cache_path = "g1_29_model_cache.pkl"
        if urdf_path is not None and model_dir is not None:
            self.urdf_path = os.path.abspath(urdf_path)
            self.model_dir = os.path.abspath(model_dir)
        else:
            self.urdf_path, self.model_dir = _get_g1_29_urdf_paths()

        use_cache = (urdf_path is None and os.path.exists(self.cache_path) and (not self.Visualization))
        if use_cache:
            logger_mp.info("[G1_29_ArmIK] >>> Loading cached model: %s", self.cache_path)
            self.robot, self.reduced_robot = self._load_cache()
        else:
            logger_mp.info("[G1_29_ArmIK] >>> Loading URDF: %s", self.urdf_path)
            self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.model_dir)
            self.reduced_robot = self.robot.buildReducedRobot(
                list_of_joints_to_lock=self.JOINTS_TO_LOCK,
                reference_configuration=np.zeros(self.robot.model.nq),
            )
            # 末端 L_ee/R_ee：与第 7 关节 left/right_wrist_yaw_joint 坐标系重合，无偏移
            self.reduced_robot.model.addFrame(
                pin.Frame(
                    "L_ee",
                    self.reduced_robot.model.getJointId("left_wrist_yaw_joint"),
                    pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.0])),
                    pin.FrameType.OP_FRAME,
                )
            )
            self.reduced_robot.model.addFrame(
                pin.Frame(
                    "R_ee",
                    self.reduced_robot.model.getJointId("right_wrist_yaw_joint"),
                    pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.0])),
                    pin.FrameType.OP_FRAME,
                )
            )
            if urdf_path is None and not os.path.exists(self.cache_path):
                self._save_cache()
                logger_mp.info("[G1_29_ArmIK] >>> Cache saved: %s", self.cache_path)

        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T),
                )
            ],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        trans_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        rot_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        reg_cost = casadi.sumsqr(self.var_q)
        smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        self.opti.minimize(50 * trans_cost + rot_cost + 0.02 * reg_cost + 0.1 * smooth_cost)

        self.opti.solver(
            "ipopt",
            {
                "expand": True,
                "detect_simple_bounds": True,
                "calc_lam_p": False,
                "print_time": False,
                "ipopt.sb": "yes",
                "ipopt.print_level": 0,
                "ipopt.max_iter": 30,
                "ipopt.tol": 1e-4,
                "ipopt.acceptable_tol": 5e-4,
                "ipopt.acceptable_iter": 5,
                "ipopt.warm_start_init_point": "yes",
                "ipopt.derivative_test": "none",
                "ipopt.jacobian_approximation": "exact",
            },
        )

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None

        if self.Visualization:
            import meshcat.geometry as mg
            from pinocchio.visualize import MeshcatVisualizer
            self.vis = MeshcatVisualizer(
                self.reduced_robot.model,
                self.reduced_robot.collision_model,
                self.reduced_robot.visual_model,
            )
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel("pinocchio")
            self.vis.display(pin.neutral(self.reduced_robot.model))
            # 基坐标系（世界系）显示在原点，轴长 0.2 m
            axis_len = 0.2
            base_pos = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float32).T * axis_len
            base_col = np.array([[0.8, 0.2, 0.2], [1, 0.5, 0.5], [0.2, 0.8, 0.2], [0.5, 1, 0.5], [0.2, 0.2, 0.8], [0.5, 0.5, 1]], dtype=np.float32).T
            self.vis.viewer["world"].set_object(
                mg.LineSegments(mg.PointsGeometry(position=base_pos, color=base_col), mg.LineBasicMaterial(linewidth=15, vertexColors=True))
            )
            self.vis.viewer["world"].set_transform(np.eye(4))
            # 左右末端目标位姿（红色/绿色粗轴，便于区分）
            axis_len_ee = 0.12
            pos_ee = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float32).T * axis_len_ee
            col_ee = np.array([[1, 0, 0], [1, 0.6, 0], [0, 1, 0], [0.6, 1, 0], [0, 0, 1], [0, 0.6, 1]], dtype=np.float32).T
            for name in ["L_ee_target", "R_ee_target"]:
                self.vis.viewer[name].set_object(
                    mg.LineSegments(mg.PointsGeometry(position=pos_ee, color=col_ee), mg.LineBasicMaterial(linewidth=20, vertexColors=True))
                )
            # 实际末端坐标系 L_ee, R_ee（逆解后的末端位姿）
            self.vis.displayFrames(True, frame_ids=[self.L_hand_id, self.R_hand_id], axis_length=0.1, axis_width=6)

    def _save_cache(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump({"robot_model": self.robot.model, "reduced_model": self.reduced_robot.model}, f)

    def _load_cache(self):
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)
        robot = pin.RobotWrapper()
        robot.model = data["robot_model"]
        robot.data = robot.model.createData()
        reduced_robot = pin.RobotWrapper()
        reduced_robot.model = data["reduced_model"]
        reduced_robot.data = reduced_robot.model.createData()
        return robot, reduced_robot

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        """
        左右末端位姿（4x4 齐次矩阵）-> 双臂关节角 q 与力矩 tau。
        left_wrist, right_wrist: (4,4) 世界系下末端位姿。
        current_lr_arm_motor_q: 热启动关节角，可选。
        """
        if current_lr_arm_motor_q is not None:
            self.init_data = np.asarray(current_lr_arm_motor_q).ravel()[: self.reduced_robot.model.nq]
        self.opti.set_initial(self.var_q, self.init_data)
        if self.Visualization:
            self.vis.viewer["L_ee_target"].set_transform(left_wrist)
            self.vis.viewer["R_ee_target"].set_transform(right_wrist)
        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)

        try:
            self.opti.solve()
            sol_q = np.asarray(self.opti.value(self.var_q)).ravel()
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data
            v = np.zeros(self.reduced_robot.model.nv) if current_lr_arm_motor_dq is None else np.asarray(current_lr_arm_motor_dq).ravel() * 0.0
            self.init_data = sol_q
            tau = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))
            if self.Visualization:
                self.vis.display(sol_q)
            return sol_q, tau
        except Exception as e:
            logger_mp.error("IK convergence failed: %s", e)
            sol_q = np.asarray(self.opti.debug.value(self.var_q)).ravel()
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data
            v = np.zeros(self.reduced_robot.model.nv)
            self.init_data = sol_q
            tau = np.zeros(self.reduced_robot.model.nv)
            if current_lr_arm_motor_q is not None:
                return np.asarray(current_lr_arm_motor_q).ravel()[: self.reduced_robot.model.nq], tau
            return sol_q, tau


if __name__ == "__main__":
    import time
    arm_ik = G1_29_ArmIK(Unit_Test=True, Visualization=False)
    L_tf = pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.25, 0.25, 0.1]))
    R_tf = pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.25, -0.25, 0.1]))
    q, tau = arm_ik.solve_ik(L_tf.homogeneous, R_tf.homogeneous)
    print("q:", q)
    print("tau:", tau)
