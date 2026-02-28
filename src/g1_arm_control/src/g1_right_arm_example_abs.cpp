#include <yaml-cpp/yaml.h>

#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <sensor_msgs/msg/joint_state.hpp>
#include <shared_mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "motor_crc_hg.h"
// ROS2
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <unitree_hg/msg/low_cmd.hpp>
#include <unitree_hg/msg/low_state.hpp>

#include "g1/g1_motion_switch_client.hpp"
using namespace std::chrono_literals;

const int G1_NUM_MOTOR = 29;

template <typename T>
class DataBuffer {
 public:
  void SetData(const T& new_data) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    data_ = std::make_shared<T>(new_data);
  }

  std::shared_ptr<const T> GetData() {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return data_ ? data_ : nullptr;
  }

  void Clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    data_ = nullptr;
  }

 private:
  std::shared_ptr<T> data_;
  std::shared_mutex mutex_;
};

struct ImuState {
  std::array<float, 3> rpy = {};
  std::array<float, 3> omega = {};
};

struct MotorCommand {
  std::array<float, G1_NUM_MOTOR> q_target = {};
  std::array<float, G1_NUM_MOTOR> dq_target = {};
  std::array<float, G1_NUM_MOTOR> kp = {};
  std::array<float, G1_NUM_MOTOR> kd = {};
  std::array<float, G1_NUM_MOTOR> tau_ff = {};
};

struct MotorState {
  std::array<float, G1_NUM_MOTOR> q = {};
  std::array<float, G1_NUM_MOTOR> dq = {};
};

enum MotorType { GEARBOX_S = 0, GEARBOX_M = 1, GEARBOX_L = 2 };

const std::array<MotorType, G1_NUM_MOTOR> G1MotorType{
    // clang-format off
    // legs
    GEARBOX_M, GEARBOX_M, GEARBOX_M, GEARBOX_L, GEARBOX_S, GEARBOX_S,
    GEARBOX_M, GEARBOX_M, GEARBOX_M, GEARBOX_L, GEARBOX_S, GEARBOX_S,
    // waist
    GEARBOX_M, GEARBOX_S, GEARBOX_S,
    // arms
    GEARBOX_S, GEARBOX_S, GEARBOX_S, GEARBOX_S, GEARBOX_S, GEARBOX_S, GEARBOX_S,
    GEARBOX_S, GEARBOX_S, GEARBOX_S, GEARBOX_S, GEARBOX_S, GEARBOX_S, GEARBOX_S
    // clang-format on
};

enum PRorAB { PR = 0, AB = 1 };

enum G1JointValidIndex {
  LEFT_SHOULDER_PITCH = 15,
  LEFT_SHOULDER_ROLL = 16,
  LEFT_SHOULDER_YAW = 17,
  LEFT_ELBOW = 18,
  LEFT_WRIST_ROLL = 19,
  LEFT_WRIST_PITCH = 20,
  LEFT_WRIST_YAW = 21,
  RIGHT_SHOULDER_PITCH = 22,
  RIGHT_SHOULDER_ROLL = 23,
  RIGHT_SHOULDER_YAW = 24,
  RIGHT_ELBOW = 25,
  RIGHT_WRIST_ROLL = 26,
  RIGHT_WRIST_PITCH = 27,
  RIGHT_WRIST_YAW = 28
};

// ---------- 右臂末端运动学 (简化 FK/IK，仅位置控制) ----------
// G1 右臂 7 关节链：肩 pitch/roll/yaw、肘、腕 roll/pitch/yaw
// 连杆长度 (米)，可根据实际 URDF 校准
constexpr std::array<float, 7> G1_RIGHT_ARM_LINK_LENGTHS = {
    0.08f, 0.25f, 0.25f, 0.15f, 0.08f, 0.08f, 0.05f};

struct Vec3 {
  float x{0.f}, y{0.f}, z{0.f};
};
struct Mat3 {
  float m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
};

static Mat3 RotX(float a) {
  float c = std::cos(a), s = std::sin(a);
  return {1, 0, 0, 0, c, -s, 0, s, c};
}
static Mat3 RotY(float a) {
  float c = std::cos(a), s = std::sin(a);
  return {c, 0, s, 0, 1, 0, -s, 0, c};
}
static Mat3 RotZ(float a) {
  float c = std::cos(a), s = std::sin(a);
  return {c, -s, 0, s, c, 0, 0, 0, 1};
}
static Mat3 Mat3Mul(const Mat3& A, const Mat3& B) {
  Mat3 C;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      C.m[i * 3 + j] =
          A.m[i * 3 + 0] * B.m[0 * 3 + j] + A.m[i * 3 + 1] * B.m[1 * 3 + j] +
          A.m[i * 3 + 2] * B.m[2 * 3 + j];
  return C;
}
static Vec3 Mat3MulVec(const Mat3& A, float x, float y, float z) {
  return {A.m[0] * x + A.m[1] * y + A.m[2] * z,
          A.m[3] * x + A.m[4] * y + A.m[5] * z,
          A.m[6] * x + A.m[7] * y + A.m[8] * z};
}
static Vec3 Vec3Add(const Vec3& a, const Vec3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

// 右臂正运动学：7 个关节角 (rad) -> 末端位置 (相对于肩部坐标系)
static Vec3 RightArmForwardKinematics(const std::array<float, 7>& q) {
  // 关节轴顺序与 G1 一致：shoulder_pitch(Y), shoulder_roll(Z), shoulder_yaw(X),
  // elbow(Y), wrist_roll(Z), wrist_pitch(Y), wrist_yaw(Z)
  Vec3 p = {0, 0, 0};
  Mat3 R = {};
  R.m[0] = R.m[4] = R.m[8] = 1.f;
  const std::array<float, 7> L = G1_RIGHT_ARM_LINK_LENGTHS;
  const Mat3 R0 = RotY(q[0]);
  const Mat3 R1 = RotZ(q[1]);
  const Mat3 R2 = RotX(q[2]);
  const Mat3 R3 = RotY(q[3]);
  const Mat3 R4 = RotZ(q[4]);
  const Mat3 R5 = RotY(q[5]);
  const Mat3 R6 = RotZ(q[6]);
  p = Vec3Add(p, Mat3MulVec(R0, 0, 0, L[0]));
  R = Mat3Mul(R0, R1);
  p = Vec3Add(p, Mat3MulVec(R, 0, 0, L[1]));
  R = Mat3Mul(R, R2);
  p = Vec3Add(p, Mat3MulVec(R, 0, 0, L[2]));
  R = Mat3Mul(R, R3);
  p = Vec3Add(p, Mat3MulVec(R, 0, 0, L[3]));
  R = Mat3Mul(R, R4);
  p = Vec3Add(p, Mat3MulVec(R, 0, 0, L[4]));
  R = Mat3Mul(R, R5);
  p = Vec3Add(p, Mat3MulVec(R, 0, 0, L[5]));
  R = Mat3Mul(R, R6);
  p = Vec3Add(p, Mat3MulVec(R, 0, 0, L[6]));
  return p;
}

// 位置误差的阻尼最小二乘 IK，迭代若干步得到关节角
static constexpr float IK_DAMPING = 1e-2f;
static constexpr float IK_MAX_DQ = 0.15f;
static constexpr int IK_MAX_ITERS = 50;
static constexpr float IK_POS_TOL = 1e-4f;

static bool RightArmInverseKinematicsPosition(
    const std::array<float, 7>& q_current, const Vec3& target_pos,
    std::array<float, 7>& q_out) {
  q_out = q_current;
  float dq[7];
  const float delta = 1e-5f;
  for (int iter = 0; iter < IK_MAX_ITERS; ++iter) {
    Vec3 pos = RightArmForwardKinematics(q_out);
    float ex = target_pos.x - pos.x, ey = target_pos.y - pos.y,
          ez = target_pos.z - pos.z;
    float err_sq = ex * ex + ey * ey + ez * ez;
    if (err_sq < IK_POS_TOL * IK_POS_TOL) return true;
    // 数值雅可比 J (3x7)
    float J[3 * 7];
    for (int j = 0; j < 7; ++j) {
      std::array<float, 7> qp = q_out;
      qp[j] += delta;
      Vec3 pp = RightArmForwardKinematics(qp);
      J[0 * 7 + j] = (pp.x - pos.x) / delta;
      J[1 * 7 + j] = (pp.y - pos.y) / delta;
      J[2 * 7 + j] = (pp.z - pos.z) / delta;
    }
    // dq = J^T (J J^T + lambda I)^-1 e
    float JJt[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 7; ++j) JJt[i * 3 + k] += J[i * 7 + j] * J[k * 7 + j];
    JJt[0] += IK_DAMPING;
    JJt[4] += IK_DAMPING;
    JJt[8] += IK_DAMPING;
    // 解 (J J^T + lambda I) y = e => y = [3x1]
    float det = JJt[0] * (JJt[4] * JJt[8] - JJt[5] * JJt[7]) -
                JJt[1] * (JJt[3] * JJt[8] - JJt[5] * JJt[6]) +
                JJt[2] * (JJt[3] * JJt[7] - JJt[4] * JJt[6]);
    if (std::fabs(det) < 1e-12f) break;
    float inv[9];
    inv[0] = (JJt[4] * JJt[8] - JJt[5] * JJt[7]) / det;
    inv[1] = (JJt[2] * JJt[7] - JJt[1] * JJt[8]) / det;
    inv[2] = (JJt[1] * JJt[5] - JJt[2] * JJt[4]) / det;
    inv[3] = (JJt[5] * JJt[6] - JJt[3] * JJt[8]) / det;
    inv[4] = (JJt[0] * JJt[8] - JJt[2] * JJt[6]) / det;
    inv[5] = (JJt[3] * JJt[2] - JJt[0] * JJt[5]) / det;
    inv[6] = (JJt[3] * JJt[7] - JJt[4] * JJt[6]) / det;
    inv[7] = (JJt[1] * JJt[6] - JJt[0] * JJt[7]) / det;
    inv[8] = (JJt[0] * JJt[4] - JJt[1] * JJt[3]) / det;
    float y0 = inv[0] * ex + inv[1] * ey + inv[2] * ez;
    float y1 = inv[3] * ex + inv[4] * ey + inv[5] * ez;
    float y2 = inv[6] * ex + inv[7] * ey + inv[8] * ez;
    for (int j = 0; j < 7; ++j)
      dq[j] = J[0 * 7 + j] * y0 + J[1 * 7 + j] * y1 + J[2 * 7 + j] * y2;
    float max_dq = 0;
    for (int j = 0; j < 7; ++j) {
      float ab = std::fabs(dq[j]);
      if (ab > max_dq) max_dq = ab;
    }
    if (max_dq > IK_MAX_DQ)
      for (int j = 0; j < 7; ++j) dq[j] *= IK_MAX_DQ / max_dq;
    for (int j = 0; j < 7; ++j) q_out[j] += dq[j];
  }
  return false;  // 未收敛
}

inline uint32_t Crc32Core(const uint32_t* ptr, uint32_t len) {
  uint32_t xbit = 0;
  uint32_t data = 0;
  uint32_t CRC32 = 0xFFFFFFFF;
  const uint32_t dwPolynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    xbit = 1 << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if ((CRC32 & 0x80000000) != 0U) {
        CRC32 <<= 1;
        CRC32 ^= dwPolynomial;
      } else {
        CRC32 <<= 1;
      }
      if ((data & xbit) != 0U) {
        CRC32 ^= dwPolynomial;
      }

      xbit >>= 1;
    }
  }
  return CRC32;
}

float GetMotorKp(MotorType type) {
  switch (type) {
    case GEARBOX_S:
    case GEARBOX_M:
      return 40;
    case GEARBOX_L:
      return 100;
    default:
      return 0;
  }
}

float GetMotorKd(MotorType type) {
  switch (type) {
    case GEARBOX_S:
    case GEARBOX_M:
    case GEARBOX_L:
      return 1;
    default:
      return 0;
  }
}

class G1Example : public rclcpp::Node {
 private:
  double time_{0.0};
  double control_dt_{0.002};  // [2ms]
  double duration_{3.0};      // [3 s]
  PRorAB mode_{PR};
  uint8_t mode_machine_{0};
  std::vector<std::vector<double>> frames_data_;

  DataBuffer<MotorState> motor_state_buffer_;
  DataBuffer<MotorCommand> motor_command_buffer_;
  DataBuffer<ImuState> imu_state_buffer_;

  // 1. 增加互斥锁定义 (解决 arm_cmd_mutex_ 报错)
  std::mutex arm_cmd_mutex_;
  // 2. 确保 has_received_arm_cmd_ 在这里定义
  bool has_received_arm_cmd_ = false;

  // 右臂目标位置存储 (7个关节)
  std::array<float, 7> right_arm_target_q_ = {0.0f, 0.0f, 0.0f, 0.0f,
                                              0.0f, 0.0f, 0.0f};
  // 末端运动：目标位姿（仅使用位置 x,y,z）
  bool use_end_effector_target_ = false;
  Vec3 end_effector_target_pos_ = {0.f, 0.f, 0.f};
  std::mutex end_effector_mutex_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr arm_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      end_effector_pose_sub_;

  rclcpp::Publisher<unitree_hg::msg::LowCmd>::SharedPtr lowcmd_publisher_;
  rclcpp::Subscription<unitree_hg::msg::LowState>::SharedPtr
      lowstate_subscriber_;
  rclcpp::TimerBase::SharedPtr command_writer_timer_;
  rclcpp::TimerBase::SharedPtr control_timer_;
  std::string resource_dir_;
  std::shared_ptr<unitree::robot::g1::MotionSwitchClient> client_;
  std::thread thread_;

 public:
  explicit G1Example(std::string resource_dir)
      : Node("g1_example"), resource_dir_(std::move(resource_dir)) {
    client_ = std::make_shared<unitree::robot::g1::MotionSwitchClient>(this);

    thread_ = std::thread([this]() {
      std::this_thread::sleep_for(1s);
      while (queryMotionStatus() != 0) {
        std::cout << "Try to deactivate the motion control-related service."
                  << std::endl;
        int32_t ret = client_->ReleaseMode();
        if (ret == 0) {
          std::cout << "ReleaseMode succeeded." << std::endl;
        } else {
          std::cout << "ReleaseMode failed. Error code: " << ret << std::endl;
        }
        std::this_thread::sleep_for(2s);
      }

      // Initialize publishers and subscribers
      lowcmd_publisher_ =
          this->create_publisher<unitree_hg::msg::LowCmd>("lowcmd", 10);

      // 订阅者初始化 (注意：这段代码必须在类成员函数内部)
      arm_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
          "/right_arm/joint_commands", 10,
          [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
            if (msg->position.size() >= 7) {
              std::unique_lock<std::mutex> lock(arm_cmd_mutex_);
              for (int i = 0; i < 7; ++i) {
                right_arm_target_q_[i] = static_cast<float>(msg->position[i]);
              }
              has_received_arm_cmd_ = true;
              {
                std::unique_lock<std::mutex> lock_ee(end_effector_mutex_);
                use_end_effector_target_ = false;  // 关节指令优先，退出末端模式
              }
            }
          });

      end_effector_pose_sub_ =
          this->create_subscription<geometry_msgs::msg::PoseStamped>(
              "/right_arm/end_effector_pose", 10,
              [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                std::unique_lock<std::mutex> lock(end_effector_mutex_);
                end_effector_target_pos_.x =
                    static_cast<float>(msg->pose.position.x);
                end_effector_target_pos_.y =
                    static_cast<float>(msg->pose.position.y);
                end_effector_target_pos_.z =
                    static_cast<float>(msg->pose.position.z);
                use_end_effector_target_ = true;
              });

      lowstate_subscriber_ =
          this->create_subscription<unitree_hg::msg::LowState>(
              "lowstate", 10,
              [this](const unitree_hg::msg::LowState::SharedPtr msg) {
                LowStateHandler(msg);
              });

      // Initialize timers
      command_writer_timer_ =
          this->create_wall_timer(2ms, [this] { LowCommandWriter(); });
      control_timer_ = this->create_wall_timer(2ms, [this] { Control(); });

      RCLCPP_INFO(this->get_logger(), "G1 Example Node Initialized");
    });
  }

  int queryMotionStatus() {
    std::string robotForm;
    std::string motionName;
    int motionStatus = 0;
    int32_t ret = client_->CheckMode(robotForm, motionName);
    if (ret == 0) {
      std::cout << "CheckMode succeeded." << std::endl;
    } else {
      std::cout << "CheckMode failed. Error code: " << ret << std::endl;
    }
    if (motionName.empty()) {
      std::cout << "The motion control-related service is deactivated."
                << std::endl;
      motionStatus = 0;
    } else {
      std::string serviceName = queryServiceName(robotForm, motionName);
      std::cout << "Service: " << serviceName << " is activate" << std::endl;
      motionStatus = 1;
    }
    return motionStatus;
  }

  static std::string queryServiceName(const std::string& form,
                                      const std::string& name) {
    if (form == "0") {
      if (name == "normal") {
        return "sport_mode";
      }
      if (name == "ai") {
        return "ai_sport";
      }
      if (name == "advanced") {
        return "advanced_sport";
      }
    } else {
      if (name == "ai-w") {
        return "wheeled_sport(go2W)";
      }
      if (name == "normal-w") {
        return "wheeled_sport(b2W)";
      }
    }
    return "";
  }

  void loadBehaviorLibrary(const std::string& behavior_name) {
    YAML::Node motion = YAML::LoadFile(resource_dir_ + behavior_name + ".seq");

    auto content = motion["components"][1]["content"].as<std::string>();
    int num_parts = motion["components"][1]["num_parts"].as<int>();
    RCLCPP_INFO(this->get_logger(), "BehaviorName: %s.seq",
                behavior_name.c_str());
    RCLCPP_INFO(this->get_logger(), "%s with %d", content.c_str(), num_parts);

    auto frames = motion["components"][1]["frames"];

    for (const auto& frame : frames) {
      std::vector<double> frame_data;
      for (const auto& element : frame) {
        frame_data.push_back(element.as<double>());
      }
      frames_data_.push_back(frame_data);
    }

    RCLCPP_INFO(this->get_logger(), "%zu knots with %zu DOF",
                frames_data_.size(), frames_data_[0].size());
  }

  void ReportRPY() {
    const std::shared_ptr<const ImuState> imu_tmp_ptr =
        imu_state_buffer_.GetData();
    if (imu_tmp_ptr) {
      RCLCPP_INFO(this->get_logger(), "rpy: [%f, %f, %f]",
                  imu_tmp_ptr->rpy.at(0), imu_tmp_ptr->rpy.at(1),
                  imu_tmp_ptr->rpy.at(2));
    }
  }

  void LowStateHandler(const unitree_hg::msg::LowState::SharedPtr msg) {
    // if (msg->crc != Crc32Core((uint32_t *)msg.get(),  // NOLINT
    //                           (sizeof(unitree_hg::msg::LowState) >> 2) - 1))
    //                           {
    //   RCLCPP_ERROR(this->get_logger(), "low_state CRC Error");
    //   return;
    // }

    // get motor state
    MotorState ms_tmp;
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      ms_tmp.q.at(i) = msg->motor_state[i].q;
      ms_tmp.dq.at(i) = msg->motor_state[i].dq;
    }
    motor_state_buffer_.SetData(ms_tmp);

    // get imu state
    ImuState imu_tmp;
    imu_tmp.omega = {msg->imu_state.gyroscope[0], msg->imu_state.gyroscope[1],
                     msg->imu_state.gyroscope[2]};
    imu_tmp.rpy = {msg->imu_state.rpy[0], msg->imu_state.rpy[1],
                   msg->imu_state.rpy[2]};
    imu_state_buffer_.SetData(imu_tmp);

    // update mode machine
    if (mode_machine_ != msg->mode_machine) {
      if (mode_machine_ == 0) {
        RCLCPP_INFO(this->get_logger(), "G1 type: %u", msg->mode_machine);
      }
      mode_machine_ = msg->mode_machine;
    }
  }

  void LowCommandWriter() {
    auto dds_low_command = unitree_hg::msg::LowCmd();
    dds_low_command.mode_pr = mode_;
    dds_low_command.mode_machine = mode_machine_;

    const std::shared_ptr<const MotorCommand> mc =
        motor_command_buffer_.GetData();
    if (mc) {
      for (size_t i = 0; i < G1_NUM_MOTOR; i++) {
        dds_low_command.motor_cmd[i].mode = 1;  // 1:Enable, 0:Disable
        dds_low_command.motor_cmd[i].tau = mc->tau_ff.at(i);
        dds_low_command.motor_cmd[i].q = mc->q_target.at(i);
        dds_low_command.motor_cmd[i].dq = mc->dq_target.at(i);
        dds_low_command.motor_cmd[i].kp = mc->kp.at(i);
        dds_low_command.motor_cmd[i].kd = mc->kd.at(i);
      }

      get_crc(dds_low_command);
      lowcmd_publisher_->publish(dds_low_command);
    }
  }

  

  void Control() {
    MotorCommand motor_command_tmp;
    const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();

    if (ms) {
      time_ += control_dt_;

      // 如果还没收到指令，先用当前实际位置“定住”手臂，防止突跳
      if (!has_received_arm_cmd_) {
        std::unique_lock<std::mutex> lock(arm_cmd_mutex_);
        for (int i = 0; i < 7; ++i) {
          right_arm_target_q_[i] = ms->q.at(RIGHT_SHOULDER_PITCH + i);
        }
        has_received_arm_cmd_ = true;  // 初始化完成
      }

      // // 全机默认设置
      // for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      //   motor_command_tmp.q_target.at(i) = 0.0f;
      //   // 假设其他部位如腿部需要 0 位维持站姿
      //   motor_command_tmp.kp.at(i) = GetMotorKp(G1MotorType[i]);
      //   motor_command_tmp.kd.at(i) = GetMotorKd(G1MotorType[i]);
      // }
      
// 全机默认设置 (非右臂关节设为软态放松，或者跟随当前位置)
      for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        motor_command_tmp.q_target.at(i) = ms->q.at(i); // 目标位置等于当前实际位置，防止突跳
        motor_command_tmp.dq_target.at(i) = 0.0f;
        motor_command_tmp.tau_ff.at(i) = 0.0f;
        
        // 安全起见，非控制关节的刚度和阻尼设为 0 (用手可以掰动)
        motor_command_tmp.kp.at(i) = 0.0f; 
        motor_command_tmp.kd.at(i) = 0.0f;
      }



      // 末端运动：若收到末端目标位姿，用 IK 解算关节角并更新目标
      {
        std::unique_lock<std::mutex> lock_ee(end_effector_mutex_);
        if (use_end_effector_target_) {
          std::array<float, 7> q_curr;
          for (int i = 0; i < 7; ++i)
            q_curr[i] = ms->q.at(RIGHT_SHOULDER_PITCH + i);
          std::array<float, 7> q_ik;
          if (RightArmInverseKinematicsPosition(
                  q_curr, end_effector_target_pos_, q_ik)) {
            std::unique_lock<std::mutex> lock_arm(arm_cmd_mutex_);
            for (int i = 0; i < 7; ++i) right_arm_target_q_[i] = q_ik[i];
          }
        }
      }

      // 右臂赋值
      {
        std::unique_lock<std::mutex> lock(arm_cmd_mutex_);
        for (int i = 0; i < 7; ++i) {
          int joint_idx = RIGHT_SHOULDER_PITCH + i;
          motor_command_tmp.q_target.at(joint_idx) = right_arm_target_q_[i];
          motor_command_tmp.kp.at(joint_idx) =
              GetMotorKp(G1MotorType[joint_idx]);
          motor_command_tmp.kd.at(joint_idx) =
              GetMotorKd(G1MotorType[joint_idx]);
        }
      }
      motor_command_buffer_.SetData(motor_command_tmp);
    }
  }
};

int main(int argc, char const* argv[]) {
  rclcpp::init(argc, argv);

  if (argc < 2) {
    RCLCPP_FATAL(rclcpp::get_logger("main"),
                 "Usage: %s <resource_directory> [behavior_name], for example: "
                 "g1_dual_arm_example ./behavior_lib/ motion",
                 argv[0]);
    return 1;
  }

  std::string resource_dir = argv[1];
  auto node = std::make_shared<G1Example>(resource_dir);

  // Optional: Load behavior if specified
  if (argc > 2) {
    std::string behavior_name = argv[2];
    node->loadBehaviorLibrary(behavior_name);
  }

  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
