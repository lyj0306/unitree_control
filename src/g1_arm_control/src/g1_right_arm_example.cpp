#include <yaml-cpp/yaml.h>

#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "motor_crc_hg.h"
// ROS2
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <unitree_hg/msg/low_cmd.hpp>
#include <unitree_hg/msg/low_state.hpp>

#include "g1/g1_motion_switch_client.hpp"
using namespace std::chrono_literals;

const int G1_NUM_MOTOR = 29;

template <typename T>
class DataBuffer {
 public:
  void SetData(const T &new_data) {
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

// 右臂关节映射表，用于将输入的JointState顺序对应到电机ID
// 顺序: Pitch, Roll, Yaw, Elbow, Wrist Roll, Wrist Pitch, Wrist Yaw
const std::vector<int> RIGHT_ARM_MOTOR_IDS = {
    RIGHT_SHOULDER_PITCH,
    RIGHT_SHOULDER_ROLL,
    RIGHT_SHOULDER_YAW,
    RIGHT_ELBOW,
    RIGHT_WRIST_ROLL,
    RIGHT_WRIST_PITCH,
    RIGHT_WRIST_YAW
};

inline uint32_t Crc32Core(const uint32_t *ptr, uint32_t len) {
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

class G1RightArmControl : public rclcpp::Node {
 private:
  PRorAB mode_{PR};
  uint8_t mode_machine_{0};

  DataBuffer<MotorState> motor_state_buffer_;
  DataBuffer<MotorCommand> motor_command_buffer_;
  DataBuffer<ImuState> imu_state_buffer_;

  rclcpp::Publisher<unitree_hg::msg::LowCmd>::SharedPtr lowcmd_publisher_;
  rclcpp::Subscription<unitree_hg::msg::LowState>::SharedPtr lowstate_subscriber_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr right_arm_cmd_subscriber_;
  rclcpp::TimerBase::SharedPtr command_writer_timer_;
  
  std::shared_ptr<unitree::robot::g1::MotionSwitchClient> client_;
  std::thread thread_;

  bool initialized_{false}; // 新增：标记是否已发送过初始命令

 public:
  explicit G1RightArmControl()
      : Node("g1_right_arm_control") {
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
      lowstate_subscriber_ =
          this->create_subscription<unitree_hg::msg::LowState>(
              "lowstate", 10,
              [this](const unitree_hg::msg::LowState::SharedPtr msg) {
                LowStateHandler(msg);
              });
      
      // 新增：订阅右臂控制指令
      right_arm_cmd_subscriber_ = 
          this->create_subscription<sensor_msgs::msg::JointState>(
              "/right_arm/joint_commands", 10,
              [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
                RightArmCmdCallback(msg);
              });

      // Initialize timers
      command_writer_timer_ =
          this->create_wall_timer(2ms, [this] { LowCommandWriter(); });

      RCLCPP_INFO(this->get_logger(), "G1 Right Arm Control Node Initialized");
      RCLCPP_INFO(this->get_logger(), "Waiting for commands on topic: /right_arm/joint_commands");
    });
  }

  int queryMotionStatus() {
    std::string robotForm;
    std::string motionName;
    int motionStatus = 0;
    int32_t ret = client_->CheckMode(robotForm, motionName);
    if (motionName.empty()) {
      motionStatus = 0;
    } else {
      motionStatus = 1;
    }
    return motionStatus;
  }

  void LowStateHandler(const unitree_hg::msg::LowState::SharedPtr msg) {
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
      mode_machine_ = msg->mode_machine;
    }
    // 【新增】 初始化逻辑：只要收到一次状态，就立刻生成默认指令，防止机器人报错
    if (!initialized_) {
      MotorCommand mc_tmp;
      // 以当前位置为目标，保持不动
      for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        mc_tmp.q_target.at(i) = ms_tmp.q.at(i); // 目标位置 = 当前位置
        mc_tmp.dq_target.at(i) = 0.0;
        mc_tmp.kp.at(i) = GetMotorKp(G1MotorType[i]);
        mc_tmp.kd.at(i) = GetMotorKd(G1MotorType[i]);
        mc_tmp.tau_ff.at(i) = 0.0;
      }
      motor_command_buffer_.SetData(mc_tmp);
      initialized_ = true;
      RCLCPP_INFO(this->get_logger(), "Robot connected. Sending keep-alive commands.");
    }
  }

  // 新增：回调函数处理外部控制指令
  void RightArmCmdCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    // 检查输入长度是否为7 (右臂7个关节)
    if (msg->position.size() != 7) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                             "Invalid command size! Expected 7, got %zu", msg->position.size());
      return;
    }

    const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();
    if (!ms) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Motor state not ready yet.");
      return;
    }

    MotorCommand mc_tmp;

    // 1. 初始化：保持所有关节当前位置（安全策略）
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      mc_tmp.q_target.at(i) = ms->q.at(i); // 保持原位
      mc_tmp.dq_target.at(i) = 0.0;
      mc_tmp.kp.at(i) = GetMotorKp(G1MotorType[i]);
      mc_tmp.kd.at(i) = GetMotorKd(G1MotorType[i]);
      mc_tmp.tau_ff.at(i) = 0.0;
    }

    // 2. 覆盖右臂指令
    for (size_t i = 0; i < 7; ++i) {
      int motor_id = RIGHT_ARM_MOTOR_IDS[i];
      
      // 位置指令
      mc_tmp.q_target.at(motor_id) = msg->position[i];
      
      // 速度指令 (如果提供)
      if (msg->velocity.size() == 7) {
        mc_tmp.dq_target.at(motor_id) = msg->velocity[i];
      }
      
      // 力矩前馈 (如果提供)
      if (msg->effort.size() == 7) {
        mc_tmp.tau_ff.at(motor_id) = msg->effort[i];
      }
    }

    motor_command_buffer_.SetData(mc_tmp);
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
};

int main(int argc, char const *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<G1RightArmControl>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
