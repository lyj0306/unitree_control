cmake_minimum_required(VERSION 3.5)
project(g1_arm_control)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


include_directories(include include/common include/nlohmann)
link_directories(src)

set (
  DEPENDENCY_LIST
  unitree_hg
  unitree_api
  rclcpp
  std_msgs
  rosbag2_cpp
  sensor_msgs  
)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(unitree_hg REQUIRED)
find_package(unitree_api REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(sensor_msgs REQUIRED)





add_executable(g1_right_arm_example src/g1_right_arm_example.cpp src/common/motor_crc_hg.cpp)
target_compile_features(g1_right_arm_example PRIVATE cxx_std_20)
ament_target_dependencies(g1_right_arm_example ${DEPENDENCY_LIST})


find_package(yaml-cpp QUIET)



install(TARGETS g1_right_arm_example
	DESTINATION lib/${PROJECT_NAME})




if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
