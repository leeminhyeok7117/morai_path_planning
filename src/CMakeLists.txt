cmake_minimum_required(VERSION 3.8)
project(slp_py)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

find_package(rosidl_default_generators REQUIRED)

install(PROGRAMS
  src/cartesian_frenet_conversion.py
  src/cubic_hermit_planner.py
  src/cubic_spline_planner.py
  src/global_path.py
  src/lookup_table_generator_no_optimize.py
  src/state_lattice_planner.py
  src/test.py
  

  DESTINATION lib/${PROJECT_NAME}
)

#install(
#    DIRECTORY path/npy_file/path/
#    DESTINATION share/${PROJECT_NAME}/path/npy_file/path
#)
