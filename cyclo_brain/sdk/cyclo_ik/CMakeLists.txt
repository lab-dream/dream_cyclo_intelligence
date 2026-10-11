cmake_minimum_required(VERSION 3.16)
project(cyclo_ik_native LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(pybind11 REQUIRED CONFIG)
find_package(pinocchio REQUIRED CONFIG)
find_package(OsqpEigen REQUIRED CONFIG)
set(CYCLO_CORE ${CMAKE_CURRENT_SOURCE_DIR}/../cyclo_control/cyclo_motion_controller_core)
# Compile the official core directly; its ROS/ament packaging is unnecessary here.
pybind11_add_module(_cyclo_ik NO_EXTRAS bindings.cpp
  ${CYCLO_CORE}/src/kinematics/kinematics_solver.cpp
  ${CYCLO_CORE}/src/controllers/ai_worker/vr_controller.cpp
  ${CYCLO_CORE}/src/controllers/ai_worker/ai_worker_movel_controller.cpp)
target_include_directories(_cyclo_ik PRIVATE ${CYCLO_CORE}/include/cyclo_motion_controller_core)
target_link_libraries(_cyclo_ik PRIVATE pinocchio::pinocchio OsqpEigen::OsqpEigen)
# Python must load the same cmeel Pinocchio libraries used to build this module.
set_target_properties(_cyclo_ik PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS _cyclo_ik LIBRARY DESTINATION .)
