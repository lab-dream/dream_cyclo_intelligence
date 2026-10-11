// Copyright 2026 ROBOTIS CO., LTD.
// SPDX-License-Identifier: Apache-2.0
// Python glue only. FK, Jacobians and QP constraints come from the Cyclo submodule.
#include <controllers/ai_worker/ai_worker_movel_controller.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <set>

namespace cc = cyclo_motion_controller;
using JointMap = std::map<std::string, double>;

class Kinematics : public cc::kinematics::KinematicsSolver
{
public:
  explicit Kinematics(const std::string &path) : KinematicsSolver(path, "") {}
  void lock(int index)
  {
    setJointVelocityBoundsByIndex(index, 0, 0);
    q_lb_[index] = -1e6;
    q_ub_[index] = 1e6;
  }
};

class CycloIK
{
  std::shared_ptr<Kinematics> robot_;
  cc::controllers::AIWorkerMoveLController controller_;
  std::vector<std::string> names_, active_, frames_;
  static constexpr double dt_ = 0.01;

public:
  CycloIK(const std::string &urdf, const std::vector<std::string> &active,
    const std::vector<std::string> &frames, double lift_weight)
  : robot_(std::make_shared<Kinematics>(urdf)), controller_(robot_, dt_),
    names_(robot_->getJointNames()), active_(active), frames_(frames)
  {
    if (!std::isfinite(lift_weight) || lift_weight <= 0) {
      throw std::invalid_argument("lift_weight must be positive and finite");
    }
    if (names_.size() != static_cast<size_t>(robot_->getDof()) || frames_.size() != 2) {
      throw std::runtime_error("Cyclo IK requires scalar joints and two EEF frames");
    }
    const std::set<std::string> active_set(active_.begin(), active_.end());
    for (const auto &name : active_) {
      if (std::find(names_.begin(), names_.end(), name) == names_.end()) {
        throw std::runtime_error("Unknown active joint: " + name);
      }
    }
    std::map<std::string, cc::common::Vector6d> weights;
    for (const auto &frame : frames_) {
      if (!robot_->hasLinkFrame(frame)) {throw std::runtime_error("Unknown EEF frame: " + frame);}
      auto weight = cc::common::Vector6d::Ones().eval();
      weight.head<3>().setConstant(10);
      weights[frame] = weight;
    }
    auto damping = Eigen::VectorXd::Constant(robot_->getDof(), 1e-4).eval();
    for (size_t index = 0; index < names_.size(); ++index) {
      if (!active_set.count(names_[index])) {robot_->lock(index);}
      if (names_[index] == "lift_joint") {damping[index] = lift_weight;}
    }
    controller_.setWeight(weights, damping);
    controller_.setControllerParams(1000, 50, 0.05, 0.02);
  }

  std::tuple<JointMap, double, double> solve(const JointMap &seed,
    const Eigen::Matrix4d &left, const Eigen::Matrix4d &right)
  {
    if (!left.allFinite() || !right.allFinite()) {throw std::runtime_error("Non-finite EEF target");}
    const auto limits = robot_->getJointPositionLimit();
    Eigen::VectorXd q = Eigen::VectorXd::Zero(robot_->getDof());
    const std::set<std::string> active_set(active_.begin(), active_.end());
    for (size_t index = 0; index < names_.size(); ++index) {
      auto value = seed.find(names_[index]);
      if (value != seed.end()) {q[index] = value->second;}
      if (active_set.count(names_[index])) {
        if (value == seed.end()) {throw std::runtime_error("Missing seed: " + names_[index]);}
        q[index] = std::clamp(q[index], limits.first[index], limits.second[index]);
      }
    }
    if (!q.allFinite()) {throw std::runtime_error("Non-finite joint seed");}
    Eigen::VectorXd velocity = Eigen::VectorXd::Zero(q.size());
    const Eigen::Matrix4d goals[] = {left, right};
    double position_error = 0, rotation_error = 0;
    for (int iteration = 0; iteration <= 100; ++iteration) {
      if (!robot_->updateState(q, velocity)) {throw std::runtime_error("Cyclo FK failed");}
      std::map<std::string, cc::common::Vector6d> targets;
      position_error = rotation_error = 0;
      for (int side = 0; side < 2; ++side) {
        const auto current = robot_->getPose(frames_[side]);
        cc::common::Vector6d error;
        error.head<3>() = goals[side].topRightCorner<3, 1>() - current.translation();
        const Eigen::AngleAxisd rotation(goals[side].topLeftCorner<3, 3>() * current.linear().transpose());
        error.tail<3>() = rotation.axis() * rotation.angle();
        position_error = std::max(position_error, error.head<3>().norm());
        rotation_error = std::max(rotation_error, error.tail<3>().norm());
        targets[frames_[side]] = 50 * error;
      }
      if ((position_error < 1e-4 && rotation_error < 1e-3) || iteration == 100) {break;}
      controller_.setDesiredTaskVel(targets);
      if (!controller_.getOptJointVel(velocity) || !velocity.allFinite()) {
        throw std::runtime_error("Cyclo MoveL QP solve failed");
      }
      for (size_t index = 0; index < names_.size(); ++index) {
        if (active_set.count(names_[index])) {
          q[index] = std::clamp(q[index] + dt_ * velocity[index], limits.first[index], limits.second[index]);
        }
      }
    }
    JointMap result;
    for (size_t index = 0; index < names_.size(); ++index) {
      if (active_set.count(names_[index])) {result[names_[index]] = q[index];}
    }
    return {result, position_error, rotation_error};
  }
};

PYBIND11_MODULE(_cyclo_ik, module)
{
  pybind11::class_<CycloIK>(module, "CycloIK")
    .def(pybind11::init<const std::string &, const std::vector<std::string> &,
      const std::vector<std::string> &, double>(), pybind11::arg("urdf"),
      pybind11::arg("active"), pybind11::arg("frames"), pybind11::arg("lift_weight") = 10.0)
    .def("solve", &CycloIK::solve, pybind11::call_guard<pybind11::gil_scoped_release>());
}
