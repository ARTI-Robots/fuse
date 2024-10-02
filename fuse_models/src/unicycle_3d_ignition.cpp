/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2024, ARTI - Autonomous Robot Technology GmbH
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */
#include <fuse_models/unicycle_3d_ignition.h>

#include <fuse_constraints/absolute_constraint.h>
#include <fuse_constraints/absolute_orientation_3d_stamped_euler_constraint.h>
#include <fuse_core/async_sensor_model.h>
#include <fuse_core/eigen.h>
#include <fuse_core/sensor_model.h>
#include <fuse_core/transaction.h>
#include <fuse_core/util.h>
#include <fuse_core/uuid.h>
#include <fuse_models/SetPose.h>
#include <fuse_models/SetPoseDeprecated.h>
#include <fuse_variables/acceleration_linear_3d_stamped.h>
#include <fuse_variables/orientation_3d_stamped.h>
#include <fuse_variables/position_3d_stamped.h>
#include <fuse_variables/velocity_angular_3d_stamped.h>
#include <fuse_variables/velocity_linear_3d_stamped.h>
#include <fuse_variables/stamped.h>

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <pluginlib/class_list_macros.hpp>
#include <std_srvs/Empty.h>
#include <tf2/convert.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <Eigen/Dense>

#include <exception>
#include <stdexcept>
#include <vector>


// Register this motion model with ROS as a plugin.
PLUGINLIB_EXPORT_CLASS(fuse_models::Unicycle3DIgnition, fuse_core::SensorModel);

namespace fuse_models
{

Unicycle3DIgnition::Unicycle3DIgnition() :
  fuse_core::AsyncSensorModel(1),
  started_(false),
  initial_transaction_sent_(false),
  device_id_(fuse_core::uuid::NIL)
{
}

void Unicycle3DIgnition::onInit()
{
  // Read settings from the parameter sever
  device_id_ = fuse_variables::loadDeviceId(private_node_handle_);

  params_.loadFromROS(private_node_handle_);

  // Connect to the reset service
  if (!params_.reset_service.empty())
  {
    reset_client_ = node_handle_.serviceClient<std_srvs::Empty>(ros::names::resolve(params_.reset_service));
  }

  // Advertise
  subscriber_ = node_handle_.subscribe(
    ros::names::resolve(params_.topic),
    params_.queue_size,
    &Unicycle3DIgnition::subscriberCallback,
    this);
  set_pose_service_ = node_handle_.advertiseService(
    ros::names::resolve(params_.set_pose_service),
    &Unicycle3DIgnition::setPoseServiceCallback,
    this);
  set_pose_deprecated_service_ = node_handle_.advertiseService(
    ros::names::resolve(params_.set_pose_deprecated_service),
    &Unicycle3DIgnition::setPoseDeprecatedServiceCallback,
    this);
}

void Unicycle3DIgnition::start()
{
  started_ = true;

  // TODO(swilliams) Should this be executed every time optimizer.reset() is called, or only once ever?
  //                 I feel like it should be "only once ever".
  // Send an initial state transaction immediately, if requested
  if (params_.publish_on_startup && !initial_transaction_sent_)
  {
    auto pose = geometry_msgs::PoseWithCovarianceStamped();
    pose.header.stamp = ros::Time::now();
    pose.pose.pose.position.x = params_.initial_state[0];
    pose.pose.pose.position.y = params_.initial_state[1];
    pose.pose.pose.position.z = params_.initial_state[2];
    tf2::Quaternion q_orientation;
    q_orientation.setEuler(params_.initial_state[3], params_.initial_state[4], params_.initial_state[5]);
    pose.pose.pose.orientation = tf2::toMsg(q_orientation);
    pose.pose.covariance[0] = params_.initial_sigma[0] * params_.initial_sigma[0];
    pose.pose.covariance[6 * 1 + 1] = params_.initial_sigma[1] * params_.initial_sigma[1];
    pose.pose.covariance[6 * 2 + 2] = params_.initial_sigma[2] * params_.initial_sigma[2];
    pose.pose.covariance[6 * 3 + 3] = params_.initial_sigma[3] * params_.initial_sigma[3];
    pose.pose.covariance[6 * 4 + 4] = params_.initial_sigma[4] * params_.initial_sigma[4];
    pose.pose.covariance[6 * 5 + 5] = params_.initial_sigma[5] * params_.initial_sigma[5];
    sendPrior(pose);
    initial_transaction_sent_ = true;
  }
}

void Unicycle3DIgnition::stop()
{
  started_ = false;
}

void Unicycle3DIgnition::subscriberCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
{
  try
  {
    process(*msg);
  }
  catch (const std::exception& e)
  {
    ROS_ERROR_STREAM(e.what() << " Ignoring message.");
  }
}

bool Unicycle3DIgnition::setPoseServiceCallback(fuse_models::SetPose::Request& req, fuse_models::SetPose::Response& res)
{
  try
  {
    process(req.pose);
    res.success = true;
  }
  catch (const std::exception& e)
  {
    res.success = false;
    res.message = e.what();
    ROS_ERROR_STREAM(e.what() << " Ignoring request.");
  }
  return true;
}

bool Unicycle3DIgnition::setPoseDeprecatedServiceCallback(
  fuse_models::SetPoseDeprecated::Request& req,
  fuse_models::SetPoseDeprecated::Response&)
{
  try
  {
    process(req.pose);
    return true;
  }
  catch (const std::exception& e)
  {
    ROS_ERROR_STREAM(e.what() << " Ignoring request.");
    return false;
  }
}

void Unicycle3DIgnition::process(const geometry_msgs::PoseWithCovarianceStamped& pose)
{
  // Verify we are in the correct state to process set pose requests
  if (!started_)
  {
    throw std::runtime_error("Attempting to set the pose while the sensor is stopped.");
  }
  // Validate the requested pose and covariance before we do anything
  if (!std::isfinite(pose.pose.pose.position.x) ||
      !std::isfinite(pose.pose.pose.position.y) ||
      !std::isfinite(pose.pose.pose.position.z))
  {
    throw std::invalid_argument("Attempting to set the pose to an invalid position (" +
                                std::to_string(pose.pose.pose.position.x) + ", " +
                                std::to_string(pose.pose.pose.position.y) + ").");
  }
  auto orientation_norm = std::sqrt(pose.pose.pose.orientation.x * pose.pose.pose.orientation.x +
                                    pose.pose.pose.orientation.y * pose.pose.pose.orientation.y +
                                    pose.pose.pose.orientation.z * pose.pose.pose.orientation.z +
                                    pose.pose.pose.orientation.w * pose.pose.pose.orientation.w);
  if (std::abs(orientation_norm - 1.0) > 1.0e-3)
  {
    throw std::invalid_argument("Attempting to set the pose to an invalid orientation (" +
                                std::to_string(pose.pose.pose.orientation.x) + ", " +
                                std::to_string(pose.pose.pose.orientation.y) + ", " +
                                std::to_string(pose.pose.pose.orientation.z) + ", " +
                                std::to_string(pose.pose.pose.orientation.w) + ").");
  }
  auto position_cov = fuse_core::Matrix3d();
  position_cov << pose.pose.covariance[0], pose.pose.covariance[1], pose.pose.covariance[2],
                  pose.pose.covariance[6], pose.pose.covariance[7], pose.pose.covariance[8],
                  pose.pose.covariance[12], pose.pose.covariance[13], pose.pose.covariance[14];
  if (!fuse_core::isSymmetric(position_cov))
  {
    throw std::invalid_argument("Attempting to set the pose with a non-symmetric position covariance matri\n " +
                                fuse_core::to_string(position_cov, Eigen::FullPrecision) + ".");
  }
  if (!fuse_core::isPositiveDefinite(position_cov))
  {
    throw std::invalid_argument("Attempting to set the pose with a non-positive-definite position covariance matrix\n" +
                                fuse_core::to_string(position_cov, Eigen::FullPrecision) + ".");
  }
  auto orientation_cov = fuse_core::Matrix3d();
  orientation_cov << pose.pose.covariance[21], pose.pose.covariance[22], pose.pose.covariance[23],
                     pose.pose.covariance[27], pose.pose.covariance[28], pose.pose.covariance[29],
                     pose.pose.covariance[33], pose.pose.covariance[34], pose.pose.covariance[35];
  if (!fuse_core::isPositiveDefinite(orientation_cov))
  {
    throw std::invalid_argument("Attempting to set the pose with a non-positive-definite orientation covariance "
                                "matrix " + fuse_core::to_string(orientation_cov, Eigen::FullPrecision) + ".");
  }

  // Tell the optimizer to reset before providing the initial state
  if (!params_.reset_service.empty())
  {
    // Wait for the reset service
    while (!reset_client_.waitForExistence(ros::Duration(10.0)) && ros::ok())
    {
      ROS_WARN_STREAM("Waiting for '" << reset_client_.getService() << "' service to become avaiable.");
    }

    auto srv = std_srvs::Empty();
    if (!reset_client_.call(srv))
    {
      // The reset() service failed. Propagate that failure to the caller of this service.
      throw std::runtime_error("Failed to call the '" + reset_client_.getService() + "' service.");
    }
  }

  // Now that the pose has been validated and the optimizer has been reset, actually send the initial state constraints
  // to the optimizer
  sendPrior(pose);
}

void Unicycle3DIgnition::sendPrior(const geometry_msgs::PoseWithCovarianceStamped& pose)
{
  const auto& stamp = pose.header.stamp;

  // Create variables for the full state.
  // The initial values of the pose are extracted from the provided PoseWithCovarianceStamped message.
  // The remaining dimensions are provided as parameters to the parameter server.
  auto position = fuse_variables::Position3DStamped::make_shared(stamp, device_id_);
  position->x() = pose.pose.pose.position.x;
  position->y() = pose.pose.pose.position.y;
  position->z() = pose.pose.pose.position.z;
  auto orientation = fuse_variables::Orientation3DStamped::make_shared(stamp, device_id_);
  orientation->w() = pose.pose.pose.orientation.w;
  orientation->x() = pose.pose.pose.orientation.x;
  orientation->y() = pose.pose.pose.orientation.y;
  orientation->z() = pose.pose.pose.orientation.z;
  auto linear_velocity = fuse_variables::VelocityLinear3DStamped::make_shared(stamp, device_id_);
  linear_velocity->x() = params_.initial_state[6];
  linear_velocity->y() = params_.initial_state[7];
  linear_velocity->z() = params_.initial_state[8];
  auto angular_velocity = fuse_variables::VelocityAngular3DStamped::make_shared(stamp, device_id_);
  angular_velocity->roll() = params_.initial_state[9];
  angular_velocity->pitch() = params_.initial_state[10];
  angular_velocity->yaw() = params_.initial_state[11];
  auto linear_acceleration = fuse_variables::AccelerationLinear3DStamped::make_shared(stamp, device_id_);
  linear_acceleration->x() = params_.initial_state[12];
  linear_acceleration->y() = params_.initial_state[13];
  linear_acceleration->z() = params_.initial_state[14];

  // Create the covariances for each variable
  // The pose covariances are extracted from the provided PoseWithCovarianceStamped message.
  // The remaining covariances are provided as parameters to the parameter server.
  auto position_cov = fuse_core::Matrix3d();
  position_cov << pose.pose.covariance[0], pose.pose.covariance[1], pose.pose.covariance[2],
                  pose.pose.covariance[6], pose.pose.covariance[7], pose.pose.covariance[8],
                  pose.pose.covariance[12], pose.pose.covariance[13], pose.pose.covariance[14];
  auto orientation_cov = fuse_core::Matrix3d();
  orientation_cov << pose.pose.covariance[21], pose.pose.covariance[22], pose.pose.covariance[23],
                     pose.pose.covariance[27], pose.pose.covariance[28], pose.pose.covariance[29],
                     pose.pose.covariance[33], pose.pose.covariance[34], pose.pose.covariance[35];
  auto linear_velocity_cov = fuse_core::Matrix3d();
  linear_velocity_cov << params_.initial_sigma[6] * params_.initial_sigma[6], 0.0, 0.0,
                         0.0, params_.initial_sigma[7] * params_.initial_sigma[7], 0.0,
                         0.0, 0.0, params_.initial_sigma[8] * params_.initial_sigma[8];
  auto angular_velocity_cov = fuse_core::Matrix3d();
  angular_velocity_cov << params_.initial_sigma[9] * params_.initial_sigma[9], 0.0, 0.0,
                          0.0, params_.initial_sigma[10] * params_.initial_sigma[10], 0.0,
                          0.0, 0.0, params_.initial_sigma[11] * params_.initial_sigma[11];
  auto linear_acceleration_cov = fuse_core::Matrix3d();
  linear_acceleration_cov << params_.initial_sigma[12] * params_.initial_sigma[12], 0.0, 0.0,
                             0.0, params_.initial_sigma[13] * params_.initial_sigma[13], 0.0,
                             0.0, 0.0, params_.initial_sigma[14] * params_.initial_sigma[14];

  // Create absolute constraints for each variable
  auto position_constraint = fuse_constraints::AbsolutePosition3DStampedConstraint::make_shared(
    name(),
    *position,
    fuse_core::Vector3d(position->x(), position->y(), position->z()),
    position_cov);
  auto orientation_constraint = fuse_constraints::AbsoluteOrientation3DStampedEulerConstraint::make_shared(
    name(),
    *orientation,
    fuse_core::Vector3d(orientation->roll(), orientation->pitch(), orientation->yaw()),
    orientation_cov,
    std::vector<fuse_variables::Orientation3DStamped::Euler>{  // NOLINT(whitespace/braces)
      fuse_variables::Orientation3DStamped::Euler::ROLL,
      fuse_variables::Orientation3DStamped::Euler::PITCH,
      fuse_variables::Orientation3DStamped::Euler::YAW});  // NOLINT(whitespace/braces)
  auto linear_velocity_constraint = fuse_constraints::AbsoluteVelocityLinear3DStampedConstraint::make_shared(
    name(),
    *linear_velocity,
    fuse_core::Vector3d(linear_velocity->x(), linear_velocity->y(), linear_velocity->z()),
    linear_velocity_cov);
  auto angular_velocity_constraint = fuse_constraints::AbsoluteVelocityAngular3DStampedConstraint::make_shared(
    name(),
    *angular_velocity,
    fuse_core::Vector3d(angular_velocity->roll(), angular_velocity->pitch(), angular_velocity->yaw()),
    angular_velocity_cov);
  auto linear_acceleration_constraint = fuse_constraints::AbsoluteAccelerationLinear3DStampedConstraint::make_shared(
    name(),
    *linear_acceleration,
    fuse_core::Vector3d(linear_acceleration->x(), linear_acceleration->y(), linear_acceleration->z()),
    linear_acceleration_cov);

  // Create the transaction
  auto transaction = fuse_core::Transaction::make_shared();
  transaction->stamp(stamp);
  transaction->addInvolvedStamp(stamp);
  transaction->addVariable(position);
  transaction->addVariable(orientation);
  transaction->addVariable(linear_velocity);
  transaction->addVariable(angular_velocity);
  transaction->addVariable(linear_acceleration);
  transaction->addConstraint(position_constraint);
  transaction->addConstraint(orientation_constraint);
  transaction->addConstraint(linear_velocity_constraint);
  transaction->addConstraint(angular_velocity_constraint);
  transaction->addConstraint(linear_acceleration_constraint);

  // Send the transaction to the optimizer.
  sendTransaction(transaction);

  ROS_INFO_STREAM("Received a set_pose request ("
                  "stamp: " << stamp << ", "
                  "x: " << position->x() << ", "
                  "y: " << position->y() << ", "
                  "z: " << position->z() << ", "
                  "roll: " << orientation->roll() << ", "
                  "pitch: " << orientation->pitch() << ", "
                  "yaw: " << orientation->yaw() << ")");
}

}  // namespace fuse_models
