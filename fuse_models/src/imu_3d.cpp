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
#include <fuse_models/common/sensor_proc.h>
#include <fuse_models/imu_3d.h>

#include <fuse_core/transaction.h>
#include <fuse_core/uuid.h>

#include <geometry_msgs/AccelWithCovarianceStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <pluginlib/class_list_macros.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>

#include <memory>
#include <utility>


// Register this sensor model with ROS as a plugin.
PLUGINLIB_EXPORT_CLASS(fuse_models::Imu3D, fuse_core::SensorModel)

namespace fuse_models
{

Imu3D::Imu3D() :
  fuse_core::AsyncSensorModel(1),
  device_id_(fuse_core::uuid::NIL),
  tf_listener_(tf_buffer_),
  throttled_callback_(std::bind(&Imu3D::process, this, std::placeholders::_1))
{
}

void Imu3D::onInit()
{
  // Read settings from the parameter sever
  device_id_ = fuse_variables::loadDeviceId(private_node_handle_);

  params_.loadFromROS(private_node_handle_);

  throttled_callback_.setThrottlePeriod(params_.throttle_period);
  throttled_callback_.setUseWallTime(params_.throttle_use_wall_time);

  if (params_.orientation_indices.empty() &&
      params_.linear_acceleration_indices.empty() &&
      params_.angular_velocity_indices.empty())
  {
    ROS_WARN_STREAM("No dimensions were specified. Data from topic " << ros::names::resolve(params_.topic) <<
                    " will be ignored.");
  }
}

void Imu3D::onStart()
{
  if (!params_.orientation_indices.empty() ||
      !params_.linear_acceleration_indices.empty() ||
      !params_.angular_velocity_indices.empty())
  {
    previous_pose_.reset();
    subscriber_ = node_handle_.subscribe<sensor_msgs::Imu>(ros::names::resolve(params_.topic), params_.queue_size,
                                                           &ImuThrottledCallback::callback, &throttled_callback_,
                                                           ros::TransportHints().tcpNoDelay(params_.tcp_no_delay));
  }
}

void Imu3D::onStop()
{
  subscriber_.shutdown();
}

void Imu3D::process(const sensor_msgs::Imu::ConstPtr& msg)
{
  // Create a transaction object
  auto transaction = fuse_core::Transaction::make_shared();
  transaction->stamp(msg->header.stamp);

  // Handle the orientation data (treat it as a pose, but with only orientation indices used)
  auto pose = std::make_unique<geometry_msgs::PoseWithCovarianceStamped>();
  pose->header = msg->header;
  pose->pose.pose.orientation = msg->orientation;

  Eigen::Map<const fuse_core::Matrix3d> orientation_covariance(msg->orientation_covariance.data());

  Eigen::Map<fuse_core::Matrix3d, Eigen::Unaligned, Eigen::OuterStride<6>>
    pose_msg_covariance(pose->pose.covariance.data() + 21);

  pose_msg_covariance = orientation_covariance.cwiseMax(params_.minimum_orientation_covariance);

  // Preprocess the twist data
  geometry_msgs::TwistWithCovarianceStamped twist;
  twist.header = msg->header;
  twist.twist.twist.angular = msg->angular_velocity;

  Eigen::Map<const fuse_core::Matrix3d> angular_velocity_covariance(msg->angular_velocity_covariance.data());

  Eigen::Map<fuse_core::Matrix3d, Eigen::Unaligned, Eigen::OuterStride<6>>
    twist_msg_covariance(twist.twist.covariance.data() + 21);

  twist_msg_covariance = angular_velocity_covariance.cwiseMax(params_.minimum_angular_velocity_covariance);

  const bool validate = !params_.disable_checks;

  if (params_.differential)
  {
    processDifferential(*pose, twist, validate, *transaction);
  }
  else
  {
    common::processAbsolutePose3DWithCovariance(
      name(),
      device_id_,
      *pose,
      params_.pose_loss,
      params_.orientation_target_frame,
      {},
      params_.orientation_indices,
      tf_buffer_,
      validate,
      *transaction,
      params_.tf_timeout);
  }

  // Handle the twist data (only include indices for angular velocity)
  common::processTwist3DWithCovariance(
    name(),
    device_id_,
    twist,
    nullptr,
    params_.angular_velocity_loss,
    params_.twist_target_frame,
    {},
    params_.angular_velocity_indices,
    tf_buffer_,
    validate,
    *transaction,
    params_.tf_timeout);

  // Handle the acceleration data
  geometry_msgs::AccelWithCovarianceStamped accel;
  accel.header = msg->header;
  accel.accel.accel.linear = msg->linear_acceleration;

  Eigen::Map<const fuse_core::Matrix3d> linear_acceleration_covariance(msg->linear_acceleration_covariance.data());

  Eigen::Map<fuse_core::Matrix3d, Eigen::Unaligned, Eigen::OuterStride<6>>
    acceleration_msg_covariance(accel.accel.covariance.data());

  acceleration_msg_covariance = linear_acceleration_covariance.cwiseMax(params_.minimum_linear_acceleration_covariance);

  // Optionally remove the acceleration due to gravity
  if (params_.remove_gravitational_acceleration)
  {
    geometry_msgs::Vector3 accel_gravity;
    accel_gravity.z = params_.gravitational_acceleration;
    geometry_msgs::TransformStamped orientation_trans;
    tf2::Quaternion imu_orientation;
    tf2::fromMsg(msg->orientation, imu_orientation);
    orientation_trans.transform.rotation = tf2::toMsg(imu_orientation.inverse());
    tf2::doTransform(accel_gravity, accel_gravity, orientation_trans);  // Doesn't use the stamp
    accel.accel.accel.linear.x -= accel_gravity.x;
    accel.accel.accel.linear.y -= accel_gravity.y;
    accel.accel.accel.linear.z -= accel_gravity.z;
  }

  common::processAccel3DWithCovariance(
    name(),
    device_id_,
    accel,
    params_.linear_acceleration_loss,
    params_.acceleration_target_frame,
    params_.linear_acceleration_indices,
    tf_buffer_,
    validate,
    *transaction,
    params_.tf_timeout);

  // Send the transaction object to the plugin's parent
  sendTransaction(transaction);
}

void Imu3D::processDifferential(const geometry_msgs::PoseWithCovarianceStamped& pose,
                                const geometry_msgs::TwistWithCovarianceStamped& twist, const bool validate,
                                fuse_core::Transaction& transaction)
{
  auto transformed_pose = std::make_unique<geometry_msgs::PoseWithCovarianceStamped>();
  transformed_pose->header.frame_id =
      params_.orientation_target_frame.empty() ? pose.header.frame_id : params_.orientation_target_frame;

  if (!common::transformMessage(tf_buffer_, pose, *transformed_pose, params_.tf_timeout))
  {
    ROS_WARN_STREAM_THROTTLE(5.0, "Cannot transform pose message with stamp " << pose.header.stamp
                                                                              << " to orientation target frame "
                                                                              << params_.orientation_target_frame);
    return;
  }

  if (!previous_pose_)
  {
    previous_pose_ = std::move(transformed_pose);
    return;
  }

  if (params_.use_twist_covariance)
  {
    geometry_msgs::TwistWithCovarianceStamped transformed_twist;
    transformed_twist.header.frame_id =
        params_.twist_target_frame.empty() ? twist.header.frame_id : params_.twist_target_frame;

    if (!common::transformMessage(tf_buffer_, twist, transformed_twist, params_.tf_timeout))
    {
      ROS_WARN_STREAM_THROTTLE(5.0, "Cannot transform twist message with stamp " << twist.header.stamp
                                                                                 << " to twist target frame "
                                                                                 << params_.twist_target_frame);
    }
    else
    {
      common::processDifferentialPoseWithTwist3DCovariance(
        name(),
        device_id_,
        *previous_pose_,
        *transformed_pose,
        twist,
        params_.minimum_pose_relative_covariance,
        params_.twist_covariance_offset,
        params_.pose_loss,
        {},
        params_.orientation_indices,
        validate,
        transaction);
    }
  }
  else
  {
    common::processDifferentialPose3DWithCovariance(
      name(),
      device_id_,
      *previous_pose_,
      *transformed_pose,
      params_.independent,
      params_.minimum_pose_relative_covariance,
      params_.pose_loss,
      {},
      params_.orientation_indices,
      validate,
      transaction);
  }

  previous_pose_ = std::move(transformed_pose);
}

}  // namespace fuse_models
