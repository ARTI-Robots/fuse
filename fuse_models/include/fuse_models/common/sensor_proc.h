/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2018, Locus Robotics
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
#ifndef FUSE_MODELS_COMMON_SENSOR_PROC_H
#define FUSE_MODELS_COMMON_SENSOR_PROC_H

#include <fuse_constraints/absolute_pose_2d_stamped_constraint.h>
#include <fuse_constraints/absolute_pose_3d_stamped_euler_constraint.h>
#include <fuse_constraints/relative_pose_2d_stamped_constraint.h>
#include <fuse_constraints/relative_pose_3d_stamped_euler_constraint.h>
#include <fuse_constraints/absolute_constraint.h>
#include <fuse_core/eigen.h>
#include <fuse_core/loss.h>
#include <fuse_core/transaction.h>
#include <fuse_core/uuid.h>
#include <fuse_variables/acceleration_linear_2d_stamped.h>
#include <fuse_variables/orientation_2d_stamped.h>
#include <fuse_variables/orientation_3d_stamped.h>
#include <fuse_variables/position_2d_stamped.h>
#include <fuse_variables/position_3d_stamped.h>
#include <fuse_variables/velocity_linear_2d_stamped.h>
#include <fuse_variables/velocity_angular_2d_stamped.h>
#include <fuse_variables/stamped.h>
#include <fuse_models/common/sensor_config.h>

#include <geometry_msgs/AccelWithCovarianceStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <ros/ros.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_2d/tf2_2d.h>
#include <tf2_2d/transform.h>
#include <tf2_eigen/tf2_eigen.h>

#include <boost/range/join.hpp>

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>


namespace tf2
{

/** \brief Apply a geometry_msgs TransformStamped to a geometry_msgs TwistWithCovarianceStamped type.
* This function is a specialization of the doTransform template defined in tf2/convert.h.
* \param t_in The twist to transform, as a timestamped TwistWithCovarianceStamped message.
* \param t_out The transformed twist, as a timestamped TwistWithCovarianceStamped message.
* \param transform The timestamped transform to apply, as a TransformStamped message.
*/
template <>
inline
void doTransform(const geometry_msgs::TwistWithCovarianceStamped& t_in, geometry_msgs::TwistWithCovarianceStamped& t_out, const geometry_msgs::TransformStamped& transform)  // NOLINT
{
  tf2::Vector3 vl;
  fromMsg(t_in.twist.twist.linear, vl);
  tf2::Vector3 va;
  fromMsg(t_in.twist.twist.angular, va);

  tf2::Transform t;
  fromMsg(transform.transform, t);
  t_out.twist.twist.linear = tf2::toMsg(t.getBasis() * vl);
  t_out.twist.twist.angular = tf2::toMsg(t.getBasis() * va);
  t_out.header.stamp = transform.header.stamp;
  t_out.header.frame_id = transform.header.frame_id;

  t_out.twist.covariance = transformCovariance(t_in.twist.covariance, t);
}

/** \brief Apply a geometry_msgs TransformStamped to a geometry_msgs AccelWithCovarianceStamped type.
* This function is a specialization of the doTransform template defined in tf2/convert.h.
* \param t_in The acceleration to transform, as a timestamped AccelWithCovarianceStamped message.
* \param t_out The transformed acceleration, as a timestamped AccelWithCovarianceStamped message.
* \param transform The timestamped transform to apply, as a TransformStamped message.
*/
template <>
inline
void doTransform(const geometry_msgs::AccelWithCovarianceStamped& t_in, geometry_msgs::AccelWithCovarianceStamped& t_out, const geometry_msgs::TransformStamped& transform)  // NOLINT
{
  tf2::Vector3 al;
  fromMsg(t_in.accel.accel.linear, al);
  tf2::Vector3 aa;
  fromMsg(t_in.accel.accel.angular, aa);

  tf2::Transform t;
  fromMsg(transform.transform, t);
  t_out.accel.accel.linear = tf2::toMsg(t.getBasis() * al);
  t_out.accel.accel.angular = tf2::toMsg(t.getBasis() * aa);
  t_out.header.stamp = transform.header.stamp;
  t_out.header.frame_id = transform.header.frame_id;

  t_out.accel.covariance = transformCovariance(t_in.accel.covariance, t);
}

}  // namespace tf2


namespace fuse_models
{

namespace common
{

/**
 * @brief Method to merge two vectors of indices adding an offset to the RHS one.
 *
 * @param[in] lhs_indices - LHS vector of indices
 * @param[in] rhs_indices - RHS vector of indices
 * @param[in] rhs_offset - RHS offset to be added to the RHS vector indices (defaults to 0)
 */
inline std::vector<size_t> mergeIndices(
  const std::vector<size_t>& lhs_indices,
  const std::vector<size_t>& rhs_indices,
  const size_t rhs_offset = 0u)
{
  auto merged_indices = boost::copy_range<std::vector<size_t>>(boost::range::join(lhs_indices, rhs_indices));

  const auto rhs_it = merged_indices.begin() + lhs_indices.size();
  std::transform(
    rhs_it,
    merged_indices.end(),
    rhs_it,
    std::bind(std::plus<size_t>(), std::placeholders::_1, rhs_offset));

  return merged_indices;
}

/**
 * @brief Method to create sub-measurements from full measurements and append them to existing partial measurements
 *
 * @param[in] mean_full - The full mean vector from which we will generate the sub-measurement
 * @param[in] covariance_full - The full covariance matrix from which we will generate the sub-measurement
 * @param[in] indices - The indices we want to include in the sub-measurement
 * @param[in,out] mean_partial - The partial measurement mean to which we want to append
 * @param[in,out] covariance_partial - The partial measurement covariance to which we want to append
 */
inline void populatePartialMeasurement(
  const fuse_core::VectorXd& mean_full,
  const fuse_core::MatrixXd& covariance_full,
  const std::vector<size_t>& indices,
  fuse_core::VectorXd& mean_partial,
  fuse_core::MatrixXd& covariance_partial)
{
  for (size_t r = 0; r < indices.size(); ++r)
  {
    mean_partial(r) = mean_full(indices[r]);

    for (size_t c = 0; c < indices.size(); ++c)
    {
      covariance_partial(r, c) = covariance_full(indices[r], indices[c]);
    }
  }
}

/**
 * @brief Method to validate partial measurements, that checks for finite values and covariance properties
 *
 * @param[in] mean_partial - The partial measurement mean we want to validate
 * @param[in] covariance_partial - The partial measurement covariance we want to validate
 * @param[in] precision - The precision to validate the partial measurements covariance is symmetric
 */
inline void validatePartialMeasurement(
  const fuse_core::VectorXd& mean_partial,
  const fuse_core::MatrixXd& covariance_partial,
  const double precision = Eigen::NumTraits<double>::dummy_precision())
{
  if (!mean_partial.allFinite())
  {
    throw std::runtime_error("Invalid partial mean " + fuse_core::to_string(mean_partial));
  }

  if (!fuse_core::isSymmetric(covariance_partial, precision))
  {
    throw std::runtime_error("Non-symmetric partial covariance matrix\n" +
                             fuse_core::to_string(covariance_partial, Eigen::FullPrecision));
  }

  if (!fuse_core::isPositiveDefinite(covariance_partial))
  {
    throw std::runtime_error("Non-positive-definite partial covariance matrix\n" +
                             fuse_core::to_string(covariance_partial, Eigen::FullPrecision));
  }
}

/**
 * @brief Transforms a ROS geometry message from its frame to the frame of the output message
 *
 * @param[in] tf_buffer - The transform buffer with which we will lookup the required transform
 * @param[in] input - The message to transform. Source frame and stamp are dictated by its header.
 * @param[in,out] output - The transformed message. Target frame is dictated by its header.
 * @param [in] timeout - Optional. The maximum time to wait for a transform to become available.
 * @return true if the transform succeeded, false otherwise
 */
template <typename T>
bool transformMessage(
  const tf2_ros::Buffer& tf_buffer,
  const T& input,
  T& output,
  const ros::Duration& tf_timeout = ros::Duration(0, 0))
{
  try
  {
    auto trans = geometry_msgs::TransformStamped();
    if (tf_timeout.isZero())
    {
      trans = tf_buffer.lookupTransform(output.header.frame_id, input.header.frame_id, input.header.stamp);
    }
    else
    {
      trans = tf_buffer.lookupTransform(output.header.frame_id, input.header.frame_id, input.header.stamp, tf_timeout);
    }
    tf2::doTransform(input, output, trans);
    return true;
  }
  catch (const tf2::TransformException& ex)
  {
    ROS_WARN_STREAM_DELAYED_THROTTLE(5.0, "Could not transform message from " << input.header.frame_id << " to " <<
      output.header.frame_id << ". Error was " << ex.what());
  }

  return false;
}

/**
 * @brief Extracts 2D pose data from a PoseWithCovarianceStamped message and adds that data to a fuse Transaction
 *
 * This method effectively adds two variables (2D position and 2D orientation) and a 2D pose constraint to the given
 * \p transaction. The pose data is extracted from the \p pose message. Only 2D data is used. The data will be
 * automatically transformed into the \p target_frame before it is used.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] pose - The PoseWithCovarianceStamped message from which we will extract the pose data
 * @param[in] loss - The loss function for the 2D pose constraint generated
 * @param[in] target_frame - The frame ID into which the pose data will be transformed before it is used
 * @param[in] tf_buffer - The transform buffer with which we will lookup the required transform
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processAbsolutePoseWithCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::PoseWithCovarianceStamped& pose,
  const fuse_core::Loss::SharedPtr& loss,
  const std::string& target_frame,
  const std::vector<size_t>& position_indices,
  const std::vector<size_t>& orientation_indices,
  const tf2_ros::Buffer& tf_buffer,
  const bool validate,
  fuse_core::Transaction& transaction,
  const ros::Duration& tf_timeout = ros::Duration(0, 0))
{
  if (position_indices.empty() && orientation_indices.empty())
  {
    return false;
  }

  geometry_msgs::PoseWithCovarianceStamped transformed_message;
  if (target_frame.empty())
  {
    transformed_message = pose;
  }
  else
  {
    transformed_message.header.frame_id = target_frame;

    if (!transformMessage(tf_buffer, pose, transformed_message, tf_timeout))
    {
      ROS_WARN_STREAM_DELAYED_THROTTLE(
        10.0,
        "Failed to transform pose message with stamp " << pose.header.stamp << ". Cannot create constraint.");
      return false;
    }
  }

  // Convert the pose into tf2_2d transform
  tf2_2d::Transform absolute_pose_2d;
  tf2::fromMsg(transformed_message.pose.pose, absolute_pose_2d);

  // Create the pose variable
  auto position = fuse_variables::Position2DStamped::make_shared(pose.header.stamp, device_id);
  auto orientation = fuse_variables::Orientation2DStamped::make_shared(pose.header.stamp, device_id);
  position->x() = absolute_pose_2d.x();
  position->y() = absolute_pose_2d.y();
  orientation->setYaw(absolute_pose_2d.yaw());

  // Create the pose for the constraint
  fuse_core::Vector3d pose_mean;
  pose_mean << absolute_pose_2d.x(), absolute_pose_2d.y(), absolute_pose_2d.yaw();

  // Create the covariance for the constraint
  fuse_core::Matrix3d pose_covariance;
  pose_covariance <<
      transformed_message.pose.covariance[0],
      transformed_message.pose.covariance[1],
      transformed_message.pose.covariance[5],
      transformed_message.pose.covariance[6],
      transformed_message.pose.covariance[7],
      transformed_message.pose.covariance[11],
      transformed_message.pose.covariance[30],
      transformed_message.pose.covariance[31],
      transformed_message.pose.covariance[35];

  // Build the sub-vector and sub-matrices based on the requested indices
  fuse_core::VectorXd pose_mean_partial(position_indices.size() + orientation_indices.size());
  fuse_core::MatrixXd pose_covariance_partial(pose_mean_partial.rows(), pose_mean_partial.rows());

  const auto indices = mergeIndices(position_indices, orientation_indices, position->size());

  populatePartialMeasurement(pose_mean, pose_covariance, indices, pose_mean_partial, pose_covariance_partial);

  if (validate)
  {
    try
    {
      validatePartialMeasurement(pose_mean_partial, pose_covariance_partial);
    }
    catch (const std::runtime_error& ex)
    {
      ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid partial absolute pose measurement from '" << source
                                                                                         << "' source: " << ex.what());
      return false;
    }
  }

  // Create an absolute pose constraint
  auto constraint = fuse_constraints::AbsolutePose2DStampedConstraint::make_shared(
    source,
    *position,
    *orientation,
    pose_mean_partial,
    pose_covariance_partial,
    position_indices,
    orientation_indices);

  constraint->loss(loss);

  transaction.addVariable(position);
  transaction.addVariable(orientation);
  transaction.addConstraint(constraint);
  transaction.addInvolvedStamp(pose.header.stamp);

  return true;
}

/**
 * @brief Extracts 3D pose data from a PoseWithCovarianceStamped message and adds that data to a fuse Transaction
 *
 * This method effectively adds two variables (3D position and 3D orientation) and a 3D pose constraint to the given
 * \p transaction. The pose data is extracted from the \p pose message. The data will be
 * automatically transformed into the \p target_frame before it is used.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] pose - The PoseWithCovarianceStamped message from which we will extract the pose data
 * @param[in] loss - The loss function for the 3D pose constraint generated
 * @param[in] target_frame - The frame ID into which the pose data will be transformed before it is used
 * @param[in] tf_buffer - The transform buffer with which we will lookup the required transform
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processAbsolutePose3DWithCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::PoseWithCovarianceStamped& pose,
  const fuse_core::Loss::SharedPtr& loss,
  const std::string& target_frame,
  const std::vector<size_t>& position_indices,
  const std::vector<size_t>& orientation_indices,
  const tf2_ros::Buffer& tf_buffer,
  const bool validate,
  fuse_core::Transaction& transaction,
  const ros::Duration& tf_timeout = ros::Duration(0, 0))
{
  if (position_indices.empty() && orientation_indices.empty())
  {
    return false;
  }

  geometry_msgs::PoseWithCovarianceStamped transformed_message;
  if (target_frame.empty())
  {
    transformed_message = pose;
  }
  else
  {
    transformed_message.header.frame_id = target_frame;

    if (!transformMessage(tf_buffer, pose, transformed_message, tf_timeout))
    {
      ROS_WARN_STREAM_DELAYED_THROTTLE(
        10.0,
        "Failed to transform pose message with stamp " << pose.header.stamp << ". Cannot create constraint.");
      return false;
    }
  }

  // Create the pose variable
  auto position = fuse_variables::Position3DStamped::make_shared(pose.header.stamp, device_id);
  auto orientation = fuse_variables::Orientation3DStamped::make_shared(pose.header.stamp, device_id);
  position->x() = transformed_message.pose.pose.position.x;
  position->y() = transformed_message.pose.pose.position.y;
  position->z() = transformed_message.pose.pose.position.z;
  orientation->w() = transformed_message.pose.pose.orientation.w;
  orientation->x() = transformed_message.pose.pose.orientation.x;
  orientation->y() = transformed_message.pose.pose.orientation.y;
  orientation->z() = transformed_message.pose.pose.orientation.z;

  // Create the pose for the constraint
  fuse_core::Vector6d pose_mean;
  pose_mean << position->x(), position->y(), position->z(),
               orientation->roll(), orientation->pitch(), orientation->yaw();

  // Create the covariance for the constraint
  Eigen::Map<const fuse_core::Matrix6d> pose_covariance(transformed_message.pose.covariance.data());

  // Build the sub-vector and sub-matrices based on the requested indices
  fuse_core::VectorXd pose_mean_partial(position_indices.size() + orientation_indices.size());
  fuse_core::MatrixXd pose_covariance_partial(pose_mean_partial.rows(), pose_mean_partial.rows());

  const auto indices = mergeIndices(position_indices, orientation_indices, position->size());

  populatePartialMeasurement(pose_mean, pose_covariance, indices, pose_mean_partial, pose_covariance_partial);

  if (validate)
  {
    try
    {
      validatePartialMeasurement(pose_mean_partial, pose_covariance_partial);
    }
    catch (const std::runtime_error& ex)
    {
      ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid partial absolute pose measurement from '" << source
                                                                                         << "' source: " << ex.what());
      return false;
    }
  }

  // Create an absolute pose constraint
  auto constraint = fuse_constraints::AbsolutePose3DStampedEulerConstraint::make_shared(
    source,
    *position,
    *orientation,
    pose_mean_partial,
    pose_covariance_partial,
    position_indices,
    toEulerEnumVector(orientation_indices));

  constraint->loss(loss);

  transaction.addVariable(position);
  transaction.addVariable(orientation);
  transaction.addConstraint(constraint);
  transaction.addInvolvedStamp(pose.header.stamp);

  return true;
}

/**
 * @brief Extracts relative 2D pose data from a PoseWithCovarianceStamped and adds that data to a fuse Transaction
 *
 * This method computes the delta between two poses and creates the required fuse variables and constraints, and then
 * adds them to the given \p transaction. Only 2D data is used. The pose delta is calculated as
 *
 * pose_relative = pose_absolute1^-1 * pose_absolute2
 *
 * Additionally, the covariance of each pose message is rotated into the robot's base frame at the time of
 * pose_absolute1. They are then added in the constraint if the pose measurements are independent.
 * Otherwise, if the pose measurements are dependent, the covariance of pose_absolute1 is substracted from the
 * covariance of pose_absolute2. A small minimum relative covariance is added to avoid getting a zero or
 * ill-conditioned covariance. This could happen if both covariance matrices are the same or very similar, e.g. when
 * pose_absolute1 == pose_absolute2, it's possible that the covariance is the same for both poses.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] pose1 - The first (and temporally earlier) PoseWithCovarianceStamped message
 * @param[in] pose2 - The second (and temporally later) PoseWithCovarianceStamped message
 * @param[in] independent - Whether the pose measurements are indepent or not
 * @param[in] minimum_pose_relative_covariance - The minimum pose relative covariance that is always added to the
 *                                               resulting pose relative covariance
 * @param[in] loss - The loss function for the 2D pose constraint generated
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processDifferentialPoseWithCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::PoseWithCovarianceStamped& pose1,
  const geometry_msgs::PoseWithCovarianceStamped& pose2,
  const bool independent,
  const fuse_core::Matrix3d& minimum_pose_relative_covariance,
  const fuse_core::Loss::SharedPtr& loss,
  const std::vector<size_t>& position_indices,
  const std::vector<size_t>& orientation_indices,
  const bool validate,
  fuse_core::Transaction& transaction)
{
  if (position_indices.empty() && orientation_indices.empty())
  {
    return false;
  }

  // Convert the poses into tf2_2d transforms
  tf2_2d::Transform pose1_2d;
  tf2::fromMsg(pose1.pose.pose, pose1_2d);

  tf2_2d::Transform pose2_2d;
  tf2::fromMsg(pose2.pose.pose, pose2_2d);

  // Create the pose variables
  auto position1 = fuse_variables::Position2DStamped::make_shared(pose1.header.stamp, device_id);
  auto orientation1 =
    fuse_variables::Orientation2DStamped::make_shared(pose1.header.stamp, device_id);
  position1->x() = pose1_2d.x();
  position1->y() = pose1_2d.y();
  orientation1->setYaw(pose1_2d.yaw());

  auto position2 = fuse_variables::Position2DStamped::make_shared(pose2.header.stamp, device_id);
  auto orientation2 = fuse_variables::Orientation2DStamped::make_shared(pose2.header.stamp, device_id);
  position2->x() = pose2_2d.x();
  position2->y() = pose2_2d.y();
  orientation2->setYaw(pose2_2d.yaw());

  // Create the delta for the constraint
  const double sy = ::sin(-pose1_2d.yaw());
  const double cy = ::cos(-pose1_2d.yaw());
  double x_diff = pose2_2d.x() - pose1_2d.x();
  double y_diff = pose2_2d.y() - pose1_2d.y();
  fuse_core::Vector3d pose_relative_mean;
  pose_relative_mean <<
    cy * x_diff - sy * y_diff,
    sy * x_diff + cy * y_diff,
    (pose2_2d.rotation() - pose1_2d.rotation()).getAngle();

  // Create the covariance components for the constraint
  fuse_core::Matrix3d cov1;
  cov1 <<
    pose1.pose.covariance[0],
    pose1.pose.covariance[1],
    pose1.pose.covariance[5],
    pose1.pose.covariance[6],
    pose1.pose.covariance[7],
    pose1.pose.covariance[11],
    pose1.pose.covariance[30],
    pose1.pose.covariance[31],
    pose1.pose.covariance[35];

  fuse_core::Matrix3d cov2;
  cov2 <<
    pose2.pose.covariance[0],
    pose2.pose.covariance[1],
    pose2.pose.covariance[5],
    pose2.pose.covariance[6],
    pose2.pose.covariance[7],
    pose2.pose.covariance[11],
    pose2.pose.covariance[30],
    pose2.pose.covariance[31],
    pose2.pose.covariance[35];

  fuse_core::Matrix3d pose_relative_covariance;
  if (independent)
  {
    // Compute Jacobians so we can rotate the covariance
    fuse_core::Matrix3d j_pose1;
    j_pose1 <<
      -cy,  sy,  sy * x_diff + cy * y_diff,
      -sy, -cy, -cy * x_diff + sy * y_diff,
        0,   0,                         -1;

    fuse_core::Matrix3d j_pose2;
    j_pose2 <<
       cy, -sy,  0,
       sy,  cy,  0,
        0,   0,  1;

    pose_relative_covariance = j_pose1 * cov1 * j_pose1.transpose() + j_pose2 * cov2 * j_pose2.transpose();
  }
  else
  {
    // For dependent pose measurements p1 and p2, we assume they're computed as:
    //
    // p2 = p1 * p12    [1]
    //
    // where p12 is the relative pose between p1 and p2, which is computed here as:
    //
    // p12 = p1^-1 * p2
    //
    // Note that the twist t12 is computed as:
    //
    // t12 = p12 / dt
    //
    // where dt = t2 - t1, for t1 and t2 being the p1 and p2 timestamps, respectively.
    //
    // The covariance propagation of p2 = p1 * p12 is:
    //
    // C2 = J_p1 * C1 * J_p1^T + J_p12 * C12 * J_p12^T
    //
    // where C1, C2, C12 are the covariance matrices of p1, p2 and dp, respectively, and J_p1 and J_p12 are the
    // jacobians of the equation wrt p1 and p12, respectively.
    //
    // Therefore, the covariance C12 of the relative pose p12 is:
    //
    // C12 = J_p12^-1 * (C2 - J_p1 * C1 * J_p1^T) * J_p12^-T    [2]
    //
    //
    //
    // In SE(2) the poses are represented by:
    //
    //     (R | t)
    // p = (-----)
    //     (0 | 1)
    //
    // where R is the rotation matrix for the yaw angle:
    //
    //     (cos(yaw) -sin(yaw))
    // R = (sin(yaw)  cos(yaw))
    //
    // and t is the translation:
    //
    //     (x)
    // t = (y)
    //
    // The pose composition/multiplication in SE(2) is defined as follows:
    //
    //           (R1 | t1)   (R2 | t2)   (R1 * R2 | R1 * t2 + t1)
    // p1 * p2 = (-------) * (-------) = (----------------------)
    //           ( 0 |  1)   ( 0 |  1)   (      0 |            1)
    //
    // which gives the following equations for each component:
    //
    // x = x2 * cos(yaw1) - y2 * sin(yaw1) + x1
    // y = x2 * sin(yaw1) + y2 * cos(yaw1) + y1
    // yaw = yaw1 + yaw2
    //
    // Since the covariance matrices are defined following that same order for the SE(2) components:
    //
    //     (xx   xy   xyaw  )
    // C = (yx   yy   yyaw  )
    //     (yawx yawy yawyaw)
    //
    // the jacobians must be defined following the same order.
    //
    // The jacobian wrt p1 is:
    //
    //        (1 0 | -sin(yaw1) * x2 - cos(yaw1) * y2)
    // J_p1 = (0 1 |  cos(yaw1) * x2 - sin(yaw1) * y2)
    //        (0 0 |                                1)
    //
    // The jacobian wrt p2 is:
    //
    //        (R1 | 0)   (cos(yaw1) -sin(yaw1) 0)
    // J_p2 = (------) = (sin(yaw1)  cos(yaw1) 0)
    //        ( 0 | 1)   (        0          0 1)
    //
    //
    //
    // Therefore, for the the covariance propagation of [1] we would get the following jacobians:
    //
    //        (1 0 | -sin(yaw1) * x12 - cos(yaw1) * y12)
    // J_p1 = (0 1 |  cos(yaw1) * x12 - sin(yaw1) * y12)
    //        (0 0 |                                  1)
    //
    //         (R1 | 0)   (cos(yaw1) -sin(yaw1) 0)
    // J_p12 = (------) = (sin(yaw1)  cos(yaw1) 0)
    //         ( 0 | 1)   (        0          0 1)
    //
    //
    //
    // At this point we could go one step further since p12 = t12 * dt and include the jacobian of this additional
    // equation:
    //
    // J_t12 = dt * Id
    //
    // where Id is a 3x3 identity matrix.
    //
    // However, that would give us the covariance of the twist t12, and here we simply need the one of the relative
    // pose p12.
    //
    //
    //
    // Finally, since we need the inverse of the jacobian J_p12, we can use the inverse directly:
    //
    //            ( cos(yaw1) sin(yaw1) 0)
    // J_p12^-1 = (-sin(yaw1) cos(yaw1) 0)
    //            (         0         0 1)
    //
    //
    //
    // In the implementation below we use:
    //
    // sy = sin(-yaw1)
    // cy = cos(-yaw1)
    //
    // which are defined before.
    //
    // Therefore, the jacobians end up with the following expressions:
    //
    //        (1 0 | sin(-yaw1) * x12 - cos(-yaw1) * y12)
    // J_p1 = (0 1 | cos(-yaw1) * x12 + sin(-yaw1) * y12)
    //        (0 0 |                                   1)
    //
    //            (cos(-yaw1) -sin(-yaw1) 0)
    // J_p12^-1 = (sin(-yaw1)  cos(-yaw1) 0)
    //            (         0           0 1)
    //
    //
    //
    // Note that the covariance propagation expression derived here for dependent pose measurements gives more accurate
    // results than simply changing the sign in the expression for independent pose measurements, which would be:
    //
    // C12 = J_p2 * C2 * J_p2^T - J_p1 * C1 * J_p1^T
    //
    // where J_p1 and J_p2 are the jacobians for p12 = p1^-1 * p2 (we're abusing the notation here):
    //
    //        (-cos(-yaw1),  sin(-yaw1),  sin(-yaw1) * x12 + cos(-yaw1) * y12)
    // J_p1 = (-sin(-yaw1), -cos(-yaw1), -cos(-yaw1) * x12 + sin(-yaw1) * y12)
    //        (          0,           0,                                   -1)
    //
    //        (R1 | 0)   (cos(yaw1) -sin(yaw1) 0)
    // J_p2 = (------) = (sin(yaw1)  cos(yaw1) 0)
    //        ( 0 | 1)   (        0          0 1)
    //
    // which are the j_pose1 and j_pose2 jacobians used above for the covariance propagation expresion for independent
    // pose measurements.
    //
    // This seems to be the approach adviced in https://github.com/cra-ros-pkg/robot_localization/issues/356, but after
    // comparing the resulting relative pose covariance C12 and the twist covariance, we can conclude that the approach
    // proposed here is the only one that allow us to get results that match.
    //
    // The relative pose covariance C12 and the twist covariance T12 can be compared with:
    //
    // T12 = J_t12 * C12 * J_t12^T
    //
    //
    //
    // In some cases the difference between the C1 and C2 covariance matrices is very small and it could yield to an
    // ill-conditioned C12 covariance. For that reason a minimum covariance is added to [2].
    fuse_core::Matrix3d j_pose1;
    j_pose1 << 1, 0, sy * pose_relative_mean(0) - cy * pose_relative_mean(1),
               0, 1, cy * pose_relative_mean(0) + sy * pose_relative_mean(1),
               0, 0, 1;

    fuse_core::Matrix3d j_pose12_inv;
    j_pose12_inv << cy, -sy, 0,
                    sy,  cy, 0,
                     0,   0, 1;

    pose_relative_covariance = j_pose12_inv * (cov2 - j_pose1 * cov1 * j_pose1.transpose()) * j_pose12_inv.transpose() +
                               minimum_pose_relative_covariance;
  }

  // Build the sub-vector and sub-matrices based on the requested indices
  fuse_core::VectorXd pose_relative_mean_partial(position_indices.size() + orientation_indices.size());
  fuse_core::MatrixXd pose_relative_covariance_partial(pose_relative_mean_partial.rows(),
                                                       pose_relative_mean_partial.rows());

  const auto indices = mergeIndices(position_indices, orientation_indices, position1->size());

  populatePartialMeasurement(
    pose_relative_mean,
    pose_relative_covariance,
    indices,
    pose_relative_mean_partial,
    pose_relative_covariance_partial);

  if (validate)
  {
    try
    {
      validatePartialMeasurement(pose_relative_mean_partial, pose_relative_covariance_partial, 1e-6);
    }
    catch (const std::runtime_error& ex)
    {
      ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid partial differential pose measurement from '"
                                          << source << "' source: " << ex.what());
      return false;
    }
  }

  // Create a relative pose constraint.
  auto constraint = fuse_constraints::RelativePose2DStampedConstraint::make_shared(
    source,
    *position1,
    *orientation1,
    *position2,
    *orientation2,
    pose_relative_mean_partial,
    pose_relative_covariance_partial,
    position_indices,
    orientation_indices);

  constraint->loss(loss);

  transaction.addVariable(position1);
  transaction.addVariable(orientation1);
  transaction.addVariable(position2);
  transaction.addVariable(orientation2);
  transaction.addConstraint(constraint);
  transaction.addInvolvedStamp(pose1.header.stamp);
  transaction.addInvolvedStamp(pose2.header.stamp);

  return true;
}

/**
 * @brief Extracts relative 3D pose data from a PoseWithCovarianceStamped and adds that data to a fuse Transaction
 *
 * This method computes the delta between two poses and creates the required fuse variables and constraints, and then
 * adds them to the given \p transaction. The pose delta is calculated as
 *
 * pose_relative = pose_absolute1^-1 * pose_absolute2
 *
 * Additionally, the covariance of each pose message is rotated into the robot's base frame at the time of
 * pose_absolute1. They are then added in the constraint if the pose measurements are independent.
 * Otherwise, if the pose measurements are dependent, the covariance of pose_absolute1 is substracted from the
 * covariance of pose_absolute2. A small minimum relative covariance is added to avoid getting a zero or
 * ill-conditioned covariance. This could happen if both covariance matrices are the same or very similar, e.g. when
 * pose_absolute1 == pose_absolute2, it's possible that the covariance is the same for both poses.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] pose1 - The first (and temporally earlier) PoseWithCovarianceStamped message
 * @param[in] pose2 - The second (and temporally later) PoseWithCovarianceStamped message
 * @param[in] independent - Whether the pose measurements are indepent or not
 * @param[in] minimum_pose_relative_covariance - The minimum pose relative covariance that is always added to the
 *                                               resulting pose relative covariance
 * @param[in] loss - The loss function for the 3D pose constraint generated
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processDifferentialPose3DWithCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::PoseWithCovarianceStamped& pose1,
  const geometry_msgs::PoseWithCovarianceStamped& pose2,
  const bool independent,
  const fuse_core::Matrix6d& minimum_pose_relative_covariance,
  const fuse_core::Loss::SharedPtr& loss,
  const std::vector<size_t>& position_indices,
  const std::vector<size_t>& orientation_indices,
  const bool validate,
  fuse_core::Transaction& transaction)
{
  if (position_indices.empty() && orientation_indices.empty())
  {
    return false;
  }

  // Create the pose variables
  auto position1 = fuse_variables::Position3DStamped::make_shared(pose1.header.stamp, device_id);
  auto orientation1 =
    fuse_variables::Orientation3DStamped::make_shared(pose1.header.stamp, device_id);
  position1->x() = pose1.pose.pose.position.x;
  position1->y() = pose1.pose.pose.position.y;
  position1->z() = pose1.pose.pose.position.z;

  orientation1->w() = pose1.pose.pose.orientation.w;
  orientation1->x() = pose1.pose.pose.orientation.x;
  orientation1->y() = pose1.pose.pose.orientation.y;
  orientation1->z() = pose1.pose.pose.orientation.z;

  auto position2 = fuse_variables::Position3DStamped::make_shared(pose2.header.stamp, device_id);
  auto orientation2 = fuse_variables::Orientation3DStamped::make_shared(pose2.header.stamp, device_id);
  position2->x() = pose2.pose.pose.position.x;
  position2->y() = pose2.pose.pose.position.y;
  position2->z() = pose2.pose.pose.position.z;

  orientation2->w() = pose2.pose.pose.orientation.w;
  orientation2->x() = pose2.pose.pose.orientation.x;
  orientation2->y() = pose2.pose.pose.orientation.y;
  orientation2->z() = pose2.pose.pose.orientation.z;

  // Method adapted from https://github.com/locusrobotics/fuse/blob/90ded4ba333b426ad56d25eddf575c9b6027dfa7/fuse_models/include/fuse_models/common/sensor_proc.hpp#L1003 NOLINT(whitespace/line_length)

  // Convert from ROS msg to Eigen
  Eigen::Isometry3d p1, p2;
  tf2::fromMsg(pose1.pose.pose, p1);
  const Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> cov1(pose1.pose.covariance.data());
  tf2::fromMsg(pose2.pose.pose, p2);
  const Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> cov2(pose2.pose.covariance.data());

  Eigen::Isometry3d p12 = p1.inverse() * p2;
  fuse_core::Matrix6d cov12;

  if (independent)
  {
    const auto cov1_inverse = fuse_core::invertPoseCovariance(p1, cov1);

    cov12 = fuse_core::composePoseCovariance(
      p1.inverse(), cov1_inverse,
      p2, cov2);
  }
  else
  {
    // If covariances of p1 and p2 are the same, then it's possible that p12 covariance will be
    // zero or ill-conditioned. To avoid this, we skip the expensive following calculations and
    // instead we just add a minimum covariance later
    if (cov1.isApprox(cov2, 1e-9))
    {
      cov12.setZero();
    }
    else
    {
      // Here we assume that poses are computed incrementally, so: p2 = p1 * p12.
      // We know cov1 and cov2 and we should subtract the first to the second in order
      // to obtain the relative pose covariance. But first the 2 of them have to be in the
      // same reference frame, moreover we need to rotate the result in p12 reference frame
      // The covariance propagation of p2 = p1 * p12 is:
      //
      // C2 = J_p1 * C1 * J_p1^T + J_p12 * C12 * J_p12^T
      //
      // where C1, C2, C12 are the covariance matrices of p1, p2 and dp, respectively, and J_p1 and
      // J_p12 are the jacobians of the equation (pose composition) wrt p1 and p12, respectively.
      //
      // Therefore, the covariance C12 of the relative pose p12 is:
      //
      // C12 = J_p12^-1 * (C2 - J_p1 * C1 * J_p1^T) * J_p12^-T

      // First we need to convert covariances from RPY (6x6) to quaternion (7x7)
      fuse_core::Matrix7d cov1_q = fuse_core::convertToPoseQuatCovariance(p1, cov1);
      fuse_core::Matrix7d cov2_q = fuse_core::convertToPoseQuatCovariance(p2, cov2);

      // Now we have to compute pose composition jacobians so we can rotate covariances
      const auto j_qn = fuse_core::jacobianQuatNormalization(Eigen::Quaterniond(p12.rotation()));

      auto j_p1 = fuse_core::jacobianPosePoseCompositionA(p1, p12);
      j_p1.block<4, 4>(3, 3).applyOnTheLeft(j_qn);
      j_p1.block<4, 3>(3, 0).setZero();

      auto j_p12 = fuse_core::jacobianPosePoseCompositionB(p1);
      j_p12.block<4, 4>(3, 3).applyOnTheLeft(j_qn);
      j_p12.block<3, 4>(0, 3).setZero();
      j_p12.block<4, 3>(3, 0).setZero();

      // TODO(giafranchini): check if faster to use j12.llt().solve() instead
      const auto j_p12_inv = j_p12.colPivHouseholderQr().solve(fuse_core::Matrix7d::Identity());
      const auto cov12_q = j_p12_inv * (cov2_q - j_p1 * cov1_q * j_p1.transpose()) * j_p12_inv.transpose();

      // Now again convert the delta pose covariance back to RPY(6x6)
      cov12 = fuse_core::convertToPoseRPYCovariance(p12, cov12_q);
    }
  }

  Eigen::Quaterniond delta_q(p12.rotation());
  double delta_q_array[4] = {delta_q.w(),  // NOLINT(whitespace/braces)
                             delta_q.x(),
                             delta_q.y(),
                             delta_q.z()};  // NOLINT(whitespace/braces)

  double delta_rpy[3];
  fuse_core::quaternion2rpy(delta_q_array, delta_rpy);

  fuse_core::Vector6d pose_relative_mean;
  pose_relative_mean <<
    p12.translation().x(), p12.translation().y(), p12.translation().z(),
    delta_rpy[0], delta_rpy[1], delta_rpy[2];

  fuse_core::Matrix6d pose_relative_covariance = cov12 + minimum_pose_relative_covariance;

  // Build the sub-vector and sub-matrices based on the requested indices
  fuse_core::VectorXd pose_relative_mean_partial(position_indices.size() + orientation_indices.size());
  fuse_core::MatrixXd pose_relative_covariance_partial(pose_relative_mean_partial.rows(),
                                                       pose_relative_mean_partial.rows());

  const auto indices = mergeIndices(position_indices, orientation_indices, position1->size());

  populatePartialMeasurement(
    pose_relative_mean,
    pose_relative_covariance,
    indices,
    pose_relative_mean_partial,
    pose_relative_covariance_partial);

  if (validate)
  {
    try
    {
      validatePartialMeasurement(pose_relative_mean_partial, pose_relative_covariance_partial, 1e-6);
    }
    catch (const std::runtime_error& ex)
    {
      ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid partial differential pose measurement from '"
                                          << source << "' source: " << ex.what());
      return false;
    }
  }

  // Create a relative pose constraint.
  auto constraint = fuse_constraints::RelativePose3DStampedEulerConstraint::make_shared(
    source,
    *position1,
    *orientation1,
    *position2,
    *orientation2,
    pose_relative_mean_partial,
    pose_relative_covariance_partial,
    position_indices,
    toEulerEnumVector(orientation_indices));

  constraint->loss(loss);

  transaction.addVariable(position1);
  transaction.addVariable(orientation1);
  transaction.addVariable(position2);
  transaction.addVariable(orientation2);
  transaction.addConstraint(constraint);
  transaction.addInvolvedStamp(pose1.header.stamp);
  transaction.addInvolvedStamp(pose2.header.stamp);

  return true;
}

/**
 * @brief Extracts relative 2D pose data from a PoseWithCovarianceStamped and adds that data to a fuse Transaction
 *
 * This method computes the delta between two poses and creates the required fuse variables and constraints, and then
 * adds them to the given \p transaction. Only 2D data is used. The pose delta is calculated as
 *
 * pose_relative = pose_absolute1^-1 * pose_absolute2
 *
 * Additionally, the twist covariance of the last message is used to compute the relative pose covariance using the time
 * difference between the pose_absolute2 and pose_absolute1 time stamps. This assumes the pose measurements are
 * dependent. A small minimum relative covariance is added to avoid getting a zero or ill-conditioned covariance. This
 * could happen if the twist covariance is very small, e.g. when the twist is zero.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] pose1 - The first (and temporally earlier) PoseWithCovarianceStamped message
 * @param[in] pose2 - The second (and temporally later) PoseWithCovarianceStamped message
 * @param[in] twist - The second (and temporally later) TwistWithCovarianceStamped message
 * @param[in] minimum_pose_relative_covariance - The minimum pose relative covariance that is always added to the
 *                                               resulting pose relative covariance
 * @param[in] twist_covariance_offset - The twist covariance offset that was added to the twist covariance and must be
 *                                       substracted from it before computing the pose relative covariance from it
 * @param[in] loss - The loss function for the 2D pose constraint generated
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processDifferentialPoseWithTwistCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::PoseWithCovarianceStamped& pose1,
  const geometry_msgs::PoseWithCovarianceStamped& pose2,
  const geometry_msgs::TwistWithCovarianceStamped& twist,
  const fuse_core::Matrix3d& minimum_pose_relative_covariance,
  const fuse_core::Matrix3d& twist_covariance_offset,
  const fuse_core::Loss::SharedPtr& loss,
  const std::vector<size_t>& position_indices,
  const std::vector<size_t>& orientation_indices,
  const bool validate,
  fuse_core::Transaction& transaction)
{
  if (position_indices.empty() && orientation_indices.empty())
  {
    return false;
  }

  // Convert the poses into tf2_2d transforms
  tf2_2d::Transform pose1_2d;
  tf2::fromMsg(pose1.pose.pose, pose1_2d);

  tf2_2d::Transform pose2_2d;
  tf2::fromMsg(pose2.pose.pose, pose2_2d);

  // Create the pose variables
  auto position1 = fuse_variables::Position2DStamped::make_shared(pose1.header.stamp, device_id);
  auto orientation1 =
    fuse_variables::Orientation2DStamped::make_shared(pose1.header.stamp, device_id);
  position1->x() = pose1_2d.x();
  position1->y() = pose1_2d.y();
  orientation1->setYaw(pose1_2d.yaw());

  auto position2 = fuse_variables::Position2DStamped::make_shared(pose2.header.stamp, device_id);
  auto orientation2 = fuse_variables::Orientation2DStamped::make_shared(pose2.header.stamp, device_id);
  position2->x() = pose2_2d.x();
  position2->y() = pose2_2d.y();
  orientation2->setYaw(pose2_2d.yaw());

  // Create the delta for the constraint
  const auto delta = pose1_2d.inverseTimes(pose2_2d);
  fuse_core::Vector3d pose_relative_mean;
  pose_relative_mean << delta.x(), delta.y(), delta.yaw();

  // Create the covariance components for the constraint
  fuse_core::Matrix3d cov;
  cov <<
    twist.twist.covariance[0],
    twist.twist.covariance[1],
    twist.twist.covariance[5],
    twist.twist.covariance[6],
    twist.twist.covariance[7],
    twist.twist.covariance[11],
    twist.twist.covariance[30],
    twist.twist.covariance[31],
    twist.twist.covariance[35];

  // For dependent pose measurements p1 and p2, we assume they're computed as:
  //
  // p2 = p1 * p12    [1]
  //
  // where p12 is the relative pose between p1 and p2, which is computed here as:
  //
  // p12 = p1^-1 * p2
  //
  // Note that the twist t12 is computed as:
  //
  // t12 = p12 / dt
  //
  // where dt = t2 - t1, for t1 and t2 being the p1 and p2 timestamps, respectively.
  //
  // Therefore, the relative pose p12 is computed as follows given the twist t12:
  //
  // p12 = t12 * dt
  //
  // The covariance propagation of this equation is:
  //
  // C12 = J_t12 * T12 * J_t12^T    [2]
  //
  // where T12 is the twist covariance and J_t12 is the jacobian of the equation wrt to t12.
  //
  // The jacobian wrt t12 is:
  //
  // J_t12 = dt * Id
  //
  // where Id is a 3x3 Identity matrix.
  //
  // In some cases the twist covariance T12 is very small and it could yield to an ill-conditioned C12 covariance. For
  // that reason a minimum covariance is added to [2].
  //
  // It is also common that for the same reason, the twist covariance T12 already has a minimum covariance offset added
  // to it by the publisher, so we have to remove it before using it.
  const auto dt = (pose2.header.stamp - pose1.header.stamp).toSec();

  if (dt < 1e-6)
  {
    ROS_ERROR_STREAM_THROTTLE(10.0, "Very small time difference " << dt << "s from '" << source << "' source.");
    return false;
  }

  fuse_core::Matrix3d j_twist;
  j_twist.setIdentity();
  j_twist *= dt;

  fuse_core::Matrix3d pose_relative_covariance =
      j_twist * (cov - twist_covariance_offset) * j_twist.transpose() + minimum_pose_relative_covariance;

  // Build the sub-vector and sub-matrices based on the requested indices
  fuse_core::VectorXd pose_relative_mean_partial(position_indices.size() + orientation_indices.size());
  fuse_core::MatrixXd pose_relative_covariance_partial(pose_relative_mean_partial.rows(),
                                                       pose_relative_mean_partial.rows());

  const auto indices = mergeIndices(position_indices, orientation_indices, position1->size());

  populatePartialMeasurement(
    pose_relative_mean,
    pose_relative_covariance,
    indices,
    pose_relative_mean_partial,
    pose_relative_covariance_partial);

  if (validate)
  {
    try
    {
      validatePartialMeasurement(pose_relative_mean_partial, pose_relative_covariance_partial, 1e-6);
    }
    catch (const std::runtime_error& ex)
    {
      ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid partial differential pose measurement using the twist covariance from '"
                                          << source << "' source: " << ex.what());
      return false;
    }
  }

  // Create a relative pose constraint.
  auto constraint = fuse_constraints::RelativePose2DStampedConstraint::make_shared(
    source,
    *position1,
    *orientation1,
    *position2,
    *orientation2,
    pose_relative_mean_partial,
    pose_relative_covariance_partial,
    position_indices,
    orientation_indices);

  constraint->loss(loss);

  transaction.addVariable(position1);
  transaction.addVariable(orientation1);
  transaction.addVariable(position2);
  transaction.addVariable(orientation2);
  transaction.addConstraint(constraint);
  transaction.addInvolvedStamp(pose1.header.stamp);
  transaction.addInvolvedStamp(pose2.header.stamp);

  return true;
}

/**
 * @brief Extracts relative 3D pose data from a PoseWithCovarianceStamped and adds that data to a fuse Transaction
 *
 * This method computes the delta between two poses and creates the required fuse variables and constraints, and then
 * adds them to the given \p transaction. The pose delta is calculated as
 *
 * pose_relative = pose_absolute1^-1 * pose_absolute2
 *
 * Additionally, the twist covariance of the last message is used to compute the relative pose covariance using the time
 * difference between the pose_absolute2 and pose_absolute1 time stamps. This assumes the pose measurements are
 * dependent. A small minimum relative covariance is added to avoid getting a zero or ill-conditioned covariance. This
 * could happen if the twist covariance is very small, e.g. when the twist is zero.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] pose1 - The first (and temporally earlier) PoseWithCovarianceStamped message
 * @param[in] pose2 - The second (and temporally later) PoseWithCovarianceStamped message
 * @param[in] twist - The second (and temporally later) TwistWithCovarianceStamped message
 * @param[in] minimum_pose_relative_covariance - The minimum pose relative covariance that is always added to the
 *                                               resulting pose relative covariance
 * @param[in] twist_covariance_offset - The twist covariance offset that was added to the twist covariance and must be
 *                                       substracted from it before computing the pose relative covariance from it
 * @param[in] loss - The loss function for the 3D pose constraint generated
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processDifferentialPoseWithTwist3DCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::PoseWithCovarianceStamped& pose1,
  const geometry_msgs::PoseWithCovarianceStamped& pose2,
  const geometry_msgs::TwistWithCovarianceStamped& twist,
  const fuse_core::Matrix6d& minimum_pose_relative_covariance,
  const fuse_core::Matrix6d& twist_covariance_offset,
  const fuse_core::Loss::SharedPtr& loss,
  const std::vector<size_t>& position_indices,
  const std::vector<size_t>& orientation_indices,
  const bool validate,
  fuse_core::Transaction& transaction)
{
  if (position_indices.empty() && orientation_indices.empty())
  {
    return false;
  }

  // Create the pose variables
  auto position1 = fuse_variables::Position3DStamped::make_shared(pose1.header.stamp, device_id);
  auto orientation1 =
    fuse_variables::Orientation3DStamped::make_shared(pose1.header.stamp, device_id);
  position1->x() = pose1.pose.pose.position.x;
  position1->y() = pose1.pose.pose.position.y;
  position1->z() = pose1.pose.pose.position.z;
  orientation1->w() = pose1.pose.pose.orientation.w;
  orientation1->x() = pose1.pose.pose.orientation.x;
  orientation1->y() = pose1.pose.pose.orientation.y;
  orientation1->z() = pose1.pose.pose.orientation.z;

  auto position2 = fuse_variables::Position3DStamped::make_shared(pose2.header.stamp, device_id);
  auto orientation2 = fuse_variables::Orientation3DStamped::make_shared(pose2.header.stamp, device_id);
  position2->x() = pose2.pose.pose.position.x;
  position2->y() = pose2.pose.pose.position.y;
  position2->z() = pose2.pose.pose.position.z;
  orientation2->w() = pose2.pose.pose.orientation.w;
  orientation2->x() = pose2.pose.pose.orientation.x;
  orientation2->y() = pose2.pose.pose.orientation.y;
  orientation2->z() = pose2.pose.pose.orientation.z;

  // Create the delta for the constraint
  tf2::Transform pose1_tf;
  tf2::fromMsg(pose1.pose.pose, pose1_tf);
  tf2::Transform pose2_tf;
  tf2::fromMsg(pose2.pose.pose, pose2_tf);

  const auto delta = pose1_tf.inverseTimes(pose2_tf);

  double delta_quat[4] = {delta.getRotation().w(),  // NOLINT(whitespace/braces)
                    delta.getRotation().x(),
                    delta.getRotation().y(),
                    delta.getRotation().z()};  // NOLINT(whitespace/braces)

  double delta_rpy[3];
  fuse_core::quaternion2rpy(delta_quat, delta_rpy);

  fuse_core::Vector6d pose_relative_mean;
  pose_relative_mean <<
    delta.getOrigin().x(), delta.getOrigin().y(), delta.getOrigin().z(),
    delta_rpy[0], delta_rpy[1], delta_rpy[2];

  // Create the covariance components for the constraint
  Eigen::Map<const fuse_core::Matrix6d> cov(twist.twist.covariance.data());

  // For dependent pose measurements p1 and p2, we assume they're computed as:
  //
  // p2 = p1 * p12    [1]
  //
  // where p12 is the relative pose between p1 and p2, which is computed here as:
  //
  // p12 = p1^-1 * p2
  //
  // Note that the twist t12 is computed as:
  //
  // t12 = p12 / dt
  //
  // where dt = t2 - t1, for t1 and t2 being the p1 and p2 timestamps, respectively.
  //
  // Therefore, the relative pose p12 is computed as follows given the twist t12:
  //
  // p12 = t12 * dt
  //
  // The covariance propagation of this equation is:
  //
  // C12 = J_t12 * T12 * J_t12^T    [2]
  //
  // where T12 is the twist covariance and J_t12 is the jacobian of the equation wrt to t12.
  //
  // The jacobian wrt t12 is:
  //
  // J_t12 = dt * Id
  //
  // where Id is a 3x3 Identity matrix.
  //
  // In some cases the twist covariance T12 is very small and it could yield to an ill-conditioned C12 covariance. For
  // that reason a minimum covariance is added to [2].
  //
  // It is also common that for the same reason, the twist covariance T12 already has a minimum covariance offset added
  // to it by the publisher, so we have to remove it before using it.
  const auto dt = (pose2.header.stamp - pose1.header.stamp).toSec();

  if (dt < 1e-6)
  {
    ROS_ERROR_STREAM_THROTTLE(10.0, "Very small time difference " << dt << "s from '" << source << "' source.");
    return false;
  }

  fuse_core::Matrix6d j_twist;
  j_twist.setIdentity();
  j_twist *= dt;

  fuse_core::Matrix6d pose_relative_covariance =
      j_twist * (cov - twist_covariance_offset) * j_twist.transpose() + minimum_pose_relative_covariance;

  // Build the sub-vector and sub-matrices based on the requested indices
  fuse_core::VectorXd pose_relative_mean_partial(position_indices.size() + orientation_indices.size());
  fuse_core::MatrixXd pose_relative_covariance_partial(pose_relative_mean_partial.rows(),
                                                       pose_relative_mean_partial.rows());

  const auto indices = mergeIndices(position_indices, orientation_indices, position1->size());

  populatePartialMeasurement(
    pose_relative_mean,
    pose_relative_covariance,
    indices,
    pose_relative_mean_partial,
    pose_relative_covariance_partial);

  if (validate)
  {
    try
    {
      validatePartialMeasurement(pose_relative_mean_partial, pose_relative_covariance_partial, 1e-6);
    }
    catch (const std::runtime_error& ex)
    {
      ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid partial differential pose measurement using the twist covariance from '"
                                          << source << "' source: " << ex.what());
      return false;
    }
  }

  // Create a relative pose constraint.
  auto constraint = fuse_constraints::RelativePose3DStampedEulerConstraint::make_shared(
    source,
    *position1,
    *orientation1,
    *position2,
    *orientation2,
    pose_relative_mean_partial,
    pose_relative_covariance_partial,
    position_indices,
    toEulerEnumVector(orientation_indices));

  constraint->loss(loss);

  transaction.addVariable(position1);
  transaction.addVariable(orientation1);
  transaction.addVariable(position2);
  transaction.addVariable(orientation2);
  transaction.addConstraint(constraint);
  transaction.addInvolvedStamp(pose1.header.stamp);
  transaction.addInvolvedStamp(pose2.header.stamp);

  return true;
}

/**
 * @brief Extracts velocity data from a TwistWithCovarianceStamped and adds that data to a fuse Transaction
 *
 * This method effectively adds two variables (2D linear velocity and 2D angular velocity) and their respective
 * constraints to the given \p transaction. The velocity data is extracted from the \p twist message. Only 2D data is
 * used. The data will be automatically transformed into the \p target_frame before it is used.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] twist - The TwistWithCovarianceStamped message from which we will extract the twist data
 * @param[in] linear_velocity_loss - The loss function for the 2D linear velocity constraint generated
 * @param[in] angular_velocity_loss - The loss function for the 2D angular velocity constraint generated
 * @param[in] target_frame - The frame ID into which the twist data will be transformed before it is used
 * @param[in] tf_buffer - The transform buffer with which we will lookup the required transform
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processTwistWithCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::TwistWithCovarianceStamped& twist,
  const fuse_core::Loss::SharedPtr& linear_velocity_loss,
  const fuse_core::Loss::SharedPtr& angular_velocity_loss,
  const std::string& target_frame,
  const std::vector<size_t>& linear_indices,
  const std::vector<size_t>& angular_indices,
  const tf2_ros::Buffer& tf_buffer,
  const bool validate,
  fuse_core::Transaction& transaction,
  const ros::Duration& tf_timeout = ros::Duration(0, 0))
{
  // Make sure we actually have work to do
  if (linear_indices.empty() && angular_indices.empty())
  {
    return false;
  }

  geometry_msgs::TwistWithCovarianceStamped transformed_message;
  if (target_frame.empty())
  {
    transformed_message = twist;
  }
  else
  {
    transformed_message.header.frame_id = target_frame;

    if (!transformMessage(tf_buffer, twist, transformed_message, tf_timeout))
    {
      ROS_WARN_STREAM_DELAYED_THROTTLE(
        10.0,
        "Failed to transform twist message with stamp " << twist.header.stamp << ". Cannot create constraint.");
      return false;
    }
  }

  bool constraints_added = false;

  // Create two absolute constraints
  if (!linear_indices.empty())
  {
    auto velocity_linear =
      fuse_variables::VelocityLinear2DStamped::make_shared(twist.header.stamp, device_id);
    velocity_linear->x() = transformed_message.twist.twist.linear.x;
    velocity_linear->y() = transformed_message.twist.twist.linear.y;

    // Create the mean twist vectors for the constraints
    fuse_core::Vector2d linear_vel_mean;
    linear_vel_mean << transformed_message.twist.twist.linear.x, transformed_message.twist.twist.linear.y;

    // Create the covariances for the constraints
    fuse_core::Matrix2d linear_vel_covariance;
    linear_vel_covariance <<
        transformed_message.twist.covariance[0],
        transformed_message.twist.covariance[1],
        transformed_message.twist.covariance[6],
        transformed_message.twist.covariance[7];

    // Build the sub-vector and sub-matrices based on the requested indices
    fuse_core::VectorXd linear_vel_mean_partial(linear_indices.size());
    fuse_core::MatrixXd linear_vel_covariance_partial(linear_vel_mean_partial.rows(), linear_vel_mean_partial.rows());

    populatePartialMeasurement(
      linear_vel_mean,
      linear_vel_covariance,
      linear_indices,
      linear_vel_mean_partial,
      linear_vel_covariance_partial);

    bool add_constraint = true;

    if (validate)
    {
      try
      {
        validatePartialMeasurement(linear_vel_mean_partial, linear_vel_covariance_partial);
      }
      catch (const std::runtime_error& ex)
      {
        ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid partial linear velocity measurement from '"
                                            << source << "' source: " << ex.what());
        add_constraint = false;
      }
    }

    if (add_constraint)
    {
      auto linear_vel_constraint = fuse_constraints::AbsoluteVelocityLinear2DStampedConstraint::make_shared(
        source, *velocity_linear, linear_vel_mean_partial, linear_vel_covariance_partial, linear_indices);

      linear_vel_constraint->loss(linear_velocity_loss);

      transaction.addVariable(velocity_linear);
      transaction.addConstraint(linear_vel_constraint);
      constraints_added = true;
    }
  }

  if (!angular_indices.empty())
  {
    // Create the twist variables
    auto velocity_angular =
      fuse_variables::VelocityAngular2DStamped::make_shared(twist.header.stamp, device_id);
    velocity_angular->yaw() = transformed_message.twist.twist.angular.z;

    fuse_core::Vector1d angular_vel_vector;
    angular_vel_vector << transformed_message.twist.twist.angular.z;

    fuse_core::Matrix1d angular_vel_covariance;
    angular_vel_covariance << transformed_message.twist.covariance[35];

    bool add_constraint = true;

    if (validate)
    {
      try
      {
        validatePartialMeasurement(angular_vel_vector, angular_vel_covariance);
      }
      catch (const std::runtime_error& ex)
      {
        ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid partial angular velocity measurement from '"
                                            << source << "' source: " << ex.what());
        add_constraint = false;
      }
    }

    if (add_constraint)
    {
      auto angular_vel_constraint = fuse_constraints::AbsoluteVelocityAngular2DStampedConstraint::make_shared(
        source, *velocity_angular, angular_vel_vector, angular_vel_covariance, angular_indices);

      angular_vel_constraint->loss(angular_velocity_loss);

      transaction.addVariable(velocity_angular);
      transaction.addConstraint(angular_vel_constraint);
      constraints_added = true;
    }
  }

  if (constraints_added)
  {
    transaction.addInvolvedStamp(twist.header.stamp);
  }

  return constraints_added;
}

/**
 * @brief Extracts velocity data from a TwistWithCovarianceStamped and adds that data to a fuse Transaction
 *
 * This method effectively adds two variables (3D linear velocity and 3D angular velocity) and their respective
 * constraints to the given \p transaction. The velocity data is extracted from the \p twist message. 
 * The data will be automatically transformed into the \p target_frame before it is used.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] twist - The TwistWithCovarianceStamped message from which we will extract the twist data
 * @param[in] linear_velocity_loss - The loss function for the 3D linear velocity constraint generated
 * @param[in] angular_velocity_loss - The loss function for the 3D angular velocity constraint generated
 * @param[in] target_frame - The frame ID into which the twist data will be transformed before it is used
 * @param[in] tf_buffer - The transform buffer with which we will lookup the required transform
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processTwist3DWithCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::TwistWithCovarianceStamped& twist,
  const fuse_core::Loss::SharedPtr& linear_velocity_loss,
  const fuse_core::Loss::SharedPtr& angular_velocity_loss,
  const std::string& target_frame,
  const std::vector<size_t>& linear_indices,
  const std::vector<size_t>& angular_indices,
  const tf2_ros::Buffer& tf_buffer,
  const bool validate,
  fuse_core::Transaction& transaction,
  const ros::Duration& tf_timeout = ros::Duration(0, 0))
{
  // Make sure we actually have work to do
  if (linear_indices.empty() && angular_indices.empty())
  {
    return false;
  }

  geometry_msgs::TwistWithCovarianceStamped transformed_message;
  if (target_frame.empty())
  {
    transformed_message = twist;
  }
  else
  {
    transformed_message.header.frame_id = target_frame;

    if (!transformMessage(tf_buffer, twist, transformed_message, tf_timeout))
    {
      ROS_WARN_STREAM_DELAYED_THROTTLE(
        10.0,
        "Failed to transform twist message with stamp " << twist.header.stamp << ". Cannot create constraint.");
      return false;
    }
  }

  bool constraints_added = false;

  // Create two absolute constraints
  if (!linear_indices.empty())
  {
    auto velocity_linear =
      fuse_variables::VelocityLinear3DStamped::make_shared(twist.header.stamp, device_id);
    velocity_linear->x() = transformed_message.twist.twist.linear.x;
    velocity_linear->y() = transformed_message.twist.twist.linear.y;
    velocity_linear->z() = transformed_message.twist.twist.linear.z;

    // Create the mean twist vectors for the constraints
    fuse_core::Vector3d linear_vel_mean;
    linear_vel_mean << transformed_message.twist.twist.linear.x,
                       transformed_message.twist.twist.linear.y,
                       transformed_message.twist.twist.linear.z;

    // Create the covariances for the constraints
    fuse_core::Matrix3d linear_vel_covariance;
    linear_vel_covariance <<
        transformed_message.twist.covariance[0], transformed_message.twist.covariance[1], transformed_message.twist.covariance[2],  // NOLINT
        transformed_message.twist.covariance[6], transformed_message.twist.covariance[7], transformed_message.twist.covariance[8],  // NOLINT
        transformed_message.twist.covariance[12], transformed_message.twist.covariance[13], transformed_message.twist.covariance[14];  // NOLINT

    // Build the sub-vector and sub-matrices based on the requested indices
    fuse_core::VectorXd linear_vel_mean_partial(linear_indices.size());
    fuse_core::MatrixXd linear_vel_covariance_partial(linear_vel_mean_partial.rows(), linear_vel_mean_partial.rows());

    populatePartialMeasurement(
      linear_vel_mean,
      linear_vel_covariance,
      linear_indices,
      linear_vel_mean_partial,
      linear_vel_covariance_partial);

    bool add_constraint = true;

    if (validate)
    {
      try
      {
        validatePartialMeasurement(linear_vel_mean_partial, linear_vel_covariance_partial);
      }
      catch (const std::runtime_error& ex)
      {
        ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid partial linear velocity measurement from '"
                                            << source << "' source: " << ex.what());
        add_constraint = false;
      }
    }

    if (add_constraint)
    {
      auto linear_vel_constraint = fuse_constraints::AbsoluteVelocityLinear3DStampedConstraint::make_shared(
        source, *velocity_linear, linear_vel_mean_partial, linear_vel_covariance_partial, linear_indices);

      linear_vel_constraint->loss(linear_velocity_loss);

      transaction.addVariable(velocity_linear);
      transaction.addConstraint(linear_vel_constraint);
      constraints_added = true;
    }
  }

  if (!angular_indices.empty())
  {
    // Create the twist variables
    auto velocity_angular =
      fuse_variables::VelocityAngular3DStamped::make_shared(twist.header.stamp, device_id);
    velocity_angular->roll() = transformed_message.twist.twist.angular.x;
    velocity_angular->pitch() = transformed_message.twist.twist.angular.y;
    velocity_angular->yaw() = transformed_message.twist.twist.angular.z;

    fuse_core::Vector3d angular_vel_mean;
    angular_vel_mean << transformed_message.twist.twist.angular.x,
                          transformed_message.twist.twist.angular.y,
                          transformed_message.twist.twist.angular.z;

    fuse_core::Matrix3d angular_vel_covariance;
    angular_vel_covariance <<
        transformed_message.twist.covariance[21], transformed_message.twist.covariance[22], transformed_message.twist.covariance[23],  // NOLINT
        transformed_message.twist.covariance[27], transformed_message.twist.covariance[28], transformed_message.twist.covariance[29],  // NOLINT
        transformed_message.twist.covariance[33], transformed_message.twist.covariance[34], transformed_message.twist.covariance[35];  // NOLINT

    // Build the sub-vector and sub-matrices based on the requested indices
    fuse_core::VectorXd angular_vel_mean_partial(angular_indices.size());
    fuse_core::MatrixXd angular_vel_covariance_partial(angular_vel_mean_partial.rows(),
                                                       angular_vel_mean_partial.rows());

    populatePartialMeasurement(
      angular_vel_mean,
      angular_vel_covariance,
      angular_indices,
      angular_vel_mean_partial,
      angular_vel_covariance_partial);

    bool add_constraint = true;

    if (validate)
    {
      try
      {
        validatePartialMeasurement(angular_vel_mean_partial, angular_vel_covariance_partial);
      }
      catch (const std::runtime_error& ex)
      {
        ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid partial angular velocity measurement from '"
                                            << source << "' source: " << ex.what());
        add_constraint = false;
      }
    }

    if (add_constraint)
    {
      auto angular_vel_constraint = fuse_constraints::AbsoluteVelocityAngular3DStampedConstraint::make_shared(
        source, *velocity_angular, angular_vel_mean_partial, angular_vel_covariance_partial, angular_indices);

      angular_vel_constraint->loss(angular_velocity_loss);

      transaction.addVariable(velocity_angular);
      transaction.addConstraint(angular_vel_constraint);
      constraints_added = true;
    }
  }

  if (constraints_added)
  {
    transaction.addInvolvedStamp(twist.header.stamp);
  }

  return constraints_added;
}

/**
 * @brief Extracts linear acceleration data from an AccelWithCovarianceStamped and adds that data to a fuse Transaction
 *
 * This method effectively adds a linear acceleration variable and constraint to the given to the given \p transaction.
 * The acceleration data is extracted from the \p acceleration message. Only 2D data is used. The data will be
 * automatically transformed into the \p target_frame before it is used.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] acceleration - The AccelWithCovarianceStamped message from which we will extract the acceleration data
 * @param[in] loss - The loss function for the 2D linear acceleration constraint generated
 * @param[in] target_frame - The frame ID into which the acceleration data will be transformed before it is used
 * @param[in] tf_buffer - The transform buffer with which we will lookup the required transform
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processAccelWithCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::AccelWithCovarianceStamped& acceleration,
  const fuse_core::Loss::SharedPtr& loss,
  const std::string& target_frame,
  const std::vector<size_t>& indices,
  const tf2_ros::Buffer& tf_buffer,
  const bool validate,
  fuse_core::Transaction& transaction,
  const ros::Duration& tf_timeout = ros::Duration(0, 0))
{
  // Make sure we actually have work to do
  if (indices.empty())
  {
    return false;
  }

  geometry_msgs::AccelWithCovarianceStamped transformed_message;
  if (target_frame.empty())
  {
    transformed_message = acceleration;
  }
  else
  {
    transformed_message.header.frame_id = target_frame;

    if (!transformMessage(tf_buffer, acceleration, transformed_message, tf_timeout))
    {
      ROS_WARN_STREAM_DELAYED_THROTTLE(
        10.0,
        "Failed to transform acceleration message with stamp " << acceleration.header.stamp
                                                               << ". Cannot create constraint.");
      return false;
    }
  }

  // Create the acceleration variables
  auto acceleration_linear =
    fuse_variables::AccelerationLinear2DStamped::make_shared(acceleration.header.stamp, device_id);
  acceleration_linear->x() = transformed_message.accel.accel.linear.x;
  acceleration_linear->y() = transformed_message.accel.accel.linear.y;

  // Create the full mean vector and covariance for the constraint
  fuse_core::Vector2d accel_mean;
  accel_mean << transformed_message.accel.accel.linear.x, transformed_message.accel.accel.linear.y;

  fuse_core::Matrix2d accel_covariance;
  accel_covariance <<
      transformed_message.accel.covariance[0],
      transformed_message.accel.covariance[1],
      transformed_message.accel.covariance[6],
      transformed_message.accel.covariance[7];

  // Build the sub-vector and sub-matrices based on the requested indices
  fuse_core::VectorXd accel_mean_partial(indices.size());
  fuse_core::MatrixXd accel_covariance_partial(accel_mean_partial.rows(), accel_mean_partial.rows());

  populatePartialMeasurement(accel_mean, accel_covariance, indices, accel_mean_partial, accel_covariance_partial);

  if (validate)
  {
    try
    {
      validatePartialMeasurement(accel_mean_partial, accel_covariance_partial);
    }
    catch (const std::runtime_error& ex)
    {
      ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid partial linear acceleration measurement from '"
                                          << source << "' source: " << ex.what());
      return false;
    }
  }

  // Create the constraint
  auto linear_accel_constraint = fuse_constraints::AbsoluteAccelerationLinear2DStampedConstraint::make_shared(
    source,
    *acceleration_linear,
    accel_mean_partial,
    accel_covariance_partial,
    indices);

  linear_accel_constraint->loss(loss);

  transaction.addVariable(acceleration_linear);
  transaction.addConstraint(linear_accel_constraint);
  transaction.addInvolvedStamp(acceleration.header.stamp);

  return true;
}

/**
 * @brief Extracts linear acceleration data from an AccelWithCovarianceStamped and adds that data to a fuse Transaction
 *
 * This method effectively adds a linear acceleration variable and constraint to the given to the given \p transaction.
 * The acceleration data is extracted from the \p acceleration message. The data will be
 * automatically transformed into the \p target_frame before it is used.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] acceleration - The AccelWithCovarianceStamped message from which we will extract the acceleration data
 * @param[in] loss - The loss function for the 3D linear acceleration constraint generated
 * @param[in] target_frame - The frame ID into which the acceleration data will be transformed before it is used
 * @param[in] tf_buffer - The transform buffer with which we will lookup the required transform
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processAccel3DWithCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::AccelWithCovarianceStamped& acceleration,
  const fuse_core::Loss::SharedPtr& loss,
  const std::string& target_frame,
  const std::vector<size_t>& indices,
  const tf2_ros::Buffer& tf_buffer,
  const bool validate,
  fuse_core::Transaction& transaction,
  const ros::Duration& tf_timeout = ros::Duration(0, 0))
{
  // Make sure we actually have work to do
  if (indices.empty())
  {
    return false;
  }

  geometry_msgs::AccelWithCovarianceStamped transformed_message;
  if (target_frame.empty())
  {
    transformed_message = acceleration;
  }
  else
  {
    transformed_message.header.frame_id = target_frame;

    if (!transformMessage(tf_buffer, acceleration, transformed_message, tf_timeout))
    {
      ROS_WARN_STREAM_DELAYED_THROTTLE(
        10.0,
        "Failed to transform acceleration message with stamp " << acceleration.header.stamp
                                                               << ". Cannot create constraint.");
      return false;
    }
  }

  // Create the acceleration variables
  auto acceleration_linear =
    fuse_variables::AccelerationLinear3DStamped::make_shared(acceleration.header.stamp, device_id);
  acceleration_linear->x() = transformed_message.accel.accel.linear.x;
  acceleration_linear->y() = transformed_message.accel.accel.linear.y;
  acceleration_linear->z() = transformed_message.accel.accel.linear.z;

  // Create the full mean vector and covariance for the constraint
  fuse_core::Vector3d accel_mean;
  accel_mean << transformed_message.accel.accel.linear.x,
                transformed_message.accel.accel.linear.y,
                transformed_message.accel.accel.linear.z;

  fuse_core::Matrix3d accel_covariance;
  accel_covariance <<
        transformed_message.accel.covariance[0], transformed_message.accel.covariance[1], transformed_message.accel.covariance[2],  // NOLINT
        transformed_message.accel.covariance[6], transformed_message.accel.covariance[7], transformed_message.accel.covariance[8],  // NOLINT
        transformed_message.accel.covariance[12], transformed_message.accel.covariance[13], transformed_message.accel.covariance[14];  // NOLINT

  // Build the sub-vector and sub-matrices based on the requested indices
  fuse_core::VectorXd accel_mean_partial(indices.size());
  fuse_core::MatrixXd accel_covariance_partial(accel_mean_partial.rows(), accel_mean_partial.rows());

  populatePartialMeasurement(accel_mean, accel_covariance, indices, accel_mean_partial, accel_covariance_partial);

  if (validate)
  {
    try
    {
      validatePartialMeasurement(accel_mean_partial, accel_covariance_partial);
    }
    catch (const std::runtime_error& ex)
    {
      ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid partial linear acceleration measurement from '"
                                          << source << "' source: " << ex.what());
      return false;
    }
  }

  // Create the constraint
  auto linear_accel_constraint = fuse_constraints::AbsoluteAccelerationLinear3DStampedConstraint::make_shared(
    source,
    *acceleration_linear,
    accel_mean_partial,
    accel_covariance_partial,
    indices);

  linear_accel_constraint->loss(loss);

  transaction.addVariable(acceleration_linear);
  transaction.addConstraint(linear_accel_constraint);
  transaction.addInvolvedStamp(acceleration.header.stamp);

  return true;
}

/**
 * @brief Scales the process noise covariance pose by the norm of the velocity
 *
 * @param[in, out] process_noise_covariance - The process noise covariance to scale. Only the pose components (x, y,
 *                                            yaw) are scaled, and they are assumed to be in the top left 3x3 corner
 * @param[in] velocity_linear - The linear velocity
 * @param[in] velocity_yaw - The yaw velocity
 * @param[in] velocity_norm_min - The minimum velocity norm
 */
inline void scaleProcessNoiseCovariance(fuse_core::Matrix8d& process_noise_covariance,
                                        const tf2_2d::Vector2& velocity_linear, const double velocity_yaw,
                                        const double velocity_norm_min)
{
  // A more principled approach would be to get the current velocity from the state, make a diagonal matrix from it,
  // and then rotate it to be in the world frame (i.e., the same frame as the pose data). We could then use this
  // rotated velocity matrix to scale the process noise covariance for the pose variables as
  // rotatedVelocityMatrix * poseCovariance * rotatedVelocityMatrix'
  // However, this presents trouble for robots that may incur rotational error as a result of linear motion (and
  // vice-versa). Instead, we create a diagonal matrix whose diagonal values are the vector norm of the state's
  // velocity. We use that to scale the process noise covariance.
  //
  // The comment above has been taken from:
  // https://github.com/cra-ros-pkg/robot_localization/blob/melodic-devel/src/filter_base.cpp#L138-L144
  //
  // We also need to make sure the norm is not zero, because otherwise the resulting process noise covariance for the
  // pose becomes zero and we get NaN when we compute the inverse to obtain the information
  fuse_core::Matrix3d velocity;
  velocity.setIdentity();
  velocity.diagonal() *=
      std::max(velocity_norm_min, fuse_core::Vector3d(velocity_linear.x(), velocity_linear.y(), velocity_yaw).norm());

  process_noise_covariance.topLeftCorner<3, 3>() =
      velocity * process_noise_covariance.topLeftCorner<3, 3>() * velocity.transpose();
}

/**
 * @brief Scales the process noise covariance pose by the norm of the velocity
 *
 * @param[in, out] process_noise_covariance - The process noise covariance to scale. Only the pose components (x, y,
 *                                            z, roll, pitch, yaw) are scaled, and they are assumed to be in the top
 *                                            left 6x6 corner
 * @param[in] velocity_linear - The linear velocity
 * @param[in] velocity_angular - The angular velocity
 * @param[in] velocity_norm_min - The minimum velocity norm
 */
inline void scaleProcessNoiseCovariance(fuse_core::Matrix15d& process_noise_covariance,
                                        const tf2::Vector3& velocity_linear, const tf2::Vector3& velocity_angular,
                                        const double velocity_norm_min)
{
  // A more principled approach would be to get the current velocity from the state, make a diagonal matrix from it,
  // and then rotate it to be in the world frame (i.e., the same frame as the pose data). We could then use this
  // rotated velocity matrix to scale the process noise covariance for the pose variables as
  // rotatedVelocityMatrix * poseCovariance * rotatedVelocityMatrix'
  // However, this presents trouble for robots that may incur rotational error as a result of linear motion (and
  // vice-versa). Instead, we create a diagonal matrix whose diagonal values are the vector norm of the state's
  // velocity. We use that to scale the process noise covariance.
  //
  // The comment above has been taken from:
  // https://github.com/cra-ros-pkg/robot_localization/blob/melodic-devel/src/filter_base.cpp#L138-L144
  //
  // We also need to make sure the norm is not zero, because otherwise the resulting process noise covariance for the
  // pose becomes zero and we get NaN when we compute the inverse to obtain the information
  fuse_core::Matrix6d velocity;
  velocity.setIdentity();

  fuse_core::Vector6d velocity_values;
  velocity_values << velocity_linear.x(), velocity_linear.y(), velocity_linear.z(),
                     velocity_angular.x(), velocity_angular.y(), velocity_angular.z();

  velocity.diagonal() *= std::max(velocity_norm_min, velocity_values.norm());

  process_noise_covariance.topLeftCorner<6, 6>() =
      velocity * process_noise_covariance.topLeftCorner<6, 6>() * velocity.transpose();
}

}  // namespace common

}  // namespace fuse_models

#endif  // FUSE_MODELS_COMMON_SENSOR_PROC_H
