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
#ifndef FUSE_CORE_UTIL_H
#define FUSE_CORE_UTIL_H

#include <ros/console.h>
#include <ros/node_handle.h>

#include <ceres/jet.h>
#include <ceres/rotation.h>
#include <Eigen/Core>

#include <cmath>

#include <fuse_core/eigen.h>


namespace fuse_core
{

/**
 * @brief Returns the Euler pitch angle from a quaternion
 *
 * @param[in] w The quaternion real-valued component
 * @param[in] x The quaternion x-axis component
 * @param[in] y The quaternion y-axis component
 * @param[in] z The quaternion z-axis component
 * @return      The quaternion's Euler pitch angle component
 */
template <typename T>
static inline T getPitch(const T w, const T x, const T y, const T z)
{
  // Adapted from https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
  const T sin_pitch = T(2.0) * (w * y - z * x);

  if (ceres::abs(sin_pitch) >= T(1.0))
  {
    return (sin_pitch >= T(0.0) ? T(1.0) : T(-1.0)) * T(M_PI / 2.0);
  }
  else
  {
    return ceres::asin(sin_pitch);
  }
}

/**
 * @brief Returns the Euler roll angle from a quaternion
 *
 * @param[in] w The quaternion real-valued component
 * @param[in] x The quaternion x-axis component
 * @param[in] y The quaternion y-axis component
 * @param[in] z The quaternion z-axis component
 * @return      The quaternion's Euler roll angle component
 */
template <typename T>
static inline T getRoll(const T w, const T x, const T y, const T z)
{
  // Adapted from https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
  const T sin_roll = T(2.0) * (w * x + y * z);
  const T cos_roll = T(1.0) - (T(2.0) * (x * x + y * y));
  return ceres::atan2(sin_roll, cos_roll);
}

/**
 * @brief Returns the Euler yaw angle from a quaternion
 *
 * Returned angle is in the range [-Pi, +Pi]
 *
 * @param[in] w The quaternion real-valued component
 * @param[in] x The quaternion x-axis component
 * @param[in] y The quaternion y-axis component
 * @param[in] z The quaternion z-axis component
 * @return      The quaternion's Euler yaw angle component
 */
template <typename T>
static inline T getYaw(const T w, const T x, const T y, const T z)
{
  // Adapted from https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
  const T sin_yaw = T(2.0) * (w * z + x * y);
  const T cos_yaw = T(1.0) - (T(2.0) * (y * y + z * z));
  return ceres::atan2(sin_yaw, cos_yaw);
}

/**
 * @brief Wrap a 2D angle to the standard [-Pi, +Pi) range.
 *
 * @param[in/out] angle Input angle to be wrapped to the [-Pi, +Pi) range. Angle is updated by this function.
 */
template <typename T>
void wrapAngle2D(T& angle)
{
  // Define some necessary variations of PI with the correct type (double or Jet)
  static const T PI = T(M_PI);
  static const T TAU = T(2 * M_PI);
  // Handle the 1*Tau roll-over (https://tauday.com/tau-manifesto)
  // Use ceres::floor because it is specialized for double and Jet types.
  angle -= TAU * ceres::floor((angle + PI) / TAU);
}

/**
 * @brief Wrap a 2D angle to the standard (-Pi, +Pi] range.
 *
 * @param[in] angle Input angle to be wrapped to the (-Pi, +Pi] range.
 * @return The equivalent wrapped angle
 */
template <typename T>
T wrapAngle2D(const T& angle)
{
  T wrapped = angle;
  wrapAngle2D(wrapped);
  return wrapped;
}

/**
 * @brief Create an 2x2 rotation matrix from an angle
 *
 * @param[in] angle The rotation angle, in radians
 * @return          The equivalent 2x2 rotation matrix
 */
template <typename T>
Eigen::Matrix<T, 2, 2, Eigen::RowMajor> rotationMatrix2D(const T angle)
{
  const T cos_angle = ceres::cos(angle);
  const T sin_angle = ceres::sin(angle);
  Eigen::Matrix<T, 2, 2, Eigen::RowMajor> rotation;
  rotation << cos_angle, -sin_angle, sin_angle, cos_angle;
  return rotation;
}


/**
 * @brief Compute roll, pitch, and yaw from a quaternion
 *
 * @param[in] q Pointer to the quaternion array (4x1 (order w, x, y, z))
 * @param[out] rpy Pointer to the roll, pitch, yaw array (3x1)
 * @param[out] jacobian Pointer to the jacobian matrix (3x4, optional)
 */
static inline void quaternion2rpy(const double * q, double * rpy, double * jacobian = nullptr)
{
  rpy[0] = fuse_core::getRoll(q[0], q[1], q[2], q[3]);
  rpy[1] = fuse_core::getPitch(q[0], q[1], q[2], q[3]);
  rpy[2] = fuse_core::getYaw(q[0], q[1], q[2], q[3]);

  if (jacobian)
  {
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_map(jacobian);
    const double qw = q[0];
    const double qx = q[1];
    const double qy = q[2];
    const double qz = q[3];
    const double discr = qw * qy - qx * qz;
    jacobian_map.setZero();

    if (discr > 0.49999)
    {
      // pitch = 90 deg
      jacobian_map(2, 0) = (2.0 * qx) / (qw * qw * ((qx * qx / qw * qw) + 1.0));
      jacobian_map(2, 1) = -2.0 / (qw * ((qx * qx / qw * qw) + 1.0));
      return;
    }
    else if (discr < -0.49999)
    {
      // pitch = -90 deg
      jacobian_map(2, 0) = (-2.0 * qx) / (qw * qw * ((qx * qx / qw * qw) + 1.0));
      jacobian_map(2, 1) = 2.0 / (qw * ((qx * qx / qw * qw) + 1.0));
      return;
    }
    else
    {
      // Non-degenerate case:
      jacobian_map(0, 0) =
        -(2.0 * qx) /
        ((std::pow((2.0 * qw * qx + 2.0 * qy * qz), 2.0) / std::pow((2.0 * qx * qx + 2.0 * qy * qy - 1.0), 2.0) +
        1.0) *
        (2.0 * qx * qx + 2.0 * qy * qy - 1.0));
      jacobian_map(0, 1) =
        -((2.0 * qw) / (2.0 * qx * qx + 2.0 * qy * qy - 1.0) -
        (4.0 * qx * (2.0 * qw * qx + 2.0 * qy * qz)) / std::pow((2.0 * qx * qx + 2.0 * qy * qy - 1.0), 2.0)) /
        (std::pow((2.0 * qw * qx + 2.0 * qy * qz), 2.0) / std::pow((2.0 * qx * qx + 2.0 * qy * qy - 1.0), 2.0) + 1.0);
      jacobian_map(0, 2) =
        -((2.0 * qz) / (2.0 * qx * qx + 2.0 * qy * qy - 1.0) -
        (4.0 * qy * (2.0 * qw * qx + 2.0 * qy * qz)) / std::pow((2.0 * qx * qx + 2.0 * qy * qy - 1.0), 2.0)) /
        (std::pow((2.0 * qw * qx + 2.0 * qy * qz), 2.0) / std::pow((2.0 * qx * qx + 2.0 * qy * qy - 1.0), 2.0) + 1.0);
      jacobian_map(0, 3) =
        -(2.0 * qy) /
        ((std::pow((2.0 * qw * qx + 2.0 * qy * qz), 2.0) / std::pow((2.0 * qx * qx + 2.0 * qy * qy - 1.0), 2.0) +
        1.0) *
        (2.0 * qx * qx + 2.0 * qy * qy - 1.0));

      jacobian_map(1, 0) = (2.0 * qy) / std::sqrt(1.0 - std::pow((2.0 * qw * qy - 2.0 * qx * qz), 2.0));
      jacobian_map(1, 1) = -(2.0 * qz) / std::sqrt(1.0 - std::pow((2.0 * qw * qy - 2.0 * qx * qz), 2.0));
      jacobian_map(1, 2) = (2.0 * qw) / std::sqrt(1.0 - std::pow((2.0 * qw * qy - 2.0 * qx * qz), 2.0));
      jacobian_map(1, 3) = -(2.0 * qx) / std::sqrt(1.0 - std::pow((2.0 * qw * qy - 2.0 * qx * qz), 2.0));

      jacobian_map(2, 0) =
        -(2.0 * qz) /
        ((std::pow((2.0 * qw * qz + 2.0 * qx * qy), 2.0) / std::pow((2.0 * qy * qy + 2.0 * qz * qz - 1.0), 2.0) +
        1.0) *
        (2.0 * qy * qy + 2.0 * qz * qz - 1.0));
      jacobian_map(2, 1) =
        -(2.0 * qy) /
        ((std::pow((2.0 * qw * qz + 2.0 * qx * qy), 2.0) / std::pow((2.0 * qy * qy + 2.0 * qz * qz - 1.0), 2.0) +
        1.0) *
        (2.0 * qy * qy + 2.0 * qz * qz - 1.0));
      jacobian_map(2, 2) =
        -((2.0 * qx) / (2.0 * qy * qy + 2.0 * qz * qz - 1.0) -
        (4.0 * qy * (2.0 * qw * qz + 2.0 * qx * qy)) / std::pow((2.0 * qy * qy + 2.0 * qz * qz - 1.0), 2.0)) /
        (std::pow((2.0 * qw * qz + 2.0 * qx * qy), 2.0) / std::pow((2.0 * qy * qy + 2.0 * qz * qz - 1.0), 2.0) + 1.0);
      jacobian_map(2, 3) =
        -((2.0 * qw) / (2.0 * qy * qy + 2.0 * qz * qz - 1.0) -
        (4.0 * qz * (2.0 * qw * qz + 2.0 * qx * qy)) / std::pow((2.0 * qy * qy + 2.0 * qz * qz - 1.0), 2.0)) /
        (std::pow((2.0 * qw * qz + 2.0 * qx * qy), 2.0) / std::pow((2.0 * qy * qy + 2.0 * qz * qz - 1.0), 2.0) + 1.0);
    }
  }
}

/**
 * @brief Compute a quaternion from roll, pitch, and yaw
 *
 * @param[in] rpy Pointer to the roll, pitch, yaw array (3x1)
 * @param[out] q Pointer to the quaternion array (4x1 (order w, x, y, z))
 * @param[out] jacobian Pointer to the jacobian matrix (4x3, optional)
 */
static inline void rpy2quaternion(const double * rpy, double * q, double * jacobian = nullptr)
{
  const double ccc = cos(rpy[0] / 2.) * cos(rpy[1] / 2.) * cos(rpy[2] / 2.);
  const double ccs = cos(rpy[0] / 2.) * cos(rpy[1] / 2.) * sin(rpy[2] / 2.);
  const double csc = cos(rpy[0] / 2.) * sin(rpy[1] / 2.) * cos(rpy[2] / 2.);
  const double scc = sin(rpy[0] / 2.) * cos(rpy[1] / 2.) * cos(rpy[2] / 2.);
  const double ssc = sin(rpy[0] / 2.) * sin(rpy[1] / 2.) * cos(rpy[2] / 2.);
  const double sss = sin(rpy[0] / 2.) * sin(rpy[1] / 2.) * sin(rpy[2] / 2.);
  const double scs = sin(rpy[0] / 2.) * cos(rpy[1] / 2.) * sin(rpy[2] / 2.);
  const double css = cos(rpy[0] / 2.) * sin(rpy[1] / 2.) * sin(rpy[2] / 2.);

  q[0] = ccc + sss;
  q[1] = scc - css;
  q[2] = csc + scs;
  q[3] = ccs - ssc;

  if (jacobian)
  {
    Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> jacobian_map(jacobian);

    // dqw/d(rpy)
    jacobian_map.row(0) <<
      (css - scc) / 2.,  // droll
      (scs - csc) / 2.,  // dpitch
      (ssc - ccs) / 2.;  // dyaw

    // dqx/d(rpy)
    jacobian_map.row(1) <<
      (ccc + sss) / 2.,  // droll
      -(ssc + ccs) / 2.,  // dpitch
      -(csc + scs) / 2.;  // dyaw

    // dqy/d(rpy)
    jacobian_map.row(2) <<
      (ccs - ssc) / 2.,  // droll
      (ccc - sss) / 2.,  // dpitch
      (scc - css) / 2.;  // dyaw

    // dqz/d(rpy)
    jacobian_map.row(3) <<
      -(csc + scs) / 2.,  // droll
      -(css + scc) / 2.,  // dpitch
      (ccc + sss) / 2.;  // dyaw
  }
}

/**
 * @brief Compute product of two quaternions and the function jacobian
 * TODO(giafranchini): parametric jacobian computation? Atm this function is only used in
 * normal_prior_pose_3d cost function. There we only need the derivatives wrt quaternion W,
 * so at the time we are only computing the jacobian wrt W
 * 
 * @param[in] z Pointer to the first quaternion array (4x1 (order w, x, y, z))
 * @param[in] w Pointer to the second quaternion array  (4x1 (order w, x, y, z))
 * @param[in] z Pointer to the first quaternion array  (4x1 (order w, x, y, z))
 * @param[in] jacobian Pointer to the jacobian matrix (4x4, optional)
 */
static inline void quaternionProduct(const double * z, const double * w, double * zw, double * jacobian = nullptr)
{
  ceres::QuaternionProduct(z, w, zw);
  if (jacobian)
  {
    Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> jacobian_map(jacobian);
    jacobian_map <<
      z[0], -z[1], -z[2], -z[3],
      z[1],  z[0], -z[3],  z[2],
      z[2],  z[3],  z[0], -z[1],
      z[3], -z[2],  z[1],  z[0];
  }
}

/**
 * @brief Compute quaternion to AngleAxis conversion and the function jacobian
 *
 * @param[in] q Pointer to the quaternion array  (4x1 (order w, x, y, z))
 * @param[in] angle_axis Pointer to the angle_axis array (3x1)
 * @param[in] jacobian Pointer to the jacobian matrix (3x4, optional)
 */
static inline void quaternionToAngleAxis(const double * q, double * angle_axis, double * jacobian = nullptr)
{
  ceres::QuaternionToAngleAxis(q, angle_axis);
  if (jacobian)
  {
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_map(jacobian);
    const double & q0 = q[0];
    const double & q1 = q[1];
    const double & q2 = q[2];
    const double & q3 = q[3];
    const double q_sum2 = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3;
    const double sin_theta2 = q1 * q1 + q2 * q2 + q3 * q3;
    const double sin_theta = std::sqrt(sin_theta2);
    const double cos_theta = q0;

    if (std::fpclassify(sin_theta) != FP_ZERO)
    {
      const double two_theta = 2.0 *
        (cos_theta < 0.0 ? std::atan2(-sin_theta, -cos_theta) : std::atan2(sin_theta, cos_theta));
      jacobian_map(0, 0) = -2.0 * q1 / q_sum2;
      jacobian_map(0, 1) =
        two_theta / sin_theta +
        (2.0 * q0 * q1 * q1) / (sin_theta2 * q_sum2) -
        (q1 * q1 * two_theta) / std::pow(sin_theta2, 1.5);
      jacobian_map(0, 2) =
        (2.0 * q0 * q1 * q2) / (sin_theta2 * q_sum2) -
        (q1 * q2 * two_theta) / std::pow(sin_theta2, 1.5);
      jacobian_map(0, 3) =
        (2.0 * q0 * q1 * q3) / (sin_theta2 * q_sum2) -
        (q1 * q3 * two_theta) / std::pow(sin_theta2, 1.5);

      jacobian_map(1, 0) = -2.0 * q2 / q_sum2;
      jacobian_map(1, 1) =
        (2.0 * q0 * q1 * q2) / (sin_theta2 * q_sum2) -
        (q1 * q2 * two_theta) / std::pow(sin_theta2, 1.5);
      jacobian_map(1, 2) =
        two_theta / sin_theta +
        (2.0 * q0 * q2 * q2) / (sin_theta2 * q_sum2) -
        (q2 * q2 * two_theta) / std::pow(sin_theta2, 1.5);
      jacobian_map(1, 3) =
        (2.0 * q0 * q2 * q3) / (sin_theta2 * q_sum2) -
        (q2 * q3 * two_theta) / std::pow(sin_theta2, 1.5);

      jacobian_map(2, 0) = -2.0 * q3 / q_sum2;
      jacobian_map(2, 1) =
        (2.0 * q0 * q1 * q3) / (sin_theta2 * q_sum2) -
        (q1 * q3 * two_theta) / std::pow(sin_theta2, 1.5);
      jacobian_map(2, 2) =
        (2.0 * q0 * q2 * q3) / (sin_theta2 * q_sum2) -
        (q2 * q3 * two_theta) / std::pow(sin_theta2, 1.5);
      jacobian_map(2, 3) =
        two_theta / sin_theta +
        (2.0 * q0 * q3 * q3) / (sin_theta2 * q_sum2) -
        (q3 * q3 * two_theta) / std::pow(sin_theta2, 1.5);
    }
    else
    {
      jacobian_map.setZero();
      jacobian_map(1, 1) = 2.0;
      jacobian_map(2, 2) = 2.0;
      jacobian_map(3, 3) = 2.0;
    }
  }
}

/**
 * @brief Compute the jacobian of a quaternion normalization
 *
 * @param[in] q The quaternion
 * @return      The jacobian of the quaternion normalization
 */
static inline fuse_core::Matrix4d jacobianQuatNormalization(Eigen::Quaterniond q)
{
  fuse_core::Matrix4d ret;
  ret <<
     q.x() * q.x() + q.y() * q.y() + q.z() * q.z(), -q.w() * q.x(), -q.w() * q.y(), -q.w() * q.z(),
     -q.x() * q.w(), q.w() * q.w() + q.y() * q.y() + q.z() * q.z(), -q.x() * q.y(), -q.x() * q.z(),
     -q.y() * q.w(), -q.y() * q.x(), q.w() * q.w() + q.x() * q.x() + q.z() * q.z(), -q.y() * q.z(),
     -q.z() * q.w(), -q.z() * q.x(), -q.z() * q.y(), q.w() * q.w() + q.x() * q.x() + q.y() * q.y();

  ret /= std::pow(q.norm(), 3.);
  return ret;
}

/**
 * @brief Convert a pose covariance in x, y, z, roll, pitch, yaw to a pose covariance in x, y, z, qw, qx, qy, qz
 *
 * @param[in] pose    The pose which covariance should be converted
 * @param[in] cov_rpy The covariance in x, y, z, roll, pitch, yaw
 * @return            The pose covariance in x, y, z, qw, qx, qy, qz
 */
static inline fuse_core::Matrix7d convertToPoseQuatCovariance(
  const Eigen::Isometry3d& pose, const fuse_core::Matrix6d& cov_rpy)
{
  double rpy[3];
  Eigen::Quaterniond q(pose.rotation());
  double q_array[4] = {  // NOLINT(whitespace/braces)
    q.w(),
    q.x(),
    q.y(),
    q.z()
  };
  quaternion2rpy(q_array, rpy);

  double J_rpy2quat[12];
  rpy2quaternion(rpy, q_array, J_rpy2quat);

  Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> J_rpy2quat_map(J_rpy2quat);

  Eigen::Matrix<double, 7, 6, Eigen::RowMajor> J;
  J.topLeftCorner<3, 3>().setIdentity();
  J.topRightCorner<3, 3>().setZero();
  J.bottomLeftCorner<4, 3>().setZero();
  J.bottomRightCorner<4, 3>() = J_rpy2quat_map;

  return J * cov_rpy * J.transpose();
}

/**
 * @brief Convert a pose covariance in x, y, z, qw, qx, qy, qz to a pose covariance in x, y, z, roll, pitch, yaw
 *
 * @param[in] pose    The pose which covariance should be converted
 * @param[in] cov_rpy The covariance in x, y, z, qw, qx, qy, qz
 * @return            The pose covariance in x, y, z, roll, pitch, yaw
 */
static inline fuse_core::Matrix6d convertToPoseRPYCovariance(
  const Eigen::Isometry3d& pose, const fuse_core::Matrix7d& cov_quat)
{
  Eigen::Quaterniond q(pose.rotation());
  double q_array[4] = {  // NOLINT(whitespace/braces)
    q.w(),
    q.x(),
    q.y(),
    q.z()
  };
  double rpy[3];
  double J_quat2rpy[12];
  quaternion2rpy(q_array, rpy, J_quat2rpy);
  Eigen::Map<fuse_core::Matrix<double, 3, 4>> J_quat2rpy_map(J_quat2rpy);
  Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J;
  J.topLeftCorner<3, 3>().setIdentity();
  J.topRightCorner<3, 4>().setZero();
  J.bottomLeftCorner<3, 3>().setZero();
  J.bottomRightCorner<3, 4>() = J_quat2rpy_map * jacobianQuatNormalization(q);

  return J * cov_quat * J.transpose();
}

/**
 * @brief Compute the pose covariance of an inverted pose
 *
 * @param[in] pose The pose which will be inverted
 * @param[in] cov  The covariance of the pose which will be inverted (order: x, y, z, roll, pitch, yaw)
 * @return         The pose covariance of the inverted pose (order: x, y, z, roll, pitch, yaw)
 */
static inline fuse_core::Matrix6d invertPoseCovariance(
  const Eigen::Isometry3d& pose, const fuse_core::Matrix6d& cov)
{
  // convert the covariances from 3D + roll-pitch-yaw to 3D + quaternion
  auto cov_quat = convertToPoseQuatCovariance(pose, cov);

  // compute the inverse pose covariance with 3D + quaternion
  const Eigen::Quaterniond q(pose.rotation());

  const double dx = (0 - pose.translation().x());
  const double dy = (0 - pose.translation().y());
  const double dz = (0 - pose.translation().z());
  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> fqrir;
  fqrir <<
    -q.y() * dz + q.z() * dy, q.y() * dy + q.z() * dz, q.x() * dy - 2 * q.y() * dx - q.w() * dz, q.x() * dz + q.w() * dy - 2 * q.z() * dx,  // NOLINT(whitespace/line_length)
    q.x() * dz - q.z() * dx, q.y() * dx - 2 * q.x() * dy + q.w() * dz, q.x() * dx + q.z() * dz, -q.w() * dx - 2 * q.z() * dy + q.y() * dz,  // NOLINT(whitespace/line_length)
    q.y() * dx - q.x() * dy, q.z() * dx - q.w() * dy - 2 * q.x() * dz, q.z() * dy + q.w() * dx - 2 * q.y() * dz, q.x() * dx + q.y() * dy;  // NOLINT(whitespace/line_length)
  fqrir *= 2;
  fqrir.applyOnTheRight(jacobianQuatNormalization(q));


  Eigen::Matrix<double, 3, 7, Eigen::RowMajor> fqri;
  fqri.leftCols<3>() <<
    2 * q.y() * q.y() + 2 * q.z() * q.z() - 1, -2 * q.w() * q.z() - 2 * q.x() * q.y(), 2 * q.w() * q.y() - 2 * q.x() * q.z(),  // NOLINT(whitespace/line_length)
    2 * q.w() * q.z() - 2 * q.x() * q.y(), 2 * q.x() * q.x() + 2 * q.z() * q.z() - 1, -2 * q.w() * q.x() - 2 * q.y() * q.z(),  // NOLINT(whitespace/line_length)
    -2 * q.w() * q.y() - 2 * q.x() * q.z(), 2 * q.w() * q.x() - 2 * q.y() * q.z(), 2 * q.x() * q.x() + 2 * q.y() * q.y() - 1;  // NOLINT(whitespace/line_length)
  fqri.rightCols<4>() = fqrir;

  fuse_core::Matrix7d fqi;
  fqi.topRows<3>() = fqri;
  fqi.bottomLeftCorner<4, 3>().setZero();
  fqi.bottomRightCorner<4, 4>() <<
    1,  0,  0,  0,
    0, -1,  0,  0,
    0,  0, -1,  0,
    0,  0,  0, -1;
  fqi.bottomRightCorner<4, 4>().applyOnTheRight(jacobianQuatNormalization(q));

  fuse_core::Matrix7d cov_inverse_quat = fqi * cov_quat * fqi.transpose();

  // convert back to 3D + roll-pitch-yaw
  const auto pose_inverse = pose.inverse();
  return convertToPoseRPYCovariance(pose_inverse, cov_inverse_quat);
}

/**
 * @brief Compute the jacobian of a pose composition wrt pose 1 (composition: pose3 = pose1 + pose2)
 *
 * @param[in] pose1 The pose 1 of the pose composition
 * @param[in] pose2 The pose 2 of the pose composition
 * @return          The jacobian of the pose composition wrt pose 1 (order: x, y, z, qw, qx, qy, qz)
 */
static inline fuse_core::Matrix7d jacobianPosePoseCompositionA(
  const Eigen::Isometry3d& pose1, const Eigen::Isometry3d& pose2)
{
  const Eigen::Quaterniond q1(pose1.rotation());
  const Eigen::Quaterniond q2(pose2.rotation());
  const Eigen::Vector3d a(pose2.translation());
  Eigen::Matrix<double, 3, 7, Eigen::RowMajor> fqr_pose1;
  fqr_pose1.leftCols<3>().setIdentity();

  fqr_pose1(0, 3) = -q1.z() * a.y() + q1.y() * a.z();
  fqr_pose1(0, 4) = q1.y() * a.y() + q1.z() * a.z();
  fqr_pose1(0, 5) = -2 * q1.y() * a.x() + q1.x() * a.y() + q1.w() * a.z();
  fqr_pose1(0, 6) = -2 * q1.z() * a.x() - q1.w() * a.y() + q1.x() * a.z();

  fqr_pose1(1, 3) = q1.z() * a.x() - q1.x() * a.z();
  fqr_pose1(1, 4) = q1.y() * a.x() - 2 * q1.x() * a.y() - q1.w() * a.z();
  fqr_pose1(1, 5) = q1.x() * a.x() + q1.z() * a.z();
  fqr_pose1(1, 6) = q1.w() * a.x() - 2 * q1.z() * a.y() + q1.y() * a.z();

  fqr_pose1(2, 3) = -q1.y() * a.x() + q1.x() * a.y();
  fqr_pose1(2, 4) = q1.z() * a.x() + q1.w() * a.y() - 2 * q1.x() * a.z();
  fqr_pose1(2, 5) = -q1.w() * a.x() + q1.z() * a.y() - 2 * q1.y() * a.z();
  fqr_pose1(2, 6) = q1.x() * a.x() + q1.y() * a.y();

  fqr_pose1.rightCols<4>() *= 2;

  fqr_pose1.rightCols<4>().applyOnTheRight(jacobianQuatNormalization(q1));

  fuse_core::Matrix7d fqc_pose1;
  fqc_pose1.topRows<3>() = fqr_pose1;
  fqc_pose1.bottomLeftCorner<4, 3>().setZero();
  fqc_pose1.bottomRightCorner<4, 4>() <<
    q2.w(), -q2.x(), -q2.y(), -q2.z(),
    q2.x(), q2.w(), q2.z(), -q2.y(),
    q2.y(), -q2.z(), q2.w(), q2.x(),
    q2.z(), q2.y(), -q2.x(), q2.w();

  return fqc_pose1;
}

/**
 * @brief Compute the jacobian of a pose composition wrt pose 2 (composition: pose3 = pose1 + pose2)
 *
 * @param[in] pose1 The pose 1 of the pose composition
 * @return          The jacobian of the pose composition wrt pose 2 (order: x, y, z, qw, qx, qy, qz)
 */
static inline fuse_core::Matrix7d jacobianPosePoseCompositionB(
  const Eigen::Isometry3d& pose1)
{
  const Eigen::Quaterniond q1(pose1.rotation());

  fuse_core::Matrix3d fqr_position2;
  fqr_position2 <<
    0.5 - q1.y() * q1.y() - q1.z() * q1.z(), q1.x() * q1.y() - q1.w() * q1.z(), q1.w() * q1.y() + q1.x() * q1.z(),
    q1.w() * q1.z() + q1.x() * q1.y(), 0.5 - q1.x() * q1.x() - q1.z() * q1.z(), q1.y() * q1.z() - q1.w() * q1.x(),
    q1.x() * q1.z() - q1.w() * q1.y(), q1.w() * q1.x() + q1.y() * q1.z(), 0.5 - q1.x() * q1.x() - q1.y() * q1.y();
  fqr_position2 *= 2;

  fuse_core::Matrix7d fqc_pose2;
  fqc_pose2.topLeftCorner<3, 3>() = fqr_position2;
  fqc_pose2.topRightCorner<3, 4>().setZero();
  fqc_pose2.bottomLeftCorner<4, 3>().setZero();
  fqc_pose2.bottomRightCorner<4, 4>() <<
    q1.w(), -q1.x(), -q1.y(), -q1.z(),
    q1.x(), q1.w(), -q1.z(), q1.y(),
    q1.y(), q1.z(), q1.w(), -q1.x(),
    q1.z(), -q1.y(), q1.x(), q1.w();

  return fqc_pose2;
}

/**
 * @brief Compute the covariance after a pose composition (composition: pose3 = pose1 + pose2)
 *
 * @param[in] pose1 The pose 1 of the pose composition
 * @param[in] cov1  The covariance of pose 1 of the pose composition (order: x, y, z, roll, pitch, yaw)
 * @param[in] pose2 The pose 2 of the pose composition
 * @param[in] cov2  The covariance of pose 2 of the pose composition (order: x, y, z, roll, pitch, yaw)
 * @return          The covariance after the pose composition (order: x, y, z, roll, pitch, yaw)
 */
static inline fuse_core::Matrix6d composePoseCovariance(
  const Eigen::Isometry3d& pose1, const fuse_core::Matrix6d& cov1,
  const Eigen::Isometry3d& pose2, const fuse_core::Matrix6d& cov2)
{
  const auto pose3 = pose1 * pose2;
  const Eigen::Quaterniond q3(pose3.rotation());

  // first, convert the covariances from 3D + roll-pitch-yaw to 3D + quaternion
  const auto cov1_quat = convertToPoseQuatCovariance(pose1, cov1);
  const auto cov2_quat = convertToPoseQuatCovariance(pose2, cov2);

  // now compose the two covariances
  const Eigen::Quaterniond q1(pose1.rotation());
  const Eigen::Quaterniond q2(pose2.rotation());
  const Eigen::Vector3d a(pose2.translation());

  fuse_core::Matrix7d fqc_pose1 = jacobianPosePoseCompositionA(pose1, pose2);

  fuse_core::Matrix7d fqc_pose2 = jacobianPosePoseCompositionB(pose1);

  const auto fqn_pose3 = jacobianQuatNormalization(q3);

  fuse_core::Matrix7d fqc_pose1_including_fqn = fqc_pose1;
  fqc_pose1_including_fqn.bottomRightCorner<4, 4>().applyOnTheLeft(fqn_pose3);

  fuse_core::Matrix7d fqc_pose2_including_fqn = fqc_pose2;
  fqc_pose2_including_fqn.bottomRightCorner<4, 4>().applyOnTheLeft(fqn_pose3);

  const fuse_core::Matrix7d cov3_quat =
    fqc_pose1_including_fqn * cov1_quat * fqc_pose1_including_fqn.transpose() +
    fqc_pose2_including_fqn * cov2_quat * fqc_pose2_including_fqn.transpose();

  // convert back to 3D + roll-pitch-yaw
  return convertToPoseRPYCovariance(pose3, cov3_quat);
}

}  // namespace fuse_core

#endif  // FUSE_CORE_UTIL_H
