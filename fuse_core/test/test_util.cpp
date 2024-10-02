/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2020, Clearpath Robotics
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
#include <fuse_core/util.h>
#include <ros/ros.h>

#include <gtest/gtest.h>
#include <fuse_core/eigen.h>
#include <fuse_core/eigen_gtest.h>

#include <cmath>

#include <covariance_geometry/pose_composition.hpp>
#include <covariance_geometry/pose_covariance_representation.hpp>
#include <covariance_geometry/pose_covariance_composition.hpp>

TEST(Util, wrapAngle2D)
{
  // Wrap angle already in [-Pi, +Pi) range
  {
    const double angle = 0.5;
    EXPECT_EQ(angle, fuse_core::wrapAngle2D(angle));
  }

  // Wrap angle equal to +Pi
  {
    const double angle = M_PI;
    EXPECT_EQ(-angle, fuse_core::wrapAngle2D(angle));
  }

  // Wrap angle equal to -Pi
  {
    const double angle = -M_PI;
    EXPECT_EQ(angle, fuse_core::wrapAngle2D(angle));
  }

  // Wrap angle greater than +Pi
  {
    const double angle = 0.5;
    EXPECT_EQ(angle, fuse_core::wrapAngle2D(angle + 3.0 * 2.0 * M_PI));
  }

  // Wrap angle smaller than -Pi
  {
    const double angle = 0.5;
    EXPECT_EQ(angle, fuse_core::wrapAngle2D(angle - 3.0 * 2.0 * M_PI));
  }
}

TEST(Util, jacobianRpyQuatYawRotation)
{
  const double rpy[3] = {  // NOLINT(whitespace/braces)
    0.0,
    0.0,
    90 * M_PI / 180.
  };

  double q_computed[4];
  double rpy_computed[3];

  double J_rpy2q[12];
  double J_q2rpy[12];

  fuse_core::rpy2quaternion(rpy, q_computed, J_rpy2q);

  fuse_core::quaternion2rpy(q_computed, rpy_computed, J_q2rpy);

  Eigen::Map<fuse_core::Matrix<double, 4, 3>> j_rpy2q_map(J_rpy2q);
  Eigen::Map<fuse_core::Matrix<double, 3, 4>> j_q2rpy_map(J_q2rpy);

  EXPECT_NEAR(q_computed[0], 0.7071068, 1e-6);
  EXPECT_NEAR(q_computed[1], 0, 1e-6);
  EXPECT_NEAR(q_computed[2], 0, 1e-6);
  EXPECT_NEAR(q_computed[3], 0.7071068, 1e-6);

  EXPECT_NEAR(rpy[0], rpy_computed[0], 1e-15);
  EXPECT_NEAR(rpy[1], rpy_computed[1], 1e-15);
  EXPECT_NEAR(rpy[2], rpy_computed[2], 1e-15);
}

TEST(Util, jacobianRpyQuatFullRotation)
{
  const double rpy[3] = {  // NOLINT(whitespace/braces)
    0.1,
    0.2,
    0.3
  };

  double q_computed[4];
  double rpy_computed[3];

  double J_rpy2q[12];
  double J_q2rpy[12];

  fuse_core::rpy2quaternion(rpy, q_computed, J_rpy2q);

  fuse_core::quaternion2rpy(q_computed, rpy_computed, J_q2rpy);

  Eigen::Map<fuse_core::Matrix<double, 4, 3>> j_rpy2q_map(J_rpy2q);
  Eigen::Map<fuse_core::Matrix<double, 3, 4>> j_q2rpy_map(J_q2rpy);

  EXPECT_NEAR(rpy[0], rpy_computed[0], 1e-15);
  EXPECT_NEAR(rpy[1], rpy_computed[1], 1e-15);
  EXPECT_NEAR(rpy[2], rpy_computed[2], 1e-15);
}

TEST(Util, convertToPoseQuatCovariance)
{
  Eigen::Isometry3d p1 = Eigen::Isometry3d::Identity();
  p1.translation() = Eigen::Vector3d(1, 2, 3);
  p1.rotate(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()));

  fuse_core::Vector6d cov1_diagonal;
  cov1_diagonal << 0.1, 0.01, 0.001, 0.01, 0.2, 0.002;

  fuse_core::Matrix6d cov1(cov1_diagonal.asDiagonal());

  // fuse's implementation
  const auto cov1_quat = fuse_core::convertToPoseQuatCovariance(p1, cov1);

  const auto cov1_quat_to_rpy = fuse_core::convertToPoseRPYCovariance(p1, cov1_quat);

  // covariance_geometry as reference
  // NOTE THAT AT COVARIANCE_GEOMETRY THE ORDER AT THE ORIENTATION COVARIANCE IS (X, Y, Z, W)
  // COMPARED TO (W, X, Y, Z) AT FUSE!

  fuse_core::Matrix7d cov1_quat_in_cg_format = cov1_quat;
  cov1_quat_in_cg_format.block<3, 3>(3, 3) = cov1_quat.block<3, 3>(4, 4);
  cov1_quat_in_cg_format(6, 6) = cov1_quat(3, 3);
  cov1_quat_in_cg_format.block<3, 1>(3, 6) = cov1_quat.block<3, 1>(4, 3);
  cov1_quat_in_cg_format.block<1, 3>(6, 3) = cov1_quat.block<1, 3>(3, 4);

  covariance_geometry::PoseQuaternionCovarianceRPY cg_p1;
  cg_p1.first.first = Eigen::Vector3d(p1.translation());
  cg_p1.first.second = Eigen::Quaterniond(p1.rotation());
  cg_p1.second = cov1;

  covariance_geometry::PoseQuaternionCovariance cg_p1_q;
  covariance_geometry::Pose3DQuaternionCovarianceRPYTo3DQuaternionCovariance(cg_p1, cg_p1_q);

  covariance_geometry::PoseQuaternionCovariance cg_p1_q_initialized_from_fuse;
  cg_p1_q_initialized_from_fuse.first.first = Eigen::Vector3d(p1.translation());
  cg_p1_q_initialized_from_fuse.first.second = Eigen::Quaterniond(p1.rotation());
  cg_p1_q_initialized_from_fuse.second = cov1_quat_in_cg_format;

  covariance_geometry::PoseQuaternionCovarianceRPY cg_p1_q_initialized_from_fuse_to_rpy;
  covariance_geometry::Pose3DQuaternionCovarianceTo3DQuaternionCovarianceRPY(
    cg_p1_q_initialized_from_fuse, cg_p1_q_initialized_from_fuse_to_rpy);

  EXPECT_MATRIX_NEAR(p1.translation(), cg_p1_q.first.first, 1e-15);
  EXPECT_MATRIX_NEAR(Eigen::Quaterniond(p1.rotation()).coeffs(), cg_p1_q.first.second.coeffs(), 1e-15);

  EXPECT_MATRIX_NEAR(cov1_quat_in_cg_format, cg_p1_q.second, 1e-15);
  EXPECT_MATRIX_NEAR(cov1_quat_to_rpy, cg_p1_q_initialized_from_fuse_to_rpy.second, 1e-15);
  EXPECT_MATRIX_NEAR(cov1, cov1_quat_to_rpy, 1e-15);
}

TEST(Util, invertPoseCovariance)
{
  Eigen::Isometry3d p1 = Eigen::Isometry3d::Identity();
  p1.translation() = Eigen::Vector3d(1, 2, 3);
  p1.rotate(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()));

  fuse_core::Vector6d cov1_diagonal;
  cov1_diagonal << 0.1, 0.01, 0.001, 0.01, 0.2, 0.002;

  fuse_core::Matrix6d cov1(cov1_diagonal.asDiagonal());

  // fuse's implementation
  const auto p1_inverse = p1.inverse();
  const auto cov1_inverse = fuse_core::invertPoseCovariance(p1, cov1);

  // covariance_geometry as reference

  covariance_geometry::PoseQuaternionCovarianceRPY cg_p1;
  cg_p1.first.first = Eigen::Vector3d(p1.translation());
  cg_p1.first.second = Eigen::Quaterniond(p1.rotation());
  cg_p1.second = cov1;

  auto cg_p1_inverted = covariance_geometry::InversePose3DQuaternionCovarianceRPY(cg_p1);

  EXPECT_MATRIX_NEAR(p1_inverse.translation(), cg_p1_inverted.first.first, 1e-15);
  EXPECT_MATRIX_NEAR(Eigen::Quaterniond(p1_inverse.rotation()).coeffs(), cg_p1_inverted.first.second.coeffs(), 1e-15);
  EXPECT_MATRIX_NEAR(cov1_inverse, cg_p1_inverted.second, 1e-15);
}

TEST(Util, composePoseCovariance)
{
  Eigen::Isometry3d p1 = Eigen::Isometry3d::Identity();
  p1.translation() = Eigen::Vector3d(1, 2, 3);
  p1.rotate(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()));

  fuse_core::Vector6d cov1_diagonal;
  cov1_diagonal << 0.1, 0.01, 0.001, 0.01, 0.2, 0.002;

  fuse_core::Matrix6d cov1(cov1_diagonal.asDiagonal());

  Eigen::Isometry3d p2 = Eigen::Isometry3d::Identity();
  p2.translation() = Eigen::Vector3d(4, 5, 6);
  p2.rotate(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitY()));

  fuse_core::Vector6d cov2_diagonal;
  cov2_diagonal << 0.2, 0.02, 0.002, 0.02, 0.4, 0.004;

  fuse_core::Matrix6d cov2(cov2_diagonal.asDiagonal());

  // fuse's implementation
  const auto cov3 = fuse_core::composePoseCovariance(
    p1, cov1,
    p2, cov2);
  const auto p3 = p1 * p2;

  // covariance_geometry as reference
  covariance_geometry::PoseQuaternionCovarianceRPY cg_p1;
  cg_p1.first.first = Eigen::Vector3d(p1.translation());
  cg_p1.first.second = Eigen::Quaterniond(p1.rotation());
  cg_p1.second = cov1;

  covariance_geometry::PoseQuaternionCovarianceRPY cg_p2;
  cg_p2.first.first = Eigen::Vector3d(p2.translation());
  cg_p2.first.second = Eigen::Quaterniond(p2.rotation());
  cg_p2.second = cov2;

  covariance_geometry::PoseQuaternionCovarianceRPY cg_p3;

  covariance_geometry::ComposePoseQuaternionCovarianceRPY(cg_p1, cg_p2, cg_p3);

  EXPECT_MATRIX_NEAR(p3.translation(), cg_p3.first.first, 1e-12);
  EXPECT_MATRIX_NEAR(Eigen::Quaterniond(p3.rotation()).coeffs(), cg_p3.first.second.coeffs(), 1e-12);
  EXPECT_MATRIX_NEAR(cov3, cg_p3.second, 1e-12);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
