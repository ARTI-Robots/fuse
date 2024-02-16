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
#ifndef FUSE_MODELS_UNICYCLE_3D_PREDICT_H
#define FUSE_MODELS_UNICYCLE_3D_PREDICT_H

#include <ceres/jet.h>
#include <ceres/rotation.h>
#include <fuse_core/util.h>
#include <fuse_core/eigen.h>

#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

#include <array>


namespace fuse_models
{

/**
 * @brief Given a state and time delta, predicts a new state
 * @param[in] position1_x - First X position
 * @param[in] position1_y - First Y position
 * @param[in] position1_z - First Z position
 * @param[in] roll1 - First roll orientation
 * @param[in] pitch1 - First pitch orientation
 * @param[in] yaw1 - First yaw orientation
 * @param[in] vel_linear1_x - First X velocity
 * @param[in] vel_linear1_y - First Y velocity
 * @param[in] vel_linear1_z - First Z velocity
 * @param[in] vel_roll1 - First roll velocity
 * @param[in] vel_pitch1 - First pitch velocity
 * @param[in] vel_yaw1 - First yaw velocity
 * @param[in] acc_linear1_x - First X acceleration
 * @param[in] acc_linear1_y - First Y acceleration
 * @param[in] acc_linear1_z - First Z acceleration
 * @param[in] dt - The time delta across which to predict the state
 * @param[out] position2_x - Second X position
 * @param[out] position2_y - Second Y position
 * @param[out] position2_z - Second Z position
 * @param[out] roll2 - Second roll orientation
 * @param[out] pitch2 - Second pitch orientation
 * @param[out] yaw2 - Second yaw orientation
 * @param[out] vel_linear2_x - Second X velocity
 * @param[out] vel_linear2_y - Second Y velocity
 * @param[out] vel_linear2_z - Second Z velocity
 * @param[out] vel_roll2 - Second roll velocity
 * @param[out] vel_pitch2 - Second pitch velocity
 * @param[out] vel_yaw2 - Second yaw velocity
 * @param[out] acc_linear2_x - Second X acceleration
 * @param[out] acc_linear2_y - Second Y acceleration
 * @param[out] acc_linear2_z - Second Z acceleration
 */
template<typename T>
inline void predict(
  const T position1_x,
  const T position1_y,
  const T position1_z,
  const T roll1,
  const T pitch1,
  const T yaw1,
  const T vel_linear1_x,
  const T vel_linear1_y,
  const T vel_linear1_z,
  const T vel_roll1,
  const T vel_pitch1,
  const T vel_yaw1,
  const T acc_linear1_x,
  const T acc_linear1_y,
  const T acc_linear1_z,
  const T dt,
  T& position2_x,
  T& position2_y,
  T& position2_z,
  T& roll2,
  T& pitch2,
  T& yaw2,
  T& vel_linear2_x,
  T& vel_linear2_y,
  T& vel_linear2_z,
  T& vel_roll2,
  T& vel_pitch2,
  T& vel_yaw2,
  T& acc_linear2_x,
  T& acc_linear2_y,
  T& acc_linear2_z)
{
  // There are better models for this projection, but this matches the one used by r_l.
  T sr = ceres::sin(roll1);
  T cr = ceres::cos(roll1);

  T sp = ceres::sin(pitch1);
  T cp = ceres::cos(pitch1);
  T cpi = T(1.0) / cp;
  T tp = sp * cpi;

  T sy = ceres::sin(yaw1);
  T cy = ceres::cos(yaw1);

  position2_x = position1_x + vel_linear1_x * cy * cp * dt
                            + vel_linear1_y * (cy * sp * sr - sy * cr) * dt
                            + vel_linear1_z * (cy * sp * cr + sy * sr) * dt
                            + acc_linear1_x * T(0.5) * cy * cp * dt * dt
                            + acc_linear1_y * T(0.5) * (cy * sp * sr - sy * cr) * dt * dt
                            + acc_linear1_z * T(0.5) * (cy * sp * cr + sy * sr) * dt * dt;

  position2_y = position1_y + vel_linear1_x * sy * cp * dt
                            + vel_linear1_y * (sy * sp * sr + cy * cr) * dt
                            + vel_linear1_z * (sy * sp * cr - cy * sr) * dt
                            + acc_linear1_x * T(0.5) * sy * cp * dt * dt
                            + acc_linear1_y * T(0.5) * (sy * sp * sr + cy * cr) * dt * dt
                            + acc_linear1_z * T(0.5) * (sy * sp * cr - cy * sr) * dt * dt;

  position2_z = position1_z + vel_linear1_x * (-sp) * dt
                            + vel_linear1_y * cp * sr * dt
                            + vel_linear1_z * cp * cr * dt
                            + acc_linear1_x * T(0.5) * (-sp) * dt * dt
                            + acc_linear1_y * T(0.5) * cp * sr * dt * dt
                            + acc_linear1_z * T(0.5) * cp * cr * dt * dt;

  roll2 = roll1 + vel_roll1 * dt
                + vel_pitch1 * sr * tp * dt
                + vel_yaw1 * cr * tp * dt;

  pitch2 = pitch1 + vel_pitch1 * cr * dt
                  + vel_yaw1 * (-sr) * dt;

  yaw2 = yaw1 + vel_pitch1 * sr * cpi * dt
              + vel_yaw1 * cr * cpi * dt;

  vel_linear2_x = vel_linear1_x + acc_linear1_x * dt;
  vel_linear2_y = vel_linear1_y + acc_linear1_y * dt;
  vel_linear2_z = vel_linear1_z + acc_linear1_z * dt;

  vel_roll2 = vel_roll1;
  vel_pitch2 = vel_pitch1;
  vel_yaw2 = vel_yaw1;

  acc_linear2_x = acc_linear1_x;
  acc_linear2_y = acc_linear1_y;
  acc_linear2_z = acc_linear1_z;

  fuse_core::wrapAngle2D(roll2);
  fuse_core::wrapAngle2D(pitch2);
  fuse_core::wrapAngle2D(yaw2);
}

/**
 * @brief Given a state and time delta, predicts a new state
 * @param[in] position1_x - First X position
 * @param[in] position1_y - First Y position
 * @param[in] position1_z - First Z position
 * @param[in] roll1 - First roll orientation
 * @param[in] pitch1 - First pitch orientation
 * @param[in] yaw1 - First yaw orientation
 * @param[in] vel_linear1_x - First X velocity
 * @param[in] vel_linear1_y - First Y velocity
 * @param[in] vel_linear1_z - First Z velocity
 * @param[in] vel_roll1 - First roll velocity
 * @param[in] vel_pitch1 - First pitch velocity
 * @param[in] vel_yaw1 - First yaw velocity
 * @param[in] acc_linear1_x - First X acceleration
 * @param[in] acc_linear1_y - First Y acceleration
 * @param[in] acc_linear1_z - First Z acceleration
 * @param[in] dt - The time delta across which to predict the state
 * @param[out] position2_x - Second X position
 * @param[out] position2_y - Second Y position
 * @param[out] position2_z - Second Z position
 * @param[out] roll2 - Second roll orientation
 * @param[out] pitch2 - Second pitch orientation
 * @param[out] yaw2 - Second yaw orientation
 * @param[out] vel_linear2_x - Second X velocity
 * @param[out] vel_linear2_y - Second Y velocity
 * @param[out] vel_linear2_z - Second Z velocity
 * @param[out] vel_roll2 - Second roll velocity
 * @param[out] vel_pitch2 - Second pitch velocity
 * @param[out] vel_yaw2 - Second yaw velocity
 * @param[out] acc_linear2_x - Second X acceleration
 * @param[out] acc_linear2_y - Second Y acceleration
 * @param[out] acc_linear2_z - Second Z acceleration
 * @param[out] jacobians - Jacobians wrt the state
 */
inline void predict(
  const double position1_x,
  const double position1_y,
  const double position1_z,
  const double roll1,
  const double pitch1,
  const double yaw1,
  const double vel_linear1_x,
  const double vel_linear1_y,
  const double vel_linear1_z,
  const double vel_roll1,
  const double vel_pitch1,
  const double vel_yaw1,
  const double acc_linear1_x,
  const double acc_linear1_y,
  const double acc_linear1_z,
  const double dt,
  double& position2_x,
  double& position2_y,
  double& position2_z,
  double& roll2,
  double& pitch2,
  double& yaw2,
  double& vel_linear2_x,
  double& vel_linear2_y,
  double& vel_linear2_z,
  double& vel_roll2,
  double& vel_pitch2,
  double& vel_yaw2,
  double& acc_linear2_x,
  double& acc_linear2_y,
  double& acc_linear2_z,
  double** jacobians)
{
  // There are better models for this projection, but this matches the one used by r_l.
  const double sr = ceres::sin(roll1);
  const double cr = ceres::cos(roll1);

  const double sp = ceres::sin(pitch1);
  const double cp = ceres::cos(pitch1);
  const double cpi = 1.0 / cp;
  const double tp = sp * cpi;

  const double sy = ceres::sin(yaw1);
  const double cy = ceres::cos(yaw1);

  position2_x = position1_x + vel_linear1_x * cy * cp * dt
                            + vel_linear1_y * (cy * sp * sr - sy * cr) * dt
                            + vel_linear1_z * (cy * sp * cr + sy * sr) * dt
                            + acc_linear1_x * 0.5 * cy * cp * dt * dt
                            + acc_linear1_y * 0.5 * (cy * sp * sr - sy * cr) * dt * dt
                            + acc_linear1_z * 0.5 * (cy * sp * cr + sy * sr) * dt * dt;

  position2_y = position1_y + vel_linear1_x * sy * cp * dt
                            + vel_linear1_y * (sy * sp * sr + cy * cr) * dt
                            + vel_linear1_z * (sy * sp * cr - cy * sr) * dt
                            + acc_linear1_x * 0.5 * sy * cp * dt * dt
                            + acc_linear1_y * 0.5 * (sy * sp * sr + cy * cr) * dt * dt
                            + acc_linear1_z * 0.5 * (sy * sp * cr - cy * sr) * dt * dt;

  position2_z = position1_z + vel_linear1_x * (-sp) * dt
                            + vel_linear1_y * cp * sr * dt
                            + vel_linear1_z * cp * cr * dt
                            + acc_linear1_x * 0.5 * (-sp) * dt * dt
                            + acc_linear1_y * 0.5 * cp * sr * dt * dt
                            + acc_linear1_z * 0.5 * cp * cr * dt * dt;

  roll2 = roll1 + vel_roll1 * dt
                + vel_pitch1 * sr * tp * dt
                + vel_yaw1 * cr * tp * dt;

  pitch2 = pitch1 + vel_pitch1 * cr * dt
                  + vel_yaw1 * (-sr) * dt;

  yaw2 = yaw1 + vel_pitch1 * sr * cpi * dt
              + vel_yaw1 * cr * cpi * dt;

  vel_linear2_x = vel_linear1_x + acc_linear1_x * dt;
  vel_linear2_y = vel_linear1_y + acc_linear1_y * dt;
  vel_linear2_z = vel_linear1_z + acc_linear1_z * dt;

  vel_roll2 = vel_roll1;
  vel_pitch2 = vel_pitch1;
  vel_yaw2 = vel_yaw1;

  acc_linear2_x = acc_linear1_x;
  acc_linear2_y = acc_linear1_y;
  acc_linear2_z = acc_linear1_z;

  fuse_core::wrapAngle2D(roll2);
  fuse_core::wrapAngle2D(pitch2);
  fuse_core::wrapAngle2D(yaw2);

  if (jacobians)
  {
    // Jacobian wrt position1
    if (jacobians[0])
    {
      Eigen::Map<fuse_core::Matrix<double, 15, 3>> jacobian(jacobians[0]);
      jacobian << 1, 0, 0,
                  0, 1, 0,
                  0, 0, 1,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0;
    }

    // Jacobian wrt orientation1 in roll, pitch, yaw (compared to quaternion components which are used from ceres)
    if (jacobians[1])
    {
      const auto dposx_droll = vel_linear1_y * (cy * sp * cr - sy * (-sr)) * dt
                               + vel_linear1_z * (cy * sp * (-sr) + sy * cr) * dt
                               + acc_linear1_y * 0.5 * (cy * sp * cr - sy * (-sr)) * dt * dt
                               + acc_linear1_z * 0.5 * (cy * sp * (-sr) + sy * cr) * dt * dt;

      const auto dposy_droll = vel_linear1_y * (sy * sp * cr + cy * (-sr)) * dt
                               + vel_linear1_z * (sy * sp * (-sr) - cy * cr) * dt
                               + acc_linear1_y * 0.5 * (sy * sp * cr + cy * (-sr)) * dt * dt
                               + acc_linear1_z * 0.5 * (sy * sp * (-sr) - cy * cr) * dt * dt;

      const auto dposz_droll = vel_linear1_y * cp * cr * dt
                               + vel_linear1_z * cp * (-sr) * dt
                               + acc_linear1_y * 0.5 * cp * cr * dt * dt
                               + acc_linear1_z * 0.5 * cp * (-sr) * dt * dt;

      const auto dposx_dpitch = vel_linear1_x * cy * (-sp) * dt
                                + vel_linear1_y * cy * cp * sr * dt
                                + vel_linear1_z * cy * cp * cr * dt
                                + acc_linear1_x * 0.5 * cy * (-sp) * dt * dt
                                + acc_linear1_y * 0.5 * cy * cp * sr * dt * dt
                                + acc_linear1_z * 0.5 * cy * cp * cr * dt * dt;

      const auto dposy_dpitch = vel_linear1_x * sy * (-sp) * dt
                                + vel_linear1_y * sy * cp * sr * dt
                                + vel_linear1_z * sy * cp * cr * dt
                                + acc_linear1_x * 0.5 * sy * (-sp) * dt * dt
                                + acc_linear1_y * 0.5 * sy * cp * sr * dt * dt
                                + acc_linear1_z * 0.5 * sy * cp * cr * dt * dt;

      const auto dposz_dpitch = vel_linear1_x * (-cp) * dt
                                + vel_linear1_y * (-sp) * sr * dt
                                + vel_linear1_z * (-sp) * cr * dt
                                + acc_linear1_x * 0.5 * (-cp) * dt * dt
                                + acc_linear1_y * 0.5 * (-sp) * sr * dt * dt
                                + acc_linear1_z * 0.5 * (-sp) * cr * dt * dt;

      const auto dposx_dyaw = vel_linear1_x * (-sy) * cp * dt
                              + vel_linear1_y * ((-sy) * sp * sr - cy * cr) * dt
                              + vel_linear1_z * ((-sy) * sp * cr + cy * sr) * dt
                              + acc_linear1_x * 0.5 * (-sy) * cp * dt * dt
                              + acc_linear1_y * 0.5 * ((-sy) * sp * sr - cy * cr) * dt * dt
                              + acc_linear1_z * 0.5 * ((-sy) * sp * cr + cy * sr) * dt * dt;

      const auto dposy_dyaw = vel_linear1_x * cy * cp * dt
                              + vel_linear1_y * (cy * sp * sr + (-sy) * cr) * dt
                              + vel_linear1_z * (cy * sp * cr - (-sy) * sr) * dt
                              + acc_linear1_x * 0.5 * cy * cp * dt * dt
                              + acc_linear1_y * 0.5 * (cy * sp * sr + (-sy) * cr) * dt * dt
                              + acc_linear1_z * 0.5 * (cy * sp * cr - (-sy) * sr) * dt * dt;

      const auto dposz_dyaw = 0.0;

      const auto droll_droll = 1.0
                               + vel_pitch1 * cr * tp * dt
                               + vel_yaw1 * (-sr) * tp * dt;

      const auto dpitch_droll = vel_pitch1 * (-sr) * dt
                                + vel_yaw1 * (-cr) * dt;

      const auto dyaw_droll = vel_pitch1 * cr * cpi * dt
                              + vel_yaw1 * (-sr) * cpi * dt;

      const auto droll_dpitch = vel_pitch1 * sr * (1.0 / (cp * cp)) * dt
                                + vel_yaw1 * cr * (1.0 / (cp * cp)) * dt;

      const auto dpitch_dpitch = 1.0;

      const auto dyaw_dpitch = vel_pitch1 * sr * (sp / (cp * cp)) * dt
                               + vel_yaw1 * cr * (sp / (cp * cp)) * dt;

      const auto droll_dyaw = 0.0;

      const auto dpitch_dyaw = 0.0;

      const auto dyaw_dyaw = 1.0;

      Eigen::Map<fuse_core::Matrix<double, 15, 3>> jacobian(jacobians[1]);

      jacobian << dposx_droll,  dposx_dpitch,  dposx_dyaw,
                  dposy_droll,  dposy_dpitch,  dposy_dyaw,
                  dposz_droll,  dposz_dpitch,  dposz_dyaw,
                  droll_droll,  droll_dpitch,  droll_dyaw,
                  dpitch_droll, dpitch_dpitch, dpitch_dyaw,
                  dyaw_droll,   dyaw_dpitch,   dyaw_dyaw,
                  0,            0,             0,
                  0,            0,             0,
                  0,            0,             0,
                  0,            0,             0,
                  0,            0,             0,
                  0,            0,             0,
                  0,            0,             0,
                  0,            0,             0,
                  0,            0,             0;
    }

    // Jacobian wrt vel_linear1
    if (jacobians[2])
    {
      const double dposx_dvel_linear_1_x = cy * cp * dt;
      const double dposx_dvel_linear_1_y = (cy * sp * sr - sy * cr) * dt;
      const double dposx_dvel_linear_1_z = (cy * sp * cr + sy * sr) * dt;

      const double dposy_dvel_linear_1_x = sy * cp * dt;
      const double dposy_dvel_linear_1_y = (sy * sp * sr + cy * cr) * dt;
      const double dposy_dvel_linear_1_z = (sy * sp * cr - cy * sr) * dt;

      const double dposz_dvel_linear_1_x = (-sp) * dt;
      const double dposz_dvel_linear_1_y = cp * sr * dt;
      const double dposz_dvel_linear_1_z = cp * cr * dt;

      Eigen::Map<fuse_core::Matrix<double, 15, 3>> jacobian(jacobians[2]);
      jacobian << dposx_dvel_linear_1_x, dposx_dvel_linear_1_y, dposx_dvel_linear_1_z,
                  dposy_dvel_linear_1_x, dposy_dvel_linear_1_y, dposy_dvel_linear_1_z,
                  dposz_dvel_linear_1_x, dposz_dvel_linear_1_y, dposz_dvel_linear_1_z,
                                      0,                     0,                     0,
                                      0,                     0,                     0,
                                      0,                     0,                     0,
                                      1,                     0,                     0,
                                      0,                     1,                     0,
                                      0,                     0,                     1,
                                      0,                     0,                     0,
                                      0,                     0,                     0,
                                      0,                     0,                     0,
                                      0,                     0,                     0,
                                      0,                     0,                     0,
                                      0,                     0,                     0;
    }

    // Jacobian wrt vel_orientation1
    if (jacobians[3])
    {
      Eigen::Map<fuse_core::Matrix<double, 15, 3>> jacobian(jacobians[3]);
      jacobian <<  0,             0,             0,
                   0,             0,             0,
                   0,             0,             0,
                  dt,  sr * tp * dt,  cr * tp * dt,
                   0,       cr * dt,    (-sr) * dt,
                   0, sr * cpi * dt, cr * cpi * dt,
                   0,             0,             0,
                   0,             0,             0,
                   0,             0,             0,
                   1,             0,             0,
                   0,             1,             0,
                   0,             0,             1,
                   0,             0,             0,
                   0,             0,             0,
                   0,             0,             0;
    }

    // Jacobian wrt acc_linear1
    if (jacobians[4])
    {
      const double dposx_dacc_linear_1_x = 0.5 * cy * cp * dt * dt;
      const double dposx_dacc_linear_1_y = 0.5 * (cy * sp * sr - sy * cr) * dt * dt;
      const double dposx_dacc_linear_1_z = 0.5 * (cy * sp * cr + sy * sr) * dt * dt;

      const double dposy_dacc_linear_1_x = 0.5 * sy * cp * dt * dt;
      const double dposy_dacc_linear_1_y = 0.5 * (sy * sp * sr + cy * cr) * dt * dt;
      const double dposy_dacc_linear_1_z = 0.5 * (sy * sp * cr - cy * sr) * dt * dt;

      const double dposz_dacc_linear_1_x = 0.5 * (-sp) * dt * dt;
      const double dposz_dacc_linear_1_y = 0.5 * cp * sr * dt * dt;
      const double dposz_dacc_linear_1_z = 0.5 * cp * cr * dt * dt;

      Eigen::Map<fuse_core::Matrix<double, 15, 3>> jacobian(jacobians[4]);
      jacobian << dposx_dacc_linear_1_x, dposx_dacc_linear_1_y, dposx_dacc_linear_1_z,
                  dposy_dacc_linear_1_x, dposy_dacc_linear_1_y, dposy_dacc_linear_1_z,
                  dposz_dacc_linear_1_x, dposz_dacc_linear_1_y, dposz_dacc_linear_1_z,
                                      0,                     0,                     0,
                                      0,                     0,                     0,
                                      0,                     0,                     0,
                                     dt,                     0,                     0,
                                      0,                    dt,                     0,
                                      0,                     0,                    dt,
                                      0,                     0,                     0,
                                      0,                     0,                     0,
                                      0,                     0,                     0,
                                      1,                     0,                     0,
                                      0,                     1,                     0,
                                      0,                     0,                     1;
    }
  }
}

/**
 * @brief Given a state and time delta, predicts a new state
 * @param[in] position1 - First position (array with x at index 0, y at index 1, z at index 2)
 * @param[in] orientation1 - First orientation (array with roll at index 0, pitch at index 1, yaw at index 2, qz at index 3)
 * @param[in] vel_linear1 - First velocity (array with x at index 0, y at index 1, z at index 2)
 * @param[in] vel_angular1 - First angular velocity (array with roll at index 0, pitch at index 1, yaw at index 2)
 * @param[in] acc_linear1 - First linear acceleration (array with x at index 0, y at index 1, z at index 2)
 * @param[in] dt - The time delta across which to predict the state
 * @param[out] position2 - Second position (array with x at index 0, y at index 1, z at index 2)
 * @param[out] orientation2 - Second orientation (array with roll at index 0, pitch at index 1, yaw at index 2)
 * @param[out] vel_linear2 - Second velocity (array with x at index 0, y at index 1, z at index 2)
 * @param[out] vel_angular2 - Second angular velocity (array with roll at index 0, pitch at index 1, yaw at index 2)
 * @param[out] acc_linear2 - Second linear acceleration (array with x at index 0, y at index 1, z at index 2)
 */
template<typename T>
inline void predict(
  const T* const position1,
  const T* const orientation1,
  const T* const vel_linear1,
  const T* const vel_angular1,
  const T* const acc_linear1,
  const T dt,
  T* const position2,
  T* const orientation2,
  T* const vel_linear2,
  T* const vel_angular2,
  T* const acc_linear2)
{
  predict(
    position1[0],
    position1[1],
    position1[2],
    orientation1[0],
    orientation1[1],
    orientation1[2],
    vel_linear1[0],
    vel_linear1[1],
    vel_linear1[2],
    vel_angular1[0],
    vel_angular1[1],
    vel_angular1[2],
    acc_linear1[0],
    acc_linear1[1],
    acc_linear1[2],
    dt,
    position2[0],
    position2[1],
    position2[2],
    orientation2[0],
    orientation2[1],
    orientation2[2],
    vel_linear2[0],
    vel_linear2[1],
    vel_linear2[2],
    vel_angular2[0],
    vel_angular2[1],
    vel_angular2[2],
    acc_linear2[0],
    acc_linear2[1],
    acc_linear2[2]);
}

/**
 * @brief Given a state and time delta, predicts a new state
 * @param[in] pose1 - The first 3D pose
 * @param[in] vel_linear_1 - The first linear velocity
 * @param[in] vel_angular1 - The first angular velocity
 * @param[in] acc_linear1 - The first linear acceleration
 * @param[in] dt - The time delta across which to predict the state
 * @param[in] pose2 - The second 3D pose
 * @param[in] vel_linear_2 - The second linear velocity
 * @param[in] vel_angular2 - The second angular velocity
 * @param[in] acc_linear2 - The second linear acceleration
 * @param[in] jacobian - The jacobian wrt the state
 */
inline void predict(
  const tf2::Transform& pose1,
  const tf2::Vector3& vel_linear1,
  const tf2::Vector3& vel_angular1,
  const tf2::Vector3& acc_linear1,
  const double dt,
  tf2::Transform& pose2,
  tf2::Vector3& vel_linear2,
  tf2::Vector3& vel_angular2,
  tf2::Vector3& acc_linear2,
  fuse_core::Matrix15d& jacobian)
{
  double x_pred {};
  double y_pred {};
  double z_pred {};
  double roll_pred {};
  double pitch_pred {};
  double yaw_pred {};
  double vel_linear_x_pred {};
  double vel_linear_y_pred {};
  double vel_linear_z_pred {};
  double vel_roll_pred {};
  double vel_pitch_pred {};
  double vel_yaw_pred {};
  double acc_linear_x_pred {};
  double acc_linear_y_pred {};
  double acc_linear_z_pred {};

  // fuse_core::Matrix15d is Eigen::RowMajor, so we cannot use pointers to the columns where each
  // parameter block starts.
  // Instead, we need to create a vector of Eigen::RowMajor matrices per parameter block and later
  // reconstruct the fuse_core::Matrix15d with the full jacobian.
  // The parameter blocks have the following sizes:
  // {position1: 3,
  //  orientation1: 3 (because of roll, pitch, yaw instead of quaternion components),
  //  vel_linear1: 3,
  //  vel_orientation1: 3,
  //  acc_linear1: 3}
  static constexpr size_t num_residuals{ 15 };
  static constexpr size_t num_parameter_blocks{ 5 };
  static const std::array<size_t, num_parameter_blocks> block_sizes = {3, 3, 3, 3, 3};

  std::array<fuse_core::MatrixXd, num_parameter_blocks> J;
  std::array<double*, num_parameter_blocks> jacobians;

  for (size_t i = 0; i < num_parameter_blocks; ++i)
  {
    J[i].resize(num_residuals, block_sizes[i]);
    jacobians[i] = J[i].data();
  }

  double quat[4] = {pose1.getRotation().w(),  // NOLINT(whitespace/braces)
                    pose1.getRotation().x(),
                    pose1.getRotation().y(),
                    pose1.getRotation().z()};  // NOLINT(whitespace/braces)

  double rpy[3];
  fuse_core::quaternion2rpy(quat, rpy);

  predict(
    pose1.getOrigin().x(),
    pose1.getOrigin().y(),
    pose1.getOrigin().z(),
    rpy[0],
    rpy[1],
    rpy[2],
    vel_linear1.x(),
    vel_linear1.y(),
    vel_linear1.z(),
    vel_angular1.x(),
    vel_angular1.y(),
    vel_angular1.z(),
    acc_linear1.x(),
    acc_linear1.y(),
    acc_linear1.z(),
    dt,
    x_pred,
    y_pred,
    z_pred,
    roll_pred,
    pitch_pred,
    yaw_pred,
    vel_linear_x_pred,
    vel_linear_y_pred,
    vel_linear_z_pred,
    vel_roll_pred,
    vel_pitch_pred,
    vel_yaw_pred,
    acc_linear_x_pred,
    acc_linear_y_pred,
    acc_linear_z_pred,
    jacobians.data());

  pose2.setOrigin(tf2::Vector3{x_pred, y_pred, z_pred});

  Eigen::Quaterniond orientation2 = Eigen::AngleAxisd(yaw_pred, Eigen::Vector3d::UnitZ()) *
                                    Eigen::AngleAxisd(pitch_pred, Eigen::Vector3d::UnitY()) *
                                    Eigen::AngleAxisd(roll_pred, Eigen::Vector3d::UnitX());

  pose2.setRotation({orientation2.x(), orientation2.y(), orientation2.z(), orientation2.w()});

  vel_linear2.setX(vel_linear_x_pred);
  vel_linear2.setY(vel_linear_y_pred);
  vel_linear2.setZ(vel_linear_z_pred);

  vel_angular2.setX(vel_roll_pred);
  vel_angular2.setY(vel_pitch_pred);
  vel_angular2.setZ(vel_yaw_pred);

  acc_linear2.setX(acc_linear_x_pred);
  acc_linear2.setY(acc_linear_y_pred);
  acc_linear2.setZ(acc_linear_z_pred);

  jacobian << J[0], J[1], J[2], J[3], J[4];
}

/**
 * @brief Given a state and time delta, predicts a new state
 * @param[in] pose1 - The first 3D pose
 * @param[in] vel_linear_1 - The first linear velocity
 * @param[in] vel_angular1 - The first angular velocity
 * @param[in] acc_linear1 - The first linear acceleration
 * @param[in] dt - The time delta across which to predict the state
 * @param[in] pose2 - The second 3D pose
 * @param[in] vel_linear_2 - The second linear velocity
 * @param[in] vel_angular2 - The second angular velocity
 * @param[in] acc_linear2 - The second linear acceleration
 */
inline void predict(
  const tf2::Transform& pose1,
  const tf2::Vector3& vel_linear1,
  const tf2::Vector3& vel_angular1,
  const tf2::Vector3& acc_linear1,
  const double dt,
  tf2::Transform& pose2,
  tf2::Vector3& vel_linear2,
  tf2::Vector3& vel_angular2,
  tf2::Vector3& acc_linear2)
{
  double x_pred {};
  double y_pred {};
  double z_pred {};
  double roll_pred {};
  double pitch_pred {};
  double yaw_pred {};
  double vel_linear_x_pred {};
  double vel_linear_y_pred {};
  double vel_linear_z_pred {};
  double vel_roll_pred {};
  double vel_pitch_pred {};
  double vel_yaw_pred {};
  double acc_linear_x_pred {};
  double acc_linear_y_pred {};
  double acc_linear_z_pred {};

  const double roll1 = fuse_core::getRoll(pose1.getRotation().w(),
                                          pose1.getRotation().x(),
                                          pose1.getRotation().y(),
                                          pose1.getRotation().z());

  const double pitch1 = fuse_core::getPitch(pose1.getRotation().w(),
                                            pose1.getRotation().x(),
                                            pose1.getRotation().y(),
                                            pose1.getRotation().z());

  const double yaw1 = fuse_core::getYaw(pose1.getRotation().w(),
                                        pose1.getRotation().x(),
                                        pose1.getRotation().y(),
                                        pose1.getRotation().z());

  predict(
    pose1.getOrigin().x(),
    pose1.getOrigin().y(),
    pose1.getOrigin().z(),
    roll1,
    pitch1,
    yaw1,
    vel_linear1.x(),
    vel_linear1.y(),
    vel_linear1.z(),
    vel_angular1.x(),
    vel_angular1.y(),
    vel_angular1.z(),
    acc_linear1.x(),
    acc_linear1.y(),
    acc_linear1.z(),
    dt,
    x_pred,
    y_pred,
    z_pred,
    roll_pred,
    pitch_pred,
    yaw_pred,
    vel_linear_x_pred,
    vel_linear_y_pred,
    vel_linear_z_pred,
    vel_roll_pred,
    vel_pitch_pred,
    vel_yaw_pred,
    acc_linear_x_pred,
    acc_linear_y_pred,
    acc_linear_z_pred);

  pose2.setOrigin(tf2::Vector3{x_pred, y_pred, z_pred});

  Eigen::Quaterniond orientation_pred_q =
    Eigen::AngleAxisd(yaw_pred, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(pitch_pred, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(roll_pred, Eigen::Vector3d::UnitX());

  pose2.setRotation(tf2::Quaternion{ orientation_pred_q.x(),  // NOLINT(whitespace/braces)
                                     orientation_pred_q.y(),
                                     orientation_pred_q.z(),
                                     orientation_pred_q.w()});  // NOLINT(whitespace/braces)

  vel_linear2.setX(vel_linear_x_pred);
  vel_linear2.setY(vel_linear_y_pred);
  vel_linear2.setZ(vel_linear_z_pred);

  vel_angular2.setX(vel_roll_pred);
  vel_angular2.setY(vel_pitch_pred);
  vel_angular2.setZ(vel_yaw_pred);

  acc_linear2.setX(acc_linear_x_pred);
  acc_linear2.setY(acc_linear_y_pred);
  acc_linear2.setZ(acc_linear_z_pred);
}

}  // namespace fuse_models

#endif  // FUSE_MODELS_UNICYCLE_3D_PREDICT_H
