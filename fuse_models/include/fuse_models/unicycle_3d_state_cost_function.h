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
#ifndef FUSE_MODELS_UNICYCLE_3D_STATE_COST_FUNCTION_H
#define FUSE_MODELS_UNICYCLE_3D_STATE_COST_FUNCTION_H

#include <fuse_models/unicycle_3d_predict.h>

#include <fuse_core/eigen.h>
#include <fuse_core/fuse_macros.h>
#include <fuse_core/util.h>

#include <Eigen/Geometry>

#include <ceres/sized_cost_function.h>


namespace fuse_models
{

/**
 * @brief Create a cost function for a 3D state vector
 * 
 * The state vector includes the following quantities, given in this order:
 *   x position
 *   y position
 *   z position
 *   roll (rotation about the x axis)
 *   pitch (rotation about the y axis)
 *   yaw (rotation about the z axis)
 *   x velocity
 *   y velocity
 *   z velocity
 *   roll velocity
 *   pitch velocity
 *   yaw velocity
 *   x acceleration
 *   y acceleration
 *   z acceleration
 *
 * The Ceres::NormalPrior cost function only supports a single variable. This is a convenience cost function that
 * applies a prior constraint on both the entire state vector.
 *
 * The cost function is of the form:
 *
 *             ||    [          x_t2 - proj(x_t1)         ] ||^2
 *   cost(x) = ||    [          y_t2 - proj(y_t1)         ] ||
 *             ||    [          z_t2 - proj(z_t1)         ] ||
 *             ||    [       roll_t2 - proj(roll_t1)      ] ||
 *             ||    [      pitch_t2 - proj(pitch_t1)     ] ||
 *             ||    [        yaw_t2 - proj(yaw_t1)       ] ||
 *             ||A * [      x_vel_t2 - proj(x_vel_t1)     ] ||
 *             ||    [      y_vel_t2 - proj(y_vel_t1)     ] ||
 *             ||    [      z_vel_t2 - proj(z_vel_t1)     ] ||
 *             ||    [   roll_vel_t2 - proj(roll_vel_t1)  ] ||
 *             ||    [  pitch_vel_t2 - proj(pitch_vel_t1) ] ||
 *             ||    [    yaw_vel_t2 - proj(yaw_vel_t1)   ] ||
 *             ||    [      x_acc_t2 - proj(x_acc_t1)     ] ||
 *             ||    [      y_acc_t2 - proj(y_acc_t1)     ] ||
 *             ||    [      z_acc_t2 - proj(z_acc_t1)     ] ||
 * 
 * where, the matrix A is fixed, the state variables are provided at two discrete time steps, and proj is a function
 * that projects the state variables from time t1 to time t2. In case the user is interested in implementing a cost
 * function of the form
 *
 *   cost(X) = (X - mu)^T S^{-1} (X - mu)
 *
 * where, mu is a vector and S is a covariance matrix, then, A = S^{-1/2}, i.e the matrix A is the square root
 * information matrix (the inverse of the covariance).
 */
class Unicycle3DStateCostFunction : public ceres::SizedCostFunction<15, 3, 4, 3, 3, 3, 3, 4, 3, 3, 3>
{
public:
  FUSE_MAKE_ALIGNED_OPERATOR_NEW();

  /**
   * @brief Construct a cost function instance
   *
   * @param[in] dt The time delta across which to generate the kinematic model cost
   * @param[in] A The residual weighting matrix, most likely the square root information matrix in order
   *              (x, y, z, roll, pitch, yaw, x_vel, y_vel, z_vel, roll_vel, pitch_vel, yaw_vel, x_acc, y_acc, z_acc)
   */
  Unicycle3DStateCostFunction(const double dt, const fuse_core::Matrix15d& A);

  /**
   * @brief Evaluate the cost function. Used by the Ceres optimization engine.
   *
   * @param[in] parameters - Parameter blocks:
   *     0 : position1 - First position (array with x at index 0, y at index 1, z at index 2)
   *     1 : orientation1 - First orientation (array with w at index 0, x at index 1, y at index 2, z at index 3)
   *     2 : vel_linear1 - First linear velocity (array with x at index 0, y at index 1)
   *     3 : vel_angular1 - First angular velocity (array with roll at index 0, pitch at index 1, yaw at index 2)
   *     4 : acc_linear1 - First linear acceleration (array with x at index 0, y at index 1, z at index 2)
   *     5 : position2 - Second position (array with x at index 0, y at index 1, z at index 2)
   *     6 : orientation2 - Second orientation (array with w at index 0, x at index 1, y at index 2, z at index 3)
   *     7 : vel_linear2 - Second linear velocity (array with x at index 0, y at index 1, z at index 2)
   *     8 : vel_angular2 - Second angular velocity (array with roll at index 0, pitch at index 1, yaw at index 2)
   *     9 : acc_linear2 - Second linear acceleration (array with x at index 0, y at index 1, z at index 2)
   * @param[out] residual - The computed residual (error)
   * @param[out] jacobians - Jacobians of the residuals wrt the parameters. Only computed if not NULL, and only
   *                         computed for the parameters where jacobians[i] is not NULL.
   * @return The return value indicates whether the computation of the residuals and/or jacobians was successful or not.
   */
  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override
  {
    double position_pred[3];
    double orientation_pred_rpy[3];
    double vel_linear_pred[3];
    double vel_angular_pred[3];
    double acc_linear_pred[3];
    double orientation1_rpy[3];
    double orientation2_rpy[3];
    double j1_quat2rpy[12];
    double j2_quat2rpy[12];
    fuse_core::quaternion2rpy(parameters[1], orientation1_rpy, j1_quat2rpy);
    fuse_core::quaternion2rpy(parameters[6], orientation2_rpy, j2_quat2rpy);

    predict(
      parameters[0][0],  // position1_x
      parameters[0][1],  // position1_y
      parameters[0][2],  // position1_z
      orientation1_rpy[0],  // roll1
      orientation1_rpy[1],  // pitch1
      orientation1_rpy[2],  // yaw1
      parameters[2][0],  // vel_linear1_x
      parameters[2][1],  // vel_linear1_y
      parameters[2][2],  // vel_linear1_z
      parameters[3][0],  // vel_angular1
      parameters[3][1],  // vel_angular1
      parameters[3][2],  // vel_angular1
      parameters[4][0],  // acc_linear1_x
      parameters[4][1],  // acc_linear1_y
      parameters[4][2],  // acc_linear1_z
      dt_,
      position_pred[0],
      position_pred[1],
      position_pred[2],
      orientation_pred_rpy[0],
      orientation_pred_rpy[1],
      orientation_pred_rpy[2],
      vel_linear_pred[0],
      vel_linear_pred[1],
      vel_linear_pred[2],
      vel_angular_pred[0],
      vel_angular_pred[1],
      vel_angular_pred[2],
      acc_linear_pred[0],
      acc_linear_pred[1],
      acc_linear_pred[2],
      jacobians);

    // Because the above function computes the jacobian wrt rpy orientation, it needs to be converted to
    // quaternion orientation.
    // This is still the case even although local parameterization is used because the jacobian matrix is the global
    // size (so 15x4) and then later for optimization the jacobian internally gets updated with the jacobian of the
    // transformation function from the global to the local size.
    // See ceres 2.0.0: internal/ceres/residual_block.cc::143
    // Also see: https://github.com/ceres-solver/ceres-solver/issues/387
    if (jacobians && jacobians[1])
    {
      // Note that this only works without an out-of-bounds memory access because jacobians[1] is originally 15x4 and
      // the rpy jacobian is smaller (15x3) while vice versa a segfault might occur because of an out-of-bounds access
      Eigen::Map<fuse_core::Matrix<double, 15, 3>> jacobian_orientation_rpy(jacobians[1]);
      Eigen::Map<fuse_core::Matrix<double, 15, 4>> jacobian_orientation_q(jacobians[1]);

      Eigen::Map<fuse_core::Matrix<double, 3, 4>> j_quat2rpy_map(j1_quat2rpy);

      jacobian_orientation_q = jacobian_orientation_rpy * j_quat2rpy_map;
    }

    residuals[0] = parameters[5][0] - position_pred[0];
    residuals[1] = parameters[5][1] - position_pred[1];
    residuals[2] = parameters[5][2] - position_pred[2];
    residuals[3] = orientation2_rpy[0] - orientation_pred_rpy[0];
    residuals[4] = orientation2_rpy[1] - orientation_pred_rpy[1];
    residuals[5] = orientation2_rpy[2] - orientation_pred_rpy[2];
    residuals[6] = parameters[7][0] - vel_linear_pred[0];
    residuals[7] = parameters[7][1] - vel_linear_pred[1];
    residuals[8] = parameters[7][2] - vel_linear_pred[2];
    residuals[9] = parameters[8][0] - vel_angular_pred[0];
    residuals[10] = parameters[8][1] - vel_angular_pred[1];
    residuals[11] = parameters[8][2] - vel_angular_pred[2];
    residuals[12] = parameters[9][0] - acc_linear_pred[0];
    residuals[13] = parameters[9][1] - acc_linear_pred[1];
    residuals[14] = parameters[9][2] - acc_linear_pred[2];

    fuse_core::wrapAngle2D(residuals[3]);
    fuse_core::wrapAngle2D(residuals[4]);
    fuse_core::wrapAngle2D(residuals[5]);

    // Scale the residuals by the square root information matrix to account for
    // the measurement uncertainty.
    Eigen::Map<fuse_core::Vector15d> residuals_map(residuals);
    residuals_map.applyOnTheLeft(A_);

    if (jacobians)
    {
      // It might be possible to simplify the code below implementing something like this but using compile-time
      // template recursion.
      //
      // // state1: (position1, orientation1, vel_linear1, vel_orientation1, acc_linear1)
      // for (size_t i = 0; i < 5; ++i)
      // {
      //   if (jacobians[i])
      //   {
      //     Eigen::Map<fuse_core::Matrix<double, 15, ParameterDims::GetDim(i)>> jacobian(jacobians[i]);
      //     jacobian.applyOnTheLeft(-A_);
      //   }
      // }

      // Update jacobian wrt position1
      if (jacobians[0])
      {
        Eigen::Map<fuse_core::Matrix<double, 15, 3>> jacobian(jacobians[0]);
        jacobian.applyOnTheLeft(-A_);
      }

      // Update jacobian wrt orientation1
      if (jacobians[1])
      {
        // Although local parameterization is used, the jacobian matrix is still the global size (so 15x4) and
        // then later for optimization the jacobian internally gets updated with the jacobian of the transformation
        // function from the global to the local size (see ceres 2.0.0: internal/ceres/residual_block.cc::143)
        // Also see: https://github.com/ceres-solver/ceres-solver/issues/387

        Eigen::Map<fuse_core::Matrix<double, 15, 4>> jacobian(jacobians[1]);
        jacobian.applyOnTheLeft(-A_);
      }

      // Update jacobian wrt vel_linear1
      if (jacobians[2])
      {
        Eigen::Map<fuse_core::Matrix<double, 15, 3>> jacobian(jacobians[2]);
        jacobian.applyOnTheLeft(-A_);
      }

      // Update jacobian wrt vel_orientation1
      if (jacobians[3])
      {
        Eigen::Map<fuse_core::Matrix<double, 15, 3>> jacobian(jacobians[3]);
        jacobian.applyOnTheLeft(-A_);
      }

      // Update jacobian wrt acc_linear1
      if (jacobians[4])
      {
        Eigen::Map<fuse_core::Matrix<double, 15, 3>> jacobian(jacobians[4]);
        jacobian.applyOnTheLeft(-A_);
      }

      // It might be possible to simplify the code below implementing something like this but using compile-time
      // template recursion.
      //
      // // state2: (position2, orientation2, vel_linear2, vel_orientation2, acc_linear2)
      // for (size_t i = 5, offset = 0; i < ParameterDims::kNumParameterBlocks; ++i)
      // {
      //   constexpr auto dim = ParameterDims::GetDim(i);

      //   if (jacobians[i])
      //   {
      //     Eigen::Map<fuse_core::Matrix<double, 15, dim>> jacobian(jacobians[i]);
      //     jacobian = A_.block<15, dim>(0, offset);
      //   }

      //   offset += dim;
      // }

      // Jacobian wrt position2
      if (jacobians[5])
      {
        Eigen::Map<fuse_core::Matrix<double, 15, 3>> jacobian(jacobians[5]);
        jacobian = A_.block<15, 3>(0, 0);
      }

      // Jacobian wrt orientation2
      if (jacobians[6])
      {
        // Although local parameterization is used, the jacobian matrix is still the global size (so 15x4) and
        // then later for optimization the jacobian internally gets updated with the jacobian of the transformation
        // function from the global to the local size (see ceres 2.0.0: internal/ceres/residual_block.cc::143)
        // Also see: https://github.com/ceres-solver/ceres-solver/issues/387

        Eigen::Map<fuse_core::Matrix<double, 15, 4>> jacobian(jacobians[6]);
        Eigen::Map<fuse_core::Matrix<double, 3, 4>> j2_quat2rpy_map(j2_quat2rpy);
        jacobian = A_.block<15, 3>(0, 3) * j2_quat2rpy_map;
      }

      // Jacobian wrt vel_linear2
      if (jacobians[7])
      {
        Eigen::Map<fuse_core::Matrix<double, 15, 3>> jacobian(jacobians[7]);
        jacobian = A_.block<15, 3>(0, 6);
      }

      // Jacobian wrt vel_orientation2
      if (jacobians[8])
      {
        Eigen::Map<fuse_core::Matrix<double, 15, 3>> jacobian(jacobians[8]);
        jacobian = A_.block<15, 3>(0, 9);
      }

      // Jacobian wrt acc_linear2
      if (jacobians[9])
      {
        Eigen::Map<fuse_core::Matrix<double, 15, 3>> jacobian(jacobians[9]);
        jacobian = A_.block<15, 3>(0, 12);
      }
    }

    return true;
  }

private:
  double dt_;
  fuse_core::Matrix15d A_;  //!< The residual weighting matrix, most likely the square root information matrix
};

Unicycle3DStateCostFunction::Unicycle3DStateCostFunction(const double dt, const fuse_core::Matrix15d& A) :
  dt_(dt),
  A_(A)
{
}

}  // namespace fuse_models

#endif  // FUSE_MODELS_UNICYCLE_3D_STATE_COST_FUNCTION_H
