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
#ifndef FUSE_CONSTRAINTS_NORMAL_PRIOR_POSE_3D_EULER_COST_FUNCTOR_H
#define FUSE_CONSTRAINTS_NORMAL_PRIOR_POSE_3D_EULER_COST_FUNCTOR_H

#include <fuse_constraints/normal_prior_orientation_3d_euler_cost_functor.h>
#include <fuse_core/eigen.h>
#include <fuse_core/fuse_macros.h>
#include <fuse_core/util.h>

#include <Eigen/Core>


namespace fuse_constraints
{

/**
 * @brief Create a prior cost function on both the 3D position and orientation variables at once.
 *
 * The Ceres::NormalPrior cost function only supports a single variable. This is a convenience cost function that
 * applies a prior constraint on both the 3D position and orientation variables at once.
 *
 * The cost function is of the form:
 *
 *   cost(x) = || A * [  p                      - b(0:2) ] ||^2
 *             ||     [  toEulerRollPitchYaw(q) - b(3:6) ] ||
 *
 * Here, the matrix A can be of variable size, thereby permitting the computation of errors for partial measurements.
 * The vector b is a fixed-size 6x1, p is the position variable, and q is the orientation variable.
 * Note that the covariance submatrix for the orientation should represent errors in roll, pitch, and yaw.
 * In case the user is interested in implementing a cost function of the form
 *
 *   cost(X) = (X - mu)^T S^{-1} (X - mu)
 *
 * where, mu is a vector and S is a covariance matrix, then, A = S^{-1/2}, i.e the matrix A is the square root
 * information matrix (the inverse of the covariance).
 */
class NormalPriorPose3DEulerCostFunctor
{
public:
  FUSE_MAKE_ALIGNED_OPERATOR_NEW();

  /**
   * @brief Construct a cost function instance
   *
   * The residual weighting matrix can vary in size, as this cost functor can be used to compute costs for partial
   * vectors. The number of rows of A will be the number of dimensions for which you want to compute the error, and the
   * number of columns in A will be fixed at 6. For example, if we just want to use the values of x and yaw, then \p A
   * will be of size 2x6 where the first row represents the weighting for x to all dimensions including x itself and
   * the second row represents the weighting for yaw to all dimensions including yaw itself. For weighting with 1 and
   * no relation to other dimensions the matrix should be:
   * [1, 0, 0, 0, 0, 0]
   * [0, 0, 0, 0, 0, 1]
   *
   * @param[in] A The residual weighting matrix, most likely the square root information matrix in order
   *              (x, y, z, roll, pitch, yaw)
   * @param[in] b The 3D pose measurement or prior in order (x, y, z, roll, pitch, yaw)
   */
  NormalPriorPose3DEulerCostFunctor(const fuse_core::MatrixXd& A, const fuse_core::Vector6d& b);

  /**
   * @brief Evaluate the cost function. Used by the Ceres optimization engine.
   */
  template <typename T>
  bool operator()(const T* const position, const T* const orientation, T* residual) const;

private:
  fuse_core::MatrixXd A_;
  fuse_core::Vector6d b_;

  NormalPriorOrientation3DEulerCostFunctor orientation_functor_;
};

NormalPriorPose3DEulerCostFunctor::NormalPriorPose3DEulerCostFunctor(
  const fuse_core::MatrixXd& A, const fuse_core::Vector6d& b)
  : A_(A),
    b_(b),
    orientation_functor_(fuse_core::Matrix3d::Identity(), b_.tail<3>())  // Delta will not be scaled
{
}

template <typename T>
bool NormalPriorPose3DEulerCostFunctor::operator()(
  const T* const position, const T* const orientation, T* residual) const
{
  Eigen::Matrix<T, 6, 1> full_residuals_vector;
  // Compute the position error
  full_residuals_vector[0] = position[0] - T(b_(0));
  full_residuals_vector[1] = position[1] - T(b_(1));
  full_residuals_vector[2] = position[2] - T(b_(2));

  // Use the 3D orientation cost functor to compute the orientation delta
  orientation_functor_(orientation, full_residuals_vector.data() + 3);

  // Scale the residuals by the square root information matrix to account for
  // the measurement uncertainty.
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> residuals_vector(residual, A_.rows());
  residuals_vector = A_.template cast<T>() * full_residuals_vector;

  return true;
}

}  // namespace fuse_constraints

#endif  // FUSE_CONSTRAINTS_NORMAL_PRIOR_POSE_3D_EULER_COST_FUNCTOR_H
