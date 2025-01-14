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
#ifndef FUSE_CONSTRAINTS_NORMAL_DELTA_POSE_3D_EULER_COST_FUNCTOR_H
#define FUSE_CONSTRAINTS_NORMAL_DELTA_POSE_3D_EULER_COST_FUNCTOR_H

#include <fuse_constraints/normal_delta_orientation_3d_euler_cost_functor.h>
#include <fuse_core/eigen.h>
#include <fuse_core/fuse_macros.h>
#include <fuse_core/util.h>

#include <ceres/rotation.h>
#include <stdexcept>


namespace fuse_constraints
{

/**
 * @brief Implements a cost function that models a difference between 3D pose variables.
 *
 * A single pose involves two variables: a 3D position and a 3D orientation. The generic NormalDelta cost function
 * only supports a single variable type, and computes the difference using per-element subtraction. This cost function
 * computes the difference using standard 3D transformation math.
 *
 *             ||     [ delta.x(p1,p2)     - b(0)] ||^2
 *             ||     [ delta.y(p1,p2)     - b(1)] ||
 *   cost(x) = || A * [ delta.z(p1,p2)     - b(2)] ||
 *             ||     [ delta.roll(q1,q2)  - b(3)] ||
 *             ||     [ delta.pitch(q1,q2) - b(4)] ||
 *             ||     [ delta.yaw(q1,q2)   - b(5)] ||
 *
 * Here, the matrix A can be of variable size, thereby permitting the computation of errors for partial measurements.
 * The vector b is a fixed-size 6x1, p1 and p2 are the position variables, and q1 and q2 are the quaternion orientation
 * variables. Note that the covariance submatrix for the orientation should represent errors in roll, pitch, and yaw.
 * In case the user is interested in implementing a cost function of the form
 *
 *   cost(X) = (X - mu)^T S^{-1} (X - mu)
 *
 * where, mu is a vector and S is a covariance matrix, then, A = S^{-1/2}, i.e the matrix A is the square root
 * information matrix (the inverse of the covariance).
 */
class NormalDeltaPose3DEulerCostFunctor
{
public:
  FUSE_MAKE_ALIGNED_OPERATOR_NEW();

  /**
   * @brief Constructor
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
   *              (dx, dy, dz, droll, dpitch, dyaw)
   * @param[in] b The exposed pose difference in order (dx, dy, dz, droll, dpitch, dyaw)
   */
  NormalDeltaPose3DEulerCostFunctor(const fuse_core::MatrixXd& A, const fuse_core::Vector6d& b);

  /**
   * @brief Compute the cost values/residuals using the provided variable/parameter values
   */
  template <typename T>
  bool operator()(
    const T* const position1,
    const T* const orientation1,
    const T* const position2,
    const T* const orientation2,
    T* residual) const;

private:
  fuse_core::MatrixXd A_;  //!< The residual weighting matrix, most likely the square root information matrix
  fuse_core::Vector6d b_;  //!< The measured difference between variable pose1 and variable pose2

  NormalDeltaOrientation3DEulerCostFunctor orientation_functor_;
};

NormalDeltaPose3DEulerCostFunctor::NormalDeltaPose3DEulerCostFunctor(
  const fuse_core::MatrixXd& A,
  const fuse_core::Vector6d& b) :
    A_(A),
    b_(b),
    orientation_functor_(fuse_core::Matrix3d::Identity(), b_.tail<3>())  // Orientation residuals will not be scaled
                                                                         // within the orientation functor but here at
                                                                         // the cost function after computation of the
                                                                         // orientation residuals without scaling.
{
  if (A.cols() != b.size())
  {
    throw std::invalid_argument("The number of columns in the residual weighting matrix A need to match the size of "
                                "the measured difference b.");
  }
}

template <typename T>
bool NormalDeltaPose3DEulerCostFunctor::operator()(
  const T* const position1,
  const T* const orientation1,
  const T* const position2,
  const T* const orientation2,
  T* residual) const
{
  Eigen::Matrix<T, 6, 1> full_residuals_vector;

  // Compute the position delta between pose1 and pose2
  T orientation1_inverse[4] =
  {
    orientation1[0],
    -orientation1[1],
    -orientation1[2],
    -orientation1[3]
  };
  T position_delta[3] =
  {
    position2[0] - position1[0],
    position2[1] - position1[1],
    position2[2] - position1[2]
  };
  T position_delta_rotated[3];
  ceres::QuaternionRotatePoint(
    orientation1_inverse,
    position_delta,
    position_delta_rotated);

  // Compute the first three residual terms as (position_delta - b)
  full_residuals_vector[0] = position_delta_rotated[0] - T(b_[0]);
  full_residuals_vector[1] = position_delta_rotated[1] - T(b_[1]);
  full_residuals_vector[2] = position_delta_rotated[2] - T(b_[2]);

  // Use the 3D orientation cost functor to compute the orientation delta
  orientation_functor_(orientation1, orientation2, full_residuals_vector.data() + 3);

  // Map it to Eigen, and weight it
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> residuals_vector(residual, A_.rows());
  residuals_vector = A_.template cast<T>() * full_residuals_vector;

  return true;
}

}  // namespace fuse_constraints

#endif  // FUSE_CONSTRAINTS_NORMAL_DELTA_POSE_3D_EULER_COST_FUNCTOR_H
