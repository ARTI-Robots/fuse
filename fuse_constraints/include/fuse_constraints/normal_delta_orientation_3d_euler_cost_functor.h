/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2019, Locus Robotics
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
#ifndef FUSE_CONSTRAINTS_NORMAL_DELTA_ORIENTATION_3D_EULER_COST_FUNCTOR_H
#define FUSE_CONSTRAINTS_NORMAL_DELTA_ORIENTATION_3D_EULER_COST_FUNCTOR_H

#include <fuse_core/eigen.h>
#include <fuse_core/fuse_macros.h>
#include <fuse_core/util.h>
#include <fuse_variables/orientation_3d_stamped.h>

#include <ceres/rotation.h>
#include <Eigen/Core>

#include <vector>


namespace fuse_constraints
{

/**
 * @brief Implements a cost function that models a difference between 3D orientation variables using Euler roll, pitch, and yaw measurements
 *
 * The functor can compute the cost of a subset of the axes, in the event that we are not interested in all the Euler
 * angles in the variable.
 *
 * So, for example, if
 * b_ = [ measured_yaw_difference  ]
 *      [ measured_roll_difference ]
 *
 * then the cost function is of the form:
 *
 *   cost(x) = || A * [ (yaw1(x) - yaw2(x))   - b_(0) ] ||^2
 *             ||     [ (roll1(x) - roll2(x)) - b_(1) ] ||
 *
 * where the matrix A and the vector b are fixed and (roll, pitch, yaw) are the components of the 3D orientation
 * variable.
 *
 * In case the user is interested in implementing a cost function of the form
 *
 *   cost(X) = (X - mu)^T S^{-1} (X - mu)
 *
 * where, mu is a vector and S is a covariance matrix, then, A = S^{-1/2}, i.e the matrix A is the square root
 * information matrix (the inverse of the covariance).
 */
class NormalDeltaOrientation3DEulerCostFunctor
{
public:
  using Euler = fuse_variables::Orientation3DStamped::Euler;
  FUSE_MAKE_ALIGNED_OPERATOR_NEW();

  /**
   * @brief Construct a cost function instance
   *
   * @param[in] A The residual weighting matrix, most likely the square root information matrix. Its order must match
   *              the values in \p axes.
   * @param[in] b The measured change between the two orientation variables. Its order must match the values in \p axes.
   * @param[in] axes The Euler angle axes for which we want to compute errors. Defaults to all axes.
   */
  NormalDeltaOrientation3DEulerCostFunctor(
    const fuse_core::MatrixXd& A,
    const fuse_core::VectorXd& b,
    const std::vector<Euler> &axes = {Euler::ROLL, Euler::PITCH, Euler::YAW}) :  //NOLINT
      A_(A),
      b_(b),
      axes_(axes)
  {
  }

  /**
   * @brief Evaluate the cost function. Used by the Ceres optimization engine.
   */
  template <typename T>
  bool operator()(const T* const orientation1, const T* const orientation2, T* residuals) const
  {
    using fuse_variables::Orientation3DStamped;

    for (size_t i = 0; i < axes_.size(); ++i)
    {
      T angle1, angle2;
      switch (axes_[i])
      {
        case Euler::ROLL:
        {
          angle1 = fuse_core::getRoll(orientation1[0], orientation1[1], orientation1[2], orientation1[3]);
          angle2 = fuse_core::getRoll(orientation2[0], orientation2[1], orientation2[2], orientation2[3]);
          break;
        }
        case Euler::PITCH:
        {
          angle1 = fuse_core::getPitch(orientation1[0], orientation1[1], orientation1[2], orientation1[3]);
          angle2 = fuse_core::getPitch(orientation2[0], orientation2[1], orientation2[2], orientation2[3]);
          break;
        }
        case Euler::YAW:
        {
          angle1 = fuse_core::getYaw(orientation1[0], orientation1[1], orientation1[2], orientation1[3]);
          angle2 = fuse_core::getYaw(orientation2[0], orientation2[1], orientation2[2], orientation2[3]);
          break;
        }
        default:
        {
          throw std::runtime_error("The provided axis specified is unknown. "
                                   "I should probably be more informative here");
        }
      }
      const auto difference = fuse_core::wrapAngle2D(angle2 - angle1);
      residuals[i] = fuse_core::wrapAngle2D(difference - T(b_[i]));
    }

    // Scale the residuals by the square root information matrix to account for the measurement uncertainty.
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> residuals_map(residuals, A_.rows());
    residuals_map.applyOnTheLeft(A_.template cast<T>());

    return true;
  }

private:
  fuse_core::MatrixXd A_;  //!< The residual weighting matrix, most likely the square root information matrix
  fuse_core::VectorXd b_;  //!< The measured difference between orientation1 and orientation2. Its order must match
                           //!< the values in \p axes.
  std::vector<Euler> axes_;  //!< The Euler angle axes that we're measuring
};

}  // namespace fuse_constraints

#endif  // FUSE_CONSTRAINTS_NORMAL_DELTA_ORIENTATION_3D_EULER_COST_FUNCTOR_H
