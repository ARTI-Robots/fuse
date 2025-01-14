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
#include <fuse_models/unicycle_3d_state_kinematic_constraint.h>

#include <fuse_models/unicycle_3d_state_cost_function.h>

#include <fuse_variables/acceleration_linear_3d_stamped.h>
#include <fuse_variables/orientation_3d_stamped.h>
#include <fuse_variables/position_3d_stamped.h>
#include <fuse_variables/velocity_angular_3d_stamped.h>
#include <fuse_variables/velocity_linear_3d_stamped.h>
#include <pluginlib/class_list_macros.hpp>

#include <boost/serialization/export.hpp>
#include <Eigen/Dense>

#include <ostream>
#include <string>


namespace fuse_models
{

Unicycle3DStateKinematicConstraint::Unicycle3DStateKinematicConstraint(
  const std::string& source,
  const fuse_variables::Position3DStamped& position1,
  const fuse_variables::Orientation3DStamped& orientation1,
  const fuse_variables::VelocityLinear3DStamped& linear_velocity1,
  const fuse_variables::VelocityAngular3DStamped& angular_velocity1,
  const fuse_variables::AccelerationLinear3DStamped& linear_acceleration1,
  const fuse_variables::Position3DStamped& position2,
  const fuse_variables::Orientation3DStamped& orientation2,
  const fuse_variables::VelocityLinear3DStamped& linear_velocity2,
  const fuse_variables::VelocityAngular3DStamped& angular_velocity2,
  const fuse_variables::AccelerationLinear3DStamped& linear_acceleration2,
  const fuse_core::Matrix15d& covariance) :
    fuse_core::Constraint(
      source,
      {position1.uuid(),
       orientation1.uuid(),
       linear_velocity1.uuid(),
       angular_velocity1.uuid(),
       linear_acceleration1.uuid(),
       position2.uuid(),
       orientation2.uuid(),
       linear_velocity2.uuid(),
       angular_velocity2.uuid(),
       linear_acceleration2.uuid()}),  // NOLINT
    dt_((position2.stamp() - position1.stamp()).toSec()),
    sqrt_information_(covariance.inverse().llt().matrixU())
{
}

void Unicycle3DStateKinematicConstraint::print(std::ostream& stream) const
{
  stream << type() << "\n"
         << "  source: " << source() << "\n"
         << "  uuid: " << uuid() << "\n"
         << "  position variable 1: " << variables().at(0) << "\n"
         << "  orientation variable 1: " << variables().at(1) << "\n"
         << "  linear velocity variable 1: " << variables().at(2) << "\n"
         << "  angular velocity variable 1: " << variables().at(3) << "\n"
         << "  linear acceleration variable 1: " << variables().at(4) << "\n"
         << "  position variable 2: " << variables().at(5) << "\n"
         << "  orientation variable 2: " << variables().at(6) << "\n"
         << "  linear velocity variable 2: " << variables().at(7) << "\n"
         << "  angular velocity variable 2: " << variables().at(8) << "\n"
         << "  linear acceleration variable 2: " << variables().at(9) << "\n"
         << "  dt: " << dt() << "\n"
         << "  sqrt_info: " << sqrtInformation() << "\n";
}

ceres::CostFunction* Unicycle3DStateKinematicConstraint::costFunction() const
{
  return new Unicycle3DStateCostFunction(dt_, sqrt_information_);
  // Here we return a cost function that computes the analytic derivatives/jacobians, but we could use automatic
  // differentiation as follows:
  //
  // return new ceres::AutoDiffCostFunction<Unicycle3DStateCostFunctor, 15, 3, 4, 3, 3, 3, 3, 4, 3, 3, 3>(
  //   new Unicycle3DStateCostFunctor(dt_, sqrt_information_));
  //
  // which requires:
  //
  // #include <fuse_models/unicycle_3d_state_cost_functor.h>
}

}  // namespace fuse_models

BOOST_CLASS_EXPORT_IMPLEMENT(fuse_models::Unicycle3DStateKinematicConstraint);
PLUGINLIB_EXPORT_CLASS(fuse_models::Unicycle3DStateKinematicConstraint, fuse_core::Constraint);
