/***************************************************************************
 * Copyright (C) 2017 Locus Robotics. All rights reserved.
 * Unauthorized copying of this file, via any medium, is strictly prohibited
 * Proprietary and confidential
 ***************************************************************************/
#include <fuse_models/unicycle_3d.h>

#include <fuse_models/unicycle_3d_state_kinematic_constraint.h>
#include <fuse_graphs/hash_graph.h>
#include <fuse_variables/acceleration_linear_3d_stamped.h>
#include <fuse_variables/orientation_3d_stamped.h>
#include <fuse_variables/position_3d_stamped.h>
#include <fuse_variables/velocity_angular_3d_stamped.h>
#include <fuse_variables/velocity_linear_3d_stamped.h>
#include <ros/duration.h>
#include <ros/time.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Quaternion.h>

#include <gtest/gtest.h>
#include <fuse_core/eigen_gtest.h>

#include <string>
#include <vector>

/**
 * @brief Derived class used in unit tests to expose protected/private functions and variables
 */
class Unicycle3DModelTest : public fuse_models::Unicycle3D
{
public:
  using fuse_models::Unicycle3D::updateStateHistoryEstimates;
  using fuse_models::Unicycle3D::StateHistoryElement;
  using fuse_models::Unicycle3D::StateHistory;
  using fuse_models::Unicycle3D::generateMotionModel;

  void setProcessNoiseCovariance(const fuse_core::Matrix15d& process_noise_covariance)
  {
    process_noise_covariance_ = process_noise_covariance;
  }

  void setStateHistory(const StateHistory& state_history)
  {
    state_history_ = state_history;
  }

  void setScaleProcessNoise(const bool& scale_process_noise)
  {
    scale_process_noise_ = scale_process_noise;
  }

  void setDisableChecks(const bool& disable_checks)
  {
    disable_checks_ = disable_checks;
  }
};

TEST(Unicycle3D, UpdateStateHistoryEstimates)
{
  // Create some variables
  auto position1 = fuse_variables::Position3DStamped::make_shared(ros::Time(1, 0));
  auto orientation1 = fuse_variables::Orientation3DStamped::make_shared(ros::Time(1, 0));
  auto linear_velocity1 = fuse_variables::VelocityLinear3DStamped::make_shared(ros::Time(1, 0));
  auto angular_velocity1 = fuse_variables::VelocityAngular3DStamped::make_shared(ros::Time(1, 0));
  auto linear_acceleration1 =
    fuse_variables::AccelerationLinear3DStamped::make_shared(ros::Time(1, 0));
  position1->x() = 1.1;
  position1->y() = 2.1;
  position1->z() = 0.0;
  orientation1->w() = 1.0;
  orientation1->x() = 0.0;
  orientation1->y() = 0.0;
  orientation1->z() = 0.0;
  linear_velocity1->x() = 1.0;
  linear_velocity1->y() = 0.0;
  linear_velocity1->z() = 0.0;
  angular_velocity1->roll() = 0.0;
  angular_velocity1->pitch() = 0.0;
  angular_velocity1->yaw() = 0.0;
  linear_acceleration1->x() = 1.0;
  linear_acceleration1->y() = 0.0;
  linear_acceleration1->z() = 0.0;
  auto position2 = fuse_variables::Position3DStamped::make_shared(ros::Time(2, 0));
  auto orientation2 = fuse_variables::Orientation3DStamped::make_shared(ros::Time(2, 0));
  auto linear_velocity2 = fuse_variables::VelocityLinear3DStamped::make_shared(ros::Time(2, 0));
  auto angular_velocity2 = fuse_variables::VelocityAngular3DStamped::make_shared(ros::Time(2, 0));
  auto linear_acceleration2 =
    fuse_variables::AccelerationLinear3DStamped::make_shared(ros::Time(2, 0));
  position2->x() = 1.2;
  position2->y() = 2.2;
  position2->z() = 0.0;
  orientation2->w() = 1.0;
  orientation2->x() = 0.0;
  orientation2->y() = 0.0;
  orientation2->z() = 0.0;
  linear_velocity2->x() = 0.0;
  linear_velocity2->y() = 1.0;
  linear_velocity2->z() = 0.0;
  angular_velocity2->roll() = 0.0;
  angular_velocity2->pitch() = 0.0;
  angular_velocity2->yaw() = 0.0;
  linear_acceleration2->x() = 0.0;
  linear_acceleration2->y() = 1.0;
  linear_acceleration2->z() = 0.0;
  auto position3 = fuse_variables::Position3DStamped::make_shared(ros::Time(3, 0));
  auto orientation3 = fuse_variables::Orientation3DStamped::make_shared(ros::Time(3, 0));
  auto linear_velocity3 = fuse_variables::VelocityLinear3DStamped::make_shared(ros::Time(3, 0));
  auto angular_velocity3 = fuse_variables::VelocityAngular3DStamped::make_shared(ros::Time(3, 0));
  auto linear_acceleration3 =
    fuse_variables::AccelerationLinear3DStamped::make_shared(ros::Time(3, 0));
  position3->x() = 1.3;
  position3->y() = 2.3;
  position3->z() = 0.0;
  orientation3->w() = 1.0;
  orientation3->x() = 0.0;
  orientation3->y() = 0.0;
  orientation3->z() = 0.0;
  linear_velocity3->x() = 4.3;
  linear_velocity3->y() = 5.3;
  linear_velocity3->z() = 0.0;
  angular_velocity3->roll() = 0.0;
  angular_velocity3->pitch() = 0.0;
  angular_velocity3->yaw() = 0.0;
  linear_acceleration3->x() = 7.3;
  linear_acceleration3->y() = 8.3;
  linear_acceleration3->z() = 0.0;
  auto position4 = fuse_variables::Position3DStamped::make_shared(ros::Time(4, 0));
  auto orientation4 = fuse_variables::Orientation3DStamped::make_shared(ros::Time(4, 0));
  auto linear_velocity4 = fuse_variables::VelocityLinear3DStamped::make_shared(ros::Time(4, 0));
  auto angular_velocity4 = fuse_variables::VelocityAngular3DStamped::make_shared(ros::Time(4, 0));
  auto linear_acceleration4 =
    fuse_variables::AccelerationLinear3DStamped::make_shared(ros::Time(4, 0));
  position4->x() = 1.4;
  position4->y() = 2.4;
  position4->z() = 0.0;
  orientation4->w() = 1.0;
  orientation4->x() = 0.0;
  orientation4->y() = 0.0;
  orientation4->z() = 0.0;
  linear_velocity4->x() = 4.4;
  linear_velocity4->y() = 5.4;
  linear_velocity4->z() = 0.0;
  angular_velocity4->roll() = 0.0;
  angular_velocity4->pitch() = 0.0;
  angular_velocity4->yaw() = 0.0;
  linear_acceleration4->x() = 7.4;
  linear_acceleration4->y() = 8.4;
  linear_acceleration4->z() = 0.0;
  auto position5 = fuse_variables::Position3DStamped::make_shared(ros::Time(5, 0));
  auto orientation5 = fuse_variables::Orientation3DStamped::make_shared(ros::Time(5, 0));
  auto linear_velocity5 = fuse_variables::VelocityLinear3DStamped::make_shared(ros::Time(5, 0));
  auto angular_velocity5 = fuse_variables::VelocityAngular3DStamped::make_shared(ros::Time(5, 0));
  auto linear_acceleration5 =
    fuse_variables::AccelerationLinear3DStamped::make_shared(ros::Time(5, 0));
  position5->x() = 1.5;
  position5->y() = 2.5;
  position5->z() = 0.0;
  orientation5->w() = 1.0;
  orientation5->x() = 0.0;
  orientation5->y() = 0.0;
  orientation5->z() = 0.0;
  linear_velocity5->x() = 4.5;
  linear_velocity5->y() = 5.5;
  linear_velocity5->z() = 0.0;
  angular_velocity5->roll() = 0.0;
  angular_velocity5->pitch() = 0.0;
  angular_velocity5->yaw() = 0.0;
  linear_acceleration5->x() = 7.5;
  linear_acceleration5->y() = 8.5;
  linear_acceleration5->z() = 0.0;

  // Add a subset of the variables to a graph
  fuse_graphs::HashGraph graph;
  graph.addVariable(position2);
  graph.addVariable(orientation2);
  graph.addVariable(linear_velocity2);
  graph.addVariable(angular_velocity2);
  graph.addVariable(linear_acceleration2);

  graph.addVariable(position4);
  graph.addVariable(orientation4);
  graph.addVariable(linear_velocity4);
  graph.addVariable(angular_velocity4);
  graph.addVariable(linear_acceleration4);

  // Add all of the variables to the state history
  Unicycle3DModelTest::StateHistory state_history;
  state_history.emplace(
    position1->stamp(),
    Unicycle3DModelTest::StateHistoryElement{  // NOLINT(whitespace/braces)
    position1->uuid(),
    orientation1->uuid(),
    linear_velocity1->uuid(),
    angular_velocity1->uuid(),
    linear_acceleration1->uuid(),
    tf2::Transform{{0.0, 0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}},
    tf2::Vector3(0.0, 0.0, 0.0),
    tf2::Vector3(0.0, 0.0, 0.0),
    tf2::Vector3(0.0, 0.0, 0.0)});    // NOLINT(whitespace/braces)
  state_history.emplace(
    position2->stamp(),
    Unicycle3DModelTest::StateHistoryElement{  // NOLINT(whitespace/braces)
    position2->uuid(),
    orientation2->uuid(),
    linear_velocity2->uuid(),
    angular_velocity2->uuid(),
    linear_acceleration2->uuid(),
    tf2::Transform{{0.0, 0.0, 0.0, 1.0}, {2.0, 0.0, 0.0}},
    tf2::Vector3(0.0, 0.0, 0.0),
    tf2::Vector3(0.0, 0.0, 0.0),
    tf2::Vector3(0.0, 0.0, 0.0)});    // NOLINT(whitespace/braces)
  state_history.emplace(
    position3->stamp(),
    Unicycle3DModelTest::StateHistoryElement{  // NOLINT(whitespace/braces)
    position3->uuid(),
    orientation3->uuid(),
    linear_velocity3->uuid(),
    angular_velocity3->uuid(),
    linear_acceleration3->uuid(),
    tf2::Transform{{0.0, 0.0, 0.0, 1.0}, {3.0, 0.0, 0.0}},
    tf2::Vector3(0.0, 0.0, 0.0),
    tf2::Vector3(0.0, 0.0, 0.0),
    tf2::Vector3(0.0, 0.0, 0.0)});    // NOLINT(whitespace/braces)
  state_history.emplace(
    position4->stamp(),
    Unicycle3DModelTest::StateHistoryElement{  // NOLINT(whitespace/braces)
    position4->uuid(),
    orientation4->uuid(),
    linear_velocity4->uuid(),
    angular_velocity4->uuid(),
    linear_acceleration4->uuid(),
    tf2::Transform{{0.0, 0.0, 0.0, 1.0}, {4.0, 0.0, 0.0}},
    tf2::Vector3(0.0, 0.0, 0.0),
    tf2::Vector3(0.0, 0.0, 0.0),
    tf2::Vector3(0.0, 0.0, 0.0)});    // NOLINT(whitespace/braces)
  state_history.emplace(
    position5->stamp(),
    Unicycle3DModelTest::StateHistoryElement{  // NOLINT(whitespace/braces)
    position5->uuid(),
    orientation5->uuid(),
    linear_velocity5->uuid(),
    angular_velocity5->uuid(),
    linear_acceleration5->uuid(),
    tf2::Transform{{0.0, 0.0, 0.0, 1.0}, {5.0, 0.0, 0.0}},
    tf2::Vector3(0.0, 0.0, 0.0),
    tf2::Vector3(0.0, 0.0, 0.0),
    tf2::Vector3(0.0, 0.0, 0.0)});    // NOLINT(whitespace/braces)

  // Update the state history
  Unicycle3DModelTest::updateStateHistoryEstimates(
    graph, state_history, ros::Duration(10.0));

  // Check the state estimates in the state history
  {
    // The first entry is missing from the graph. It will not get updated.
    auto expected_position = fuse_core::Vector3d(1.0, 0.0, 0.0);
    auto expected_orientation = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
    auto actual_position = state_history[ros::Time(1, 0)].pose.getOrigin();
    auto actual_orientation = state_history[ros::Time(1, 0)].pose.getRotation();
    EXPECT_NEAR(expected_position.x(), actual_position.x(), 1.0e-9);
    EXPECT_NEAR(expected_position.y(), actual_position.y(), 1.0e-9);
    EXPECT_NEAR(expected_position.z(), actual_position.z(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.x(), actual_orientation.x(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.y(), actual_orientation.y(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.z(), actual_orientation.z(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.w(), actual_orientation.w(), 1.0e-9);

    auto expected_linear_velocity = fuse_core::Vector3d(0.0, 0.0, 0.0);
    auto actual_linear_velocity = state_history[ros::Time(1, 0)].velocity_linear;
    EXPECT_NEAR(expected_linear_velocity.x(), actual_linear_velocity.x(), 1.0e-9);
    EXPECT_NEAR(expected_linear_velocity.y(), actual_linear_velocity.y(), 1.0e-9);
    EXPECT_NEAR(expected_linear_velocity.z(), actual_linear_velocity.z(), 1.0e-9);

    auto expected_angular_velocity = fuse_core::Vector3d(0.0, 0.0, 0.0);
    auto actual_angular_velocity = state_history[ros::Time(1, 0)].velocity_angular;
    EXPECT_NEAR(expected_angular_velocity.x(), actual_angular_velocity.x(), 1.0e-9);
    EXPECT_NEAR(expected_angular_velocity.y(), actual_angular_velocity.y(), 1.0e-9);
    EXPECT_NEAR(expected_angular_velocity.z(), actual_angular_velocity.z(), 1.0e-9);

    auto expected_linear_acceleration = fuse_core::Vector3d(0.0, 0.0, 0.0);
    auto actual_linear_acceleration = state_history[ros::Time(1, 0)].acceleration_linear;
    EXPECT_NEAR(expected_linear_acceleration.x(), actual_linear_acceleration.x(), 1.0e-9);
    EXPECT_NEAR(expected_linear_acceleration.y(), actual_linear_acceleration.y(), 1.0e-9);
    EXPECT_NEAR(expected_linear_acceleration.z(), actual_linear_acceleration.z(), 1.0e-9);
  }
  {
    // The second entry is included in the graph. It will get updated directly.
    auto expected_position = fuse_core::Vector3d(1.2, 2.2, 0.0);  // <-- value in the Graph
    auto expected_orientation = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
    auto actual_position = state_history[ros::Time(2, 0)].pose.getOrigin();
    auto actual_orientation = state_history[ros::Time(2, 0)].pose.getRotation();
    EXPECT_NEAR(expected_position.x(), actual_position.x(), 1.0e-9);
    EXPECT_NEAR(expected_position.y(), actual_position.y(), 1.0e-9);
    EXPECT_NEAR(expected_position.z(), actual_position.z(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.x(), actual_orientation.x(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.y(), actual_orientation.y(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.z(), actual_orientation.z(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.w(), actual_orientation.w(), 1.0e-9);

    auto expected_linear_velocity = fuse_core::Vector3d(0.0, 1.0, 0.0);
    auto actual_linear_velocity = state_history[ros::Time(2, 0)].velocity_linear;
    EXPECT_NEAR(expected_linear_velocity.x(), actual_linear_velocity.x(), 1.0e-9);
    EXPECT_NEAR(expected_linear_velocity.y(), actual_linear_velocity.y(), 1.0e-9);
    EXPECT_NEAR(expected_linear_velocity.z(), actual_linear_velocity.z(), 1.0e-9);

    auto expected_angular_velocity = fuse_core::Vector3d(0.0, 0.0, 0.0);
    auto actual_angular_velocity = state_history[ros::Time(2, 0)].velocity_angular;
    EXPECT_NEAR(expected_angular_velocity.x(), actual_angular_velocity.x(), 1.0e-9);
    EXPECT_NEAR(expected_angular_velocity.y(), actual_angular_velocity.y(), 1.0e-9);
    EXPECT_NEAR(expected_angular_velocity.z(), actual_angular_velocity.z(), 1.0e-9);

    auto expected_linear_acceleration = fuse_core::Vector3d(0.0, 1.0, 0.0);
    auto actual_linear_acceleration = state_history[ros::Time(2, 0)].acceleration_linear;
    EXPECT_NEAR(expected_linear_acceleration.x(), actual_linear_acceleration.x(), 1.0e-9);
    EXPECT_NEAR(expected_linear_acceleration.y(), actual_linear_acceleration.y(), 1.0e-9);
    EXPECT_NEAR(expected_linear_acceleration.z(), actual_linear_acceleration.z(), 1.0e-9);
  }
  {
    // The third entry is missing from the graph. It will get predicted from previous state.
    auto expected_position = fuse_core::Vector3d(1.2, 3.7, 0.0);
    auto expected_orientation =
      Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());
    auto actual_position = state_history[ros::Time(3, 0)].pose.getOrigin();
    auto actual_orientation = state_history[ros::Time(3, 0)].pose.getRotation();
    EXPECT_NEAR(expected_position.x(), actual_position.x(), 1.0e-9);
    EXPECT_NEAR(expected_position.y(), actual_position.y(), 1.0e-9);
    EXPECT_NEAR(expected_position.z(), actual_position.z(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.x(), actual_orientation.x(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.y(), actual_orientation.y(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.z(), actual_orientation.z(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.w(), actual_orientation.w(), 1.0e-9);

    auto expected_linear_velocity = fuse_core::Vector3d(0.0, 2.0, 0.0);
    auto actual_linear_velocity = state_history[ros::Time(3, 0)].velocity_linear;
    EXPECT_NEAR(expected_linear_velocity.x(), actual_linear_velocity.x(), 1.0e-9);
    EXPECT_NEAR(expected_linear_velocity.y(), actual_linear_velocity.y(), 1.0e-9);
    EXPECT_NEAR(expected_linear_velocity.z(), actual_linear_velocity.z(), 1.0e-9);

    auto expected_angular_velocity = fuse_core::Vector3d(0.0, 0.0, 0.0);
    auto actual_angular_velocity = state_history[ros::Time(3, 0)].velocity_angular;
    EXPECT_NEAR(expected_angular_velocity.x(), actual_angular_velocity.x(), 1.0e-9);
    EXPECT_NEAR(expected_angular_velocity.y(), actual_angular_velocity.y(), 1.0e-9);
    EXPECT_NEAR(expected_angular_velocity.z(), actual_angular_velocity.z(), 1.0e-9);

    auto expected_linear_acceleration = fuse_core::Vector3d(0.0, 1.0, 0.0);
    auto actual_linear_acceleration = state_history[ros::Time(3, 0)].acceleration_linear;
    EXPECT_NEAR(expected_linear_acceleration.x(), actual_linear_acceleration.x(), 1.0e-9);
    EXPECT_NEAR(expected_linear_acceleration.y(), actual_linear_acceleration.y(), 1.0e-9);
    EXPECT_NEAR(expected_linear_acceleration.z(), actual_linear_acceleration.z(), 1.0e-9);
  }
  {
    // The forth entry is included in the graph. It will get updated directly.
    auto expected_position = fuse_core::Vector3d(1.4, 2.4, 0.0);  // <-- value in the Graph
    auto expected_orientation =
      Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());
    auto actual_position = state_history[ros::Time(4, 0)].pose.getOrigin();
    auto actual_orientation = state_history[ros::Time(4, 0)].pose.getRotation();
    EXPECT_NEAR(expected_position.x(), actual_position.x(), 1.0e-9);
    EXPECT_NEAR(expected_position.y(), actual_position.y(), 1.0e-9);
    EXPECT_NEAR(expected_position.z(), actual_position.z(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.x(), actual_orientation.x(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.y(), actual_orientation.y(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.z(), actual_orientation.z(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.w(), actual_orientation.w(), 1.0e-9);

    auto expected_linear_velocity = fuse_core::Vector3d(4.4, 5.4, 0.0);
    auto actual_linear_velocity = state_history[ros::Time(4, 0)].velocity_linear;
    EXPECT_NEAR(expected_linear_velocity.x(), actual_linear_velocity.x(), 1.0e-9);
    EXPECT_NEAR(expected_linear_velocity.y(), actual_linear_velocity.y(), 1.0e-9);
    EXPECT_NEAR(expected_linear_velocity.z(), actual_linear_velocity.z(), 1.0e-9);

    auto expected_angular_velocity = fuse_core::Vector3d(0.0, 0.0, 0.0);
    auto actual_angular_velocity = state_history[ros::Time(4, 0)].velocity_angular;
    EXPECT_NEAR(expected_angular_velocity.x(), actual_angular_velocity.x(), 1.0e-9);
    EXPECT_NEAR(expected_angular_velocity.y(), actual_angular_velocity.y(), 1.0e-9);
    EXPECT_NEAR(expected_angular_velocity.z(), actual_angular_velocity.z(), 1.0e-9);

    auto expected_linear_acceleration = fuse_core::Vector3d(7.4, 8.4, 0.0);
    auto actual_linear_acceleration = state_history[ros::Time(4, 0)].acceleration_linear;
    EXPECT_NEAR(expected_linear_acceleration.x(), actual_linear_acceleration.x(), 1.0e-9);
    EXPECT_NEAR(expected_linear_acceleration.y(), actual_linear_acceleration.y(), 1.0e-9);
    EXPECT_NEAR(expected_linear_acceleration.z(), actual_linear_acceleration.z(), 1.0e-9);
  }
  {
    // The fifth entry is missing from the graph. It will get predicted from previous state.
    auto expected_position = fuse_core::Vector3d(9.5, 12.0, 0.0);
    auto expected_orientation =
      Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());
    auto actual_position = state_history[ros::Time(5, 0)].pose.getOrigin();
    auto actual_orientation = state_history[ros::Time(5, 0)].pose.getRotation();
    EXPECT_NEAR(expected_position.x(), actual_position.x(), 1.0e-9);
    EXPECT_NEAR(expected_position.y(), actual_position.y(), 1.0e-9);
    EXPECT_NEAR(expected_position.z(), actual_position.z(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.x(), actual_orientation.x(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.y(), actual_orientation.y(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.z(), actual_orientation.z(), 1.0e-9);
    EXPECT_NEAR(expected_orientation.w(), actual_orientation.w(), 1.0e-9);

    auto expected_linear_velocity = fuse_core::Vector3d(11.8, 13.8, 0.0);
    auto actual_linear_velocity = state_history[ros::Time(5, 0)].velocity_linear;
    EXPECT_NEAR(expected_linear_velocity.x(), actual_linear_velocity.x(), 1.0e-9);
    EXPECT_NEAR(expected_linear_velocity.y(), actual_linear_velocity.y(), 1.0e-9);
    EXPECT_NEAR(expected_linear_velocity.z(), actual_linear_velocity.z(), 1.0e-9);

    auto expected_angular_velocity = fuse_core::Vector3d(0.0, 0.0, 0.0);
    auto actual_angular_velocity = state_history[ros::Time(5, 0)].velocity_angular;
    EXPECT_NEAR(expected_angular_velocity.x(), actual_angular_velocity.x(), 1.0e-9);
    EXPECT_NEAR(expected_angular_velocity.y(), actual_angular_velocity.y(), 1.0e-9);
    EXPECT_NEAR(expected_angular_velocity.z(), actual_angular_velocity.z(), 1.0e-9);

    auto expected_linear_acceleration = fuse_core::Vector3d(7.4, 8.4, 0.0);
    auto actual_linear_acceleration = state_history[ros::Time(5, 0)].acceleration_linear;
    EXPECT_NEAR(expected_linear_acceleration.x(), actual_linear_acceleration.x(), 1.0e-9);
    EXPECT_NEAR(expected_linear_acceleration.y(), actual_linear_acceleration.y(), 1.0e-9);
    EXPECT_NEAR(expected_linear_acceleration.z(), actual_linear_acceleration.z(), 1.0e-9);
  }
}

// TODO(fhirmann): adapt test copied from test_unicycle2D.cpp to 3D
/* TEST(Unicycle3D, generateMotionModel)
{
  // Create some variables
  auto position1 = fuse_variables::Position3DStamped::make_shared(ros::Time(1, 0));
  auto yaw1 = fuse_variables::Orientation3DStamped::make_shared(ros::Time(1, 0));
  auto linear_velocity1 = fuse_variables::VelocityLinear3DStamped::make_shared(ros::Time(1, 0));
  auto yaw_velocity1 = fuse_variables::VelocityAngular3DStamped::make_shared(ros::Time(1, 0));
  auto linear_acceleration1 = fuse_variables::AccelerationLinear3DStamped::make_shared(ros::Time(1, 0));
  position1->x() = 1.1;
  position1->y() = 2.1;
  yaw1->setYaw(1.2);
  linear_velocity1->x() = 1.0;
  linear_velocity1->y() = 0.0;
  yaw_velocity1->yaw() = 0.0;
  linear_acceleration1->x() = 1.0;
  linear_acceleration1->y() = 0.0;

  const ros::Time beginning_stamp = position1->stamp();
  const ros::Time ending_stamp = beginning_stamp + ros::Duration{1.0};

  Unicycle3DModelTest::StateHistory state_history;
  state_history.emplace(
    position1->stamp(),
    Unicycle3DModelTest::StateHistoryElement{  // NOLINT(whitespace/braces)
      position1->uuid(),
      yaw1->uuid(),
      linear_velocity1->uuid(),
      yaw_velocity1->uuid(),
      linear_acceleration1->uuid(),
      tf2::Transform(position1->x(), position1->y(), yaw1->getYaw()),
      tf2::Vector2(linear_velocity1->x(), linear_velocity1->y()),
      yaw_velocity1->yaw(),
      tf2::Vector2(linear_acceleration1->x(), linear_acceleration1->y())});  // NOLINT(whitespace/braces)

  Unicycle3DModelTest unicycle_3d_model_test;

  unicycle_3d_model_test.setStateHistory(state_history);

  std::vector<double> process_noise_diagonal{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  unicycle_3d_model_test.setProcessNoiseCovariance(fuse_core::Vector8d(process_noise_diagonal.data()).asDiagonal());

  unicycle_3d_model_test.setScaleProcessNoise(false);
  unicycle_3d_model_test.setDisableChecks(false);

  std::vector<fuse_core::Constraint::SharedPtr> constraints;
  std::vector<fuse_core::Variable::SharedPtr> variables;

  // Generate the motion model
  unicycle_3d_model_test.generateMotionModel(
    beginning_stamp,
    ending_stamp,
    constraints,
    variables);

  // Check first the created variables
  {
    // check the first entry (the original value before prediction)

    auto position = std::static_pointer_cast<fuse_variables::Position3DStamped>(variables.at(0));
    auto orientation = std::static_pointer_cast<fuse_variables::Orientation3DStamped>(variables.at(1));
    auto velocity_linear = std::static_pointer_cast<fuse_variables::VelocityLinear3DStamped>(variables.at(2));
    auto velocity_yaw = std::static_pointer_cast<fuse_variables::VelocityAngular3DStamped>(variables.at(3));
    auto acceleration_linear = std::static_pointer_cast<fuse_variables::AccelerationLinear3DStamped>(variables.at(4));

    auto expected_pose = tf2::Transform(position1->x(), position1->y(), yaw1->getYaw());
    auto expected_linear_velocity = tf2::Vector2(linear_velocity1->x(), linear_velocity1->y());
    auto expected_yaw_velocity = yaw_velocity1->yaw();
    auto expected_linear_acceleration = tf2::Vector2(linear_acceleration1->x(), linear_acceleration1->y());

    EXPECT_EQ(position->x(), expected_pose.x());
    EXPECT_EQ(position->y(), expected_pose.y());
    EXPECT_EQ(orientation->getYaw(), expected_pose.getYaw());

    EXPECT_EQ(velocity_linear->x(), expected_linear_velocity.x());
    EXPECT_EQ(velocity_linear->y(), expected_linear_velocity.y());

    EXPECT_EQ(velocity_yaw->yaw(), expected_yaw_velocity);

    EXPECT_EQ(acceleration_linear->x(), expected_linear_acceleration.x());
    EXPECT_EQ(acceleration_linear->y(), expected_linear_acceleration.y());
  }

  {
    // check the second entry (the prediction)

    auto position = std::static_pointer_cast<fuse_variables::Position3DStamped>(variables.at(5));
    auto orientation = std::static_pointer_cast<fuse_variables::Orientation3DStamped>(variables.at(6));
    auto velocity_linear = std::static_pointer_cast<fuse_variables::VelocityLinear3DStamped>(variables.at(7));
    auto velocity_yaw = std::static_pointer_cast<fuse_variables::VelocityAngular3DStamped>(variables.at(8));
    auto acceleration_linear = std::static_pointer_cast<fuse_variables::AccelerationLinear3DStamped>(variables.at(9));

    auto expected_pose = tf2::Transform(1.643536632, 3.498058629, 1.2);
    auto expected_linear_velocity = tf2::Vector2(2.0, 0.0);
    auto expected_yaw_velocity = 0.0;
    auto expected_linear_acceleration = tf2::Vector2(1.0, 0.0);

    EXPECT_NEAR(position->x(), expected_pose.x(), 1.0e-9);
    EXPECT_NEAR(position->y(), expected_pose.y(), 1.0e-9);
    EXPECT_NEAR(orientation->getYaw(), expected_pose.getYaw(), 1.0e-9);

    EXPECT_NEAR(velocity_linear->x(), expected_linear_velocity.x(), 1.0e-9);
    EXPECT_NEAR(velocity_linear->y(), expected_linear_velocity.y(), 1.0e-9);

    EXPECT_NEAR(velocity_yaw->yaw(), expected_yaw_velocity, 1.0e-9);

    EXPECT_NEAR(acceleration_linear->x(), expected_linear_acceleration.x(), 1.0e-9);
    EXPECT_NEAR(acceleration_linear->y(), expected_linear_acceleration.y(), 1.0e-9);
  }

  // Now check the created constraint
  {
    auto new_constraint =
      std::static_pointer_cast<fuse_models::Unicycle3DStateKinematicConstraint>(constraints.back());

    // As based on the yaw angle we are mainly moving upwards and the process noise covariance
    // is defined in the robot frame (original: cov_xx=1.0, cov_yy=2.0) it is expected that the final
    // process noise covariance in the world frame (so the frame where optimization happens) is higher
    // at cov_xx than in cov_yy.
    // Additionally, cov_xy and cov_yx are also no more zero (so no more independent between x and y) because
    // again it is moving simultaneously in the x- and y-direction.
    fuse_core::Matrix8d expected_cov;
    expected_cov <<  1.868696858, -0.33773159,        0,       0,       0,       0,       0,       0,
                    -0.33773159,   1.131303142,       0,       0,       0,       0,       0,       0,
                              0,             0,       3,       0,       0,       0,       0,       0,
                              0,             0,       0,       4,       0,       0,       0,       0,
                              0,             0,       0,       0,       5,       0,       0,       0,
                              0,             0,       0,       0,       0,       6,       0,       0,
                              0,             0,       0,       0,       0,       0,       7,       0,
                              0,             0,       0,       0,       0,       0,       0,       8;

    EXPECT_MATRIX_NEAR(expected_cov, new_constraint->covariance(), 1.0e-9);
  }
} */

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "unicycle_3d_test");
  auto spinner = ros::AsyncSpinner(1);
  spinner.start();
  int ret = RUN_ALL_TESTS();
  spinner.stop();
  ros::shutdown();
  return ret;
}
