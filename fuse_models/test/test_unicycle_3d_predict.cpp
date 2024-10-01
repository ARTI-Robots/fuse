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

#include <fuse_models/unicycle_3d_predict.h>

#include <gtest/gtest.h>
#include <fuse_core/eigen_gtest.h>

#include <array>
#include <limits>
#include <vector>

#include <ceres/jet.h>

TEST(Predict, predictDirectVals)
{
  tf2::Transform pose1 { tf2::Quaternion{ 0.0, 0.0, 0.0, 1.0},
                         tf2::Vector3{0.0, 0.0, 0.0} };
  tf2::Vector3 vel_linear1 { 1.0, 0.0, 0.0};
  tf2::Vector3 vel_angular1 { 0.0, 0.0, 1.570796327 };
  tf2::Vector3 acc_linear1 {1.0, 0.0, 0.0 };
  double dt = 0.1;
  tf2::Transform pose2;
  tf2::Vector3 vel_linear2;
  tf2::Vector3 vel_angular2;
  tf2::Vector3 acc_linear2;

  fuse_models::predict(
    pose1,
    vel_linear1,
    vel_angular1,
    acc_linear1,
    dt,
    pose2,
    vel_linear2,
    vel_angular2,
    acc_linear2);

  Eigen::Quaterniond q;
  q = Eigen::AngleAxisd(0.1570796327, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());

  EXPECT_DOUBLE_EQ(0.105, pose2.getOrigin().x());
  EXPECT_DOUBLE_EQ(0.0, pose2.getOrigin().y());
  EXPECT_DOUBLE_EQ(0.0, pose2.getOrigin().z());
  EXPECT_DOUBLE_EQ(q.w(), pose2.getRotation().w());
  EXPECT_DOUBLE_EQ(q.x(), pose2.getRotation().x());
  EXPECT_DOUBLE_EQ(q.y(), pose2.getRotation().y());
  EXPECT_DOUBLE_EQ(q.z(), pose2.getRotation().z());
  EXPECT_DOUBLE_EQ(1.1, vel_linear2.x());
  EXPECT_DOUBLE_EQ(0.0, vel_linear2.y());
  EXPECT_DOUBLE_EQ(0.0, vel_linear2.z());
  EXPECT_DOUBLE_EQ(0.0, vel_angular2.x());
  EXPECT_DOUBLE_EQ(0.0, vel_angular2.y());
  EXPECT_DOUBLE_EQ(1.570796327, vel_angular2.z());
  EXPECT_DOUBLE_EQ(1.0, acc_linear2.x());
  EXPECT_DOUBLE_EQ(0.0, acc_linear2.y());
  EXPECT_DOUBLE_EQ(0.0, acc_linear2.z());

  // // Carry on with the output state from last time - show in-place update support
  fuse_models::predict(
    pose2,
    vel_linear2,
    vel_angular2,
    acc_linear2,
    dt,
    pose2,
    vel_linear2,
    vel_angular2,
    acc_linear2);

  q = Eigen::AngleAxisd(0.3141592654, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());

  EXPECT_DOUBLE_EQ(0.21858415916807189, pose2.getOrigin().x());
  EXPECT_DOUBLE_EQ(0.017989963481956205, pose2.getOrigin().y());
  EXPECT_DOUBLE_EQ(0.0, pose2.getOrigin().z());
  EXPECT_DOUBLE_EQ(q.w(), pose2.getRotation().w());
  EXPECT_DOUBLE_EQ(q.x(), pose2.getRotation().x());
  EXPECT_DOUBLE_EQ(q.y(), pose2.getRotation().y());
  EXPECT_DOUBLE_EQ(q.z(), pose2.getRotation().z());
  EXPECT_DOUBLE_EQ(1.2, vel_linear2.x());
  EXPECT_DOUBLE_EQ(0.0, vel_linear2.y());
  EXPECT_DOUBLE_EQ(0.0, vel_linear2.z());
  EXPECT_DOUBLE_EQ(0.0, vel_angular2.x());
  EXPECT_DOUBLE_EQ(0.0, vel_angular2.y());
  EXPECT_DOUBLE_EQ(1.570796327, vel_angular2.z());
  EXPECT_DOUBLE_EQ(1.0, acc_linear2.x());
  EXPECT_DOUBLE_EQ(0.0, acc_linear2.y());
  EXPECT_DOUBLE_EQ(0.0, acc_linear2.z());

  // Use non-zero Y values
  vel_linear1.setY(-1.0);
  vel_angular1.setZ(-1.570796327);
  acc_linear1.setY(-1.0);

  fuse_models::predict(
    pose1,
    vel_linear1,
    vel_angular1,
    acc_linear1,
    dt,
    pose2,
    vel_linear2,
    vel_angular2,
    acc_linear2);

  q = Eigen::AngleAxisd(-0.1570796327, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());

  Eigen::Quaterniond q_2{ pose2.getRotation().getW(),  // NOLINT(whitespace/braces)
                          pose2.getRotation().getX(),
                          pose2.getRotation().getY(),
                          pose2.getRotation().getZ()};  // NOLINT(whitespace/braces)

  EXPECT_DOUBLE_EQ(0.105, pose2.getOrigin().x());
  EXPECT_DOUBLE_EQ(-0.105, pose2.getOrigin().y());
  EXPECT_DOUBLE_EQ(0.0, pose2.getOrigin().z());
  EXPECT_TRUE(q.isApprox(q_2));
  EXPECT_DOUBLE_EQ(1.1, vel_linear2.x());
  EXPECT_DOUBLE_EQ(-1.1, vel_linear2.y());
  EXPECT_DOUBLE_EQ(0.0, vel_linear2.z());
  EXPECT_DOUBLE_EQ(0.0, vel_angular2.x());
  EXPECT_DOUBLE_EQ(0.0, vel_angular2.y());
  EXPECT_DOUBLE_EQ(-1.570796327, vel_angular2.z());
  EXPECT_DOUBLE_EQ(1.0, acc_linear2.x());
  EXPECT_DOUBLE_EQ(-1.0, acc_linear2.y());
  EXPECT_DOUBLE_EQ(0.0, acc_linear2.z());

  // Out of plane motion
  pose1.setOrigin({0.0, 0.0, 0.0});
  pose1.setRotation({ 0.0, 0.0, 0.0, 1.0});
  vel_linear1 = {0.0, 0.0, 0.1};
  vel_angular1 = {1.570796327, 0.0, 0.0};
  acc_linear1 = {0.0, 0.0, 1.0};
  dt = 0.1;

  fuse_models::predict(
    pose1,
    vel_linear1,
    vel_angular1,
    acc_linear1,
    dt,
    pose2,
    vel_linear2,
    vel_angular2,
    acc_linear2
  );  // NOLINT(whitespace/parens)

  EXPECT_DOUBLE_EQ(0.0, pose2.getOrigin().x());
  EXPECT_DOUBLE_EQ(0.0, pose2.getOrigin().y());
  EXPECT_DOUBLE_EQ(0.015, pose2.getOrigin().z());
  EXPECT_DOUBLE_EQ(0.99691733373232339, pose2.getRotation().w());
  EXPECT_DOUBLE_EQ(0.078459095738068516, pose2.getRotation().x());
  EXPECT_DOUBLE_EQ(0.0, pose2.getRotation().y());
  EXPECT_DOUBLE_EQ(0.0, pose2.getRotation().z());
  EXPECT_DOUBLE_EQ(0.0, vel_linear2.x());
  EXPECT_DOUBLE_EQ(0.0, vel_linear2.y());
  EXPECT_DOUBLE_EQ(0.2, vel_linear2.z());
  EXPECT_DOUBLE_EQ(1.570796327, vel_angular2.x());
  EXPECT_DOUBLE_EQ(0.0, vel_angular2.y());
  EXPECT_DOUBLE_EQ(0.0, vel_angular2.z());
  EXPECT_DOUBLE_EQ(0.0, acc_linear2.x());
  EXPECT_DOUBLE_EQ(0.0, acc_linear2.y());
  EXPECT_DOUBLE_EQ(1.0, acc_linear2.z());

  // General 3D motion (these value are checked against rl predict() equations)
  pose1.setOrigin({0.0, 0.0, 0.0});
  pose1.setRotation({-0.003, -0.943, 0.314, 0.110});  // RPY {-2.490, -0.206, 3.066}
  vel_linear1 = {0.1, 0.2, 0.1};
  vel_angular1 = {1.570796327, 1.570796327, -1.570796327};
  acc_linear1 = {-0.5, 1.0, 1.0};
  dt = 0.1;

  fuse_models::predict(
    pose1,
    vel_linear1,
    vel_angular1,
    acc_linear1,
    dt,
    pose2,
    vel_linear2,
    vel_angular2,
    acc_linear2
  );  // NOLINT(whitespace/parens)

  // TODO(fhirmann): The resulting quaternion might have all components negated but effectively
  // this is the same rotation but this is not checked in the tests
  EXPECT_NEAR(-0.012044123300410431, pose2.getOrigin().x(), 1e-4);
  EXPECT_NEAR(0.011755776496514461, pose2.getOrigin().y(), 1e-4);
  EXPECT_NEAR(-0.024959783911094033, pose2.getOrigin().z(), 1e-4);
  EXPECT_NEAR(-0.20388993714859482, pose2.getRotation().w(), 1e-4);
  EXPECT_NEAR(-0.061993007799788086, pose2.getRotation().x(), 1e-4);
  EXPECT_NEAR(0.90147820778463239, pose2.getRotation().y(), 1e-4);
  EXPECT_NEAR(-0.3767264277999153, pose2.getRotation().z(), 1e-4);
  EXPECT_DOUBLE_EQ(0.05, vel_linear2.x());
  EXPECT_DOUBLE_EQ(0.3, vel_linear2.y());
  EXPECT_DOUBLE_EQ(0.2, vel_linear2.z());
  EXPECT_DOUBLE_EQ(1.570796327, vel_angular2.x());
  EXPECT_DOUBLE_EQ(1.570796327, vel_angular2.y());
  EXPECT_DOUBLE_EQ(-1.570796327, vel_angular2.z());
  EXPECT_DOUBLE_EQ(-0.5, acc_linear2.x());
  EXPECT_DOUBLE_EQ(1.0, acc_linear2.y());
  EXPECT_DOUBLE_EQ(1.0, acc_linear2.z());
}

TEST(Predict, predictFromDoublePointers)
{
  double position1[3] {0.0, 0.0, 0.0};
  double orientation1[3] {0.0, 0.0, 0.0};
  double vel_linear1[3] {1.0, 0.0, 0.0};
  double vel_angular1[3] {0.0, 0.0, 1.570796327};
  double acc_linear1[3] {1.0, 0.0, 0.0};
  double dt = 0.1;
  double position2[3];
  double orientation2[3];
  double vel_linear2[3];
  double vel_angular2[3];
  double acc_linear2[3];

  fuse_models::predict(
    position1,
    orientation1,
    vel_linear1,
    vel_angular1,
    acc_linear1,
    dt,
    position2,
    orientation2,
    vel_linear2,
    vel_angular2,
    acc_linear2);

  EXPECT_DOUBLE_EQ(0.105, position2[0]);
  EXPECT_DOUBLE_EQ(0.0,   position2[1]);
  EXPECT_DOUBLE_EQ(0.0,   position2[2]);
  EXPECT_DOUBLE_EQ(0.0, orientation2[0]);
  EXPECT_DOUBLE_EQ(0.0, orientation2[1]);
  EXPECT_DOUBLE_EQ(0.1570796327, orientation2[2]);
  EXPECT_DOUBLE_EQ(1.1,   vel_linear2[0]);
  EXPECT_DOUBLE_EQ(0.0,   vel_linear2[1]);
  EXPECT_DOUBLE_EQ(0.0,   vel_linear2[2]);
  EXPECT_DOUBLE_EQ(0.0,   vel_angular2[0]);
  EXPECT_DOUBLE_EQ(0.0,   vel_angular2[1]);
  EXPECT_DOUBLE_EQ(1.570796327, vel_angular2[2]);
  EXPECT_DOUBLE_EQ(1.0, acc_linear2[0]);
  EXPECT_DOUBLE_EQ(0.0, acc_linear2[1]);
  EXPECT_DOUBLE_EQ(0.0, acc_linear2[2]);

  // Carry on with the output state from last time - show in-place update support
  fuse_models::predict(
      position2,
      orientation2,
      vel_linear2,
      vel_angular2,
      acc_linear2,
      dt,
      position2,
      orientation2,
      vel_linear2,
      vel_angular2,
      acc_linear2);

  EXPECT_DOUBLE_EQ(0.21858415916807189,  position2[0]);
  EXPECT_DOUBLE_EQ(0.017989963481956205, position2[1]);
  EXPECT_DOUBLE_EQ(0.0, position2[2]);
  EXPECT_DOUBLE_EQ(0.0, orientation2[0]);
  EXPECT_DOUBLE_EQ(0.0, orientation2[1]);
  EXPECT_DOUBLE_EQ(0.3141592654, orientation2[2]);
  EXPECT_DOUBLE_EQ(1.2,   vel_linear2[0]);
  EXPECT_DOUBLE_EQ(0.0,   vel_linear2[1]);
  EXPECT_DOUBLE_EQ(0.0,   vel_linear2[2]);
  EXPECT_DOUBLE_EQ(0.0,   vel_angular2[0]);
  EXPECT_DOUBLE_EQ(0.0,   vel_angular2[1]);
  EXPECT_DOUBLE_EQ(1.570796327, vel_angular2[2]);
  EXPECT_DOUBLE_EQ(1.0, acc_linear2[0]);
  EXPECT_DOUBLE_EQ(0.0, acc_linear2[1]);
  EXPECT_DOUBLE_EQ(0.0, acc_linear2[2]);

  // Use non-zero Y values
  vel_linear1[1] = -1.0;
  vel_angular1[2] = -1.570796327;
  acc_linear1[1] = -1.0;

  fuse_models::predict(
      position1,
      orientation1,
      vel_linear1,
      vel_angular1,
      acc_linear1,
      dt,
      position2,
      orientation2,
      vel_linear2,
      vel_angular2,
      acc_linear2);

  EXPECT_DOUBLE_EQ(0.105,  position2[0]);
  EXPECT_DOUBLE_EQ(-0.105, position2[1]);
  EXPECT_DOUBLE_EQ(0.0,    position2[2]);
  EXPECT_DOUBLE_EQ(0.0, orientation2[0]);
  EXPECT_DOUBLE_EQ(0.0, orientation2[1]);
  EXPECT_DOUBLE_EQ(-0.1570796327, orientation2[2]);
  EXPECT_DOUBLE_EQ(1.1,   vel_linear2[0]);
  EXPECT_DOUBLE_EQ(-1.1,  vel_linear2[1]);
  EXPECT_DOUBLE_EQ(0.0,   vel_linear2[2]);
  EXPECT_DOUBLE_EQ(0.0,   vel_angular2[0]);
  EXPECT_DOUBLE_EQ(0.0,   vel_angular2[1]);
  EXPECT_DOUBLE_EQ(-1.570796327, vel_angular2[2]);
  EXPECT_DOUBLE_EQ(1.0,  acc_linear2[0]);
  EXPECT_DOUBLE_EQ(-1.0, acc_linear2[1]);
  EXPECT_DOUBLE_EQ(0.0,  acc_linear2[2]);

  // Out of plane motion
  position1[0] = 0.0;
  position1[1] = 0.0;
  position1[2] = 0.0;
  orientation1[0] = 0.0;
  orientation1[1] = 0.0;
  orientation1[2] = 0.0;
  vel_linear1[0] = 0.0;
  vel_linear1[1] = 0.0;
  vel_linear1[2] = 0.1;
  vel_angular1[0] = 1.570796327;
  vel_angular1[1] = 0.0;
  vel_angular1[2] = 0.0;
  acc_linear1[0] = 0.0;
  acc_linear1[1] = 0.0;
  acc_linear1[2] = 1.0;
  dt = 0.1;

  fuse_models::predict(
    position1,
    orientation1,
    vel_linear1,
    vel_angular1,
    acc_linear1,
    dt,
    position2,
    orientation2,
    vel_linear2,
    vel_angular2,
    acc_linear2
  );  // NOLINT(whitespace/parens)

  EXPECT_DOUBLE_EQ(0.0, position2[0]);
  EXPECT_DOUBLE_EQ(0.0, position2[1]);
  EXPECT_DOUBLE_EQ(0.015, position2[2]);
  EXPECT_DOUBLE_EQ(0.15707963270000003, orientation2[0]);
  EXPECT_DOUBLE_EQ(0.0, orientation2[1]);
  EXPECT_DOUBLE_EQ(0.0, orientation2[2]);
  EXPECT_DOUBLE_EQ(0.0, vel_linear2[0]);
  EXPECT_DOUBLE_EQ(0.0, vel_linear2[1]);
  EXPECT_DOUBLE_EQ(0.2, vel_linear2[2]);
  EXPECT_DOUBLE_EQ(1.570796327, vel_angular2[0]);
  EXPECT_DOUBLE_EQ(0.0, vel_angular2[1]);
  EXPECT_DOUBLE_EQ(0.0, vel_angular2[2]);
  EXPECT_DOUBLE_EQ(0.0, acc_linear2[0]);
  EXPECT_DOUBLE_EQ(0.0, acc_linear2[1]);
  EXPECT_DOUBLE_EQ(1.0, acc_linear2[2]);

  // General 3D motion (these value are checked against rl predict() equations)
  position1[0] = 0.0;
  position1[1] = 0.0;
  position1[2] = 0.0;
  orientation1[0] = -2.490;
  orientation1[1] = -0.206;
  orientation1[2] = 3.066;
  vel_linear1[0] = 0.1;
  vel_linear1[1] = 0.2;
  vel_linear1[2] = 0.1;
  vel_angular1[0] = 1.570796327;
  vel_angular1[1] = 1.570796327;
  vel_angular1[2] = -1.570796327;
  acc_linear1[0] = -0.5;
  acc_linear1[1] = 1.0;
  acc_linear1[2] = 1.0;
  dt = 0.1;

  fuse_models::predict(
    position1,
    orientation1,
    vel_linear1,
    vel_angular1,
    acc_linear1,
    dt,
    position2,
    orientation2,
    vel_linear2,
    vel_angular2,
    acc_linear2
  );  // NOLINT(whitespace/parens)

  EXPECT_DOUBLE_EQ(-0.012031207341885572, position2[0]);
  EXPECT_DOUBLE_EQ(0.011723254405731805,  position2[1]);
  EXPECT_DOUBLE_EQ(-0.024981300126995967, position2[2]);
  EXPECT_DOUBLE_EQ(-2.3391131265098766,   orientation2[0]);
  EXPECT_DOUBLE_EQ(-0.4261584872792554,  orientation2[1]);
  EXPECT_DOUBLE_EQ(3.0962756133525855,  orientation2[2]);
  EXPECT_DOUBLE_EQ(0.05, vel_linear2[0]);
  EXPECT_DOUBLE_EQ(0.3, vel_linear2[1]);
  EXPECT_DOUBLE_EQ(0.2, vel_linear2[2]);
  EXPECT_DOUBLE_EQ(1.570796327, vel_angular2[0]);
  EXPECT_DOUBLE_EQ(1.570796327, vel_angular2[1]);
  EXPECT_DOUBLE_EQ(-1.570796327, vel_angular2[2]);
  EXPECT_DOUBLE_EQ(-0.5, acc_linear2[0]);
  EXPECT_DOUBLE_EQ(1.0, acc_linear2[1]);
  EXPECT_DOUBLE_EQ(1.0, acc_linear2[2]);
}

TEST(Predict, predictFromJetPointers)
{
  using Jet = ceres::Jet<double, 32>;

  Jet position1[3] = {Jet(0.0), Jet(0.0), Jet(0.0)};
  Jet orientation1[3] = {Jet(0.0), Jet(0.0), Jet(0.0)};
  Jet vel_linear1[3] = {Jet(1.0), Jet(0.0), Jet(0.0)};
  Jet vel_angular1[3] = {Jet(0.0), Jet(0.0), Jet(1.570796327)};
  Jet acc_linear1[3] = {Jet(1.0), Jet(0.0), Jet(0.0)};
  Jet dt = Jet(0.1);
  Jet position2[3];
  Jet orientation2[3];
  Jet vel_linear2[3];
  Jet vel_angular2[3];
  Jet acc_linear2[3];

  fuse_models::predict(
    position1,
    orientation1,
    vel_linear1,
    vel_angular1,
    acc_linear1,
    dt,
    position2,
    orientation2,
    vel_linear2,
    vel_angular2,
    acc_linear2);

  EXPECT_DOUBLE_EQ(Jet(0.105).a, position2[0].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a,   position2[1].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a,   position2[2].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, orientation2[0].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, orientation2[1].a);
  EXPECT_DOUBLE_EQ(Jet(0.1570796327).a, orientation2[2].a);
  EXPECT_DOUBLE_EQ(Jet(1.1).a,   vel_linear2[0].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a,   vel_linear2[1].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a,   vel_linear2[2].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a,   vel_angular2[0].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a,   vel_angular2[1].a);
  EXPECT_DOUBLE_EQ(Jet(1.570796327).a, vel_angular2[2].a);
  EXPECT_DOUBLE_EQ(Jet(1.0).a, acc_linear2[0].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, acc_linear2[1].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, acc_linear2[2].a);

  // Carry on with the output state from last time - show in-place update support
  fuse_models::predict(
    position2,
    orientation2,
    vel_linear2,
    vel_angular2,
    acc_linear2,
    dt,
    position2,
    orientation2,
    vel_linear2,
    vel_angular2,
    acc_linear2);

  EXPECT_DOUBLE_EQ(Jet(0.21858415916807189).a,  position2[0].a);
  EXPECT_DOUBLE_EQ(Jet(0.017989963481956205).a, position2[1].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, position2[2].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, orientation2[0].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, orientation2[1].a);
  EXPECT_DOUBLE_EQ(Jet(0.3141592654).a, orientation2[2].a);
  EXPECT_DOUBLE_EQ(Jet(1.2).a,   vel_linear2[0].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a,   vel_linear2[1].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a,   vel_linear2[2].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a,   vel_angular2[0].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a,   vel_angular2[1].a);
  EXPECT_DOUBLE_EQ(Jet(1.570796327).a, vel_angular2[2].a);
  EXPECT_DOUBLE_EQ(Jet(1.0).a, acc_linear2[0].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, acc_linear2[1].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, acc_linear2[2].a);

  // // Use non-zero Y values
  vel_linear1[1]  = Jet(-1.0);
  vel_angular1[2] = Jet(-1.570796327);
  acc_linear1[1]  = Jet(-1.0);

  fuse_models::predict(
    position1,
    orientation1,
    vel_linear1,
    vel_angular1,
    acc_linear1,
    dt,
    position2,
    orientation2,
    vel_linear2,
    vel_angular2,
    acc_linear2);

  EXPECT_DOUBLE_EQ(Jet(0.105).a,  position2[0].a);
  EXPECT_DOUBLE_EQ(Jet(-0.105).a, position2[1].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a,    position2[2].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, orientation2[0].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, orientation2[1].a);
  EXPECT_DOUBLE_EQ(Jet(-0.1570796327).a, orientation2[2].a);
  EXPECT_DOUBLE_EQ(Jet(1.1).a,   vel_linear2[0].a);
  EXPECT_DOUBLE_EQ(Jet(-1.1).a,  vel_linear2[1].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a,   vel_linear2[2].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a,   vel_angular2[0].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a,   vel_angular2[1].a);
  EXPECT_DOUBLE_EQ(Jet(-1.570796327).a, vel_angular2[2].a);
  EXPECT_DOUBLE_EQ(Jet(1.0).a,  acc_linear2[0].a);
  EXPECT_DOUBLE_EQ(Jet(-1.0).a, acc_linear2[1].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a,  acc_linear2[2].a);

// Out of plane motion
  position1[0] = Jet(0.0);
  position1[1] = Jet(0.0);
  position1[2] = Jet(0.0);
  orientation1[0] = Jet(0.0);
  orientation1[1] = Jet(0.0);
  orientation1[2] = Jet(0.0);
  vel_linear1[0] = Jet(0.0);
  vel_linear1[1] = Jet(0.0);
  vel_linear1[2] = Jet(0.1);
  vel_angular1[0] = Jet(1.570796327);
  vel_angular1[1] = Jet(0.0);
  vel_angular1[2] = Jet(0.0);
  acc_linear1[0] = Jet(0.0);
  acc_linear1[1] = Jet(0.0);
  acc_linear1[2] = Jet(1.0);
  dt = Jet(0.1);

  fuse_models::predict(
    position1,
    orientation1,
    vel_linear1,
    vel_angular1,
    acc_linear1,
    dt,
    position2,
    orientation2,
    vel_linear2,
    vel_angular2,
    acc_linear2
  );  // NOLINT(whitespace/parens)

  EXPECT_DOUBLE_EQ(Jet(0.0).a, position2[0].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, position2[1].a);
  EXPECT_DOUBLE_EQ(Jet(0.015).a, position2[2].a);
  EXPECT_DOUBLE_EQ(Jet(0.15707963270000003).a, orientation2[0].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, orientation2[1].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, orientation2[2].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, vel_linear2[0].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, vel_linear2[1].a);
  EXPECT_DOUBLE_EQ(Jet(0.2).a, vel_linear2[2].a);
  EXPECT_DOUBLE_EQ(Jet(1.570796327).a, vel_angular2[0].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, vel_angular2[1].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, vel_angular2[2].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, acc_linear2[0].a);
  EXPECT_DOUBLE_EQ(Jet(0.0).a, acc_linear2[1].a);
  EXPECT_DOUBLE_EQ(Jet(1.0).a, acc_linear2[2].a);

  // General 3D motion (these value are checked against rl predict() equations)
  position1[0] = Jet(0.0);
  position1[1] = Jet(0.0);
  position1[2] = Jet(0.0);
  orientation1[0] = Jet(-2.490);
  orientation1[1] = Jet(-0.206);
  orientation1[2] = Jet(3.066);
  vel_linear1[0] = Jet(0.1);
  vel_linear1[1] = Jet(0.2);
  vel_linear1[2] = Jet(0.1);
  vel_angular1[0] = Jet(1.570796327);
  vel_angular1[1] = Jet(1.570796327);
  vel_angular1[2] = Jet(-1.570796327);
  acc_linear1[0] = Jet(-0.5);
  acc_linear1[1] = Jet(1.0);
  acc_linear1[2] = Jet(1.0);
  dt = Jet(0.1);

  fuse_models::predict(
    position1,
    orientation1,
    vel_linear1,
    vel_angular1,
    acc_linear1,
    dt,
    position2,
    orientation2,
    vel_linear2,
    vel_angular2,
    acc_linear2
  );  // NOLINT(whitespace/parens)

  EXPECT_DOUBLE_EQ(Jet(-0.012031207341885572).a, position2[0].a);
  EXPECT_DOUBLE_EQ(Jet(0.011723254405731805).a,  position2[1].a);
  EXPECT_DOUBLE_EQ(Jet(-0.024981300126995967).a, position2[2].a);
  EXPECT_DOUBLE_EQ(Jet(-2.3391131265098766).a,   orientation2[0].a);
  EXPECT_DOUBLE_EQ(Jet(-0.4261584872792554).a,  orientation2[1].a);
  EXPECT_DOUBLE_EQ(Jet(3.0962756133525855).a,  orientation2[2].a);
  EXPECT_DOUBLE_EQ(Jet(0.05).a, vel_linear2[0].a);
  EXPECT_DOUBLE_EQ(Jet(0.3).a, vel_linear2[1].a);
  EXPECT_DOUBLE_EQ(Jet(0.2).a, vel_linear2[2].a);
  EXPECT_DOUBLE_EQ(Jet(1.570796327).a, vel_angular2[0].a);
  EXPECT_DOUBLE_EQ(Jet(1.570796327).a, vel_angular2[1].a);
  EXPECT_DOUBLE_EQ(Jet(-1.570796327).a, vel_angular2[2].a);
  EXPECT_DOUBLE_EQ(Jet(-0.5).a, acc_linear2[0].a);
  EXPECT_DOUBLE_EQ(Jet(1.0).a, acc_linear2[1].a);
  EXPECT_DOUBLE_EQ(Jet(1.0).a, acc_linear2[2].a);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
