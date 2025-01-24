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
#include <fuse_core/async_sensor_model.h>

#include <fuse_core/callback_wrapper.h>
#include <fuse_core/graph.h>
#include <fuse_core/transaction.h>
#include <ros/callback_queue.h>

#include <boost/make_shared.hpp>

#include <functional>
#include <utility>
#include <string>


namespace fuse_core
{

AsyncSensorModel::AsyncSensorModel(size_t thread_count) :
  name_("uninitialized"),
  spinner_(thread_count, &callback_queue_), active_(true)
{
}

AsyncSensorModel::~AsyncSensorModel()
{
  if (start_stop_thread_.joinable())
  {
    {
      std::lock_guard<std::mutex> lock(start_stop_mutex_);
      start_stop_request_ = StartStopEnum::STOP_THREAD;
      start_stop_condition_.notify_all();
    }
    start_stop_thread_.join();
  }
}

void AsyncSensorModel::graphCallback(Graph::ConstSharedPtr graph)
{
  callback_queue_.addCallback(
    boost::make_shared<CallbackWrapper<void>>(std::bind(&AsyncSensorModel::onGraphUpdate, this, std::move(graph))),
    reinterpret_cast<uint64_t>(this));
}

void AsyncSensorModel::initialize(
  const std::string& name,
  TransactionCallback transaction_callback)
{
  // Initialize internal state
  name_ = name;
  node_handle_.setCallbackQueue(&callback_queue_);
  private_node_handle_ = ros::NodeHandle("~/" + name_);
  private_node_handle_.setCallbackQueue(&callback_queue_);
  transaction_callback_ = transaction_callback;

  enabled_service_server_ = private_node_handle_.advertiseService("enable_sensor",
                                                                  &AsyncSensorModel::enableSensorCallback, this);

  start_stop_thread_ = std::thread(&AsyncSensorModel::startStopThreadLoop, this);

  sensor_enabled_pub_ = private_node_handle_.advertise<std_msgs::Bool>("sensor_enabled", 3, true);

  active_ = private_node_handle_.param<bool>("auto_enabled", true);

  // Call the derived onInit() function to perform implementation-specific initialization
  onInit();

  // Start the async spinner to service the local callback queue
  spinner_.start();
}

void AsyncSensorModel::sendTransaction(Transaction::SharedPtr transaction)
{
  if (!active_)
  {
    ROS_WARN_STREAM_THROTTLE(1.0, "sensor: '" << name_ << "' is not active but tries to add transaction");
    return;
  }
  transaction_callback_(std::move(transaction));
}

void AsyncSensorModel::start()
{
  publishSensorsEnabled();

  if (!active_)
  {
    // the sensor should not be active so do not start the sensor
    return;
  }

  auto callback = boost::make_shared<CallbackWrapper<void>>(std::bind(&AsyncSensorModel::onStart, this));
  auto result = callback->getFuture();
  callback_queue_.addCallback(callback, reinterpret_cast<uint64_t>(this));
  result.wait();
}

void AsyncSensorModel::stop()
{

  if (active_)
  {
    active_ = false;
  }

  if (ros::ok())
  {
    publishSensorsEnabled();

    auto callback = boost::make_shared<CallbackWrapper<void>>(std::bind(&AsyncSensorModel::onStop, this));
    auto result = callback->getFuture();
    callback_queue_.addCallback(callback, reinterpret_cast<uint64_t>(this));
    result.wait();
  }
  else
  {
    spinner_.stop();
    onStop();
  }
}


bool AsyncSensorModel::enableSensorCallback(std_srvs::SetBoolRequest& request, std_srvs::SetBoolResponse& response)
{
  ROS_INFO_STREAM("enable sensor: '" << name_ << "' called with: " << static_cast<int>(request.data));

  response.success = false;

  std::lock_guard<std::mutex> lock(start_stop_mutex_);

  if (active_ && !static_cast<bool>(request.data))
  {
    ROS_INFO_STREAM("stop sensor: '" << name_ << "'");
    active_ = false;
    start_stop_request_ = StartStopEnum::STOP_SENSOR;
    start_stop_condition_.notify_all();
    response.success = true;
  }
  else if (!active_ && static_cast<bool>(request.data))
  {
    ROS_INFO_STREAM("start sensor: '" << name_ << "'");
    active_ = true;
    start_stop_request_ = StartStopEnum::START_SENSOR;
    start_stop_condition_.notify_all();
    response.success = true;
  }
  else
  {
    ROS_WARN_STREAM("no change due to service call for sensor '" << name_ << "'");
  }

  return true;
}

void AsyncSensorModel::startStopThreadLoop()
{
  while(true)
  {
    std::unique_lock<std::mutex> lock(start_stop_mutex_);
    while (start_stop_request_ == StartStopEnum::KEEP_SENSOR_STATE)
    {
      start_stop_condition_.wait(lock);
    }

    switch (start_stop_request_)
    {
      case StartStopEnum::START_SENSOR:
        start();
        break;
      case StartStopEnum::STOP_SENSOR:
        stop();
        break;
      case StartStopEnum::KEEP_SENSOR_STATE:
        //nothing to do just wait for the next call
        break;
      case StartStopEnum::STOP_THREAD:
        return;
    }

    start_stop_request_ = StartStopEnum::KEEP_SENSOR_STATE;
  }
}

void AsyncSensorModel::publishSensorsEnabled()
{
  std_msgs::Bool msg;
  msg.data = static_cast<uint8_t>(active_);
  sensor_enabled_pub_.publish(msg);
}


}  // namespace fuse_core
