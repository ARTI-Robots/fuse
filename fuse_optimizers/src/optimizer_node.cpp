//
// Created by cmuehlbacher on 11.04.25.
//

#include "fuse_optimizers/optimizer_node.h"

namespace fuse_optimizers {
OptimizerNode::OptimizerNode(const std::string &node_name)
        : LifecycleNode(node_name) {
}

OptimizerNode::~OptimizerNode() = default;

void OptimizerNode::setOptimizer(const Optimizer::SharedPtr &optimizer) {
    optimizer_ = optimizer;
}

OptimizerNode::CallbackReturn OptimizerNode::on_configure(const rclcpp_lifecycle::State &state) {
    if (!optimizer_) {
        RCLCPP_FATAL(get_logger(), "No optimizer set for node can not perform configuration");
        return CallbackReturn::FAILURE;
    }

    const auto callback_return = nav2_util::LifecycleNode::on_configure(state);
    if (callback_return != CallbackReturn::SUCCESS) {
        return callback_return;
    }

    RCLCPP_INFO(get_logger(), "Configuring node");

    if (!optimizer_->configure()) {
        RCLCPP_ERROR(get_logger(), "Configuration of optimizer failed");
        return CallbackReturn::FAILURE;
    }

    RCLCPP_INFO(get_logger(), "Configuring node finished");

    return CallbackReturn::SUCCESS;
}

OptimizerNode::CallbackReturn OptimizerNode::on_activate(const rclcpp_lifecycle::State &state) {
    if (!optimizer_) {
        RCLCPP_FATAL(get_logger(), "No optimizer set for node can not perform configuration");
        return CallbackReturn::FAILURE;
    }

    const auto callback_return = nav2_util::LifecycleNode::on_activate(state);
    if (callback_return != CallbackReturn::SUCCESS) {
        return callback_return;
    }

    RCLCPP_INFO(get_logger(), "Activating node");

    if (!optimizer_->activate()) {
        RCLCPP_ERROR(get_logger(), "Activation of optimizer failed");
        return CallbackReturn::FAILURE;
    }

    createBond();

    RCLCPP_INFO(get_logger(), "Activating node finished");

    return CallbackReturn::SUCCESS;
}

OptimizerNode::CallbackReturn OptimizerNode::on_deactivate(const rclcpp_lifecycle::State &state) {
    if (!optimizer_) {
        RCLCPP_FATAL(get_logger(), "No optimizer set for node can not perform configuration");
        return CallbackReturn::FAILURE;
    }

    const auto callback_return = nav2_util::LifecycleNode::on_deactivate(state);
    if (callback_return != CallbackReturn::SUCCESS) {
        return callback_return;
    }

    RCLCPP_INFO(get_logger(), "Deactivating node");

    if (!optimizer_->deactivate()) {
        RCLCPP_ERROR(get_logger(), "Deactivation of optimizer failed");
        return CallbackReturn::FAILURE;
    }

    destroyBond();

    RCLCPP_INFO(get_logger(), "Deactivating node finished");

    return CallbackReturn::SUCCESS;
}

}  // namespace fuse_optimizers