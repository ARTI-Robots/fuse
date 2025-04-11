//
// Created by cmuehlbacher on 11.04.25.
//

#ifndef FUSE_OPTIMIZERS_OPTIMIZER_NODE_H
#define FUSE_OPTIMIZERS_OPTIMIZER_NODE_H

#include <nav2_util/lifecycle_node.hpp>
#include <fuse_optimizers/optimizer.hpp>

namespace fuse_optimizers {
class OptimizerNode : public nav2_util::LifecycleNode {
public:
    using nav2_util::LifecycleNode::CallbackReturn;

    explicit OptimizerNode(const std::string &node_name);

    ~OptimizerNode() override;

    void setOptimizer(const Optimizer::SharedPtr &optimizer);

    CallbackReturn on_configure(const rclcpp_lifecycle::State &state) override;

    CallbackReturn on_activate(const rclcpp_lifecycle::State &state) override;

    CallbackReturn on_deactivate(const rclcpp_lifecycle::State &state) override;

private:
    Optimizer::SharedPtr optimizer_;
};
}  // namespace fuse_optimizers

#endif //FUSE_OPTIMIZERS_OPTIMIZER_NODE_H
