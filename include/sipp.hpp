#pragma once

#include "planner.hpp"

namespace parking_lot_planner {

class SippPlanner : public LowLevelPlanner {
 public:
  PlanResult plan(
      const Pose& start,
      const Pose& goal,
      const Instance& instance,
      const std::vector<Trajectory>& dynamic_obstacles,
      const ConstraintSet& constraints) const override;
};

}  // namespace parking_lot_planner
