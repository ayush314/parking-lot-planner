#pragma once

#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace parking_lot_planner {

struct Pose {
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
};

struct TrajectoryPoint {
  int t = 0;
  Pose pose;
};

using Trajectory = std::vector<TrajectoryPoint>;
using TrajectoryMap = std::unordered_map<std::string, Trajectory>;

struct MapParams {
  double width = 0.0;
  double height = 0.0;
  double resolution = 0.5;
};

struct VehicleParams {
  double length = 4.8;
  double width = 2.0;
  double min_turning_radius = 5.5;
  double step_size = 0.5;
  int heading_bins = 72;
  int max_time_steps = 300;
};

struct StaticCar {
  std::string id;
  Pose pose;
};

struct AgentSpec {
  std::string id;
  Pose start;
  Pose goal;
};

struct PlannerConfig {
  double goal_position_tolerance = std::numeric_limits<double>::quiet_NaN();
  double goal_yaw_tolerance = std::numeric_limits<double>::quiet_NaN();
  double collision_check_resolution = std::numeric_limits<double>::quiet_NaN();
  int max_expansions = -1;
};

struct ParkingLayout {
  bool enabled = false;
  int grid_cols = 0;
  int grid_rows = 0;
  int stalls_per_side = 0;
  double stall_width = 0.0;
  double stall_depth = 0.0;
  double zone_gap = 0.0;
  double perimeter_gap = 0.0;
};

struct Instance {
  MapParams map;
  VehicleParams vehicle;
  PlannerConfig planner;
  ParkingLayout parking_layout;
  std::vector<StaticCar> static_cars;
  std::vector<AgentSpec> agents;
};

struct Constraint {
  int time_step = 0;
  Pose pose;
  double position_tolerance = 0.0;
  double yaw_tolerance = 0.0;
};

using ConstraintSet = std::vector<Constraint>;

struct PlanStats {
  int expanded_states = 0;
};

struct PlanResult {
  bool success = false;
  Trajectory trajectory;
  std::string failure_reason;
  PlanStats stats;
};

struct Metrics {
  int num_agents = 0;
  int makespan = 0;
  int total_low_level_expanded_states = 0;
};

struct Solution {
  bool success = false;
  double runtime_sec = 0.0;
  Metrics metrics;
  TrajectoryMap trajectories;
  std::string failure_reason;
};

class CollisionChecker {
 public:
  explicit CollisionChecker(const Instance& instance);

  bool collides(const Pose& pose) const;
  bool collides(
      const Pose& pose,
      const std::vector<Trajectory>& dynamic_obstacles,
      const ConstraintSet& constraints,
      int time_step) const;
  bool overlaps(const Pose& lhs, const Pose& rhs) const;

 private:
  const Instance& instance_;
};

class LowLevelPlanner {
 public:
  virtual ~LowLevelPlanner() = default;
  virtual PlanResult plan(
      const Pose& start,
      const Pose& goal,
      const Instance& instance,
      const std::vector<Trajectory>& dynamic_obstacles,
      const ConstraintSet& constraints) const = 0;
};

class HybridAStarPlanner : public LowLevelPlanner {
 public:
  PlanResult plan(
      const Pose& start,
      const Pose& goal,
      const Instance& instance,
      const std::vector<Trajectory>& dynamic_obstacles,
      const ConstraintSet& constraints) const override;
};

class MultiAgentSolver {
 public:
  virtual ~MultiAgentSolver() = default;
  virtual Solution solve(const Instance& instance) const = 0;
};

class PrioritizedPlanner : public MultiAgentSolver {
 public:
  PrioritizedPlanner() = default;
  explicit PrioritizedPlanner(const LowLevelPlanner& low_level_planner);

  Solution solve(const Instance& instance) const override;

 private:
  HybridAStarPlanner default_low_level_planner_;
  const LowLevelPlanner* low_level_planner_ = &default_low_level_planner_;
};

void validateInstance(const Instance& instance);

Instance loadInstance(const std::string& path);

void writeSolution(const Solution& solution, const std::string& path);

Metrics computeMetrics(
    const Instance& instance,
    const TrajectoryMap& trajectories,
    int total_low_level_expanded_states);

}  // namespace parking_lot_planner
