#include "planner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace parking_lot_planner {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kStateResolutionFloor = 1e-3;
constexpr double kCurvatureEpsilon = 1e-9;
constexpr double kCostEpsilon = 1e-9;
constexpr double kDefaultCollisionCheckResolutionMin = 0.1;
constexpr double kDefaultCollisionCheckMapScale = 0.5;
constexpr int kDefaultMaxExpansions = 200000;
constexpr int kMinPrimitiveSamples = 2;
constexpr std::size_t kSearchReserveHint = 8192;

struct Primitive {
  int direction = 0;
  int steer = 0;
  bool is_wait = false;
};

constexpr std::array<Primitive, 7> kPrimitives = {{
    Primitive{1, -1, false},
    Primitive{1, 0, false},
    Primitive{1, 1, false},
    Primitive{-1, -1, false},
    Primitive{-1, 0, false},
    Primitive{-1, 1, false},
    Primitive{0, 0, true},
}};

struct SearchNode {
  Pose pose;
  int time_step = 0;
  double g_cost = 0.0;
  double f_cost = 0.0;
  int parent_index = -1;
};

struct StateKey {
  int x_bin = 0;
  int y_bin = 0;
  int yaw_bin = 0;
  int time_step = 0;

  bool operator==(const StateKey& other) const {
    return x_bin == other.x_bin && y_bin == other.y_bin &&
           yaw_bin == other.yaw_bin && time_step == other.time_step;
  }
};

struct StateKeyHash {
  std::size_t operator()(const StateKey& key) const {
    std::size_t seed = 0;
    auto hash_mix = [&](int value) {
      seed ^= static_cast<std::size_t>(value) + 0x9e3779b9U + (seed << 6U) +
              (seed >> 2U);
    };
    hash_mix(key.x_bin);
    hash_mix(key.y_bin);
    hash_mix(key.yaw_bin);
    hash_mix(key.time_step);
    return seed;
  }
};

struct OpenEntry {
  double f_cost = 0.0;
  int node_index = -1;
};

struct OpenCompare {
  bool operator()(const OpenEntry& lhs, const OpenEntry& rhs) const {
    return lhs.f_cost > rhs.f_cost;
  }
};

double normalizeAngle(double angle) {
  while (angle >= kPi) {
    angle -= kTwoPi;
  }
  while (angle < -kPi) {
    angle += kTwoPi;
  }
  return angle;
}

double angleDifference(double lhs, double rhs) {
  return std::abs(normalizeAngle(lhs - rhs));
}

double distance(const Pose& lhs, const Pose& rhs) {
  const double dx = lhs.x - rhs.x;
  const double dy = lhs.y - rhs.y;
  return std::sqrt(dx * dx + dy * dy);
}

double heuristic(const Pose& pose, const Pose& goal) {
  return distance(pose, goal);
}

bool isConfigured(double value) {
  return !std::isnan(value);
}

PlannerConfig resolvePlannerConfig(const Instance& instance) {
  PlannerConfig config = instance.planner;
  if (!isConfigured(config.goal_position_tolerance)) {
    config.goal_position_tolerance = instance.vehicle.step_size;
  }
  if (!isConfigured(config.goal_yaw_tolerance)) {
    config.goal_yaw_tolerance =
        kTwoPi / static_cast<double>(instance.vehicle.heading_bins);
  }
  if (!isConfigured(config.collision_check_resolution)) {
    config.collision_check_resolution = std::max(
        kDefaultCollisionCheckResolutionMin,
        instance.map.resolution * kDefaultCollisionCheckMapScale);
  }
  if (config.max_expansions <= 0) {
    config.max_expansions = kDefaultMaxExpansions;
  }
  return config;
}

int quantize(double value, double resolution) {
  return static_cast<int>(std::llround(value / resolution));
}

int headingToBin(double yaw, int heading_bins) {
  if (heading_bins <= 0) {
    throw std::invalid_argument("heading_bins must be positive");
  }

  double wrapped = normalizeAngle(yaw);
  if (wrapped < 0.0) {
    wrapped += kTwoPi;
  }
  const double bin_size = kTwoPi / static_cast<double>(heading_bins);
  return static_cast<int>(std::llround(wrapped / bin_size)) % heading_bins;
}

StateKey makeStateKey(
    const Pose& pose,
    double resolution,
    int heading_bins,
    int time_step) {
  return StateKey{
      quantize(pose.x, resolution),
      quantize(pose.y, resolution),
      headingToBin(pose.yaw, heading_bins),
      time_step,
  };
}

StateKey makeStateKey(
    const Pose& pose,
    const Instance& instance,
    int time_step) {
  const double resolution = std::max(kStateResolutionFloor, instance.map.resolution);
  return makeStateKey(pose, resolution, instance.vehicle.heading_bins, time_step);
}

bool goalReached(
    const Pose& pose,
    const Pose& goal,
    const PlannerConfig& planner_config) {
  return distance(pose, goal) <= planner_config.goal_position_tolerance &&
         angleDifference(pose.yaw, goal.yaw) <= planner_config.goal_yaw_tolerance;
}

Pose integrateMotion(
    const Pose& pose,
    int direction,
    int steer,
    const VehicleParams& vehicle,
    double travel) {
  const double curvature = steer == 0 ? 0.0
                                      : static_cast<double>(steer) /
                                            vehicle.min_turning_radius;
  const double delta_yaw = static_cast<double>(direction) * curvature * travel;

  Pose next = pose;
  if (std::abs(curvature) < kCurvatureEpsilon) {
    next.x += static_cast<double>(direction) * travel * std::cos(pose.yaw);
    next.y += static_cast<double>(direction) * travel * std::sin(pose.yaw);
  } else {
    const double avg_yaw = pose.yaw + 0.5 * delta_yaw;
    next.x += static_cast<double>(direction) * travel * std::cos(avg_yaw);
    next.y += static_cast<double>(direction) * travel * std::sin(avg_yaw);
  }
  next.yaw = normalizeAngle(pose.yaw + delta_yaw);
  return next;
}

Pose integratePrimitive(
    const Pose& pose,
    int direction,
    int steer,
    const VehicleParams& vehicle,
    double fraction) {
  return integrateMotion(
      pose,
      direction,
      steer,
      vehicle,
      vehicle.step_size * fraction);
}

Pose interpolatePose(
    const Pose& start,
    const Pose& end,
    double fraction) {
  return Pose{
      start.x + (end.x - start.x) * fraction,
      start.y + (end.y - start.y) * fraction,
      normalizeAngle(start.yaw + normalizeAngle(end.yaw - start.yaw) * fraction),
  };
}

Pose trajectoryPoseAtContinuousTime(
    const Trajectory& trajectory,
    double time_step) {
  if (trajectory.empty()) {
    return Pose{};
  }
  if (time_step <= static_cast<double>(trajectory.front().t)) {
    return trajectory.front().pose;
  }
  for (std::size_t i = 1; i < trajectory.size(); ++i) {
    const TrajectoryPoint& previous = trajectory[i - 1];
    const TrajectoryPoint& current = trajectory[i];
    if (time_step <= static_cast<double>(current.t)) {
      const double duration =
          static_cast<double>(current.t - previous.t);
      if (duration <= kCostEpsilon) {
        return current.pose;
      }
      const double fraction =
          (time_step - static_cast<double>(previous.t)) / duration;
      return interpolatePose(previous.pose, current.pose, fraction);
    }
  }
  return trajectory.back().pose;
}

int primitiveSampleCount(
    const Instance& instance,
    const PlannerConfig& planner_config) {
  return std::max(
      kMinPrimitiveSamples,
      static_cast<int>(
          std::ceil(instance.vehicle.step_size / planner_config.collision_check_resolution)));
}

bool dynamicCollisionFreeAtTime(
    const Pose& pose,
    double time_step,
    const CollisionChecker& collision_checker,
    const std::vector<Trajectory>& dynamic_obstacles) {
  for (const Trajectory& obstacle : dynamic_obstacles) {
    if (obstacle.empty()) {
      continue;
    }
    if (collision_checker.overlaps(
            pose,
            trajectoryPoseAtContinuousTime(obstacle, time_step))) {
      return false;
    }
  }
  return true;
}

bool primitiveKeepsGoalSafe(
    const Pose& pose,
    int arrival_time_step,
    const Instance& instance,
    const PlannerConfig& planner_config,
    const CollisionChecker& collision_checker,
    const std::vector<Trajectory>& dynamic_obstacles,
    const ConstraintSet& constraints) {
  if (collision_checker.collides(pose)) {
    return false;
  }

  const int samples = primitiveSampleCount(instance, planner_config);
  for (int time_step = arrival_time_step;
       time_step <= instance.vehicle.max_time_steps;
       ++time_step) {
    if (collision_checker.collides(pose, {}, constraints, time_step)) {
      return false;
    }
    if (!dynamicCollisionFreeAtTime(
            pose,
            static_cast<double>(time_step),
            collision_checker,
            dynamic_obstacles)) {
      return false;
    }
    if (time_step == instance.vehicle.max_time_steps) {
      continue;
    }
    for (int sample_index = 1; sample_index < samples; ++sample_index) {
      const double fraction =
          static_cast<double>(sample_index) / static_cast<double>(samples);
      const double query_time =
          static_cast<double>(time_step) + fraction;
      if (!dynamicCollisionFreeAtTime(
              pose,
              query_time,
              collision_checker,
              dynamic_obstacles)) {
        return false;
      }
    }
  }
  return true;
}

bool primitiveIsValid(
    const Pose& start_pose,
    const Pose& end_pose,
    int start_time_step,
    int next_time_step,
    const Primitive& primitive,
    const Instance& instance,
    const PlannerConfig& planner_config,
    const CollisionChecker& collision_checker,
    const std::vector<Trajectory>& dynamic_obstacles,
    const ConstraintSet& constraints) {
  const int samples = primitiveSampleCount(instance, planner_config);

  for (int i = 1; i <= samples; ++i) {
    const double fraction = static_cast<double>(i) / static_cast<double>(samples);
    const Pose sample = primitive.is_wait
                            ? start_pose
                            : integratePrimitive(
                                  start_pose,
                                  primitive.direction,
                                  primitive.steer,
                                  instance.vehicle,
                                  fraction);
    if (collision_checker.collides(sample)) {
      return false;
    }
    if (!dynamicCollisionFreeAtTime(
            sample,
            static_cast<double>(start_time_step) + fraction,
            collision_checker,
            dynamic_obstacles)) {
      return false;
    }
  }

  return !collision_checker.collides(
      end_pose,
      {},
      constraints,
      next_time_step);
}

double transitionCost(double step_size) {
  return step_size;
}

Trajectory reconstructTrajectory(
    const std::vector<SearchNode>& nodes,
    int goal_index) {
  Trajectory trajectory;
  for (int current = goal_index; current >= 0; current = nodes[current].parent_index) {
    trajectory.push_back(TrajectoryPoint{
        nodes[current].time_step,
        nodes[current].pose,
    });
  }
  std::reverse(trajectory.begin(), trajectory.end());
  return trajectory;
}

PlanResult finalizePlanResult(
    bool success,
    Trajectory trajectory,
    std::string failure_reason,
    PlanStats stats) {
  PlanResult result;
  result.success = success;
  result.trajectory = std::move(trajectory);
  result.failure_reason = std::move(failure_reason);
  result.stats = stats;
  return result;
}

}  // namespace

PlanResult HybridAStarPlanner::plan(
    const Pose& start,
    const Pose& goal,
    const Instance& instance,
    const std::vector<Trajectory>& dynamic_obstacles,
    const ConstraintSet& constraints) const {
  validateInstance(instance);

  PlanStats stats;
  const auto fail = [&](std::string failure_reason) {
    return finalizePlanResult(false, {}, std::move(failure_reason), stats);
  };

  const PlannerConfig planner_config = resolvePlannerConfig(instance);
  const CollisionChecker collision_checker(instance);

  if (collision_checker.collides(start, dynamic_obstacles, constraints, 0)) {
    return fail("start pose is in collision");
  }
  if (collision_checker.collides(goal)) {
    return fail("goal pose is in static collision");
  }

  bool goal_window_available = false;
  for (int time_step = 0; time_step <= instance.vehicle.max_time_steps; ++time_step) {
    if (!collision_checker.collides(goal, dynamic_obstacles, constraints, time_step)) {
      goal_window_available = true;
      break;
    }
  }
  if (!goal_window_available) {
    return fail("goal pose is blocked for the full planning horizon");
  }

  std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenCompare> open;
  std::vector<SearchNode> nodes;
  nodes.reserve(kSearchReserveHint);

  std::unordered_map<StateKey, double, StateKeyHash> best_costs;
  best_costs.reserve(kSearchReserveHint);

  nodes.push_back(SearchNode{
      start,
      0,
      0.0,
      heuristic(start, goal),
      -1,
  });
  open.push(OpenEntry{nodes.front().f_cost, 0});
  best_costs.emplace(makeStateKey(start, instance, 0), 0.0);

  while (!open.empty()) {
    const OpenEntry current_entry = open.top();
    open.pop();

    const SearchNode current = nodes[current_entry.node_index];
    const StateKey current_key =
        makeStateKey(current.pose, instance, current.time_step);
    const auto best_it = best_costs.find(current_key);
    if (best_it == best_costs.end() ||
        current.g_cost > best_it->second + kCostEpsilon) {
      continue;
    }

    if (goalReached(current.pose, goal, planner_config) &&
        primitiveKeepsGoalSafe(
            current.pose,
            current.time_step,
            instance,
            planner_config,
            collision_checker,
            dynamic_obstacles,
            constraints)) {
      return finalizePlanResult(
          true,
          reconstructTrajectory(nodes, current_entry.node_index),
          "",
          stats);
    }

    if (current.time_step >= instance.vehicle.max_time_steps) {
      continue;
    }

    if (stats.expanded_states >= planner_config.max_expansions) {
      return fail("search reached max expansions");
    }
    ++stats.expanded_states;

    for (const Primitive& primitive : kPrimitives) {
      const Pose next_pose = primitive.is_wait
                                 ? current.pose
                                 : integratePrimitive(
                                       current.pose,
                                       primitive.direction,
                                       primitive.steer,
                                       instance.vehicle,
                                       1.0);
      const int next_time = current.time_step + 1;

      if (!primitiveIsValid(
              current.pose,
              next_pose,
              current.time_step,
              next_time,
              primitive,
              instance,
              planner_config,
              collision_checker,
              dynamic_obstacles,
              constraints)) {
        continue;
      }

      const double new_g = current.g_cost + transitionCost(instance.vehicle.step_size);
      const StateKey next_key = makeStateKey(next_pose, instance, next_time);
      const auto existing = best_costs.find(next_key);
      if (existing != best_costs.end() &&
          new_g >= existing->second - kCostEpsilon) {
        continue;
      }

      best_costs[next_key] = new_g;
      const double new_f = new_g + heuristic(next_pose, goal);

      nodes.push_back(SearchNode{
          next_pose,
          next_time,
          new_g,
          new_f,
          current_entry.node_index,
      });
      open.push(OpenEntry{new_f, static_cast<int>(nodes.size() - 1)});
    }
  }

  return fail("search exhausted without reaching goal");
}

Metrics computeMetrics(
    const Instance& instance,
    const TrajectoryMap& trajectories,
    int total_low_level_expanded_states) {
  Metrics metrics;
  metrics.num_agents = static_cast<int>(instance.agents.size());
  metrics.total_low_level_expanded_states = total_low_level_expanded_states;

  for (const auto& [agent_id, trajectory] : trajectories) {
    (void)agent_id;
    if (trajectory.empty()) {
      continue;
    }
    metrics.makespan = std::max(metrics.makespan, trajectory.back().t);
  }

  return metrics;
}

PrioritizedPlanner::PrioritizedPlanner(const LowLevelPlanner& low_level_planner)
    : low_level_planner_(&low_level_planner) {}

void validateInstance(const Instance& instance) {
  if (instance.map.width <= 0.0 || instance.map.height <= 0.0) {
    throw std::invalid_argument("map dimensions must be positive");
  }
  if (instance.map.resolution <= 0.0) {
    throw std::invalid_argument("map.resolution must be positive");
  }
  if (instance.vehicle.length <= 0.0 || instance.vehicle.width <= 0.0) {
    throw std::invalid_argument("vehicle dimensions must be positive");
  }
  if (instance.vehicle.step_size <= 0.0) {
    throw std::invalid_argument("vehicle.step_size must be positive");
  }
  if (instance.vehicle.min_turning_radius <= 0.0) {
    throw std::invalid_argument("vehicle.min_turning_radius must be positive");
  }
  if (instance.vehicle.heading_bins <= 0) {
    throw std::invalid_argument("vehicle.heading_bins must be positive");
  }
  if (instance.vehicle.max_time_steps < 0) {
    throw std::invalid_argument("vehicle.max_time_steps must be non-negative");
  }
  if (isConfigured(instance.planner.goal_position_tolerance) &&
      instance.planner.goal_position_tolerance <= 0.0) {
    throw std::invalid_argument(
        "planner.goal_position_tolerance must be positive when provided");
  }
  if (isConfigured(instance.planner.goal_yaw_tolerance) &&
      instance.planner.goal_yaw_tolerance <= 0.0) {
    throw std::invalid_argument(
        "planner.goal_yaw_tolerance must be positive when provided");
  }
  if (isConfigured(instance.planner.collision_check_resolution) &&
      instance.planner.collision_check_resolution <= 0.0) {
    throw std::invalid_argument(
        "planner.collision_check_resolution must be positive when provided");
  }
  if (instance.planner.max_expansions != -1 &&
      instance.planner.max_expansions <= 0) {
    throw std::invalid_argument(
        "planner.max_expansions must be positive when provided");
  }
  if (instance.parking_layout.enabled) {
    if (instance.parking_layout.grid_cols <= 0 ||
        instance.parking_layout.grid_rows <= 0 ||
        instance.parking_layout.stalls_per_side <= 0) {
      throw std::invalid_argument(
          "parking_layout grid sizes and stalls_per_side must be positive");
    }
    if (instance.parking_layout.stall_width <= 0.0 ||
        instance.parking_layout.stall_depth <= 0.0 ||
        instance.parking_layout.zone_gap < 0.0 ||
        instance.parking_layout.perimeter_gap < 0.0) {
      throw std::invalid_argument(
          "parking_layout dimensions must be non-negative and stall sizes positive");
    }
  }
}

Solution PrioritizedPlanner::solve(const Instance& instance) const {
  validateInstance(instance);

  const auto start_time = std::chrono::steady_clock::now();
  Solution solution;
  std::vector<Trajectory> planned_trajectories;
  planned_trajectories.reserve(instance.agents.size());
  int total_low_level_expanded_states = 0;

  for (const AgentSpec& agent : instance.agents) {
    const ConstraintSet empty_constraints;
    const PlanResult result = low_level_planner_->plan(
        agent.start,
        agent.goal,
        instance,
        planned_trajectories,
        empty_constraints);
    total_low_level_expanded_states += result.stats.expanded_states;

    if (!result.success) {
      solution.success = false;
      solution.failure_reason =
          "agent " + agent.id + " failed: " + result.failure_reason;
      break;
    }

    solution.trajectories[agent.id] = result.trajectory;
    planned_trajectories.push_back(result.trajectory);
  }

  if (solution.trajectories.size() == instance.agents.size()) {
    solution.success = true;
  }

  solution.metrics = computeMetrics(
      instance,
      solution.trajectories,
      total_low_level_expanded_states);
  const auto end_time = std::chrono::steady_clock::now();
  solution.runtime_sec =
      std::chrono::duration<double>(end_time - start_time).count();
  return solution;
}

}  // namespace parking_lot_planner
