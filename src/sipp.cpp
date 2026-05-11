#include "sipp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <queue>
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
};

constexpr std::array<Primitive, 6> kPrimitives = {{
    Primitive{1, -1},
    Primitive{1, 0},
    Primitive{1, 1},
    Primitive{-1, -1},
    Primitive{-1, 0},
    Primitive{-1, 1},
}};

struct SafeInterval {
  int lo = 0;
  int hi = 0;
};

struct CellKey {
  int x_bin = 0;
  int y_bin = 0;
  int yaw_bin = 0;

  bool operator==(const CellKey& other) const {
    return x_bin == other.x_bin && y_bin == other.y_bin &&
           yaw_bin == other.yaw_bin;
  }
};

struct CellKeyHash {
  std::size_t operator()(const CellKey& key) const {
    std::size_t seed = 0;
    auto mix = [&](int value) {
      seed ^= static_cast<std::size_t>(value) + 0x9e3779b9U + (seed << 6U) +
              (seed >> 2U);
    };
    mix(key.x_bin);
    mix(key.y_bin);
    mix(key.yaw_bin);
    return seed;
  }
};

struct StateKey {
  CellKey cell;
  int interval_index = 0;

  bool operator==(const StateKey& other) const {
    return cell == other.cell && interval_index == other.interval_index;
  }
};

struct StateKeyHash {
  std::size_t operator()(const StateKey& key) const {
    std::size_t seed = CellKeyHash{}(key.cell);
    seed ^= static_cast<std::size_t>(key.interval_index) + 0x9e3779b9U +
            (seed << 6U) + (seed >> 2U);
    return seed;
  }
};

struct SearchNode {
  Pose pose;
  int arrival_time = 0;
  int interval_index = 0;
  double g_cost = 0.0;
  double f_cost = 0.0;
  int parent_index = -1;
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
  double wrapped = normalizeAngle(yaw);
  if (wrapped < 0.0) {
    wrapped += kTwoPi;
  }
  const double bin_size = kTwoPi / static_cast<double>(heading_bins);
  return static_cast<int>(std::llround(wrapped / bin_size)) % heading_bins;
}

CellKey makeCellKey(const Pose& pose, const Instance& instance) {
  const double resolution =
      std::max(kStateResolutionFloor, instance.map.resolution);
  return CellKey{
      quantize(pose.x, resolution),
      quantize(pose.y, resolution),
      headingToBin(pose.yaw, instance.vehicle.heading_bins),
  };
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
      const double duration = static_cast<double>(current.t - previous.t);
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
      static_cast<int>(std::ceil(
          instance.vehicle.step_size / planner_config.collision_check_resolution)));
}

Pose poseAtIntegerTime(const Trajectory& trajectory, int time_step) {
  if (trajectory.empty()) {
    return Pose{};
  }
  if (time_step <= trajectory.front().t) {
    return trajectory.front().pose;
  }
  if (time_step >= trajectory.back().t) {
    return trajectory.back().pose;
  }
  for (const TrajectoryPoint& point : trajectory) {
    if (point.t == time_step) {
      return point.pose;
    }
  }
  return trajectory.back().pose;
}

// Precomputed obstacle pose snapshots indexed by integer time, plus the
// closest squared distance at which a footprint overlap is still possible.
// Building this once per agent removes O(L) trajectory scans from the
// inner loop and lets us prune obviously-distant obstacles cheaply.
struct ObstacleSnapshot {
  std::vector<Pose> poses;
  double cull_distance_sq = 0.0;
};

std::vector<ObstacleSnapshot> buildObstacleSnapshots(
    const Instance& instance,
    const std::vector<Trajectory>& dynamic_obstacles) {
  const double half_length = 0.5 * instance.vehicle.length;
  const double half_width = 0.5 * instance.vehicle.width;
  const double cull_radius =
      2.0 * std::sqrt(half_length * half_length + half_width * half_width) +
      1.0;
  std::vector<ObstacleSnapshot> snapshots;
  snapshots.reserve(dynamic_obstacles.size());
  for (const Trajectory& trajectory : dynamic_obstacles) {
    ObstacleSnapshot snapshot;
    snapshot.cull_distance_sq = cull_radius * cull_radius;
    snapshot.poses.resize(instance.vehicle.max_time_steps + 1);
    if (!trajectory.empty()) {
      for (int t = 0; t <= instance.vehicle.max_time_steps; ++t) {
        snapshot.poses[t] = poseAtIntegerTime(trajectory, t);
      }
    }
    snapshots.push_back(std::move(snapshot));
  }
  return snapshots;
}

std::vector<std::vector<const Constraint*>> bucketConstraintsByTime(
    const ConstraintSet& constraints,
    int max_time_steps) {
  std::vector<std::vector<const Constraint*>> buckets(max_time_steps + 1);
  for (const Constraint& constraint : constraints) {
    if (constraint.time_step >= 0 && constraint.time_step <= max_time_steps) {
      buckets[constraint.time_step].push_back(&constraint);
    }
  }
  return buckets;
}

// Lazily computes and caches safe intervals per discretized pose cell.
// A cell's safe intervals are the maximal time spans during which the
// vehicle footprint is free of static obstacles, integer-time dynamic
// obstacles, and active constraints. Static collision is checked once
// per cell; dynamic and constraint checks use the precomputed snapshots
// so each interval sweep is linear in (time x obstacles) without any
// repeated trajectory traversal.
class SafeIntervalTable {
 public:
  SafeIntervalTable(
      const Instance& instance,
      const CollisionChecker& checker,
      const std::vector<ObstacleSnapshot>& obstacle_snapshots,
      const std::vector<std::vector<const Constraint*>>& constraints_by_time)
      : instance_(instance),
        checker_(checker),
        obstacle_snapshots_(obstacle_snapshots),
        constraints_by_time_(constraints_by_time) {}

  const std::vector<SafeInterval>& intervals(
      const CellKey& key,
      const Pose& pose) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }
    return cache_.emplace(key, compute(pose)).first->second;
  }

 private:
  bool unsafeAtTime(const Pose& pose, int time_step) const {
    for (const Constraint* constraint : constraints_by_time_[time_step]) {
      const double dx = pose.x - constraint->pose.x;
      const double dy = pose.y - constraint->pose.y;
      if (dx * dx + dy * dy <=
              (constraint->position_tolerance + 1e-9) *
                  (constraint->position_tolerance + 1e-9) &&
          std::abs(normalizeAngle(pose.yaw - constraint->pose.yaw)) <=
              constraint->yaw_tolerance + 1e-9) {
        return true;
      }
    }
    for (const ObstacleSnapshot& snapshot : obstacle_snapshots_) {
      if (snapshot.poses.empty()) {
        continue;
      }
      const Pose& obstacle_pose = snapshot.poses[time_step];
      const double dx = pose.x - obstacle_pose.x;
      const double dy = pose.y - obstacle_pose.y;
      if (dx * dx + dy * dy > snapshot.cull_distance_sq) {
        continue;
      }
      if (checker_.overlaps(pose, obstacle_pose)) {
        return true;
      }
    }
    return false;
  }

  std::vector<SafeInterval> compute(const Pose& pose) const {
    std::vector<SafeInterval> intervals;
    if (checker_.collides(pose)) {
      return intervals;
    }
    if (obstacle_snapshots_.empty() && constraints_by_time_.empty()) {
      intervals.push_back(
          SafeInterval{0, instance_.vehicle.max_time_steps});
      return intervals;
    }
    int run_start = -1;
    for (int t = 0; t <= instance_.vehicle.max_time_steps; ++t) {
      const bool safe = !unsafeAtTime(pose, t);
      if (safe && run_start < 0) {
        run_start = t;
      } else if (!safe && run_start >= 0) {
        intervals.push_back(SafeInterval{run_start, t - 1});
        run_start = -1;
      }
    }
    if (run_start >= 0) {
      intervals.push_back(
          SafeInterval{run_start, instance_.vehicle.max_time_steps});
    }
    return intervals;
  }

  const Instance& instance_;
  const CollisionChecker& checker_;
  const std::vector<ObstacleSnapshot>& obstacle_snapshots_;
  const std::vector<std::vector<const Constraint*>>& constraints_by_time_;
  std::unordered_map<CellKey, std::vector<SafeInterval>, CellKeyHash> cache_;
};

int findIntervalIndex(
    const std::vector<SafeInterval>& intervals,
    int time_step) {
  for (std::size_t i = 0; i < intervals.size(); ++i) {
    if (time_step >= intervals[i].lo && time_step <= intervals[i].hi) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

bool transitionIsSafe(
    const Pose& start_pose,
    const Pose& end_pose,
    const Primitive& primitive,
    int depart_time,
    int arrival_time,
    const Instance& instance,
    const PlannerConfig& planner_config,
    const CollisionChecker& checker,
    const std::vector<Trajectory>& dynamic_obstacles,
    const ConstraintSet& constraints) {
  const int samples = primitiveSampleCount(instance, planner_config);
  for (int i = 1; i <= samples; ++i) {
    const double fraction =
        static_cast<double>(i) / static_cast<double>(samples);
    const Pose sample = integratePrimitive(
        start_pose,
        primitive.direction,
        primitive.steer,
        instance.vehicle,
        fraction);
    if (checker.collides(sample)) {
      return false;
    }
    const double query_time =
        static_cast<double>(depart_time) + fraction;
    for (const Trajectory& obstacle : dynamic_obstacles) {
      if (obstacle.empty()) {
        continue;
      }
      if (checker.overlaps(
              sample,
              trajectoryPoseAtContinuousTime(obstacle, query_time))) {
        return false;
      }
    }
  }
  return !checker.collides(end_pose, {}, constraints, arrival_time);
}

// Walks the feasible departure window for a single SIPP successor edge
// and returns the earliest integer departure time whose continuous
// motion is collision-free, or -1 if none exists.
int findEarliestSafeDeparture(
    const Pose& start_pose,
    const Pose& end_pose,
    const Primitive& primitive,
    int depart_lo,
    int depart_hi,
    const Instance& instance,
    const PlannerConfig& planner_config,
    const CollisionChecker& checker,
    const std::vector<Trajectory>& dynamic_obstacles,
    const ConstraintSet& constraints) {
  for (int depart = depart_lo; depart <= depart_hi; ++depart) {
    if (transitionIsSafe(
            start_pose,
            end_pose,
            primitive,
            depart,
            depart + 1,
            instance,
            planner_config,
            checker,
            dynamic_obstacles,
            constraints)) {
      return depart;
    }
  }
  return -1;
}

// SIPP nodes only record motion at departure/arrival times, so fill in
// the waiting steps in between so the trajectory has an entry for every
// integer time step. This keeps downstream collision checks (which look
// up obstacle poses by integer time) consistent.
Trajectory reconstructTrajectory(
    const std::vector<SearchNode>& nodes,
    int goal_index) {
  Trajectory sparse;
  for (int current = goal_index; current >= 0;
       current = nodes[current].parent_index) {
    sparse.push_back(TrajectoryPoint{
        nodes[current].arrival_time,
        nodes[current].pose,
    });
  }
  std::reverse(sparse.begin(), sparse.end());

  Trajectory dense;
  dense.reserve(sparse.size());
  for (std::size_t i = 0; i < sparse.size(); ++i) {
    if (i > 0) {
      const TrajectoryPoint& previous = sparse[i - 1];
      for (int t = previous.t + 1; t < sparse[i].t; ++t) {
        dense.push_back(TrajectoryPoint{t, previous.pose});
      }
    }
    dense.push_back(sparse[i]);
  }
  return dense;
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

PlanResult SippPlanner::plan(
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
  const CollisionChecker checker(instance);

  if (checker.collides(start, dynamic_obstacles, constraints, 0)) {
    return fail("start pose is in collision");
  }
  if (checker.collides(goal)) {
    return fail("goal pose is in static collision");
  }

  const std::vector<ObstacleSnapshot> obstacle_snapshots =
      buildObstacleSnapshots(instance, dynamic_obstacles);
  const std::vector<std::vector<const Constraint*>> constraints_by_time =
      bucketConstraintsByTime(constraints, instance.vehicle.max_time_steps);
  SafeIntervalTable intervals(
      instance, checker, obstacle_snapshots, constraints_by_time);

  const CellKey start_cell = makeCellKey(start, instance);
  const std::vector<SafeInterval>& start_intervals =
      intervals.intervals(start_cell, start);
  const int start_interval_index = findIntervalIndex(start_intervals, 0);
  if (start_interval_index < 0) {
    return fail("start pose has no safe interval at t=0");
  }

  std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenCompare> open;
  std::vector<SearchNode> nodes;
  nodes.reserve(kSearchReserveHint);

  std::unordered_map<StateKey, double, StateKeyHash> best_costs;
  best_costs.reserve(kSearchReserveHint);

  nodes.push_back(SearchNode{
      start,
      0,
      start_interval_index,
      0.0,
      heuristic(start, goal),
      -1,
  });
  open.push(OpenEntry{nodes.front().f_cost, 0});
  best_costs.emplace(StateKey{start_cell, start_interval_index}, 0.0);

  while (!open.empty()) {
    const OpenEntry current_entry = open.top();
    open.pop();

    const SearchNode current = nodes[current_entry.node_index];
    const CellKey current_cell = makeCellKey(current.pose, instance);
    const StateKey current_key{current_cell, current.interval_index};

    const auto best_it = best_costs.find(current_key);
    if (best_it == best_costs.end() ||
        current.g_cost > best_it->second + kCostEpsilon) {
      continue;
    }

    const std::vector<SafeInterval>& cur_intervals =
        intervals.intervals(current_cell, current.pose);
    const SafeInterval cur_interval = cur_intervals[current.interval_index];

    if (goalReached(current.pose, goal, planner_config) &&
        cur_interval.hi >= instance.vehicle.max_time_steps) {
      return finalizePlanResult(
          true,
          reconstructTrajectory(nodes, current_entry.node_index),
          "",
          stats);
    }

    if (current.arrival_time >= instance.vehicle.max_time_steps) {
      continue;
    }

    if (stats.expanded_states >= planner_config.max_expansions) {
      return fail("search reached max expansions");
    }
    ++stats.expanded_states;

    for (const Primitive& primitive : kPrimitives) {
      const Pose next_pose = integrateMotion(
          current.pose,
          primitive.direction,
          primitive.steer,
          instance.vehicle,
          instance.vehicle.step_size);
      const CellKey next_cell = makeCellKey(next_pose, instance);
      const std::vector<SafeInterval>& next_intervals =
          intervals.intervals(next_cell, next_pose);

      for (std::size_t k = 0; k < next_intervals.size(); ++k) {
        const SafeInterval& next_interval = next_intervals[k];
        const int depart_lo =
            std::max(current.arrival_time, next_interval.lo - 1);
        const int depart_hi =
            std::min(cur_interval.hi, next_interval.hi - 1);
        if (depart_lo > depart_hi) {
          continue;
        }

        const int depart = findEarliestSafeDeparture(
            current.pose,
            next_pose,
            primitive,
            depart_lo,
            depart_hi,
            instance,
            planner_config,
            checker,
            dynamic_obstacles,
            constraints);
        if (depart < 0) {
          continue;
        }

        const int arrival = depart + 1;
        const double new_g =
            static_cast<double>(arrival) * instance.vehicle.step_size;
        const StateKey next_key{next_cell, static_cast<int>(k)};
        const auto existing = best_costs.find(next_key);
        if (existing != best_costs.end() &&
            new_g >= existing->second - kCostEpsilon) {
          continue;
        }

        best_costs[next_key] = new_g;
        const double new_f = new_g + heuristic(next_pose, goal);

        nodes.push_back(SearchNode{
            next_pose,
            arrival,
            static_cast<int>(k),
            new_g,
            new_f,
            current_entry.node_index,
        });
        open.push(OpenEntry{new_f, static_cast<int>(nodes.size() - 1)});
      }
    }
  }

  return fail("search exhausted without reaching goal");
}

}  // namespace parking_lot_planner
