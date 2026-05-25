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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kStateResolutionFloor = 1e-3;
constexpr double kCurvatureEpsilon = 1e-9;
constexpr double kDefaultCollisionCheckResolutionMin = 0.1;
constexpr double kDefaultCollisionCheckMapScale = 0.5;
constexpr int kDefaultMaxExpansions = 200000;
constexpr int kMinPrimitiveSamples = 2;
constexpr std::size_t kSearchReserveHint = 8192;

// ---------------------------------------------------------------------------
// Motion primitives
//
// Six kinematic primitives (3 steer angles x 2 directions). Note: SIPP does
// NOT need an explicit "wait" primitive the way Hybrid A* does. Waiting is
// expressed implicitly by the departure-time search inside each safe interval:
// a single successor node represents "arrive in interval k as early as
// possible", which already accounts for any amount of waiting in the current
// cell. This is the structural reason SIPP expands far fewer states than
// Hybrid A* on long horizons -- one node per (cell, interval) rather than one
// node per (cell, timestep).
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Safe interval: an inclusive [lo, hi] window of integer time steps during
// which a cell is free of dynamic obstacles and time-point constraints.
// ---------------------------------------------------------------------------

struct SafeInterval {
  int lo = 0;
  int hi = 0;
};

// ---------------------------------------------------------------------------
// Cell key: the discretized (x, y, yaw) configuration. This is the spatial
// state. The full SIPP state is (cell, interval_index).
// ---------------------------------------------------------------------------

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
      seed ^= static_cast<std::size_t>(static_cast<unsigned int>(value)) +
              0x9e3779b9U + (seed << 6U) + (seed >> 2U);
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
  // Fine-pose bins: a second, finer quantization of the continuous pose. Two
  // search nodes share a StateKey only if they land in the same search cell,
  // the same safe interval, AND the same fine-pose bin -- making dominance an
  // exact, transitive equivalence rather than a non-convergent proximity test.
  int fine_x = 0;
  int fine_y = 0;
  int fine_yaw = 0;

  bool operator==(const StateKey& other) const {
    return cell == other.cell && interval_index == other.interval_index &&
           fine_x == other.fine_x && fine_y == other.fine_y &&
           fine_yaw == other.fine_yaw;
  }
};

struct StateKeyHash {
  std::size_t operator()(const StateKey& key) const {
    std::size_t seed = CellKeyHash{}(key.cell);
    auto mix = [&](int value) {
      seed ^= static_cast<std::size_t>(static_cast<unsigned int>(value)) +
              0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    };
    mix(key.interval_index);
    mix(key.fine_x);
    mix(key.fine_y);
    mix(key.fine_yaw);
    return seed;
  }
};

// Fine-pose dominance key.
//
// The search cell (x,y at 0.5 m, yaw at 5 deg) is too coarse to be a sound
// dominance key: kinematically distinct poses collapse onto one cell and the
// earliest-arrival rule then discards a pose a later maneuver needs. The fix
// is a SECOND, finer quantization used purely for dominance: snap the pose to
// a fine grid and treat (cell, interval, fine-pose-bin) as the state. This is
// an exact, transitive equivalence (unlike a tolerance test, which is not
// transitive and so never converges -- it lets a chain of pairwise-close
// poses accumulate without bound). One label per fine bin.
//
// The fine resolution is chosen finer than the search cell so genuinely
// distinct poses stay separate, but coarse enough to still merge the
// near-identical poses different primitive sequences produce.
constexpr double kFinePositionRes = 0.35;  // metres
constexpr double kFineYawRes = 0.06;       // radians (~3.4 deg)

// ---------------------------------------------------------------------------
// Search node.
//
// g_cost is stored as arrival_time (integer time steps). Keeping g and h in
// the same units (time steps) makes f = g + h consistent and the dominance
// check exact (no floating-point ambiguity).
// ---------------------------------------------------------------------------

struct SearchNode {
  Pose pose;
  int arrival_time = 0;
  int interval_index = 0;
  CellKey cell;
  double f_cost = 0.0;
  int parent_index = -1;
};

struct OpenEntry {
  double f_cost = 0.0;
  int arrival_time = 0;  // tie-breaker: prefer earlier arrival (smaller makespan)
  int node_index = -1;
};

struct OpenCompare {
  bool operator()(const OpenEntry& lhs, const OpenEntry& rhs) const {
    if (lhs.f_cost != rhs.f_cost) {
      return lhs.f_cost > rhs.f_cost;
    }
    // Tie-break toward earlier arrival. This nudges the search toward lower
    // makespan solutions when multiple paths have equal f.
    return lhs.arrival_time > rhs.arrival_time;
  }
};

// ---------------------------------------------------------------------------
// Angle / geometry helpers
// ---------------------------------------------------------------------------

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

// Builds the full dominance key for a pose: the search cell, the safe-interval
// index, and the fine-pose bin. Two nodes with equal StateKey are exact
// duplicates for dominance purposes; the earlier arrival dominates.
StateKey makeStateKey(
    const Pose& pose, const CellKey& cell, int interval_index) {
  StateKey key;
  key.cell = cell;
  key.interval_index = interval_index;
  key.fine_x = static_cast<int>(std::llround(pose.x / kFinePositionRes));
  key.fine_y = static_cast<int>(std::llround(pose.y / kFinePositionRes));
  double wrapped = pose.yaw;
  while (wrapped < 0.0) {
    wrapped += kTwoPi;
  }
  while (wrapped >= kTwoPi) {
    wrapped -= kTwoPi;
  }
  key.fine_yaw = static_cast<int>(std::llround(wrapped / kFineYawRes));
  return key;
}

// Heuristic in time-step units: a straight-line dash at one step_size per
// step is an admissible lower bound on the number of steps remaining, so
// f = g + h stays consistent with g (also in time-step units).
double heuristic(const Pose& pose, const Pose& goal, double step_size) {
  return distance(pose, goal) / step_size;
}

bool isConfigured(double value) { return !std::isnan(value); }

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

double cellResolution(const Instance& instance) {
  return std::max(kStateResolutionFloor, instance.map.resolution);
}

CellKey makeCellKey(const Pose& pose, const Instance& instance) {
  const double res = cellResolution(instance);
  return CellKey{
      quantize(pose.x, res),
      quantize(pose.y, res),
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

// ---------------------------------------------------------------------------
// Kinematic integration (constant-curvature arc), identical to the Hybrid A*
// baseline so that the two planners explore the same continuous state space
// and the comparison is apples-to-apples.
// ---------------------------------------------------------------------------

Pose integrateMotion(
    const Pose& pose,
    int direction,
    int steer,
    const VehicleParams& vehicle,
    double travel) {
  const double curvature =
      steer == 0 ? 0.0
                 : static_cast<double>(steer) / vehicle.min_turning_radius;
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
      pose, direction, steer, vehicle, vehicle.step_size * fraction);
}

Pose interpolatePose(const Pose& start, const Pose& end, double fraction) {
  return Pose{
      start.x + (end.x - start.x) * fraction,
      start.y + (end.y - start.y) * fraction,
      normalizeAngle(start.yaw + normalizeAngle(end.yaw - start.yaw) * fraction),
  };
}

// Obstacle pose at a continuous time (linear interpolation between the
// integer-time trajectory points), used for sub-step swept collision checks.
Pose trajectoryPoseAtContinuousTime(
    const Trajectory& trajectory, double time_step) {
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
      if (duration <= 1e-9) {
        return current.pose;
      }
      const double fraction =
          (time_step - static_cast<double>(previous.t)) / duration;
      return interpolatePose(previous.pose, current.pose, fraction);
    }
  }
  return trajectory.back().pose;
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
  // Trajectories are densified to one point per step, so a direct probe by
  // offset is exact; fall back to a scan if that assumption ever breaks.
  const int offset = time_step - trajectory.front().t;
  if (offset >= 0 && offset < static_cast<int>(trajectory.size()) &&
      trajectory[offset].t == time_step) {
    return trajectory[offset].pose;
  }
  for (const TrajectoryPoint& point : trajectory) {
    if (point.t == time_step) {
      return point.pose;
    }
  }
  return trajectory.back().pose;
}

int primitiveSampleCount(
    const Instance& instance, const PlannerConfig& planner_config) {
  return std::max(
      kMinPrimitiveSamples,
      static_cast<int>(std::ceil(
          instance.vehicle.step_size /
          planner_config.collision_check_resolution)));
}

// ---------------------------------------------------------------------------
// Precomputed obstacle snapshots, indexed by integer time. Each snapshot also
// carries a squared cull distance so the per-time sweep can skip obstacles
// that are obviously too far to overlap before doing the exact SAT check.
// ---------------------------------------------------------------------------

struct ObstacleSnapshot {
  std::vector<Pose> poses;
  bool active = false;
  double cull_distance_sq = 0.0;
};

std::vector<ObstacleSnapshot> buildObstacleSnapshots(
    const Instance& instance,
    const std::vector<Trajectory>& dynamic_obstacles) {
  const double half_length = 0.5 * instance.vehicle.length;
  const double half_width = 0.5 * instance.vehicle.width;
  // Two vehicle diagonals plus a small margin: a safe upper bound on the
  // centre-to-centre distance at which two footprints can possibly overlap.
  const double cull_radius =
      2.0 * std::sqrt(half_length * half_length + half_width * half_width) +
      1.0;

  std::vector<ObstacleSnapshot> snapshots;
  snapshots.reserve(dynamic_obstacles.size());
  for (const Trajectory& trajectory : dynamic_obstacles) {
    ObstacleSnapshot snapshot;
    snapshot.cull_distance_sq = cull_radius * cull_radius;
    snapshot.active = !trajectory.empty();
    if (snapshot.active) {
      snapshot.poses.resize(instance.vehicle.max_time_steps + 1);
      for (int t = 0; t <= instance.vehicle.max_time_steps; ++t) {
        snapshot.poses[t] = poseAtIntegerTime(trajectory, t);
      }
    }
    snapshots.push_back(std::move(snapshot));
  }
  return snapshots;
}

std::vector<std::vector<const Constraint*>> bucketConstraintsByTime(
    const ConstraintSet& constraints, int max_time_steps) {
  std::vector<std::vector<const Constraint*>> buckets(max_time_steps + 1);
  for (const Constraint& constraint : constraints) {
    if (constraint.time_step >= 0 && constraint.time_step <= max_time_steps) {
      buckets[constraint.time_step].push_back(&constraint);
    }
  }
  return buckets;
}

// ---------------------------------------------------------------------------
// SafeIntervalTable
//
// Lazily computes and caches the safe intervals of each discretized cell.
//
// DESIGN (the core of this rewrite):
//
//   The table answers exactly ONE question: "if the vehicle footprint sits
//   at this cell, during which integer-time windows is it free of dynamic
//   obstacles and time-point constraints?" It does NOT decide whether an
//   edge can be traversed -- that is the sole job of the swept-pose primitive
//   check in transitionIsSafe(). The previous implementation let the table
//   and the swept check both judge collision, and they could disagree;
//   here each concern has exactly one owner.
//
//   Intervals are keyed on the cell and computed from the FIRST actual pose
//   that reaches that cell. There is no synthetic "cell-centre pose". At the
//   benchmark resolution (0.5 m, 72 heading bins => 5 deg) any two poses that
//   quantize to the same cell are within ~0.35 m and ~2.5 deg of each other;
//   for a 4.77 m x 1.86 m vehicle the resulting footprint difference is
//   negligible, so caching by cell is sound and -- because the computing pose
//   is a real, free pose the search actually visited -- there is no cache
//   poisoning.
//
//   When the environment is fully static the whole cell is trivially one
//   interval [0, max_time_steps]; the per-time sweep is skipped entirely.
// ---------------------------------------------------------------------------

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
        constraints_by_time_(constraints_by_time) {
    for (const ObstacleSnapshot& snapshot : obstacle_snapshots_) {
      if (snapshot.active) {
        has_dynamic_info_ = true;
        break;
      }
    }
    if (!has_dynamic_info_) {
      for (const auto& bucket : constraints_by_time_) {
        if (!bucket.empty()) {
          has_dynamic_info_ = true;
          break;
        }
      }
    }
  }

  // Returns the safe intervals for the cell that `pose` quantizes to. The
  // result is computed once (from `pose`) and cached. A cell that is in
  // static collision yields an empty interval list.
  const std::vector<SafeInterval>& intervals(
      const CellKey& key, const Pose& pose) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }
    return cache_.emplace(key, compute(pose)).first->second;
  }

 private:
  // True iff `pose` is in collision with a dynamic obstacle or violates a
  // time-point constraint at integer time `time_step`. Static geometry is
  // intentionally NOT considered here -- it is handled once in compute().
  bool dynamicallyUnsafeAt(const Pose& pose, int time_step) const {
    for (const Constraint* constraint : constraints_by_time_[time_step]) {
      const double dx = pose.x - constraint->pose.x;
      const double dy = pose.y - constraint->pose.y;
      const double tol = constraint->position_tolerance + 1e-9;
      if (dx * dx + dy * dy <= tol * tol &&
          angleDifference(pose.yaw, constraint->pose.yaw) <=
              constraint->yaw_tolerance + 1e-9) {
        return true;
      }
    }
    for (const ObstacleSnapshot& snapshot : obstacle_snapshots_) {
      if (!snapshot.active) {
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
    std::vector<SafeInterval> result;

    // Static obstacles never move: a statically-blocked cell is blocked for
    // all time and has no safe interval.
    if (checker_.collides(pose)) {
      return result;
    }

    // Fully static environment: the entire horizon is one safe interval.
    if (!has_dynamic_info_) {
      result.push_back(SafeInterval{0, instance_.vehicle.max_time_steps});
      return result;
    }

    // Sweep integer time, accumulating maximal runs of safe steps.
    int run_start = -1;
    for (int t = 0; t <= instance_.vehicle.max_time_steps; ++t) {
      const bool safe = !dynamicallyUnsafeAt(pose, t);
      if (safe && run_start < 0) {
        run_start = t;
      } else if (!safe && run_start >= 0) {
        result.push_back(SafeInterval{run_start, t - 1});
        run_start = -1;
      }
    }
    if (run_start >= 0) {
      result.push_back(
          SafeInterval{run_start, instance_.vehicle.max_time_steps});
    }
    return result;
  }

  const Instance& instance_;
  const CollisionChecker& checker_;
  const std::vector<ObstacleSnapshot>& obstacle_snapshots_;
  const std::vector<std::vector<const Constraint*>>& constraints_by_time_;
  bool has_dynamic_info_ = false;
  std::unordered_map<CellKey, std::vector<SafeInterval>, CellKeyHash> cache_;
};

int findIntervalIndex(
    const std::vector<SafeInterval>& intervals, int time_step) {
  for (std::size_t i = 0; i < intervals.size(); ++i) {
    if (time_step >= intervals[i].lo && time_step <= intervals[i].hi) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

// ---------------------------------------------------------------------------
// Edge validity: the SINGLE authority on whether a motion can be executed.
//
// A primitive departs `start_pose` at integer time `depart_time` and arrives
// at `end_pose` at `depart_time + 1`. The motion is valid iff every swept
// sub-sample is free of static geometry AND free of every dynamic obstacle at
// the matching continuous time, and the arrival pose satisfies the time-point
// constraints at the arrival step.
// ---------------------------------------------------------------------------

bool transitionIsSafe(
    const Pose& start_pose,
    const Pose& end_pose,
    const Primitive& primitive,
    int depart_time,
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
    const double query_time = static_cast<double>(depart_time) + fraction;
    for (const Trajectory& obstacle : dynamic_obstacles) {
      if (obstacle.empty()) {
        continue;
      }
      if (checker.overlaps(
              sample, trajectoryPoseAtContinuousTime(obstacle, query_time))) {
        return false;
      }
    }
  }
  // Arrival pose vs. time-point constraints at the arrival step. Dynamic
  // obstacles at the arrival step are already covered by the fraction == 1.0
  // sample above.
  return !checker.collides(end_pose, {}, constraints, depart_time + 1);
}

// Finds the earliest integer departure time in [depart_lo, depart_hi] for
// which the transition is safe. When the environment is static, transition
// safety is time-independent, so a single probe at depart_lo suffices.
int findEarliestSafeDeparture(
    const Pose& start_pose,
    const Pose& end_pose,
    const Primitive& primitive,
    int depart_lo,
    int depart_hi,
    bool has_dynamic_info,
    const Instance& instance,
    const PlannerConfig& planner_config,
    const CollisionChecker& checker,
    const std::vector<Trajectory>& dynamic_obstacles,
    const ConstraintSet& constraints) {
  if (depart_lo > depart_hi) {
    return -1;
  }
  if (!has_dynamic_info) {
    return transitionIsSafe(
               start_pose, end_pose, primitive, depart_lo, instance,
               planner_config, checker, dynamic_obstacles, constraints)
               ? depart_lo
               : -1;
  }
  for (int depart = depart_lo; depart <= depart_hi; ++depart) {
    if (transitionIsSafe(
            start_pose, end_pose, primitive, depart, instance, planner_config,
            checker, dynamic_obstacles, constraints)) {
      return depart;
    }
  }
  return -1;
}

// ---------------------------------------------------------------------------
// Goal validity: the goal must be collision-free from the arrival step all
// the way to the end of the horizon, otherwise a later obstacle would force
// the parked agent to move. This forward scan matches HybridAStarPlanner's
// primitiveKeepsGoalSafe semantics and, unlike a pre-computed interval test,
// is robust to an earlier agent transiently crossing the goal cell (such a
// crossing splits the goal cell's interval list so no single interval reaches
// the horizon, even though the goal is genuinely clear from arrival onward).
// ---------------------------------------------------------------------------

bool goalIsClearFromArrival(
    const Pose& goal_pose,
    int arrival_time,
    const Instance& instance,
    const CollisionChecker& checker,
    const std::vector<Trajectory>& dynamic_obstacles,
    const ConstraintSet& constraints) {
  for (int t = arrival_time; t <= instance.vehicle.max_time_steps; ++t) {
    if (checker.collides(goal_pose, dynamic_obstacles, constraints, t)) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Trajectory reconstruction. SIPP nodes record motion only at departure /
// arrival times, so the waiting steps in between are filled in (the agent
// holds its pose) to produce one trajectory point per integer time step --
// the dense format the rest of the system expects.
// ---------------------------------------------------------------------------

Trajectory reconstructTrajectory(
    const std::vector<SearchNode>& nodes, int goal_index) {
  Trajectory sparse;
  for (int current = goal_index; current >= 0;
       current = nodes[current].parent_index) {
    sparse.push_back(
        TrajectoryPoint{nodes[current].arrival_time, nodes[current].pose});
  }
  std::reverse(sparse.begin(), sparse.end());

  Trajectory dense;
  if (!sparse.empty()) {
    dense.reserve(static_cast<std::size_t>(sparse.back().t) + 1);
  }
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

// ---------------------------------------------------------------------------
// SippPlanner::plan
//
// Safe-Interval Path Planning over the Hybrid-A* continuous state space.
//
// The search state is (cell, safe-interval). For each motion primitive and
// each safe interval of the resulting successor cell, the planner generates
// AT MOST ONE successor node, representing "wait in the current cell as long
// as necessary, then execute the primitive so as to arrive in that interval
// as early as possible". Waiting therefore costs no extra nodes -- this is
// what gives SIPP its expanded-states advantage over a Hybrid A* that must
// spend one node per waited time step.
// ---------------------------------------------------------------------------

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

  bool has_dynamic_info = false;
  for (const ObstacleSnapshot& snapshot : obstacle_snapshots) {
    if (snapshot.active) {
      has_dynamic_info = true;
      break;
    }
  }
  if (!has_dynamic_info) {
    for (const auto& bucket : constraints_by_time) {
      if (!bucket.empty()) {
        has_dynamic_info = true;
        break;
      }
    }
  }

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

  // best_costs maps each dominance state -- (cell, interval, fine-pose-bin) --
  // to the best (earliest) arrival time settled for it. Because the fine-pose
  // bin is part of the key, two kinematically distinct poses in the same
  // search cell occupy different entries and neither is pruned by the other;
  // within a single fine bin, where poses are interchangeable, the earliest
  // arrival dominates. Integer arrival times make the check exact.
  std::unordered_map<StateKey, int, StateKeyHash> best_costs;
  best_costs.reserve(kSearchReserveHint);

  const double start_h = heuristic(start, goal, instance.vehicle.step_size);
  nodes.push_back(SearchNode{
      start,
      0,                     // arrival_time
      start_interval_index,  // interval_index
      start_cell,            // cell
      start_h,               // f_cost (g = 0)
      -1,                    // parent_index
  });
  open.push(OpenEntry{start_h, 0, 0});
  best_costs.emplace(
      makeStateKey(start, start_cell, start_interval_index), 0);

  while (!open.empty()) {
    const OpenEntry current_entry = open.top();
    open.pop();

    const SearchNode current = nodes[current_entry.node_index];
    const StateKey current_key =
        makeStateKey(current.pose, current.cell, current.interval_index);

    // Stale-node check: skip if a strictly better path to this exact state
    // (same cell, interval, AND fine-pose bin) was already settled.
    const auto best_it = best_costs.find(current_key);
    if (best_it == best_costs.end() ||
        current.arrival_time > best_it->second) {
      continue;
    }

    const std::vector<SafeInterval>& cur_intervals =
        intervals.intervals(current.cell, current.pose);
    if (current.interval_index >=
        static_cast<int>(cur_intervals.size())) {
      // Defensive: the interval list is path-independent, so this should not
      // happen, but guard against an out-of-range index rather than risk UB.
      continue;
    }
    const SafeInterval cur_interval = cur_intervals[current.interval_index];

    // Goal test: the agent is at the goal pose AND the goal stays clear from
    // the arrival step to the end of the horizon.
    if (goalReached(current.pose, goal, planner_config) &&
        goalIsClearFromArrival(
            current.pose,
            current.arrival_time,
            instance,
            checker,
            dynamic_obstacles,
            constraints)) {
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

    // The agent may wait in the current cell anywhere within its current
    // safe interval. The latest it can still depart is bounded by the
    // interval's upper edge and by the planning horizon.
    const int latest_depart = std::min(
        cur_interval.hi, instance.vehicle.max_time_steps - 1);

    for (const Primitive& primitive : kPrimitives) {
      const Pose next_pose = integratePrimitive(
          current.pose,
          primitive.direction,
          primitive.steer,
          instance.vehicle,
          1.0);

      // Reject successors that are in static collision before touching the
      // interval table -- a statically-blocked cell has no intervals and
      // would only waste a cache slot.
      if (checker.collides(next_pose)) {
        continue;
      }

      const CellKey next_cell = makeCellKey(next_pose, instance);
      const std::vector<SafeInterval>& next_intervals =
          intervals.intervals(next_cell, next_pose);

      for (std::size_t k = 0; k < next_intervals.size(); ++k) {
        const SafeInterval& next_interval = next_intervals[k];

        // The primitive takes one time step: depart at T, arrive at T + 1.
        // The arrival T + 1 must land inside next_interval, so the feasible
        // departure window is the current interval intersected with the
        // (shifted-back) successor interval and the horizon.
        const int depart_lo =
            std::max(current.arrival_time, next_interval.lo - 1);
        const int depart_hi = std::min(latest_depart, next_interval.hi - 1);
        if (depart_lo > depart_hi) {
          continue;
        }

        const int depart = findEarliestSafeDeparture(
            current.pose,
            next_pose,
            primitive,
            depart_lo,
            depart_hi,
            has_dynamic_info,
            instance,
            planner_config,
            checker,
            dynamic_obstacles,
            constraints);
        if (depart < 0) {
          continue;
        }

        const int arrival = depart + 1;
        const StateKey next_key =
            makeStateKey(next_pose, next_cell, static_cast<int>(k));

        // Dominance: skip the successor if an equal-or-better arrival is
        // already recorded for this exact state (cell, interval, fine-pose
        // bin). Otherwise relax the state and push the node.
        const auto existing = best_costs.find(next_key);
        if (existing != best_costs.end() && arrival >= existing->second) {
          continue;
        }
        best_costs[next_key] = arrival;

        const double new_f =
            static_cast<double>(arrival) +
            heuristic(next_pose, goal, instance.vehicle.step_size);

        nodes.push_back(SearchNode{
            next_pose,
            arrival,
            static_cast<int>(k),
            next_cell,
            new_f,
            current_entry.node_index,
        });
        open.push(OpenEntry{
            new_f, arrival, static_cast<int>(nodes.size() - 1)});
      }
    }
  }

  return fail("search exhausted without reaching goal");
}

}  // namespace parking_lot_planner