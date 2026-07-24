#include "planner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

// ---------------------------------------------------------------------------
// Ablation toggles.
//
// Each of the three collision-checking optimizations can be independently
// switched off at COMPILE TIME so their individual and combined impact can be
// measured. All default to ON (1), so a normal build is the fully-optimized
// production checker; the OFF (0) paths reproduce the original pre-optimization
// behavior exactly.
//
//   PLP_OPT_UNIT_AXIS   1: 4 unit SAT axes derived from yaw (no sqrt)
//                       0: 8 edge-normal axes, each normalized (legacy)
//   PLP_OPT_PRECOMPUTE  1: parked-car boxes built once in the constructor
//                       0: parked-car boxes rebuilt on every query (legacy)
//   PLP_OPT_BROADPHASE  1: cheap AABB reject before the exact SAT test
//                       0: no broad phase; always run exact SAT (legacy)
// ---------------------------------------------------------------------------
#ifndef PLP_OPT_UNIT_AXIS
#define PLP_OPT_UNIT_AXIS 1
#endif
#ifndef PLP_OPT_PRECOMPUTE
#define PLP_OPT_PRECOMPUTE 1
#endif
#ifndef PLP_OPT_BROADPHASE
#define PLP_OPT_BROADPHASE 1
#endif

namespace parking_lot_planner {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kEpsilon = 1e-9;
constexpr double kCollisionBuffer = 0.2;

struct Vec2 {
  double x = 0.0;
  double y = 0.0;
};

Vec2 operator-(const Vec2& lhs, const Vec2& rhs) {
  return Vec2{lhs.x - rhs.x, lhs.y - rhs.y};
}

double dot(const Vec2& lhs, const Vec2& rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y;
}

#if !PLP_OPT_UNIT_AXIS
double norm(const Vec2& value) {
  return std::sqrt(dot(value, value));
}

Vec2 normalized(const Vec2& value) {
  const double length = norm(value);
  if (length < kEpsilon) {
    return Vec2{0.0, 0.0};
  }
  return Vec2{value.x / length, value.y / length};
}
#endif

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

// ---------------------------------------------------------------------------
// OrientedBox: a vehicle footprint as an oriented rectangle, precomputed so
// the hot path never recomputes corners or trig.
// ---------------------------------------------------------------------------
struct OrientedBox {
  std::array<Vec2, 4> corners{};
  std::array<Vec2, 2> axes{};
  double min_x = 0.0;
  double min_y = 0.0;
  double max_x = 0.0;
  double max_y = 0.0;
};

OrientedBox makeBox(
    const Pose& pose,
    const VehicleParams& vehicle,
    double buffer) {
  const double half_length = vehicle.length * 0.5 + buffer;
  const double half_width = vehicle.width * 0.5 + buffer;
  const double cos_yaw = std::cos(pose.yaw);
  const double sin_yaw = std::sin(pose.yaw);

  const std::array<Vec2, 4> local_corners = {{
      {half_length, half_width},
      {half_length, -half_width},
      {-half_length, -half_width},
      {-half_length, half_width},
  }};

  OrientedBox box;
  box.min_x = std::numeric_limits<double>::infinity();
  box.min_y = std::numeric_limits<double>::infinity();
  box.max_x = -std::numeric_limits<double>::infinity();
  box.max_y = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < local_corners.size(); ++i) {
    const Vec2& local = local_corners[i];
    const Vec2 world{
        pose.x + local.x * cos_yaw - local.y * sin_yaw,
        pose.y + local.x * sin_yaw + local.y * cos_yaw,
    };
    box.corners[i] = world;
    box.min_x = std::min(box.min_x, world.x);
    box.min_y = std::min(box.min_y, world.y);
    box.max_x = std::max(box.max_x, world.x);
    box.max_y = std::max(box.max_y, world.y);
  }
#if PLP_OPT_UNIT_AXIS
  // Unit separating axes straight from the orientation -- no normalisation.
  box.axes[0] = Vec2{cos_yaw, sin_yaw};
  box.axes[1] = Vec2{-sin_yaw, cos_yaw};
#endif
  return box;
}

bool overlapsOnAxis(
    const OrientedBox& lhs,
    const OrientedBox& rhs,
    const Vec2& axis) {
  double lhs_min = std::numeric_limits<double>::infinity();
  double lhs_max = -std::numeric_limits<double>::infinity();
  double rhs_min = std::numeric_limits<double>::infinity();
  double rhs_max = -std::numeric_limits<double>::infinity();

  for (const Vec2& point : lhs.corners) {
    const double projection = dot(point, axis);
    lhs_min = std::min(lhs_min, projection);
    lhs_max = std::max(lhs_max, projection);
  }
  for (const Vec2& point : rhs.corners) {
    const double projection = dot(point, axis);
    rhs_min = std::min(rhs_min, projection);
    rhs_max = std::max(rhs_max, projection);
  }

  return lhs_max + kEpsilon >= rhs_min && rhs_max + kEpsilon >= lhs_min;
}

#if PLP_OPT_BROADPHASE
// Cheap, conservative broad phase: if the axis-aligned bounds are disjoint the
// oriented boxes cannot overlap, so the exact SAT test can be skipped. Never
// produces a false negative (disjoint AABBs => disjoint boxes).
bool aabbDisjoint(const OrientedBox& lhs, const OrientedBox& rhs) {
  return lhs.max_x + kEpsilon < rhs.min_x || rhs.max_x + kEpsilon < lhs.min_x ||
         lhs.max_y + kEpsilon < rhs.min_y || rhs.max_y + kEpsilon < lhs.min_y;
}
#endif

#if PLP_OPT_UNIT_AXIS
// Exact overlap using the two precomputed unit axes per box (4 axes total).
bool satOverlap(const OrientedBox& lhs, const OrientedBox& rhs) {
  return overlapsOnAxis(lhs, rhs, lhs.axes[0]) &&
         overlapsOnAxis(lhs, rhs, lhs.axes[1]) &&
         overlapsOnAxis(lhs, rhs, rhs.axes[0]) &&
         overlapsOnAxis(lhs, rhs, rhs.axes[1]);
}
#else
// Legacy exact overlap: derive all 8 edge normals from the corners and
// normalize each (sqrt per axis), then look for a separating axis.
bool satOverlap(const OrientedBox& lhs, const OrientedBox& rhs) {
  std::array<Vec2, 8> axes{};
  int axis_index = 0;
  auto collect_axes = [&](const OrientedBox& box) {
    for (std::size_t i = 0; i < box.corners.size(); ++i) {
      const Vec2 edge =
          box.corners[(i + 1) % box.corners.size()] - box.corners[i];
      axes[axis_index++] = normalized(Vec2{-edge.y, edge.x});
    }
  };
  collect_axes(lhs);
  collect_axes(rhs);

  for (const Vec2& axis : axes) {
    if (norm(axis) < kEpsilon) {
      continue;
    }
    if (!overlapsOnAxis(lhs, rhs, axis)) {
      return false;
    }
  }
  return true;
}
#endif

bool boxesIntersect(const OrientedBox& lhs, const OrientedBox& rhs) {
#if PLP_OPT_BROADPHASE
  if (aabbDisjoint(lhs, rhs)) {
    return false;
  }
#endif
  return satOverlap(lhs, rhs);
}

bool violatesConstraint(
    const Pose& pose,
    const ConstraintSet& constraints,
    int time_step) {
  for (const Constraint& constraint : constraints) {
    if (constraint.time_step != time_step) {
      continue;
    }
    const double dx = pose.x - constraint.pose.x;
    const double dy = pose.y - constraint.pose.y;
    const double distance = std::sqrt(dx * dx + dy * dy);
    if (distance <= constraint.position_tolerance + kEpsilon &&
        angleDifference(pose.yaw, constraint.pose.yaw) <=
            constraint.yaw_tolerance + kEpsilon) {
      return true;
    }
  }
  return false;
}

Pose poseAtTimeInternal(const Trajectory& trajectory, int time_step) {
  if (trajectory.empty()) {
    return Pose{};
  }
  if (time_step <= trajectory.front().t) {
    return trajectory.front().pose;
  }
  for (const TrajectoryPoint& point : trajectory) {
    if (point.t == time_step) {
      return point.pose;
    }
  }
  return trajectory.back().pose;
}

}  // namespace

// ---------------------------------------------------------------------------
// StaticIndex: the parked cars and map bounds resolved into precomputed
// oriented boxes. Built once per instance; queried (read-only) on every
// collision check.
// ---------------------------------------------------------------------------
struct CollisionChecker::StaticIndex {
  std::vector<OrientedBox> static_boxes;
  double map_width = 0.0;
  double map_height = 0.0;
};

CollisionChecker::CollisionChecker(const Instance& instance)
    : instance_(instance),
      static_index_(std::make_unique<StaticIndex>()) {
  static_index_->map_width = instance.map.width;
  static_index_->map_height = instance.map.height;
#if PLP_OPT_PRECOMPUTE
  static_index_->static_boxes.reserve(instance.static_cars.size());
  for (const StaticCar& parked_car : instance.static_cars) {
    static_index_->static_boxes.push_back(
        makeBox(parked_car.pose, instance.vehicle, kCollisionBuffer));
  }
#endif
}

CollisionChecker::~CollisionChecker() = default;

bool CollisionChecker::collides(const Pose& pose) const {
  const OrientedBox active = makeBox(pose, instance_.vehicle, kCollisionBuffer);

  // Map bounds: every corner must lie inside the lot.
  if (active.min_x < -kEpsilon || active.max_x > static_index_->map_width + kEpsilon ||
      active.min_y < -kEpsilon || active.max_y > static_index_->map_height + kEpsilon) {
    return true;
  }

#if PLP_OPT_PRECOMPUTE
  for (const OrientedBox& obstacle : static_index_->static_boxes) {
    if (boxesIntersect(active, obstacle)) {
      return true;
    }
  }
#else
  for (const StaticCar& parked_car : instance_.static_cars) {
    const OrientedBox obstacle =
        makeBox(parked_car.pose, instance_.vehicle, kCollisionBuffer);
    if (boxesIntersect(active, obstacle)) {
      return true;
    }
  }
#endif
  return false;
}

bool CollisionChecker::collides(
    const Pose& pose,
    const std::vector<Trajectory>& dynamic_obstacles,
    const ConstraintSet& constraints,
    int time_step) const {
  if (collides(pose)) {
    return true;
  }

  if (!dynamic_obstacles.empty()) {
    const OrientedBox active =
        makeBox(pose, instance_.vehicle, kCollisionBuffer);
    for (const Trajectory& obstacle : dynamic_obstacles) {
      if (obstacle.empty()) {
        continue;
      }
      const OrientedBox obstacle_box = makeBox(
          poseAtTimeInternal(obstacle, time_step),
          instance_.vehicle,
          kCollisionBuffer);
      if (boxesIntersect(active, obstacle_box)) {
        return true;
      }
    }
  }

  return violatesConstraint(pose, constraints, time_step);
}

bool CollisionChecker::overlaps(const Pose& lhs, const Pose& rhs) const {
  return boxesIntersect(
      makeBox(lhs, instance_.vehicle, kCollisionBuffer),
      makeBox(rhs, instance_.vehicle, kCollisionBuffer));
}

}  // namespace parking_lot_planner
