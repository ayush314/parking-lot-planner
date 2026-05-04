#include "planner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

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

using Footprint = std::array<Vec2, 4>;

Vec2 operator-(const Vec2& lhs, const Vec2& rhs) {
  return Vec2{lhs.x - rhs.x, lhs.y - rhs.y};
}

double dot(const Vec2& lhs, const Vec2& rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y;
}

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

bool overlapsOnAxis(
    const Footprint& lhs,
    const Footprint& rhs,
    const Vec2& axis) {
  double lhs_min = std::numeric_limits<double>::infinity();
  double lhs_max = -std::numeric_limits<double>::infinity();
  double rhs_min = std::numeric_limits<double>::infinity();
  double rhs_max = -std::numeric_limits<double>::infinity();

  for (const Vec2& point : lhs) {
    const double projection = dot(point, axis);
    lhs_min = std::min(lhs_min, projection);
    lhs_max = std::max(lhs_max, projection);
  }

  for (const Vec2& point : rhs) {
    const double projection = dot(point, axis);
    rhs_min = std::min(rhs_min, projection);
    rhs_max = std::max(rhs_max, projection);
  }

  return lhs_max + kEpsilon >= rhs_min && rhs_max + kEpsilon >= lhs_min;
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

Footprint makeFootprint(
    const Pose& pose,
    const VehicleParams& vehicle,
    double buffer = 0.0) {
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

  Footprint footprint{};
  for (std::size_t i = 0; i < local_corners.size(); ++i) {
    const Vec2& local = local_corners[i];
    footprint[i] = Vec2{
        pose.x + local.x * cos_yaw - local.y * sin_yaw,
        pose.y + local.x * sin_yaw + local.y * cos_yaw,
    };
  }
  return footprint;
}

bool rectanglesIntersect(const Footprint& lhs, const Footprint& rhs) {
  std::array<Vec2, 8> axes{};
  int axis_index = 0;

  auto collect_axes = [&](const Footprint& footprint) {
    for (std::size_t i = 0; i < footprint.size(); ++i) {
      const Vec2 edge = footprint[(i + 1) % footprint.size()] - footprint[i];
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

bool footprintWithinBounds(const Footprint& footprint, const MapParams& map) {
  return std::all_of(
      footprint.begin(),
      footprint.end(),
      [&](const Vec2& point) {
        return point.x >= -kEpsilon && point.x <= map.width + kEpsilon &&
               point.y >= -kEpsilon && point.y <= map.height + kEpsilon;
      });
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

bool posesConflict(
    const Pose& lhs,
    const Pose& rhs,
    const VehicleParams& vehicle) {
  return rectanglesIntersect(
      makeFootprint(lhs, vehicle, kCollisionBuffer),
      makeFootprint(rhs, vehicle, kCollisionBuffer));
}

bool collidesWithStaticEnvironment(
    const Pose& pose,
    const Instance& instance) {
  const Footprint active =
      makeFootprint(pose, instance.vehicle, kCollisionBuffer);
  if (!footprintWithinBounds(active, instance.map)) {
    return true;
  }

  for (const StaticCar& parked_car : instance.static_cars) {
    const Footprint obstacle =
        makeFootprint(parked_car.pose, instance.vehicle, kCollisionBuffer);
    if (rectanglesIntersect(active, obstacle)) {
      return true;
    }
  }
  return false;
}

}  // namespace

CollisionChecker::CollisionChecker(const Instance& instance)
    : instance_(instance) {}

bool CollisionChecker::collides(const Pose& pose) const {
  return collidesWithStaticEnvironment(pose, instance_);
}

bool CollisionChecker::collides(
    const Pose& pose,
    const std::vector<Trajectory>& dynamic_obstacles,
    const ConstraintSet& constraints,
    int time_step) const {
  if (collidesWithStaticEnvironment(pose, instance_)) {
    return true;
  }

  for (const Trajectory& obstacle : dynamic_obstacles) {
    if (obstacle.empty()) {
      continue;
    }
    const Pose obstacle_pose = poseAtTimeInternal(obstacle, time_step);
    if (posesConflict(pose, obstacle_pose, instance_.vehicle)) {
      return true;
    }
  }

  return violatesConstraint(pose, constraints, time_step);
}

bool CollisionChecker::overlaps(const Pose& lhs, const Pose& rhs) const {
  return posesConflict(lhs, rhs, instance_.vehicle);
}

}  // namespace parking_lot_planner
