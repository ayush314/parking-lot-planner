#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "planner.hpp"

namespace {

using parking_lot_planner::AgentSpec;
using parking_lot_planner::CollisionChecker;
using parking_lot_planner::Constraint;
using parking_lot_planner::ConstraintSet;
using parking_lot_planner::HybridAStarPlanner;
using parking_lot_planner::Instance;
using parking_lot_planner::LowLevelPlanner;
using parking_lot_planner::PlanStats;
using parking_lot_planner::PlanResult;
using parking_lot_planner::Pose;
using parking_lot_planner::PrioritizedPlanner;
using parking_lot_planner::Solution;
using parking_lot_planner::StaticCar;
using parking_lot_planner::Trajectory;
using parking_lot_planner::TrajectoryMap;
using parking_lot_planner::TrajectoryPoint;
using parking_lot_planner::loadInstance;
using parking_lot_planner::validateInstance;

double yawDifference(double lhs, double rhs) {
  constexpr double kPi = 3.14159265358979323846;
  double delta = lhs - rhs;
  while (delta >= kPi) {
    delta -= 2.0 * kPi;
  }
  while (delta < -kPi) {
    delta += 2.0 * kPi;
  }
  return std::abs(delta);
}

void expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

Instance makeInstance() {
  Instance instance;
  instance.map.width = 40.0;
  instance.map.height = 30.0;
  instance.map.resolution = 0.5;
  instance.vehicle.length = 4.8;
  instance.vehicle.width = 2.0;
  instance.vehicle.min_turning_radius = 5.5;
  instance.vehicle.step_size = 0.5;
  instance.vehicle.heading_bins = 72;
  instance.vehicle.max_time_steps = 140;
  return instance;
}

Trajectory makeTrajectory(const std::vector<Pose>& poses) {
  Trajectory trajectory;
  trajectory.reserve(poses.size());
  for (std::size_t i = 0; i < poses.size(); ++i) {
    trajectory.push_back(TrajectoryPoint{static_cast<int>(i), poses[i]});
  }
  return trajectory;
}

class StubLowLevelPlanner : public LowLevelPlanner {
 public:
  mutable int call_count = 0;

  PlanResult plan(
      const Pose& start,
      const Pose& goal,
      const Instance& instance,
      const std::vector<Trajectory>& dynamic_obstacles,
      const ConstraintSet& constraints) const override {
    (void)instance;
    (void)dynamic_obstacles;
    (void)constraints;
    ++call_count;
    return PlanResult{
        true,
        makeTrajectory({start, goal}),
        "",
        PlanStats{3},
    };
  }
};

void testSingleAgentPlannerAvoidsStaticCar() {
  Instance instance = makeInstance();
  instance.map.height = 35.0;
  instance.static_cars = {
      StaticCar{"blocker", Pose{12.0, 10.0, 0.0}},
  };
  const HybridAStarPlanner planner;

  const PlanResult result = planner.plan(
      Pose{4.0, 10.0, 0.0},
      Pose{20.0, 10.0, 0.0},
      instance,
      {},
      ConstraintSet{});

  expect(result.success, "single-agent planner failed around static obstacle");
  bool deviated = false;
  for (const TrajectoryPoint& point : result.trajectory) {
    if (std::abs(point.pose.y - 10.0) > 0.75 ||
        yawDifference(point.pose.yaw, 0.0) > 0.15) {
      deviated = true;
      break;
    }
  }
  expect(deviated, "planner did not deviate around the static obstacle");
}

void testDynamicObstacleOccupiesGoal() {
  Instance instance = makeInstance();
  instance.vehicle.max_time_steps = 90;
  const HybridAStarPlanner planner;

  const Trajectory occupied_goal = makeTrajectory({Pose{12.0, 10.0, 0.0}});
  const PlanResult result = planner.plan(
      Pose{4.0, 10.0, 0.0},
      Pose{12.0, 10.0, 0.0},
      instance,
      {occupied_goal},
      ConstraintSet{});

  expect(!result.success, "planner should fail when the goal remains occupied");
}

void testWaitPrimitiveSupportsTimedDelay() {
  Instance instance = makeInstance();
  instance.map.width = 24.0;
  instance.map.height = 20.0;
  instance.vehicle.step_size = 6.0;
  instance.vehicle.max_time_steps = 3;
  instance.planner.goal_position_tolerance = 0.25;

  const HybridAStarPlanner planner;
  const Trajectory timed_blocker = {
      TrajectoryPoint{0, Pose{14.0, 14.0, 0.0}},
      TrajectoryPoint{1, Pose{9.0, 10.0, 0.0}},
      TrajectoryPoint{2, Pose{14.0, 14.0, 0.0}},
      TrajectoryPoint{3, Pose{14.0, 14.0, 0.0}},
  };

  const Pose start{3.0, 10.0, 0.0};
  const PlanResult result = planner.plan(
      start,
      Pose{9.0, 10.0, 0.0},
      instance,
      {timed_blocker},
      ConstraintSet{});

  expect(result.success, "planner should succeed by waiting for the blocker");
  expect(
      result.trajectory.size() >= 3U,
      "expected a delayed trajectory before moving");
  expect(std::abs(result.trajectory[1].pose.x - start.x) < 1e-9 &&
             std::abs(result.trajectory[1].pose.y - start.y) < 1e-9,
         "expected explicit wait at t=1");
}

void testWaitConstraintIsRejected() {
  Instance instance = makeInstance();
  instance.map.width = 24.0;
  instance.map.height = 20.0;
  instance.vehicle.step_size = 6.0;
  instance.vehicle.max_time_steps = 3;
  instance.planner.goal_position_tolerance = 0.25;

  const HybridAStarPlanner planner;
  const Trajectory timed_blocker = {
      TrajectoryPoint{0, Pose{14.0, 14.0, 0.0}},
      TrajectoryPoint{1, Pose{9.0, 10.0, 0.0}},
      TrajectoryPoint{2, Pose{14.0, 14.0, 0.0}},
      TrajectoryPoint{3, Pose{14.0, 14.0, 0.0}},
  };

  const PlanResult result = planner.plan(
      Pose{3.0, 10.0, 0.0},
      Pose{9.0, 10.0, 0.0},
      instance,
      {timed_blocker},
      ConstraintSet{Constraint{1, Pose{3.0, 10.0, 0.0}, 0.25, 0.1}});

  expect(!result.success, "planner should fail when waiting violates constraints");
}

void testPlannerRejectsMidStepDynamicCollision() {
  Instance instance = makeInstance();
  instance.map.width = 24.0;
  instance.map.height = 20.0;
  instance.vehicle.step_size = 5.0;
  instance.vehicle.max_time_steps = 1;
  instance.planner.goal_position_tolerance = 0.25;

  const HybridAStarPlanner planner;
  const Trajectory crossing_blocker = {
      TrajectoryPoint{0, Pose{7.5, 15.0, -1.57079632679}},
      TrajectoryPoint{1, Pose{12.0, 5.0, -1.57079632679}},
  };

  const PlanResult result = planner.plan(
      Pose{5.0, 10.0, 0.0},
      Pose{10.0, 10.0, 0.0},
      instance,
      {crossing_blocker},
      ConstraintSet{});

  expect(
      !result.success,
      "planner should reject a motion that collides mid-step with a dynamic obstacle");
}

void testPlannerRejectsGoalThatBecomesOccupiedLater() {
  Instance instance = makeInstance();
  instance.map.width = 20.0;
  instance.map.height = 20.0;
  instance.vehicle.step_size = 5.0;
  instance.vehicle.max_time_steps = 3;
  instance.planner.goal_position_tolerance = 0.25;

  const HybridAStarPlanner planner;
  const Trajectory future_goal_blocker = {
      TrajectoryPoint{0, Pose{15.0, 15.0, 0.0}},
      TrajectoryPoint{1, Pose{15.0, 15.0, 0.0}},
      TrajectoryPoint{2, Pose{10.0, 10.0, 0.0}},
      TrajectoryPoint{3, Pose{10.0, 10.0, 0.0}},
  };

  const PlanResult result = planner.plan(
      Pose{5.0, 10.0, 0.0},
      Pose{10.0, 10.0, 0.0},
      instance,
      {future_goal_blocker},
      ConstraintSet{});

  expect(!result.success, "planner should reject a goal that becomes occupied later");
}

void testPrioritizedPlannerBaseline() {
  Instance instance = makeInstance();
  instance.static_cars = {
      StaticCar{"s0", Pose{18.0, 8.0, 1.57079632679}},
      StaticCar{"s1", Pose{22.0, 8.0, 1.57079632679}},
      StaticCar{"s2", Pose{18.0, 22.0, 1.57079632679}},
      StaticCar{"s3", Pose{22.0, 22.0, 1.57079632679}},
  };
  instance.agents = {
      AgentSpec{"0", Pose{6.0, 6.0, 0.0}, Pose{30.0, 8.0, 1.57079632679}},
      AgentSpec{"1", Pose{6.0, 18.0, 0.0}, Pose{30.0, 22.0, 1.57079632679}},
  };

  PrioritizedPlanner solver;
  const Solution solution = solver.solve(instance);
  expect(solution.success, "prioritized planner failed on baseline instance");
  expect(solution.trajectories.size() == 2U, "expected trajectories for both agents");
  expect(solution.metrics.total_low_level_expanded_states > 0,
         "expected aggregated low-level expansion stats");
}

void testPrioritizedPlannerAcceptsInjectedLowLevelPlanner() {
  Instance instance = makeInstance();
  instance.agents = {
      AgentSpec{"0", Pose{4.0, 4.0, 0.0}, Pose{6.0, 4.0, 0.0}},
      AgentSpec{"1", Pose{10.0, 10.0, 0.0}, Pose{12.0, 10.0, 0.0}},
  };

  StubLowLevelPlanner low_level_planner;
  const PrioritizedPlanner solver(low_level_planner);
  const Solution solution = solver.solve(instance);

  expect(solution.success, "solver with injected low-level planner should succeed");
  expect(low_level_planner.call_count == 2, "expected one low-level call per agent");
  expect(solution.metrics.total_low_level_expanded_states == 6,
         "expected aggregated expanded-state count from stub planner");
}

void testValidateInstanceRejectsInvalidResolution() {
  Instance instance = makeInstance();
  instance.map.resolution = 0.0;

  bool threw = false;
  try {
    validateInstance(instance);
  } catch (const std::invalid_argument&) {
    threw = true;
  }

  expect(threw, "validateInstance should reject zero map resolution");
}

void testPlannerConfigParsing() {
  const std::string path = "planner_config_test.yaml";
  std::ofstream output(path);
  output << "map:\n";
  output << "  width: 20.0\n";
  output << "  height: 20.0\n";
  output << "  resolution: 0.5\n";
  output << "vehicle:\n";
  output << "  length: 4.8\n";
  output << "  width: 2.0\n";
  output << "  min_turning_radius: 5.5\n";
  output << "  step_size: 0.5\n";
  output << "  heading_bins: 72\n";
  output << "  max_time_steps: 50\n";
  output << "planner:\n";
  output << "  goal_position_tolerance: 0.75\n";
  output << "  goal_yaw_tolerance: 0.35\n";
  output << "  collision_check_resolution: 0.2\n";
  output << "  max_expansions: 1234\n";
  output << "parking_layout:\n";
  output << "  grid_cols: 2\n";
  output << "  grid_rows: 2\n";
  output << "  stalls_per_side: 6\n";
  output << "  stall_width: 3.35\n";
  output << "  stall_depth: 6.3\n";
  output << "  zone_gap: 8.375\n";
  output << "  perimeter_gap: 8.375\n";
  output << "agents:\n";
  output << "  - id: 0\n";
  output << "    start: [1.0, 1.0, 0.0]\n";
  output << "    goal: [2.0, 2.0, 0.0]\n";
  output.close();

  const Instance instance = loadInstance(path);
  std::remove(path.c_str());

  expect(std::abs(instance.planner.goal_position_tolerance - 0.75) < 1e-9,
         "expected goal_position_tolerance to parse");
  expect(std::abs(instance.planner.goal_yaw_tolerance - 0.35) < 1e-9,
         "expected goal_yaw_tolerance to parse");
  expect(std::abs(instance.planner.collision_check_resolution - 0.2) < 1e-9,
         "expected collision_check_resolution to parse");
  expect(instance.planner.max_expansions == 1234,
         "expected max_expansions to parse");
  expect(instance.parking_layout.enabled, "expected parking_layout to parse");
  expect(instance.parking_layout.grid_cols == 2,
         "expected grid_cols to parse");
  expect(instance.parking_layout.grid_rows == 2,
         "expected grid_rows to parse");
  expect(instance.parking_layout.stalls_per_side == 6,
         "expected stalls_per_side to parse");
  expect(std::abs(instance.parking_layout.stall_width - 3.35) < 1e-9,
         "expected stall_width to parse");
  expect(std::abs(instance.parking_layout.stall_depth - 6.3) < 1e-9,
         "expected stall_depth to parse");
}

void testMaxExpansionsStopsSearch() {
  Instance instance = makeInstance();
  instance.planner.max_expansions = 1;
  instance.planner.goal_position_tolerance = 0.1;
  const HybridAStarPlanner planner;

  const PlanResult result = planner.plan(
      Pose{4.0, 10.0, 0.0},
      Pose{10.0, 10.0, 0.0},
      instance,
      {},
      ConstraintSet{});

  expect(!result.success, "planner should stop once max_expansions is reached");
  expect(result.failure_reason == "search reached max expansions",
         "expected max-expansion failure reason");
  expect(result.stats.expanded_states == 1,
         "expected exactly one expanded state before stopping");
}

}  // namespace

int main() {
  try {
    testSingleAgentPlannerAvoidsStaticCar();
    testDynamicObstacleOccupiesGoal();
    testWaitPrimitiveSupportsTimedDelay();
    testWaitConstraintIsRejected();
    testPlannerRejectsMidStepDynamicCollision();
    testPlannerRejectsGoalThatBecomesOccupiedLater();
    testPrioritizedPlannerBaseline();
    testPrioritizedPlannerAcceptsInjectedLowLevelPlanner();
    testValidateInstanceRejectsInvalidResolution();
    testPlannerConfigParsing();
    testMaxExpansionsStopsSearch();
    std::cout << "planner_tests: all tests passed\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "planner_tests: " << ex.what() << "\n";
    return 1;
  }
}
