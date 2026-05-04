#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

#include "planner.hpp"

#include <yaml-cpp/yaml.h>

namespace parking_lot_planner {

namespace {

Pose parsePose(const YAML::Node& node) {
  if (!node || !node.IsSequence() || node.size() != 3U) {
    throw std::runtime_error("pose must be a sequence of [x, y, yaw]");
  }
  return Pose{
      node[0].as<double>(),
      node[1].as<double>(),
      node[2].as<double>(),
  };
}

std::string scalarToString(const YAML::Node& node, const std::string& field_name) {
  if (!node || !node.IsScalar()) {
    throw std::runtime_error(field_name + " must be a scalar");
  }
  return node.Scalar();
}

std::string escapeJson(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    switch (c) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += c;
        break;
    }
  }
  return escaped;
}

}  // namespace

Instance loadInstance(const std::string& path) {
  const YAML::Node root = YAML::LoadFile(path);

  Instance instance;

  const YAML::Node map = root["map"];
  if (!map) {
    throw std::runtime_error("missing required 'map' section");
  }
  instance.map.width = map["width"].as<double>();
  instance.map.height = map["height"].as<double>();
  instance.map.resolution = map["resolution"].as<double>();

  const YAML::Node vehicle = root["vehicle"];
  if (!vehicle) {
    throw std::runtime_error("missing required 'vehicle' section");
  }
  instance.vehicle.length = vehicle["length"].as<double>();
  instance.vehicle.width = vehicle["width"].as<double>();
  instance.vehicle.min_turning_radius = vehicle["min_turning_radius"].as<double>();
  instance.vehicle.step_size = vehicle["step_size"].as<double>();
  instance.vehicle.heading_bins = vehicle["heading_bins"].as<int>();
  instance.vehicle.max_time_steps = vehicle["max_time_steps"].as<int>();

  const YAML::Node planner = root["planner"];
  if (planner) {
    if (planner["goal_position_tolerance"]) {
      instance.planner.goal_position_tolerance =
          planner["goal_position_tolerance"].as<double>();
    }
    if (planner["goal_yaw_tolerance"]) {
      instance.planner.goal_yaw_tolerance =
          planner["goal_yaw_tolerance"].as<double>();
    }
    if (planner["collision_check_resolution"]) {
      instance.planner.collision_check_resolution =
          planner["collision_check_resolution"].as<double>();
    }
    if (planner["max_expansions"]) {
      instance.planner.max_expansions = planner["max_expansions"].as<int>();
    }
  }

  YAML::Node parking_layout = root["parking_layout"];
  if (parking_layout) {
    instance.parking_layout.enabled = true;
    instance.parking_layout.grid_cols = parking_layout["grid_cols"].as<int>();
    instance.parking_layout.grid_rows = parking_layout["grid_rows"].as<int>();
    instance.parking_layout.stalls_per_side =
        parking_layout["stalls_per_side"].as<int>();
    instance.parking_layout.stall_width =
        parking_layout["stall_width"].as<double>();
    instance.parking_layout.stall_depth =
        parking_layout["stall_depth"].as<double>();
    instance.parking_layout.zone_gap = parking_layout["zone_gap"].as<double>();
    instance.parking_layout.perimeter_gap =
        parking_layout["perimeter_gap"].as<double>();
  }

  const YAML::Node static_cars = root["static_cars"];
  if (static_cars) {
    for (const YAML::Node& node : static_cars) {
      instance.static_cars.push_back(StaticCar{
          scalarToString(node["id"], "static_cars.id"),
          parsePose(node["pose"]),
      });
    }
  }

  const YAML::Node agents = root["agents"];
  if (!agents || !agents.IsSequence() || agents.size() == 0U) {
    throw std::runtime_error("missing required non-empty 'agents' section");
  }
  for (const YAML::Node& node : agents) {
    instance.agents.push_back(AgentSpec{
        scalarToString(node["id"], "agents.id"),
        parsePose(node["start"]),
        parsePose(node["goal"]),
    });
  }

  return instance;
}

void writeSolution(const Solution& solution, const std::string& path) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to open output file: " + path);
  }

  output << std::fixed << std::setprecision(6);
  output << "{\n";
  output << "  \"success\": " << (solution.success ? "true" : "false") << ",\n";
  output << "  \"runtime_sec\": " << solution.runtime_sec;
  if (!solution.failure_reason.empty()) {
    output << ",\n  \"failure_reason\": \"" << escapeJson(solution.failure_reason)
           << "\"";
  }
  output << ",\n";
  output << "  \"metrics\": {\n";
  output << "    \"num_agents\": " << solution.metrics.num_agents << ",\n";
  output << "    \"makespan\": " << solution.metrics.makespan << ",\n";
  output << "    \"total_low_level_expanded_states\": "
         << solution.metrics.total_low_level_expanded_states << "\n";
  output << "  },\n";

  output << "  \"trajectories\": {\n";

  std::vector<std::string> agent_ids;
  agent_ids.reserve(solution.trajectories.size());
  for (const auto& [agent_id, trajectory] : solution.trajectories) {
    (void)trajectory;
    agent_ids.push_back(agent_id);
  }
  std::sort(agent_ids.begin(), agent_ids.end());

  for (std::size_t i = 0; i < agent_ids.size(); ++i) {
    const auto it = solution.trajectories.find(agent_ids[i]);
    output << "    \"" << escapeJson(agent_ids[i]) << "\": [\n";
    for (std::size_t j = 0; j < it->second.size(); ++j) {
      const TrajectoryPoint& point = it->second[j];
      output << "      {\"t\": " << point.t << ", \"x\": " << point.pose.x
             << ", \"y\": " << point.pose.y << ", \"yaw\": " << point.pose.yaw
             << "}";
      if (j + 1 < it->second.size()) {
        output << ",";
      }
      output << "\n";
    }
    output << "    ]";
    if (i + 1 < agent_ids.size()) {
      output << ",";
    }
    output << "\n";
  }

  output << "  }\n";
  output << "}\n";
}

}  // namespace parking_lot_planner
