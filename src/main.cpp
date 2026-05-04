#include <exception>
#include <iostream>
#include <string>

#include "planner.hpp"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <instance.yaml> <output.json>\n";
    return 1;
  }

  try {
    const std::string instance_path = argv[1];
    const std::string output_path = argv[2];

    const parking_lot_planner::Instance instance =
        parking_lot_planner::loadInstance(instance_path);
    const parking_lot_planner::PrioritizedPlanner solver;
    const parking_lot_planner::Solution solution = solver.solve(instance);
    parking_lot_planner::writeSolution(solution, output_path);

    std::cout << "success=" << (solution.success ? "true" : "false")
              << " agents=" << solution.metrics.num_agents
              << " runtime_sec=" << solution.runtime_sec << "\n";

    if (!solution.failure_reason.empty()) {
      std::cout << "failure_reason=" << solution.failure_reason << "\n";
    }

    return solution.success ? 0 : 2;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
