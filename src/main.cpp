#include <exception>
#include <iostream>
#include <string>

#include "planner.hpp"
#include "sipp.hpp"

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: " << argv[0]
              << " <instance.yaml> <output.json> [sipp|hybrid]\n";
    return 1;
  }

  try {
    const std::string instance_path = argv[1];
    const std::string output_path = argv[2];
    const std::string low_level = argc == 4 ? argv[3] : "sipp";

    const parking_lot_planner::Instance instance =
        parking_lot_planner::loadInstance(instance_path);
    const parking_lot_planner::SippPlanner sipp_planner;
    const parking_lot_planner::PrioritizedPlanner solver =
        low_level == "hybrid"
            ? parking_lot_planner::PrioritizedPlanner()
            : parking_lot_planner::PrioritizedPlanner(sipp_planner);
    if (low_level != "sipp" && low_level != "hybrid") {
      std::cerr << "error: low-level must be 'sipp' or 'hybrid'\n";
      return 1;
    }
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
