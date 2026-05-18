cmake_minimum_required(VERSION 3.5)

project(parking_lot_planner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
include(FetchContent)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0
)

FetchContent_MakeAvailable(yaml-cpp)

add_library(
  parking_lot_core
  src/planner.cpp
  src/geometry.cpp
  src/sipp.cpp
  src/io.cpp
)

target_include_directories(
  parking_lot_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(parking_lot_core PUBLIC yaml-cpp)

target_compile_features(parking_lot_core PUBLIC cxx_std_17)

add_executable(parking_lot_planner src/main.cpp)
target_link_libraries(parking_lot_planner PRIVATE parking_lot_core)

if(MSVC)
  target_compile_options(parking_lot_core PRIVATE /W4 /permissive-)
  target_compile_options(parking_lot_planner PRIVATE /W4 /permissive-)
else()
  target_compile_options(parking_lot_core PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(parking_lot_planner PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BUILD_TESTING)
  add_executable(planner_tests tests/planner_tests.cpp)
  target_link_libraries(planner_tests PRIVATE parking_lot_core)

  if(MSVC)
    target_compile_options(planner_tests PRIVATE /W4 /permissive-)
  else()
    target_compile_options(planner_tests PRIVATE -Wall -Wextra -Wpedantic)
  endif()

  add_test(NAME planner_tests COMMAND planner_tests)

  set(SMOKE_INSTANCE ${CMAKE_CURRENT_BINARY_DIR}/parking_benchmark.yaml)
  set(SMOKE_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/smoke_output.json)
  add_test(
    NAME smoke_generate
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_parking_scenario.py
      --layout ${CMAKE_CURRENT_SOURCE_DIR}/examples/parking_layout.yaml
      --config ${CMAKE_CURRENT_SOURCE_DIR}/examples/parking_benchmark_config.yaml
      --output ${SMOKE_INSTANCE}
  )
  add_test(
    NAME smoke_run
    COMMAND parking_lot_planner
      ${SMOKE_INSTANCE}
      ${SMOKE_OUTPUT}
  )
  set_tests_properties(smoke_run PROPERTIES DEPENDS smoke_generate)
  add_test(
    NAME smoke_validate
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/validate_smoke_output.py
      --instance ${SMOKE_INSTANCE}
      --solution ${SMOKE_OUTPUT}
  )
  set_tests_properties(smoke_validate PROPERTIES DEPENDS smoke_run)
endif()
