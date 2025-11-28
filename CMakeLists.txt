cmake_minimum_required(VERSION 3.12)
project(ZoomParentGridder CXX)

# ========== Version Macros ==========
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION_PATCH 0)

# ========== C++ Standard ==========
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========== Options ==========
option(ENABLE_MPI "Enable MPI support" OFF)

# Add custom target to show configuration options
add_custom_target(help-config
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "======================================"
    COMMAND ${CMAKE_COMMAND} -E echo "  CMake Configuration Options"
    COMMAND ${CMAKE_COMMAND} -E echo "======================================"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "ENABLE_MPI - default: OFF"
    COMMAND ${CMAKE_COMMAND} -E echo "  Enable MPI support for parallel processing"
    COMMAND ${CMAKE_COMMAND} -E echo "  Usage: cmake -B build -DENABLE_MPI=ON"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "CMAKE_BUILD_TYPE - default: Release"
    COMMAND ${CMAKE_COMMAND} -E echo "  Build type: Release, Debug, RelWithDebInfo, MinSizeRel"
    COMMAND ${CMAKE_COMMAND} -E echo "  - Release:        -O3 -march=native optimized"
    COMMAND ${CMAKE_COMMAND} -E echo "  - Debug:          -g -O0 with DEBUGGING_CHECKS"
    COMMAND ${CMAKE_COMMAND} -E echo "  - RelWithDebInfo: -O2 -g optimized + debug symbols"
    COMMAND ${CMAKE_COMMAND} -E echo "  - MinSizeRel:     -Os optimize for size"
    COMMAND ${CMAKE_COMMAND} -E echo "  Usage: cmake -B build -DCMAKE_BUILD_TYPE=Debug"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Examples:"
    COMMAND ${CMAKE_COMMAND} -E echo "  cmake -B build -DENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release"
    COMMAND ${CMAKE_COMMAND} -E echo "  cmake -B build -DCMAKE_BUILD_TYPE=Debug"
    COMMAND ${CMAKE_COMMAND} -E echo "  cmake --build build"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Current configuration:"
    COMMAND ${CMAKE_COMMAND} -E echo "  ENABLE_MPI:       ${ENABLE_MPI}"
    COMMAND ${CMAKE_COMMAND} -E echo "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMENT "Displaying configuration options"
)

# ========== Build Type ==========
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_DEBUG          "-g -O0 -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -march=native -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -Wall -Wextra -Wpedantic")

# ========== Dependencies ==========
find_package(OpenMP REQUIRED)

# HDF5 (always serial - we use per-rank files instead of parallel HDF5)
find_package(HDF5 REQUIRED COMPONENTS C HL)

# MPI (if enabled, only for communication - not for HDF5)
if(ENABLE_MPI)
  find_package(MPI REQUIRED)
  message(STATUS "MPI C compiler: ${MPI_C_COMPILER}")
  message(STATUS "MPI CXX compiler: ${MPI_CXX_COMPILER}")
endif()

# Detect HDF5 version with multiple fallback methods
set(HDF5_VERSION "unknown")
if(HDF5_VERSION_STRING)
  set(HDF5_VERSION "${HDF5_VERSION_STRING}")
elseif(HDF5_VERSION_MAJOR AND HDF5_VERSION_MINOR)
  if(HDF5_VERSION_PATCH)
    set(HDF5_VERSION "${HDF5_VERSION_MAJOR}.${HDF5_VERSION_MINOR}.${HDF5_VERSION_PATCH}")
  else()
    set(HDF5_VERSION "${HDF5_VERSION_MAJOR}.${HDF5_VERSION_MINOR}")
  endif()
else()
  # Try to extract version from h5cc or pkg-config
  find_program(H5CC_EXECUTABLE h5cc)
  if(H5CC_EXECUTABLE)
    execute_process(
      COMMAND ${H5CC_EXECUTABLE} -showconfig
      OUTPUT_VARIABLE H5CC_OUTPUT
      ERROR_QUIET
    )
    string(REGEX MATCH "HDF5 Version: ([0-9]+\\.[0-9]+\\.[0-9]+)" HDF5_VERSION_MATCH "${H5CC_OUTPUT}")
    if(HDF5_VERSION_MATCH)
      set(HDF5_VERSION "${CMAKE_MATCH_1}")
    endif()
  endif()
  
  # Last resort: try pkg-config
  if(HDF5_VERSION STREQUAL "unknown")
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
      pkg_check_modules(HDF5_PC QUIET hdf5)
      if(HDF5_PC_VERSION)
        set(HDF5_VERSION "${HDF5_PC_VERSION}")
      endif()
    endif()
  endif()
endif()

message(STATUS "Found HDF5 version: ${HDF5_VERSION}")
message(STATUS "HDF5 usage: Serial (per-rank files + virtual combination)")

# ========== Metadata ==========
execute_process(
  COMMAND git rev-parse --short HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_REVISION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

execute_process(
  COMMAND git log -1 --format=%cd --date=short HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_DATE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

# Handle case where git commands fail (e.g., not a git repo)
if(NOT GIT_REVISION)
  set(GIT_REVISION "unknown")
endif()
if(NOT GIT_BRANCH)
  set(GIT_BRANCH "unknown")
endif()
if(NOT GIT_DATE)
  set(GIT_DATE "unknown")
endif()

if(CMAKE_CXX_COMPILER_VERSION AND CMAKE_CXX_COMPILER_ID)
  set(COMPILER_INFO "${CMAKE_CXX_COMPILER_ID}@${CMAKE_CXX_COMPILER_VERSION}")
else()
  set(COMPILER_INFO "Unknown Compiler")
endif()

string(REPLACE ";" " " CFLAGS_INFO "${CMAKE_CXX_FLAGS}")
# Note: HDF5_VERSION is now set above after HDF5 detection

# ========== Version Header ==========
configure_file(
  ${CMAKE_SOURCE_DIR}/src/config/version.h.in
  ${CMAKE_SOURCE_DIR}/src/version.h
)

# ========== Sources ==========
set(SOURCE_FILES
  src/gridder.cpp
  src/cell.cpp
  src/cell_search.cpp
  src/construct_grid_points.cpp
  src/construct_cells.cpp
  src/grid_point.cpp
  src/hdf_io.cpp
  src/logger.cpp
  src/metadata.cpp
  src/output.cpp
  src/partition.cpp
  src/params.cpp
  src/simulation.cpp
  src/debugging_utils.cpp
)

# ========== Target ==========
add_executable(parent_gridder ${SOURCE_FILES})

# Base include directories
target_include_directories(parent_gridder PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${HDF5_INCLUDE_DIRS}
)

# Base libraries
target_link_libraries(parent_gridder PRIVATE
  ${HDF5_LIBRARIES}  # Use HDF5_LIBRARIES instead of HDF5::HDF5 for broader compatibility
  OpenMP::OpenMP_CXX
)

# MPI-specific configuration
if(ENABLE_MPI)
  target_include_directories(parent_gridder PRIVATE ${MPI_CXX_INCLUDE_DIRS})
  target_link_libraries(parent_gridder PRIVATE ${MPI_CXX_LIBRARIES})
  target_compile_definitions(parent_gridder PRIVATE WITH_MPI)
  
  # Add MPI compile flags if needed
  if(MPI_CXX_COMPILE_FLAGS)
    set_target_properties(parent_gridder PROPERTIES
      COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
  endif()
  
  # Add MPI link flags if needed  
  if(MPI_CXX_LINK_FLAGS)
    set_target_properties(parent_gridder PROPERTIES
      LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
  endif()
endif()

# Debug-specific definitions
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(parent_gridder PRIVATE DEBUGGING_CHECKS)
endif()

# ========== Status Messages ==========
message(STATUS "=== Configuration Summary ===")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Version: ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${COMPILER_INFO}")
message(STATUS "MPI enabled: ${ENABLE_MPI}")
if(ENABLE_MPI)
  message(STATUS "  - MPI version: ${MPI_CXX_VERSION}")
  message(STATUS "  - MPI include dirs: ${MPI_CXX_INCLUDE_DIRS}")
  message(STATUS "  - HDF5 strategy: Per-rank files + virtual combination")
endif()
message(STATUS "HDF5 version: ${HDF5_VERSION}")
message(STATUS "OpenMP enabled: ${OpenMP_CXX_FOUND}")
message(STATUS "Git revision: ${GIT_REVISION}")
message(STATUS "Git branch: ${GIT_BRANCH}")
message(STATUS "Git date: ${GIT_DATE}")
message(STATUS "===============================")

