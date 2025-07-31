cmake_minimum_required(VERSION 3.12)
project(ZoomParentGridder CXX)

# C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================
# User Options
# ============================================
option(ENABLE_MPI "Enable MPI support" OFF)

# ============================================
# Build Types and Compiler Flags
# ============================================
# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

# Set compile flags per build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -Wall -Wextra -Wpedantic")

# ============================================
# Dependencies
# ============================================
find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)

if(ENABLE_MPI)
  find_package(MPI REQUIRED)

  if(NOT HDF5_IS_PARALLEL)
    message(FATAL_ERROR "MPI requested but HDF5 was not built with parallel support")
  endif()
endif()

# ============================================
# Source Files
# ============================================
set(SOURCE_FILES
    src/gridder.cpp
    src/cell.cpp
    src/cell_search.cpp
    src/construct_grid_points.cpp
    src/construct_cells.cpp
    src/grid_point.cpp
    src/hdf_io.cpp
    src/metadata.cpp
    src/output.cpp
    src/partition.cpp
    src/params.cpp
    src/simulation.cpp
)

# ============================================
# Executable and Definitions
# ============================================
add_executable(parent_gridder ${SOURCE_FILES})

# Debug macro
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(parent_gridder PRIVATE DEBUGGING_CHECKS)
endif()

# MPI macro
if(ENABLE_MPI)
  target_compile_definitions(parent_gridder PRIVATE WITH_MPI)
endif()

# ============================================
# Includes and Linkage
# ============================================
target_link_libraries(parent_gridder PRIVATE
  HDF5::HDF5
  OpenMP::OpenMP_CXX
)

if(ENABLE_MPI)
  target_link_libraries(parent_gridder PRIVATE MPI::MPI_CXX)
  target_include_directories(parent_gridder PRIVATE ${MPI_CXX_INCLUDE_PATH})
endif()

# ============================================
# Git Metadata Header Generation
# ============================================
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND git log -1 --format=%cd --date=short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Fallback compiler info
if(CMAKE_CXX_COMPILER_VERSION AND CMAKE_CXX_COMPILER_ID)
  set(COMPILER_INFO "${CMAKE_CXX_COMPILER_ID}@${CMAKE_CXX_COMPILER_VERSION}")
else()
  set(COMPILER_INFO "Unknown Compiler")
endif()

# Configure version header
configure_file(
    "${CMAKE_SOURCE_DIR}/src/config/version.h.in"
    "${CMAKE_SOURCE_DIR}/src/version.h"
)

# ============================================
# Output Summary
# ============================================
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "MPI Enabled: ${ENABLE_MPI}")
