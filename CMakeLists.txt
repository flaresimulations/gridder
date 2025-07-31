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
find_package(HDF5 REQUIRED COMPONENTS C HL)

if(ENABLE_MPI)
  find_package(MPI REQUIRED)
  if(NOT HDF5_IS_PARALLEL)
    message(FATAL_ERROR "MPI enabled but HDF5 is not parallel")
  endif()
endif()

# ========== Metadata ==========
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

if(CMAKE_CXX_COMPILER_VERSION AND CMAKE_CXX_COMPILER_ID)
  set(COMPILER_INFO "${CMAKE_CXX_COMPILER_ID}@${CMAKE_CXX_COMPILER_VERSION}")
else()
  set(COMPILER_INFO "Unknown Compiler")
endif()

string(REPLACE ";" " " CFLAGS_INFO "${CMAKE_CXX_FLAGS}")
set(HDF5_VERSION "${HDF5_VERSION_STRING}")

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
  src/metadata.cpp
  src/output.cpp
  src/partition.cpp
  src/params.cpp
  src/simulation.cpp
)

# ========== Target ==========
add_executable(parent_gridder ${SOURCE_FILES})

target_include_directories(parent_gridder PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${HDF5_INCLUDE_DIRS}
)

if(ENABLE_MPI)
  target_include_directories(parent_gridder PRIVATE ${MPI_CXX_INCLUDE_PATH})
endif()

target_link_libraries(parent_gridder PRIVATE
  HDF5::HDF5
  OpenMP::OpenMP_CXX
)

if(ENABLE_MPI)
  target_link_libraries(parent_gridder PRIVATE MPI::MPI_CXX)
  target_compile_definitions(parent_gridder PRIVATE WITH_MPI)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(parent_gridder PRIVATE DEBUGGING_CHECKS)
endif()

# ========== Status ==========
message(STATUS "MPI enabled: ${ENABLE_MPI}")
message(STATUS "Compiler: ${COMPILER_INFO}")
