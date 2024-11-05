# Define the minimum cmake version
cmake_minimum_required(VERSION 3.12)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Project metadata
set(TARGET parent_gridder)
project(ZoomParentGridder CXX)

# Set the version
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION_PATCH 0)

# ================= BUILD MODE =================

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Define possible build types
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ================= DEFINE OPTIONS =================
option(DISABLE_OPTIMIZATIONS "Disable optimizations" OFF)
option(ENABLE_DEBUG "Enable debugging symbols" OFF)
option(ENABLE_MPI "Enable MPI support" OFF)

# ================= DEPENDANCIES =================

# Find HDF5 with parallel support if available
find_package(HDF5 REQUIRED COMPONENTS C CXX HL)

# Check for OpenMP support
find_package(OpenMP REQUIRED)

# MPI
if(ENABLE_MPI)
  find_package(MPI REQUIRED)
endif()

# ================= COMPILER FLAGS =================

# Set the compiler flags
# -Wall: Enable all warnings
# -Wextra: Enable extra warnings
# -Wpedantic: Enable pedantic warnings
# -Werror: Treat warnings as errors
# -O3: Optimize for speed
if(DISABLE_OPTIMIZATIONS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
# -g: Generate debug information
if(ENABLE_DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# ================= SOURCE FILES =================

# Create a list of source files
set(SOURCE_FILES
    src/gridder.cpp
)

# ================= COMPILATION =================

# Create the executable using the source files
add_executable(${TARGET} ${SOURCE_FILES})

# ================= DEFINE PREPROCESSOR DIRECTIVES =================

# Define DEBUGGING_CHECKS if the build type is Debug
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  target_compile_definitions(${TARGET} PRIVATE DEBUGGING_CHECKS)
endif()

# Are we using MPI? 
if(ENABLE_MPI)
  target_compile_definitions(${TARGET} PRIVATE WITH_MPI)
endif()


# ================= LIBRARY INCLUSION =================

# HDF5
target_link_libraries(${TARGET} PRIVATE HDF5::HDF5)

# MPI
if(ENABLE_MPI)
  target_link_libraries(${TARGET} PRIVATE MPI::MPI_CXX)
  target_include_directories(${TARGET} PRIVATE ${MPI_CXX_INCLUDE_PATH})
endif()

# OpenMP
target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)

# Display compiler flags being used.
message(STATUS "CXXFLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CXXFLAGS for ${CMAKE_BUILD_TYPE}:")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message(STATUS "  ${CMAKE_CXX_FLAGS_DEBUG}")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  message(STATUS "  ${CMAKE_CXX_FLAGS_RELEASE}")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel")
  message(STATUS "  ${CMAKE_CXX_FLAGS_MINSIZEREL}")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
  message(STATUS "  ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

# ================= METADATA COLLECTION =================

# Execute Git commands to get revision and branch
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

# Get compiler information including version
if (CMAKE_CXX_COMPILER_VERSION AND CMAKE_CXX_COMPILER_ID)
  set(COMPILER_INFO "${CMAKE_CXX_COMPILER_ID}@${CMAKE_CXX_COMPILER_VERSION}")
elseif (NOT CMAKE_CXX_COMPILER_VERSION AND CMAKE_CXX_COMPILER_ID)
  set(COMPILER_INFO "${CMAKE_CXX_COMPILER_ID}")
else()
  set(COMPILER_INFO "Unknown Compiler")
endif()

# Get compiler flags (CFLAGS)
string(REPLACE ";" " " CFLAGS_INFO "${CMAKE_CXX_FLAGS}")

# Add the necessary version paths
configure_file(
    "${CMAKE_SOURCE_DIR}/src/config/version.h.in"
    "${CMAKE_SOURCE_DIR}/src/version.h"
)

