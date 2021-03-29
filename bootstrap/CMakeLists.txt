cmake_minimum_required(VERSION 3.15)

set(_TARGET_NAME %USER_PROJECT_NAME)

project(${_TARGET_NAME} LANGUAGES CXX)

# Add from environment after specific env vars
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")
if(POLICY CMP0074) # Search in PACKAGENAME_ROOT
  cmake_policy(SET CMP0074 NEW)
endif()

find_package(alpaka REQUIRED)
add_subdirectory(FastParticleToolkit)

################################################################################
# Find Other Dependencies
################################################################################

find_package(MPI REQUIRED)
find_package(OpenMP)

################################################################################
# List out targets
################################################################################

alpaka_add_executable(${_TARGET_NAME} src/main.cpp)
target_link_libraries(${_TARGET_NAME} PUBLIC fpt OpenMP::OpenMP_CXX MPI::MPI_CXX)

install(TARGETS ${_TARGET_NAME} DESTINATION bin)
