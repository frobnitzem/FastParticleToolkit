cmake_minimum_required(VERSION 3.11.4)

project(FPTapp LANGUAGES CXX)

set(PROJNAME %USER_PROJECT_NAME)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# Add from environment after specific env vars
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")

set(FPT_DIR ${FPTapp_SOURCE_DIR}/FastParticleToolkit)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Search in PACKAGENAME_ROOT
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

add_subdirectory(FastParticleToolkit)

################################################################################
# Find MPI
################################################################################

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
set(LIBS ${LIBS} ${MPI_C_LIBRARIES})

################################################################################
# Find PThreads
################################################################################

find_package(Threads REQUIRED)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

################################################################################
# Find OpenMP
################################################################################

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

################################################################################
# List out targets
################################################################################

alpaka_add_executable(%USER_PROJECT_NAME src/main.cpp)
target_link_libraries(%USER_PROJECT_NAME PUBLIC fpt)

install(TARGETS %USER_PROJECT_NAME DESTINATION bin)
