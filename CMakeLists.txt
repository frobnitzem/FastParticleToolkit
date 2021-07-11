cmake_minimum_required(VERSION 3.15)

project(FPT LANGUAGES CXX VERSION 0.0.1)
set(CMAKE_MODULE_PATH 
    "${CMAKE_CURRENT_LIST_DIR}/cmake"
    ${CMAKE_MODULE_PATH}
    "${CMAKE_CURRENT_LIST_DIR}/lib/Catch2/extras")

option(BUILD_TESTS "Build the tests accompanying this library." ON)
option(BUILD_DOCS "Build the documentation accompanying this library." ON)

#--------------------------------------
# External Packages

find_package(alpaka REQUIRED)

#--------------------------------------
# Define this header-only target

add_library(fpt INTERFACE)
target_compile_features(fpt INTERFACE cxx_std_14)

target_include_directories(fpt INTERFACE include) # <fpt/Atoms.hh>, etc.
target_link_libraries(fpt INTERFACE alpaka::alpaka)

#--------------------------------------
# Options

if(BUILD_TESTS)
  add_subdirectory(lib/Catch2)
  add_subdirectory(tests)
endif()
if(BUILD_DOCS)
  add_subdirectory(docs)
endif()

# user-code:
#alpaka_add_executable(${_TARGET_NAME} helloWorld.cpp)
