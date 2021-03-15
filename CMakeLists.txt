
find_package(alpaka REQUIRED)

add_library(fpt INTERFACE)
target_compile_features(fpt INTERFACE cxx_std_14)

target_include_directories(fpt INTERFACE include) # <fpt/Atoms.hh>, etc.
target_link_libraries(fpt INTERFACE alpaka::alpaka)

# user-code:
#alpaka_add_executable(${_TARGET_NAME} helloWorld.cpp)
