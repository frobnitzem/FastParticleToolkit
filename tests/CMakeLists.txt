# Tests
include(CTest)
include(Catch)

alpaka_add_executable(test test.cpp testAlloc.cpp)
target_link_libraries(test PRIVATE fpt Catch2::Catch2)

catch_discover_tests(test)

