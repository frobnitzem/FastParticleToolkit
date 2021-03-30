#include <catch2/internal/catch_main.cpp>
#include <catch2/catch_all.hpp>
#include <fpt/Alloc.hpp>

TEST_CASE( "allocator indices", "[allocator]") {
    uint32_t N = 540;

    SECTION( "fpt::searchNext returns in-range" ) {
        std::vector<uint32_t> trials{0,3,10,100,N-32,N-1};

        for(uint32_t blockId : trials) {
            uint32_t next = blockId % N;
            REQUIRE(next < N);
            for(uint32_t j=0; j<10; j++) {
                next = fpt::searchNext(next, blockId, N);
                REQUIRE(next < N);
            }
        }
    }
}
