#include <catch2/catch_all.hpp>

#include <fpt/Alloc.hpp>
#include "TestAlpaka.hpp"

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

template <typename Dev>
class EmptyTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success,
        uint32_t N0,
        fpt::Alloc_d<int,Dev> alloc) const
    -> void
    {
        const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        //std::int32_t const warpExtent = alpaka::warp::getSize(acc);

        ALPAKA_CHECK(*success, false);
        for(int i=0; i<32; i++) {
            uint32_t k = idx*32 + i;
            ALPAKA_CHECK(*success, alloc.is_free(idx*32 + i) == (k >= N0) && (k < alloc.N));
        }
    }
};

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "fpt::Alloc returns empty block", "[warp]", alpaka::test::TestAccs) {
    using Acc = TestType;
    using Dev = alpaka::Dev<Acc>;
    using Pltf = alpaka::Pltf<Dev>;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    using Vec = alpaka::Vec<Dim, Idx>;

    using Queue = alpaka::Queue<Acc, alpaka::Blocking>;

    uint32_t N0 = 100;
    uint32_t N = 1029;

    Dev const dev = alpaka::getDevByIdx<Pltf>(0u);
    auto A = fpt::Alloc<int,Acc>(dev, N);
    auto Q = Queue(dev);
    A.reinit(N0, Q);
    alpaka::wait(Q);

    auto const warpExtent = alpaka::getWarpSize(dev);

    using ExecutionFixture = alpaka::test::KernelExecutionFixture<Acc,Queue>;
    // Enforce one warp per thread block
    auto workDiv = typename ExecutionFixture::WorkDiv{
        Vec::all(A.M),
        Vec::all(warpExtent),
        Vec::ones()};
    auto fixture = ExecutionFixture{ workDiv, Q };

    SECTION( "fpt::Alloc.reinit initializes correctly" ) {
        EmptyTestKernel<Dev> kernel;
        REQUIRE( fixture( kernel, N0, A.device() ) );
    }
    SECTION( "fpt::Alloc.alloc test" ) {
        //EmptyTestKernel<Dev> kernel;
        //REQUIRE( fixture( kernel, N0, A.device() ) );
    }
    SECTION( "fpt::Alloc.free test" ) {
        //EmptyTestKernel<Dev> kernel;
        //REQUIRE( fixture( kernel, N0, A.device() ) );
    }
}

// TODO: - test alloc.alloc(acc, start)
//       - create new allocated data-layout and test read-speed
