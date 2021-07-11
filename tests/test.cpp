#include <catch2/internal/catch_main.cpp>
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

//#############################################################################
template <typename Dev>
class SingleThreadEmptyTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success,
        fpt::Alloc_d<int,Dev> alloc) const
    -> void
    {
        std::int32_t const warpExtent = alpaka::warp::getSize(acc);
        ALPAKA_CHECK(*success, warpExtent == 1);

        ALPAKA_CHECK(*success, alpaka::warp::all(acc, 42) != 0);
        ALPAKA_CHECK(*success, alpaka::warp::all(acc, 0) == 0);
    }
};

//#############################################################################
template <typename Dev>
class MultipleThreadEmptyTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success,
        fpt::Alloc_d<int,Dev> alloc) const
    -> void
    {
        std::int32_t const warpExtent = alpaka::warp::getSize(acc);
        ALPAKA_CHECK(*success, warpExtent > 1);

        ALPAKA_CHECK(*success, alpaka::warp::all(acc, 0) == 0);
        ALPAKA_CHECK(*success, alpaka::warp::all(acc, 42) != 0);

        // Test relies on having a single warp per thread block
        auto const blockExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        ALPAKA_CHECK(*success, static_cast<std::int32_t>(blockExtent.prod()) == warpExtent);
        auto const localThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const threadIdxInWarp = static_cast<std::int32_t>(alpaka::mapIdx<1u>(
            localThreadIdx,
            blockExtent)[0]);

        // Some threads quit the kernel to test that the warp operations
        // properly operate on the active threads only
        if (threadIdxInWarp % 3)
            return;

        for (auto idx = 0; idx < warpExtent; idx++)
        {
            ALPAKA_CHECK(
                *success,
                alpaka::warp::all(acc, threadIdxInWarp == idx ? 1 : 0) == 0);
            std::int32_t const expected = idx % 3 ? 1 : 0;
            ALPAKA_CHECK(
                *success,
                alpaka::warp::all(acc, threadIdxInWarp == idx ? 0 : 1) == expected);
        }
    }
};

template <typename Acc, typename Dim, typename Idx>
void warp_cfg_test() {
    return;
}

template <typename Acc>
void warp_cfg_test<Acc, alpaka::DimInt<1u>, uint32_t>() {
    using Dev = alpaka::Dev<Acc>;
    using Pltf = alpaka::Pltf<Dev>;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    using Queue = alpaka::Queue<Acc, alpaka::Blocking>;

    Dev const dev(alpaka::getDevByIdx<Pltf>(0u));
    auto A = fpt::Alloc<int,Acc>(dev, 1029);
    auto Q = Queue(dev);
    A.reinit(Q);
    alpaka::wait(Q);

    auto const warpExtent = alpaka::getWarpSize(dev);
    //std::cout << "Testing with warpExtent = " << warpExtent << "\n";
    if (warpExtent == 1) {
        Idx const gridThreadExtentPerDim = 4;
        alpaka::test::KernelExecutionFixture<Acc,Queue> fixture(
            alpaka::Vec<Dim, Idx>::all(gridThreadExtentPerDim), Q);
        SingleThreadEmptyTestKernel<Dev> kernel;
        REQUIRE( fixture( kernel, A.device() ) );
    }
    else
    {
        // Work around gcc 7.5 trying and failing to offload for OpenMP 4.0
#if BOOST_COMP_GNUC && (BOOST_COMP_GNUC == BOOST_VERSION_NUMBER(7, 5, 0)) && defined ALPAKA_ACC_ANY_BT_OMP5_ENABLED
        return;
#else
        using ExecutionFixture = alpaka::test::KernelExecutionFixture<Acc,Queue>;
        auto const gridBlockExtent = alpaka::Vec<Dim, Idx>::all(2);
        // Enforce one warp per thread block
        auto blockThreadExtent = alpaka::Vec<Dim, Idx>::ones();
        blockThreadExtent[0] = static_cast<Idx>(warpExtent);
        auto const threadElementExtent = alpaka::Vec<Dim, Idx>::ones();
        auto workDiv = typename ExecutionFixture::WorkDiv{
            gridBlockExtent,
            blockThreadExtent,
            threadElementExtent};
        auto fixture = ExecutionFixture{ workDiv };
        MultipleThreadEmptyTestKernel<Dev> kernel;
        REQUIRE( fixture( kernel, A.device() ) );
#endif
    }
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "fpt::Alloc returns empty block", "[warp]", alpaka::test::TestAccs) {
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    warp_cfg_test<Acc,Dim,Idx>();
}
