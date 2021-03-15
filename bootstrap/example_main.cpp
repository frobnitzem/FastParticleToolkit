#include <algorithm>
#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <cassert>
#include <iostream>
#include <mallocMC/mallocMC.hpp>
#include <numeric>
#include <vector>

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

// Define the device accelerator
using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;

struct ScatterConfig
{
    static constexpr auto pagesize = 4096;
    static constexpr auto accessblocks = 8;
    static constexpr auto regionsize = 16;
    static constexpr auto wastefactor = 2;
    static constexpr auto resetfreedpages = false;
};

struct ScatterHashParams
{
    static constexpr auto hashingK = 38183;
    static constexpr auto hashingDistMP = 17497;
    static constexpr auto hashingDistWP = 1;
    static constexpr auto hashingDistWPRel = 1;
};

struct AlignmentConfig
{
    static constexpr auto dataAlignment = 32;
};

using ScatterAllocator = mallocMC::Allocator<
    Acc,
    mallocMC::CreationPolicies::Scatter<ScatterConfig, ScatterHashParams>,
    mallocMC::DistributionPolicies::Noop,
    mallocMC::OOMPolicies::ReturnNull,
    mallocMC::ReservePoolPolicies::AlpakaBuf<Acc>,
    mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>>;

ALPAKA_STATIC_ACC_MEM_GLOBAL int* arA = nullptr;

struct SortKernel
{
    int n;
    SortKernel(int n_) : n(n_) {}

    ALPAKA_FN_ACC void operator()(const Acc& acc, ScatterAllocator::AllocatorHandle allocHandle) const
    {
        const auto id = static_cast<uint32_t>(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0]);
        if(id < n)
            arA = (int*) allocHandle.malloc(acc, sizeof(int) * 32);
        // wait the the malloc from thread zero is not changing the result for some threads
        alpaka::syncBlockThreads(acc);
        const auto slots = allocHandle.getAvailableSlots(acc, 1);
        if(arA != nullptr)
        {
            arA[id] = id;
            printf("id: %u array: %d slots %u\n", id, arA[id], slots);
        }
        else
            printf("error: device size allocation failed");

        // wait that all thread read from `arA`
        alpaka::syncBlockThreads(acc);
        if(id == 0)
            allocHandle.free(acc, arA);
    }
};

int main(int argc, char *argv[]) {
    const auto dev = alpaka::getDevByIdx<Acc>(0);
    auto queue = alpaka::Queue<Acc, alpaka::Blocking>{dev};
    auto const devProps = alpaka::getAccDevProps<Acc>(dev);
    unsigned const block = std::min(static_cast<size_t>(32u), static_cast<size_t>(devProps.m_blockThreadCountMax));

    std::cout << "block = " << block << std::endl;
    ScatterAllocator scatterAlloc(dev, queue, 1U * 1024U * 1024U * 1024U); // 1GB for device-side malloc

    const auto workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{block}, Idx{1}};

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc>(workDiv, SortKernel{1}, scatterAlloc.getAllocatorHandle()));
    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc>(workDiv, SortKernel{4}, scatterAlloc.getAllocatorHandle()));
    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc>(workDiv, SortKernel{16}, scatterAlloc.getAllocatorHandle()));

    std::cout << "Slots from Host: " << scatterAlloc.getAvailableSlots(dev, queue, 1) << '\n';

    return 0;
}
