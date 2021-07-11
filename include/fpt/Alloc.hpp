#pragma once

#include <alpaka/alpaka.hpp>

namespace fpt {

    // from code.google.com/p/smhasher/wiki/MurmurHash3
    ALPAKA_FN_HOST_ACC inline static uint32_t searchNext(uint32_t a, uint32_t b, uint32_t N) {
        uint32_t h = a | (b << 16);
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        return h % N;
    }

    /**  Device-resident copy of Alloc.
     *   This is created by calling an Alloc's device() method.
     */
    template <typename A, typename Dev>
    class Alloc_d {
    private:
        using Dim = alpaka::DimInt<1u>;
        using BufDev = alpaka::Buf<Dev, A, Dim, uint32_t>;
        using FreeDev = alpaka::Buf<Dev, uint32_t, Dim, uint32_t>;

        const uint32_t N, M; // N = #arr, M = #blocks
        const uint32_t warp; // number of threads in a warp
        uint32_t *frl; // free list, size = M*warp
        A *arr; // allocatable array blocks

    public:
        Alloc_d(const uint32_t N_, const uint32_t M_, const uint32_t warp_,
                    uint32_t *frl_, A *arr_)
                        : N(N_), M(M_), warp(warp_), frl(frl_), arr(arr_) {}

        // Device-accessible function to allocate.
        // Searches for blocks starting with searchNext(`start`,blk,N)
        //
        // The entire point of this routine is to find any bit set to 1
        // in the frl.  Setting it to 0 will indicate it's claimed.
        template <typename Acc>
        ALPAKA_FN_ACC uint32_t alloc(Acc const& acc, uint32_t start) {
            const auto idx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
            const auto blk = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];

            uint32_t n = start;
            
            for(int i=0; i<10; i++) {
                n = searchNext(n, blk, N);

                uint32_t m0 = n/32;
                uint32_t k0 = n%32;
                uint32_t base = m0 - (m0%warp); // warp-align the next read
                uint32_t F = frl[base+idx]; // read a warp-sized chunk of frl

                // nonzero for ea. thread finding an empty cell (nonzero bit)
                // (i.e. if there are any 1-s in the thread's search space)
                auto mask = alpaka::warp::ballot(acc, F != 0);
                if(mask == 0) {
                    continue;
                }
                // Do a linear search from the starting (m0,k0)
                for(int j=0; j<warp; j++) {
                    int thr = (m0+j)%warp; // search this thread's space
                    uint32_t m = base + thr;
                    int b = 32; // indicates no bit found

                    if((mask & (1<<thr)) == 0) continue;

                    if(idx == thr) { // this thread does up to 32 atomic operations
                        for(b=0; b<32; b++) {
                            int k = (k0+b)%32;
                            uint32_t nF = (1<<k);
                            uint32_t ans = alpaka::atomicOp<alpaka::AtomicAnd>(acc, &frl[m], ~nF);
                            if(ans & nF) { // still 1 => success
                                break;
                            }
                        }
                    }
                    b = alpaka::warp::shfl(acc, b, thr);
                    if(b != 32)
                        return m*32 + (k0+b)%32; // 32*m+k
                }
            }
        }

        // Device-accessible function to de-allocate.
        template <typename Acc>
        ALPAKA_FN_ACC void free(Acc const& acc, uint32_t n) {
            const auto idx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
            uint32_t m = n/32;
            uint32_t k = n%32;

            if(idx != m%32) return; // only need 1 thread to run this

            uint32_t nF = (1<<k);
            alpaka::atomicOp<alpaka::AtomicOr>(acc, &frl[m], nF);
            // Note: return value should have had a 0 at position k,
            // or else we just double freed.
        }

        // Device-accessible function to lookup an element.
        ALPAKA_FN_HOST_ACC inline A &operator[](uint32_t n) {
            return arr[n];
        }

    };

    //#############################################################################
    //! Kernel clearing the allocation space on the device
    struct ClearAllocKernel {
        using Dim = alpaka::DimInt<1u>;
        using Vec = alpaka::Vec<Dim,uint32_t>;

        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                const uint32_t N,
                uint32_t *frl
                ) const {
            const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
            uint32_t ans = 0;
            // N/32, N%32 is the bit-number at which we start placing 0-s
            if(idx <= N/32) {
                if(idx == N/32) {
                    ans = (1<<(N%32)) - 1;
                } else {
                    ans = 0xFFFFFFFF;
                }
            }
            frl[idx] = ans;
        }
    };

    /**  Allocate up to N members of type A on Dev.
     */
    template <typename A, typename Acc>
    class Alloc {
    private:
        using Dev = alpaka::Dev<Acc>;
        using Dim = alpaka::DimInt<1u>;
        using Idx = uint32_t; // not modifiable
        using Vec = alpaka::Vec<Dim,uint32_t>;
        using Device = Dev;
        using FreeDev = alpaka::Buf<Dev, uint32_t, Dim, Idx>;
        using BufDev = alpaka::Buf<Dev, A, Dim, Idx>;

        const uint32_t N; // N = #arr
        const uint32_t warp; // number of threads in a warp
        const uint32_t M; // M = #blocks
        const Dev &devAcc;
        FreeDev frl; // free list, size = M*warp
        BufDev arr; // allocatable array blocks

    public:
        Alloc(const Dev &devAcc_, const uint32_t N_)
            : N(N_)
            , warp( alpaka::getWarpSize(devAcc_) )
            , M((N+32*warp-1)/(32*warp))
            , devAcc(devAcc_)
            , frl( FreeDev{alpaka::allocBuf<uint32_t, Idx>(
                                    devAcc_, M * warp)} )
            , arr( BufDev{alpaka::allocBuf<A, Idx>(devAcc_, N)} ) {
        // FIXME: kernel call to inialize frl (all 0, except last few
        // cells when 32*warp doesn't divide N = 1)
        }

        // Create device-resident allocator class.
        Alloc_d<A,Dev> device() {
            return Alloc_d<A,Dev>(N, M, warp,
                                alpaka::getPtrNative(frl), 
                                alpaka::getPtrNative(arr) );
        }

        template <typename Queue>
        void reinit(Queue &Q) {
            auto K = initKernel();
            alpaka::enqueue(Q, K);
        }

        // Kernel launch to (re)initialize free-list.
        auto initKernel() {
            // Launch with one warp per thread block:
            auto const gridBlockExtent = Vec::all(M);
            auto blockThreadExtent = Vec::all( alpaka::getWarpSize(devAcc) );

            alpaka::WorkDivMembers<Dim, uint32_t>
                    workDiv{gridBlockExtent, blockThreadExtent, Vec::all(1)};

            // Create the kernel execution task.
            ClearAllocKernel K{};
            return alpaka::createTaskKernel<Acc>(workDiv, K, N, alpaka::getPtrNative(frl));
        }
    };

}
