#include <fpt/Cell.hpp>

namespace fpt {
/**
 * Compute a 1-body operator.
 */
template <typename Oper1, typename Vec>
struct Oper1Kernel {
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                const Cell *__restrict__ X,
                typename Oper1::Output *__restrict__ const out
                ) const {
        const int idx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
        auto cell = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];

        const Cell &A = X[cell];
        uint32_t n = A.n[idx];
        float x = A.x[idx];
        float y = A.y[idx];
        float z = A.z[idx];
        Oper1::f(out[cell], idx, n, x, y, z);
    }
};

/** 1-body operator to count the total number of particles per cell
 */
struct NumCellOper {
    using Output = uint32_t;

    ALPAKA_NO_HOST_ACC_WARNING
    static inline ALPAKA_FN_ACC void f(
            Output& out, int idx,
            uint32_t n, float x, float y, float z) {
        out += (n != 0);
    }
};

/** 1-body operator to initialize cells to zero.
 */
struct ZeroCellOper {
    using Output = Cell;

    ALPAKA_NO_HOST_ACC_WARNING
    static inline ALPAKA_FN_ACC void f(
            Output& out, int idx,
            uint32_t n, float x, float y, float z) {
        out.n[idx] = 0;
        out.x[idx] = 0.0;
        out.y[idx] = 0.0;
        out.z[idx] = 0.0;
    }
};

/** 1-body operator to initialize energies to zero.
 */
struct ZeroEnOper {
    using Output = CellEnergy;

    ALPAKA_NO_HOST_ACC_WARNING
    static inline ALPAKA_FN_ACC void f(
            Output& out, int idx,
            uint32_t n, float x, float y, float z) {
        out.n[idx] = 0;
        out.en[idx] = 0.0;
    }
};

/** Create a 1-body operation.  Oper1 has f : out[cell],idx,n,x,y,z -> out[cell]
 *  cell = cell(x,y,z), the cell that the particle lies within
 *
 * Example enque calls
 *  ZeroEnK = mk1Body<ZeroEnOper,Acc,Dim,Idx>(devAcc, X, out);
 *  alpaka::enqueue(queue, ZeroEnK);
 *
 *  ZeroCellK = mk1Body<ZeroCellOper,Acc,Dim,Idx>(devAcc, X, X);
 *  alpaka::enqueue(queue, ZeroCellK);
*/
template <typename Oper1, typename Acc, typename Dim, typename Idx, typename Dev>
auto mk1Body(const Dev &devAcc, const alpaka::Buf<Dev, Cell, Dim, Idx> &X,
             alpaka::Buf<Dev, typename Oper1::Output, Dim, Idx> &out) {
    using Vec = alpaka::Vec<Dim,Idx>;

    // Launch with one warp per thread block
    Idx const warpExtent  = alpaka::getWarpSize(devAcc);

    // Spaces must match.
    Idx const ncells = alpaka::extent::getExtent<0>(X);
    assert( ncells == alpaka::extent::getExtent<0>(out) );

    Vec const gridBlockExtent = Vec::all(ncells);
    // min of 2
    Vec blockThreadExtent = Vec::all(
            warpExtent < ATOMS_PER_CELL ?
            warpExtent : ATOMS_PER_CELL);

    alpaka::WorkDivMembers<Dim, Idx> workDiv{
                gridBlockExtent,
                blockThreadExtent,
                Vec::all(1)};

    std::cout << "Creating 1-body kernel for " << ncells << " cells.\n";
    Oper1Kernel<Oper1,Vec> K{};
    return alpaka::createTaskKernel<Acc>(workDiv, K,
                alpaka::getPtrNative(X), alpaka::getPtrNative(out));
}

}

// e.g. call_1body<ZeroEnOper>();

/* Initialize particle positions from a uniform distribution.
struct Gas1Oper {
    using Output = Cell;
    using State = decltype(alpaka::rand::generator::createDefault(acc,0,0);
    // There is no way to write this line of code without going deep into the alpaka library source to find the eventual type of the random number generator state used.  This search yielded a device-specialized struct in the "rand::generator::traits" namespace. It could certainly be created by a device function and stored, but there's no easy way to retrieve its type from this (outer) level.  Perhaps decltype magic would succeed.  Then again, the mapping from threads to state initialization is still unclear.  This situation appears to be a symptom of the alpaka::rand class not being used much by the developers.  Storing the type seems to be required by https://github.com/alpaka-group/alpaka/issues/1210

    ALPAKA_NO_HOST_ACC_WARNING
    static inline ALPAKA_FN_ACC void operator()(
        Output& out,
        int j, uint32_t n, float x, float y, float z) {
        out.n[j] = 1;
        out.x[j] = 1.0;
        out.y[j] = 1.0;
        out.z[j] = 1.0;
    }
    ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                T* const out,
                uint32_t const extent,
                ) const -> void {
        auto dist = rand::distribution::createUniformReal<T>(acc);
        auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto stride = alpaka::getWorkdiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        // Must be declared in device memory
        // but set as a block in the curand docs...
        auto gen = alpaka::rand::generator::createDefault(acc, 6789u, idx);

        genNumbers(acc, success, genDefault);
        for(auto i = idx; i < extent; i += stride) {
            out[i] = dist(gen);
        }
    }
};
*/
