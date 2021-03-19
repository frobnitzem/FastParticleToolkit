#pragma once

#include <fpt/Cell.hpp>

namespace fpt {

template <typename Vec>
class sortAtomsKernel {
public:
    const CellSorter_d srt;
    sortAtomsKernel(const CellSorter &srt_) : srt(srt_.device()) {}

    //-----------------------------------------------------------------------------
    //! The kernel entry point.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \param acc The accelerator to be executed on.
    //! \param X Input particle locations.
    //! \param Y Output particle locations.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            Cell *X, Cell *Y
            //const Cell *__restrict__ X,
            //Cell *__restrict__ Y
            ) const {
        using Idx = typename Vec::Val;
        using Dim = typename Vec::Dim;
        Idx const bin(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]); // blockIdx.x
        Idx const idx(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]); // threadIdx.x

        auto& far = alpaka::declareSharedVar<CellTranspose, __COUNTER__>(acc);
        load_cell(acc,X,bin,far);
        alpaka::syncBlockThreads(acc);

        /*uint32_t n = X[bin].n[idx];
        float x = X[bin].x[idx];
        float y = X[bin].y[idx];
        float z = X[bin].z[idx];
        const unsigned int to_bin = srt.calcBinF(x, y, z);

        if(n != 0) {
            const unsigned lane = srt.addToBin(Y, to_bin); // successful lane
            Y[to_bin].x[lane] = x;
            Y[to_bin].y[lane] = y;
            Y[to_bin].z[lane] = z;
        }*/
    }
};

/* Return a sorting kernel */
template<typename Acc, typename Dim, typename Idx, typename Dev>
auto mkSorter(const Dev &devAcc,
              const CellSorter &srt,
              alpaka::Buf<Dev, Cell, Dim, Idx> &X,
              alpaka::Buf<Dev, Cell, Dim, Idx> &Y) {
    using Vec = alpaka::Vec<Dim,Idx>;

    // Launch with one warp per thread block
    Idx const warpExtent  = alpaka::getWarpSize(devAcc);

    Vec const gridBlockExtent = Vec::all(srt.cells);
    // min of 2
    Vec blockThreadExtent = Vec::all(
            warpExtent < ATOMS_PER_CELL ?
            warpExtent : ATOMS_PER_CELL);

    alpaka::WorkDivMembers<Dim, Idx> workDiv{
                gridBlockExtent,
                blockThreadExtent,
                Vec::all(1)};

    // Create the kernel execution task.
    fpt::sortAtomsKernel<Vec> K{srt};
    return alpaka::createTaskKernel<Acc>(workDiv, K,
                alpaka::getPtrNative(X), alpaka::getPtrNative(Y));
}

}
