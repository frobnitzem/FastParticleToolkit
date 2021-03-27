#pragma once

#include <fpt/Cell.hpp>

#define SQR(x) ((x)*(x))

/*  E = eps ( s/r^12 - 2 s/r^6 ) = 4 eps (s1/r^12 - s1/r^6)
 *  s = 2^(1/6) s1
 *
 *  for s = eps = 1
 *  E = r2^(-6) - 2 r2^(-3)
 *
 *  dE / d(r2) = -6 r2^(-7) + 6 r2^(-4)
 *    given r2 = (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2
 *
 *  d(r2) / dx1 = 2 (x1-x2)
 *
 */

ALPAKA_FN_HOST_ACC inline float lj_en(const float r2) {
    float ir2 = 1.0/r2;
    float ir6 = ir2*ir2*ir2;
    float ir12 = ir6*ir6;
    return fmaf(-2.0, ir6, ir12);
}

ALPAKA_FN_HOST_ACC inline float lj_deriv(const float &dx, const float &dy, float &dz) {
    float r2 = dx*dx;
    r2 = fmaf(dy, dy, r2);
    r2 = fmaf(dz, dz, r2);

    float ir2 = 1.0/r2;
    float ir4 = ir2*ir2;
    float ir8 = ir4*ir4;

    float ir6 = ir2*ir4;
    float ir14 = ir6*ir8;
    //float en = fmaf(-2.0, ir6, ir12);

    float two_dEdr2 = 12.0f*(ir8 - ir14);
    return two_dEdr2;
    /*dx *= two_dEdr2;
    dy *= two_dEdr2;
    dz *= two_dEdr2;*/
}

/** Pair computation leaving the LJ energy on every particle.
 */
struct LJEnOper {
    using Output = fpt::CellEnergy;
    using Accum = double[1];

    static inline ALPAKA_FN_ACC void pair(Accum en, float dx, float dy, float dz) {
        float r2 = SQR(dx) + SQR(dy) + SQR(dz);
        en[0] += lj_en(r2); //erfcf(sqrtf(r2));
    }
    static inline ALPAKA_FN_ACC void finalize(Output &E, Accum en, uint32_t n, int j) {
        if(n == 0)
            en[0] = 0.0;
        E.n[j] = n;
        E.en[j] = en[0]*0.5; // half due to double-iterating over all-pairs
    }
};

/** Pair computation leaving the derivative of the LJ energy on every particle.
 */
struct LJDerivOper {
    using Output = fpt::Cell;
    using Accum = float[3];

    static inline ALPAKA_FN_ACC void pair(Accum de, float dx, float dy, float dz) {
        float scale = lj_deriv(dx, dy, dz);
        de[0] = fmaf(scale, dx, de[0]);
        de[1] = fmaf(scale, dy, de[1]);
        de[2] = fmaf(scale, dz, de[2]);
    }
    static inline ALPAKA_FN_ACC void finalize(Output &dE, Accum de, uint32_t n, int j) {
        if(n == 0) {
            dE.x[j] = 0.0;
            dE.y[j] = 0.0;
            dE.z[j] = 0.0;
        }
        dE.n[j] = n;
        dE.x[j] = de[0];
        dE.y[j] = de[1];
        dE.z[j] = de[2];
    }
};

namespace fpt {

/**
  Compute a pairwise function by summing over all atoms in a far cell.
  Work is distributed such that every thread is associated with
  an atom in the `near' cell.  We therefore loop over all `far' cells,
  and all atoms in those far cells.

  Loading of data from the `next' far cell is overlapped with computations
  on the `current' far cell.
 
  Data Layout Schematic
  (WARNING: outdated, currently using 1 thread per 'near' atom slot):
      c0 = blockIdx.x*ATOMS_PER_CELL
      r0 = blockIdx.y*ATOMS_PER_CELL

              --x--> (a)
     .              c0, c0+1, ..., c0+ATOMS_PER_CELL
     y   r0          0    1    ...   31
     |   r0+1        0    1    ...   31
     v   ...
    (b)  r0+ATOMS_PER_CELL 0    1    ...   31

 */
// pairFunc
template <typename Oper2, typename Vec>
struct Oper2Kernel {
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                const CellSorter_d box,
                const CellRange *__restrict__ nbr,
                const Cell *__restrict__ X,
                typename Oper2::Output *__restrict__ const out
                ) const {
        auto const j = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
        auto const bin = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];

        // far cell read repeatedly
        auto& far = alpaka::declareSharedVar<CellTranspose, __COUNTER__>(acc);
        // local copy for overlapping:
        uint32_t an[ATOMS_PER_CELL];
        float ax[ATOMS_PER_CELL], ay[ATOMS_PER_CELL], az[ATOMS_PER_CELL];

        // atom belonging to this thread
        uint32_t bn;
        float bx, by, bz;
        int bi, bj, bk;
        box.decodeBin(bin, bi, bj, bk);
        // prevent modulo wrapping issues
        bi += box.n[0]; bj += box.n[1]; bk += box.n[2];

        const Cell &B = X[bin];
        bn = B.n[j];
        bx = B.x[j];
        by = B.y[j];
        bz = B.z[j];

        typename Oper2::Accum ans{};

        CellRange off = nbr[0];

        unsigned int start = box.calcBin(0, (bj+off.j)%box.n[1], (bk+off.k)%box.n[2]);
        int self2 = load_cell(acc, X, start + (bi + off.i0)%box.n[0], far);

        for(int k=0; off.i0 <= off.i1; k++) { // offsets define a valid range
            for(int i = off.i0; i <= off.i1; i++) {
                alpaka::syncBlockThreads(acc);

                // Copy last far cell
                const int self = self2;
                for(int m=0; m<ATOMS_PER_CELL; m++) {
                    an[m] = far.n[m];
                    ax[m] = far.x[m];
                    ay[m] = far.y[m];
                    az[m] = far.z[m];
                }
                // Load next far cell (A) as a group
                if(i < off.i1) {
                    self2 = load_cell(acc, X, start + (bi + i+1)%box.n[0], far);
                } else {
                    off = nbr[k+1]; // the old 'off' isn't useful anymore
                    if(off.i0 <= off.i1) {
                        start = box.calcBin(0, (bj+off.j)%box.n[1], (bk+off.k)%box.n[2]);
                        self2 = load_cell(acc, X, start + (bi + off.i0)%box.n[0], far);
                    }
                }

                /*if(bn != 0) {
                    const int i0 = self*(j+1);
                    for (int i = i0; i < ATOMS_PER_CELL; i++) {
                        if(an[i] == 0) continue;
                    //for (int i = 0; i < ATOMS_PER_CELL; i++) { // this code is technically correct,
                    //    if(i < i0 || an[i] == 0) continue;     // but some compiler transform evaluates ans += nan ...
                        float r2 = SQR(ax[i]-bx) + SQR(ay[i]-by) + SQR(az[i]-bz);
                        ans += lj_en(r2); //erfcf(sqrtf(r2));
                    }}
                }*/
                // The loop above would be twice as fast, but should exclude some far cells
                // and I don't want to bother, since the weird loop starting at i0 makes it buggy / slow!
                //if(bn != 0) {
                    for (int m = 0; m < ATOMS_PER_CELL; m++) {
                        if(an[m] == 0 || self*(m==j)) continue;

                        float dx = bx - ax[m];
                        float dy = by - ay[m];
                        float dz = bz - az[m];
                        Oper2::pair(ans, dx, dy, dz);
                    }
                //}
            }
        }

        Oper2::finalize(out[bin], ans, bn, j);
    }
};

/** Create a 2-body operation.
 *
 * Oper2 must be a class including members:
 *     type Output = type of output per cell
 *     type Accum  = local variable to pass to f
 *     pair : Accum, dx, dy, dz -> void
 *     finalize : Output,Accum,n,idx -> void
 *
 * Example enque calls:
 *
 *  LJEnK = mk2Body<LJEnKernel,Acc,Dim,Idx>(devAcc, srt, nbr, X, out);
 *  alpaka::enqueue(queue, LJEnK);
 *
*/
template <typename Oper2, typename Acc, typename Dim, typename Idx, typename Dev>
auto mk2Body(const Dev &devAcc, const CellSorter &srt, const alpaka::Buf<Dev, CellRange, Dim, Idx> &nbr,
             const alpaka::Buf<Dev, Cell, Dim, Idx> &X,
             alpaka::Buf<Dev, typename Oper2::Output, Dim, Idx> &out) {
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

    std::cout << "Creating 2-body kernel for " << ncells << " cells.\n";
    Oper2Kernel<Oper2,Vec> K{};
    return alpaka::createTaskKernel<Acc>(workDiv, K,
                srt.device(), alpaka::getPtrNative(nbr),
                alpaka::getPtrNative(X), alpaka::getPtrNative(out));
}

}
