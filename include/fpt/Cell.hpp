#pragma once

#include <algorithm>
#include <alpaka/alpaka.hpp>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

#define ATOMS_PER_CELL 32

/** Round up list of CellRange to this size */
#define CELL_LIST_PAD  32

namespace fpt {
    struct CellEnergy {
        uint32_t n[ATOMS_PER_CELL];
        double  en[ATOMS_PER_CELL];
    };


    /** Transpose of a cell -- holding max.
        number of atoms per cell.
      
        Algorithms work with this transpose.
     */
    struct CellTranspose {
        uint32_t n[ATOMS_PER_CELL]; // per-thread control block / atom number, indicating continuation, etc.
        float x[ATOMS_PER_CELL]; // currently 1 for "present", 0 for "absent"
        float y[ATOMS_PER_CELL];
        float z[ATOMS_PER_CELL];
    };
    using Cell = CellTranspose; // keep it simple for now

    /** Specifies a strip of cells indices along the x-direction.
     *  Would be nice to make an iterator.
     */
    struct CellRange {
        union {
            struct {signed char i0, i1, j, k; };
            uint32_t r;
        };
        CellRange(signed char _i0, signed char _i1, signed char _j, signed char _k) : i0(_i0),i1(_i1),j(_j),k(_k) {};
    };

    /** Flat copy of CellSorter class to be passed by value to device
     */
    struct CellSorter_d {
        const float h[3];
        const int n[3];

        CellSorter_d(float Lx, float Ly, float Lz, int nx, int ny, int nz)
            : h{Lx/nx, Ly/ny, Lz/nz}, n{nx, ny, nz} {}

        ALPAKA_FN_HOST_ACC inline
            void decodeBin(const unsigned int bin, int& i, int& j, int& k) const { 
                i = bin % n[0];
                j = (bin/n[0])%n[1];
                k = bin/(n[0]*n[1]);
        }

        ALPAKA_FN_HOST_ACC inline
            unsigned int calcBin(const int i, const int j, const int k) const { 
                return (k*n[1] + j)*n[0] + i;
        }

        ALPAKA_FN_HOST_ACC inline
            unsigned int calcBinF(const float x, const float y, const float z) const { 
                return calcBin(x/h[0], y/h[1], z/h[2]);
        }

        /* Returns the new atom count within the bin. */
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TAcc>
        ALPAKA_FN_ACC inline unsigned addToBin(
                TAcc const &acc, Cell *aosoa,
                const unsigned int bin) const {
            uint32_t cont = 1;
            unsigned winner;
            Cell &cell = aosoa[bin];

            while(cont) {
                uint32_t n = cell.n[threadIdx.x];
                auto mask = alpaka::warp::ballot(acc, n != 0);

                if(mask == alpaka::warp::activemask(acc)) { // no open slots
                    // FIXME: allocate continuation
                    return alpaka::warp::getSize(acc)-1;
                }
                for(winner=0; (1<<winner) & mask; winner++); // find winning thread (first 0)
            
                if(threadIdx.x == winner) {
                    cont = alpaka::atomicOp<alpaka::AtomicCas>(acc,
                                  &cell.n[threadIdx.x], 0, 1);
                }
                cont = alpaka::warp::shfl(acc, cont, winner);
            }
            return winner;
        }
    };

    struct CellSorter {
        const float L[6]; // x,y,z,yx,zx,zy
        const int n[3];
        const unsigned int cells;

        CellSorter(float Lx, float Ly, float Lz, int nx, int ny, int nz, float Lyx=0.0, float Lzx=0.0, float Lzy=0.0)
            : L{Lx, Ly, Lz, Lyx,Lzx,Lzy}, n{nx, ny, nz}, cells(nx*ny*nz) { }

        CellSorter_d device() { // return device-accessible copy of this class
            return CellSorter_d(L[0], L[1], L[2], n[0], n[1], n[2]);
        }

        // Create list of cells within cutoff Rc
        // inclusive range to iterate over (i0,i1),(j,k)
        std::vector<CellRange> list_cells(const float Rc) {
            const float eps = 1e-6;
            std::vector<CellRange> cell_list;
            cell_list.reserve(28);
            const float h_yx = L[3]/n[1];
            const float h_zx = L[4]/n[2], h_zy = L[5]/n[2];

            const float hz = L[2]/n[2], hy = L[1]/n[1], hx = L[0]/n[0];

            float R2 = Rc*Rc;
            signed char k0 = ceil(eps-(hz + Rc)/hz);
            signed char k1 = floor((hz + Rc)/hz-eps);
            //for(int k=0; ; k = -k+(k<=0)) { // 0, 1, -1, 2, -2, 3, ...
            for(signed char k=k0; k<=k1; k++) {
                signed char absk = k > 0 ? k-1 : (k < 0 ? -k-1 : 0);
                float dz = absk*hz; // z-dist. to this pt

                float R2z = R2 - dz*dz;
                if(R2z < 0.0) break;

                float y0 = -k*h_zy; // shifted base-pt.

                signed char j0 = ceil((y0 - hy - sqrt(R2z))/hy+eps);
                signed char j1 = floor((y0 + hy + sqrt(R2z))/hy-eps);
                for(signed char j=j0; j<=j1; j++) {
                    float dy = fabs(j*hy - y0) - hy;
                    dy -= dy*(dy < 0.0);

                    float R2y = R2z - dy*dy;
                    if(R2y < 0.0) continue;

                    float x0 = -j*h_yx-k*h_zx; // shifted base-pt.

                    signed char i0 = ceil((x0 - hx - sqrt(R2y))/hx+eps);
                    signed char i1 = floor((x0 + hx + sqrt(R2y))/hx-eps);
                    if(i1 >= i0) {
                        cell_list.push_back(CellRange(i0,i1,j,k));
                    }
                }
            }
            cell_list.push_back(CellRange(1,0,0,0)); // end-terminator
            if(cell_list.size()%CELL_LIST_PAD != 0) {
              for(int k = cell_list.size()%CELL_LIST_PAD; k<CELL_LIST_PAD; k++) { // pad to end of warp size
                cell_list.push_back(CellRange(1,0,0,0)); // end-terminator
              }
            }
            return cell_list;
        }
    };

}
