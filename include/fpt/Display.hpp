#pragma once

#include <fpt/Cell.hpp>
#include <iostream>

namespace fpt {
void print_list(const std::vector<CellRange> &lst) {
    std::cout << "Nbr List:" << std::endl;
    for(auto r : lst) {
        std::cout << "  " << r.i0 << " : " << r.i1 << " "
                          << r.j << ", "   << r.k << std::endl;
    }
}

/* simple routine to print contents of a vector */
template <typename Vector>
void print_vector(const std::string& name, const Vector& v) {
  typedef typename Vector::value_type T;
  std::cout << "  " << name << "  ";
  for(auto &x : v) {
      std::cout << x << " ";
  }
  std::cout << std::endl;
}

void print_cells(const Cell *aosoa, const unsigned int cells) {
    unsigned int N = 0;
    double sum[3] = {0., 0., 0.};

    std::cout << cells << " cells:\n";
    for(int i = 0; i < cells; i++) {
        if(N < 20 && i < 50)
            std::cout << "Bin " << i << std::endl;
        for(int j=0; j<ATOMS_PER_CELL; j++) {
            if(aosoa[i].n[j] == 0) continue; // absent
            if(N < 20)
                std::cout << "  " << aosoa[i].x[j] << " "
                                  << aosoa[i].y[j] << " "
                                  << aosoa[i].z[j] << std::endl;
            sum[0] += aosoa[i].x[j];
            sum[1] += aosoa[i].y[j];
            sum[2] += aosoa[i].z[j];
            N++;
        }
    }
    std::cout << N << " total atoms." << std::endl;
    std::cout << "sums = "
              << sum[0] << ", "
              << sum[1] << ", "
              << sum[2] << std::endl;
}

template <typename Acc>
void print_Ecells(const Acc &devAcc,
                  const alpaka::Buf<Acc, CellEnergy, alpaka::DimInt<1u>, uint32_t> &aosoa,
                  const unsigned int cells) {
    //thrust::host_vector<CellEnergy> aosoa(en);
    alpaka::Buf<alpaka::DevCpu, CellEnergy, alpaka::DimInt<1u>, uint32_t>
        bhost{alpaka::allocBuf<CellEnergy, uint32_t>(devAcc, cells)};
    auto queue = alpaka::Queue<Acc, alpaka::Blocking>(devAcc);
    alpaka::memcpy(queue, bhost, aosoa, cells);
    CellEnergy* host = alpaka::getPtrNative( bhost );

    unsigned int N = 0;
    double sum = 0.0;

    for(int i = 0; i < cells; i++) {
        if(N < 20 && i < 50)
            std::cout << "Bin " << i << std::endl;
        for(int j=0; j<ATOMS_PER_CELL; j++) {
            if(host[i].n[j] == 0) continue; // absent
            if(N < 20)
                std::cout << "  " << host[i].en[j] << std::endl;
            sum += host[i].en[j];
            N++;
        }
    }
    std::cout << N << " total atoms." << std::endl;
    std::cout << "total energy = " << sum << std::endl;
}
}
