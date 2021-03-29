#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <fpt/Cell.hpp>
#include <fpt/Display.hpp>
#include <fpt/Sort.hpp>
#include <fpt/Singles.hpp>
#include <fpt/Pairs.hpp>
#include <fpt/Timer.hpp>

#include <assert.h>
#include <random>

template <typename A, typename Dim, typename Idx, typename Dev>
auto alloc(const Dev &devAcc, Idx n) {
    using  BufDev = alpaka::Buf<Dev, A, Dim, Idx>;
    return BufDev{alpaka::allocBuf<A, Idx>(devAcc, n)};
}

int main(int argc, char *argv[]) {
    // Spin up devices
    using DevHost = alpaka::DevCpu;

    using Idx = uint32_t;
    using Dim = alpaka::DimInt<1u>; // can potentially use 3D...
    using Vec = alpaka::Vec<Dim, Idx>;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;

    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>()
              << std::endl;

    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    const alpaka::DevCpu devHost = alpaka::getDevByIdx<DevHost>(0u);
    auto queue = alpaka::Queue<Acc, alpaka::NonBlocking>(devAcc);

    // Create cell geometry
    const int nx = 16;
    const float hx = 2.65625;
    auto srt = fpt::CellSorter(nx*hx, nx*hx, nx*hx, nx, nx, nx); // define cell sizes
    //auto srt = fpt::CellSorter(45.0, 85.0, 85.0, 32, 32, 32); // define cell sizes
    const int N = 10240; // number of atoms to add
    auto nbr = srt.list_cells(3.5);

    std::cout << nbr.size() << " Neighbor cells" << std::endl;
    //print_list(srt.list_cells(0.0, 0.0, 0.0, 0.1));
    //print_list(srt.list_cells(0.0, 0.0, 0.0, 1.0));
    //print_list(srt.list_cells(0.0, 0.0, 0.0, 2.0));
    //print_list(srt.list_cells(0.1, 0.2, 0.0, 2.0));

    // Initialize host-buffer
    auto xHost         = alloc<fpt::Cell, Dim, Idx>(devHost, srt.cells);
    fpt::Cell* pHost   = alpaka::getPtrNative( xHost );

    assert(N < srt.cells*ATOMS_PER_CELL);

    // Step 1: generate random particle positions on the host
    //auto rng = alpaka::rand::generator::createDefault(devHost, 61251u, 0u);
    //auto U = alpaka::rand::distribution::createUniformReal<float>(devHost);
    //auto number = U(rng);
    // Seed with a real random value, if available
    //std::random_device r;
    std::default_random_engine rng(1729);
    std::uniform_real_distribution<float> U(0.0, 1.0);
    std::cout << "Random numbers:\n";
    std::cout << U(rng) << " " << U(rng) << " " << U(rng) << std::endl;

    // TODO: benchmark sort times for different starting distributions
    #pragma omp parallel for
    for(int i = 0; i < srt.cells; i++) {
        for(int j=0; j<ATOMS_PER_CELL; j++) {
            pHost[i].n[j] = 0;
        }
    }
    //for(int i = 0; i < srt.cells; i++) {
    //  #pragma omp parallel for
    //  for(int j=0; j<ATOMS_PER_CELL; j++) {
    //      aosoa[i].n[j] = 0;
    //  }
    //}
    //#pragma omp parallel for
    for(int i = 0; i < N; i++) {
        pHost[i/ATOMS_PER_CELL].n[i%ATOMS_PER_CELL] = 1;
        pHost[i/ATOMS_PER_CELL].x[i%ATOMS_PER_CELL] = U(rng)*srt.L[0];
        pHost[i/ATOMS_PER_CELL].y[i%ATOMS_PER_CELL] = U(rng)*srt.L[1];
        pHost[i/ATOMS_PER_CELL].z[i%ATOMS_PER_CELL] = U(rng)*srt.L[2];
    }
    fpt::print_cells(pHost, srt.cells);

    // Step 2: copy to device
    auto nbr1 = alloc<fpt::CellRange, Dim, Idx>(devAcc, nbr.size());
    auto en = alloc<fpt::CellEnergy, Dim, Idx>(devAcc, srt.cells);
    // Create Cell buffers
    auto xNextAcc = alloc<fpt::Cell, Dim, Idx>(devAcc, srt.cells);
    auto xCurrAcc = alloc<fpt::Cell, Dim, Idx>(devAcc, srt.cells);

    alpaka::memcpy(queue, xCurrAcc, xHost, srt.cells);
    alpaka::memcpy(queue, nbr1, nbr, nbr.size());

    //auto const warpExtent = alpaka::getWarpSize(dev);
    auto const gridBlockExtent = Vec::all( nbr.size() );
    auto const blockThreadExtent = Vec::all( ATOMS_PER_CELL );

    auto const sortKernel = fpt::mkSorter<Acc,Dim,Idx>(
                    devAcc, srt, xCurrAcc, xNextAcc);
    auto const ZeroCellK = fpt::mk1Body<fpt::ZeroCellOper,Acc,Dim,Idx>(
                    devAcc, xNextAcc, xNextAcc);

    // Step 3: sort atoms into bins
    fpt::time_kernel(queue, "Bin Atoms (no zero)", [&] {
            alpaka::enqueue(queue, sortKernel);
        }, 1000);
    fpt::time_kernel(queue, "Bin Atoms (zero)", [&] {
            alpaka::enqueue(queue, ZeroCellK);
            alpaka::enqueue(queue, sortKernel);
        }, 1000);

    alpaka::memcpy(queue, xHost, xNextAcc, srt.cells);

    // Copy back the re-partitioned atoms
    fpt::print_cells(pHost, srt.cells);

    // Step 4: calculate pair energies
    //auto const ZeroEnK = fpt::mk1Body<fpt::ZeroEnOper,Acc,Dim,Idx>(
    //                devAcc, xNextAcc, en);
    auto const LJEnK = fpt::mk2Body<LJEnOper,Acc,Dim,Idx>(
                            devAcc, srt, nbr1, xNextAcc, en);

    fpt::time_kernel(queue, "Pair Energy", [&] {
            //alpaka::enqueue(queue, ZeroEnK);
            alpaka::enqueue(queue, LJEnK);
        }, 100);

    // Copy back the per-atom energies:
    // Note: this causes a performance bug, since it creates
    // a separate stream that causes a conflict of some sort.
    //fpt::print_Ecells(devAcc, en);

    // Step 5: calculate pair forces
    auto const LJDEK = fpt::mk2Body<LJDerivOper,Acc,Dim,Idx>(
                             devAcc, srt, nbr1, xNextAcc, xCurrAcc);
    fpt::time_kernel(queue, "Pair Force", [&] {
            //alpaka::enqueue(queue, ZeroCellK);
            alpaka::enqueue(queue, LJDEK);
        }, 100);

    // Copy back the per-atom derivatives:
    alpaka::memcpy(queue, xHost, xCurrAcc, srt.cells);
    fpt::print_cells(pHost, srt.cells);

    /*
    //unsigned int ctr = srt_d.calcBin(4,15,2);
    test_deriv(srt, aosoa1, aosoa2, en, nbr1, 0, 0);
    */
  
    return 0;
}

