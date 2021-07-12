How to Write a Main Function
############################

The example directory contains a simple
Lennard Jones simulation.  You can see that
its main function does the following tasks::

    // Declare an Accelerator & Queue
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto queue = alpaka::Queue<Acc, alpaka::NonBlocking>(devAcc);

    // Initialize Buffers
    auto xHost         = alloc<fpt::Cell, Dim, Idx>(devHost, srt.cells);
    fpt::Cell* pHost   = alpaka::getPtrNative( xHost );
    for(int i = 0; i < N; i++) {
        pHost[i/ATOMS_PER_CELL].n[i%ATOMS_PER_CELL] = 1;
    }

    // Copy to Device
    auto xCurrAcc = alloc<fpt::Cell, Dim, Idx>(devAcc, srt.cells);
    alpaka::memcpy(queue, xCurrAcc, xHost, srt.cells);

    // Declare Kernels
    auto const LJDEK = fpt::mk2Body<LJDerivOper,Acc,Dim,Idx>(
                             devAcc, srt, nbr1, xNextAcc, xCurrAcc);

    // Execute Kernels
    alpaka::enqueue(queue, LJDEK);

    // Copy back results
    alpaka::memcpy(queue, xHost, xCurrAcc, srt.cells);

..
    .. doxygenfile:: proto.hh
    .. doxygenclass:: dwork::TaskDB
       :members:
    .. doxygenclass:: dwork::TaskHeap
       :members:

