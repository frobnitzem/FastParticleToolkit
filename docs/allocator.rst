Cell Memory Allocator
#####################

Most users won't have to deal with the cell memory allocator itself.
The documentation here is for those interested in the design
choices made inside FastParticleToolkit.

The allocator is based on the concepts from `SlabHash`.
An entire warp works synchronously, so every operation
is called by every thread, but only one logical
allocation/de-allocation takes place during a call.

The allocator is designed to allocate/de-allocate
one `Cell` data structure at a time.  This is because
the expected number of overflows is usually 0, or 1,
but could sometimes be more.

It maintains 2 data structures:

  * A giant vector of `Cell` data.

  * A *free-list* of `uint32_t` bitmasks.  Each bit corresponds to
    a cell.  If a bit is on, then the cell is empty.

Memory addresses `uin32_t n`, index cells, and are decoded as
you might expect::

    // Pseudo-code ignoring race-conditions.
    // Try allocation at n.

    bool occupied = (free_list[n/32]>>(n%32)) & 1; // bit n of free_list

    if( ! occupied ) {
        free_list[n/32] |= 1<<(n%32);
        return cell[n];
    } else {
        return "Failed Allocation at n";
    }

Because FPT always launches its kernels with 1 block = 1 base cell,
the kernel's blockId is hashed to determine the search sequence::

    N = sizeof(cell / sizeof(Cell));   // allocatable cells
    n_0     = blockId;                 // first index linearly
    n_{i+1} = hash(n_i | (blockId<<16)) % N = searchNext(n_i,blockId,N);

The hash output space is 2^16.  This limits the maximum addressable space to N <= 2^16 cells.
These addresses belong to just one MPI rank, so more space is not needed for this application.


.. doxygenclass:: fpt::Alloc
   :members:

