Single-Particle Kernels
#######################

Here's an example single-particle (aka 1-body) kernel::

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

All single-particle kernels include the following:

  * `Output` - a user-defined type to store the result of the kernel in each cell.

  * `f` - function to be called on every particle in the cell.

Single Kernel Operation
------------------------

Behind the scenes, your single-particle kernel is being invoked inside
a loop over all particles within the cell.
