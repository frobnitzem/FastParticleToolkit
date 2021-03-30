Pair Kernels
#############

Here's an example pair kernel::

    ALPAKA_FN_HOST_ACC inline float lj_en(const float r2) {
        float ir2 = 1.0/r2;
        float ir6 = ir2*ir2*ir2;
        float ir12 = ir6*ir6;
        return fmaf(-2.0, ir6, ir12);
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

All pair kernels include the following:

  * `Output` - a user-defined type to store the result of the kernel in each cell.

  * `Accum` - a user-defined type to store the intermediate result during computations.

  * `pair` - function to be called on every pair of particles

  * `finalize` - function responsible for storing `Accum` into `Output`

Pair Kernel Operation
---------------------

Behind the scenes, your pair kernel is being invoked inside
a loop over neighboring cells.
