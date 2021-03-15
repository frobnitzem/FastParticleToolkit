FastParticleToolkit - Licenses
===============================================================================

**Copyright 2021**

Oak Ridge National Laboratory

FastParticleToolkit is a program collection containing the main simulation, independent
scripts and auxiliary libraries. If not stated otherwise explicitly, the
following licenses apply:

### Documentation

Documentation is licensed under CC-BY 4.0.
See https://creativecommons.org/licenses/by/4.0/ for the license.

If not stated otherwise explicitly, that affects files in:

- `docs`


### Third party software and other licenses

We include a list of (GPL-) compatible third party software for the sake
of an easier install of `FastParticleToolkit`. Contributions to these parts of the
repository should *not* be made in the `thirdParty/` directory but in
*their according repositories* (that we import).

 - `thirdParty/alpaka`:
   alpaka is a header-only C++11 abstraction library for parallel
   kernel development on accelerator hardware. It provides a single-source,
   performance portable programming model for FastParticleToolkit.
   Please visit
     https://github.com/alpaka-group/alpaka
   for further details and contributions.

 - `thirdParty/mallocMC`:
   mallocMC is a fast memory allocator for many core accelerators and was
   originally forked from the `ScatterAlloc` project.
   It is licensed under the *MIT License*.
   Please visit
     https://github.com/alpaka-group/mallocMC
   for further details and contributions.

- `thirdParty/cupla`:
   cupla is a simple user interface for alpaka. It provides a software layer
   that follows a similar concept as the Nvidia CUDA API, allowing to write
   kernels more efficiently.
   Please visit
     https://github.com/alpaka-group/cupla
   for further details and contributions.

- `thirdParty/cmake-modules`:
   a set of useful CMake modules that are not in the
   CMake mainline under the *ISC license* at
     https://github.com/ComputationalRadiationPhysics/cmake-modules
   for contributions or inclusion in FPT and other projects.

