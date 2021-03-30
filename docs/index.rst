Fast Particle Toolkit
=====================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installing
   main

   atoms
   cells
   singles
   pairs
   allocator

:ref:`genindex`

Fast Particle Toolkit is intended to be a Swiss-Army knife for computing
spatial functions on particle systems.  It's guiding principle is
to let the user write functions that work on the particles themselves.

This library compiles them to GPU kernels and runs them.  It also handles
sorting all the particles into their distributed spatial bins.

The documentation 
