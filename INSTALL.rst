.. _install-dependencies:

.. seealso::

   You will need to understand how to use `the terminal <http://www.ks.uiuc.edu/Training/Tutorials/Reference/unixprimer.html>`_, what are `environment variables <https://unix.stackexchange.com/questions/44990/what-is-the-difference-between-path-and-ld-library-path/45106#45106>`_ and please read our :ref:`compiling introduction <install-source>`.

.. note::

   If you are a scientific user at a supercomputing facility we might have already prepared a software setup for you.
   See the :ref:`following chapter <install-profile>` if you can skip this step fully or in part by loading existing modules on those systems.

Dependencies
============

.. sectionauthor:: Axel Huebl

Overview
--------

.. figure:: libraryDependencies.png
   :alt: overview of FPT library dependencies

   Overview of inter-library dependencies for parallel execution of FPT on a typical HPC system. Due to common binary incompatibilities between compilers, MPI and boost versions, we recommend to organize software with a version-aware package manager such as `spack <https://github.com/spack/spack>`_ and to deploy a hierarchical module system such as `lmod <https://github.com/TACC/Lmod>`_.
   An Lmod example setup can be found `here <https://github.com/ComputationalRadiationPhysics/compileNode>`_.

Requirements
------------

Mandatory
^^^^^^^^^

gcc
"""
- 4.9 - 7 (if you want to build for Nvidia GPUs, supported compilers depend on your current `CUDA version <https://gist.github.com/ax3l/9489132>`_)

  - CUDA 9.2 - 10.0: Use gcc 4.9 - 7
  - CUDA 10.1/10.2: Use gcc 4.9 - 8
- *note:* be sure to build all libraries/dependencies with the *same* gcc version; GCC 5 or newer is recommended
- *Debian/Ubuntu:*
  
  - ``sudo apt-get install gcc-5.3 g++-5.3 build-essential``
  - ``sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5.3 60 --slave /usr/bin/g++ g++ /usr/bin/g++-5.3``
- *Arch Linux:*
  
  - ``sudo pacman --sync base-devel``
  - if the installed version of **gcc** is too new, `compile an older gcc <https://gist.github.com/slizzered/a9dc4e13cb1c7fffec53>`_
- *Spack:*
  
  - ``spack install gcc@5.3.0``
  - make it the default in your `packages.yaml <http://spack.readthedocs.io/en/latest/getting_started.html#compiler-configuration>`_ or *suffix* `all following <http://spack.readthedocs.io/en/latest/features.html#simple-package-installation>`_ ``spack install`` commands with a *space* and ``%gcc@5.3.0``

CMake
"""""
- 3.11.4 or higher
- *Debian/Ubuntu:* ``sudo apt-get install cmake file cmake-curses-gui``
- *Arch Linux:* ``sudo pacman --sync cmake``
- *Spack:* ``spack install cmake``

MPI 2.3+
""""""""
- **OpenMPI** 1.7+ / **MVAPICH2** 1.8+ or similar
- for running on Nvidia GPUs, perform a `GPU aware MPI install <https://devblogs.nvidia.com/parallelforall/introduction-cuda-aware-mpi/>`_ *after* installing CUDA
- *Debian/Ubuntu:* ``sudo apt-get install libopenmpi-dev``
- *Arch Linux:* ``sudo pacman --sync openmpi``
- *Spack:*

  - *GPU support:* ``spack install openmpi+cuda``
  - *CPU only:* ``spack install openmpi``
- *environment:*

  - ``export MPI_ROOT=<MPI_INSTALL>``
  - as long as CUDA awareness (``openmpi+cuda``) is missing: ``export OMPI_MCA_mpi_leave_pinned=0``

zlib
""""
- *Debian/Ubuntu:* ``sudo apt-get install zlib1g-dev``
- *Arch Linux:* ``sudo pacman --sync zlib``
- *Spack:* ``spack install zlib``
- *from source:*

  - ``./configure --prefix=$HOME/lib/zlib``
  - ``make && make install``
- *environent:* (assumes install from source in ``$HOME/lib/zlib``)

  - ``export ZLIB_ROOT=$HOME/lib/zlib``
  - ``export LD_LIBRARY_PATH=$ZLIB_ROOT/lib:$LD_LIBRARY_PATH``
  - ``export CMAKE_PREFIX_PATH=$ZLIB_ROOT:$CMAKE_PREFIX_PATH``

boost
"""""
- 1.65.1 - 1.70.0 (``program_options``, ``filesystem``, ``system``, ``math``, ``serialization`` and header-only libs, optional: ``fiber`` with ``context``, ``thread``, ``chrono``, ``atomic``, ``date_time``)
- *Debian/Ubuntu:* ``sudo apt-get install libboost-program-options-dev libboost-filesystem-dev libboost-system-dev libboost-thread-dev libboost-chrono-dev libboost-atomic-dev libboost-date-time-dev libboost-math-dev libboost-serialization-dev libboost-fiber-dev libboost-context-dev``
- *Arch Linux:* ``sudo pacman --sync boost``
- *Spack:* ``spack install boost``
- *from source:*

  - ``curl -Lo boost_1_65_1.tar.gz https://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.gz``
  - ``tar -xzf boost_1_65_1.tar.gz``
  - ``cd boost_1_65_1``
  - ``./bootstrap.sh --with-libraries=atomic,chrono,context,date_time,fiber,filesystem,math,program_options,serialization,system,thread --prefix=$HOME/lib/boost``
  - ``./b2 cxxflags="-std=c++11" -j4 && ./b2 install``
- *environment:* (assumes install from source in ``$HOME/lib/boost``)

  - ``export BOOST_ROOT=$HOME/lib/boost``
  - ``export LD_LIBRARY_PATH=$BOOST_ROOT/lib:$LD_LIBRARY_PATH``

git
"""
- 1.7.9.5 or `higher <https://help.github.com/articles/https-cloning-errors>`_
- *Debian/Ubuntu:* ``sudo apt-get install git``
- *Arch Linux:* ``sudo pacman --sync git``
- *Spack:* ``spack install git``

rsync
"""""
- *Debian/Ubuntu:* ``sudo apt-get install rsync``
- *Arch Linux:* ``sudo pacman --sync rsync``
- *Spack:* ``spack install rsync``

.. _install-dependencies-fpt:

FastParticleToolkit Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``mkdir myProject && cd myProject``
- ``git init```
- ``git submodule add https://github.com/frobnitzem/FastParticleToolkit.git``
- ``FastParticleToolkit/bootstrap.sh myProgram``

- *environment*:

  - ``export PATH=$PWD/FastParticleToolkit/bin:$PATH`` (optional)

Optional Libraries
^^^^^^^^^^^^^^^^^^

CUDA
""""
- `9.2 - 10.2 <https://developer.nvidia.com/cuda-downloads>`_
- required if you want to run on Nvidia GPUs
- *Debian/Ubuntu:* ``sudo apt-get install nvidia-cuda-toolkit``
- *Arch Linux:* ``sudo pacman --sync cuda``
- *Spack:* ``spack install cuda``
- at least one **CUDA** capable **GPU**
- *compute capability*: ``sm_30`` or higher
- `full list <https://developer.nvidia.com/cuda-gpus>`_ of CUDA GPUs and their *compute capability*
- `More <http://www.olcf.ornl.gov/summit/>`_ is always `better <http://www.cscs.ch/computers/piz_daint/index.html>`_. Especially, if we are talking GPUs :-)
- *environment:*

  - ``export CUDA_ROOT=<CUDA_INSTALL>``

If you do not install the following libraries, you will not have the full amount of FPT plugins.
We recommend to install **HDF5** or **ADIOS**.

HDF5
""""
- 1.8.13+
- standard shared version (no C++, enable parallel)
- *Debian/Ubuntu:* ``sudo apt-get install libhdf5-openmpi-dev``
- *Arch Linux:* ``sudo pacman --sync hdf5-openmpi``
- *Spack:* ``spack install hdf5~fortran``
- *from source:*

  - ``mkdir -p ~/src ~/lib``
  - ``cd ~/src``
  - download hdf5 source code from `release list of the HDF5 group <https://www.hdfgroup.org/ftp/HDF5/releases/>`_, for example:

  - ``curl -Lo hdf5-1.8.20.tar.gz https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.20/src/hdf5-1.8.20.tar.gz``
  - ``tar -xzf hdf5-1.8.20.tar.gz``
  - ``cd hdf5-1.8.20``
  - ``./configure --enable-parallel --enable-shared --prefix $HOME/lib/hdf5/``
  - ``make``
  - *optional:* ``make test``
  - ``make install``
  - If you encounter errors related to linking MPI during ``./configure``, you might try setting the compiler manually via ``./configure --enable-parallel --enable-shared --prefix $HOME/lib/hdf5/ CC=mpicc CXX=mpic++``.
- *environment:* (assumes install from source in ``$HOME/lib/hdf5``)

  - ``export HDF5_ROOT=$HOME/lib/hdf5``
  - ``export LD_LIBRARY_PATH=$HDF5_ROOT/lib:$LD_LIBRARY_PATH``

ADIOS
"""""
- 1.13.1+ (requires *MPI*, *zlib* and *c-blosc*)
- *Debian/Ubuntu:* ``sudo apt-get install libadios-dev libadios-bin``
- *Arch Linux* using an `AUR helper <https://wiki.archlinux.org/index.php/AUR_helpers>`_: ``pacaur --sync libadios``
- *Arch Linux* using the `AUR <https://wiki.archlinux.org/index.php/Arch_User_Repository>`_ manually:

  - ``sudo pacman --sync --needed base-devel``
  - ``git clone https://aur.archlinux.org/libadios.git``
  - ``cd libadios``
  - ``makepkg -sri``
- *Spack:* ``spack install adios``
- *from source:*

  - ``mkdir -p ~/src ~/lib``
  - ``cd ~/src``
  - ``curl -Lo adios-1.13.1.tar.gz http://users.nccs.gov/~pnorbert/adios-1.13.1.tar.gz``
  - ``tar -xzf adios-1.13.1.tar.gz``
  - ``cd adios-1.13.1``
  - ``CFLAGS="-fPIC" ./configure --enable-static --enable-shared --prefix=$HOME/lib/adios --with-mpi=$MPI_ROOT --with-zlib=$HOME/lib/zlib --with-blosc=$HOME/lib/c-blosc``
  - ``make``
  - ``make install``
- *environment:* (assumes install from source in ``$HOME/lib/adios``)

  - ``export ADIOS_ROOT=$HOME/lib/adios``
  - ``export LD_LIBRARY_PATH=$ADIOS_ROOT/lib:$LD_LIBRARY_PATH``

VampirTrace
"""""""""""
- for developers: performance tracing support
- download 5.14.4 or higher, e.g. from `www.tu-dresden.de <https://tu-dresden.de/zih/forschung/projekte/vampirtrace>`_
- *from source:*

  - ``mkdir -p ~/src ~/build ~/lib``
  - ``cd ~/src``
  - ``curl -Lo VampirTrace-5.14.4.tar.gz "http://wwwpub.zih.tu-dresden.de/~mlieber/dcount/dcount.php?package=vampirtrace&get=VampirTrace-5.14.4.tar.gz"``
  - ``tar -xzf VampirTrace-5.14.4.tar.gz``
  - ``cd VampirTrace-5.14.4``
  - ``./configure --prefix=$HOME/lib/vampirtrace --with-cuda-dir=<CUDA_ROOT>``
  - ``make all -j``
  - ``make install``
- *environment:* (assumes install from source in ``$HOME/lib/vampirtrace``)

  - ``export VT_ROOT=$HOME/lib/vampirtrace``
  - ``export PATH=$VT_ROOT/bin:$PATH``
