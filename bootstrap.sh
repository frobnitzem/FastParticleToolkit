#!/bin/bash

FPT_DIR="$(cd $(dirname $0) && pwd)"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <project executable name>"
    exit 1
fi
proj="$1"

# 1. check that the current directory is a git repo pointing at FastParticleToolkit as a submodule
if [ ! -d FastParticleToolkit ] || [[ "$FPT_DIR" != "$(cd FastParticleToolkit && pwd)" ]]; then
    echo 'Error: This script must be run as "FastParticleToolkit/bootstrap.sh"'
    exit 1
fi

if [ -s CMakeLists.txt ]; then
    echo "Error: CMakeLists.txt already exists."
    exit 2
fi

# 2. copy bootsrap/CMakeLists.txt to root and substitute user's project name
sed -e "s/%USER_PROJECT_NAME/$proj/g" FastParticleToolkit/bootstrap/CMakeLists.txt >CMakeLists.txt
mkdir -p src

# 3. copy boostrap/example_main.cpp to src/main.cpp
if [ -e src/main.cpp ]; then
    echo "src/main.cpp exists: leaving alone"
    echo "For an example program, see FastParticleToolkit/bootstrap/example_main.cpp"
else
    echo "creating example src/main.cpp"
    cp FastParticleToolkit/bootstrap/example_main.cpp src/main.cpp
fi
