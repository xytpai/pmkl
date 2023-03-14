#!/bin/bash

find . -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.cu" | xargs clang-format -style=file -i

mkdir -p build
cd build

if [ $INSTALL ]; then
    cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx ..; make install
elif [ $TEST ]; then
    cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx ..; make -j16; make test
else
    cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx ..; make -j16
fi
