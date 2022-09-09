#!/bin/bash

find . -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.cu" | xargs clang-format -style=file -i

mkdir -p build
cd build

if [ $INSTALL ]; then
    cmake ..; make install
elif [ $TEST ]; then
    cmake ..; make -j16; make test
else
    cmake ..; make -j16
fi