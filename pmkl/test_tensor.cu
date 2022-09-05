#include "pmkl.h"
#include <iostream>

using namespace pmkl;

int main() {
    Tensor t = empty({1, 2, 3}, ScalarType::Float, Device(DeviceType::CUDA, 0));
    {
        Tensor t1 = t;
    }
    Tensor t2 = t;
}
