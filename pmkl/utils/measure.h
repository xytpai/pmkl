#pragma once

#include <iostream>
#include <iomanip>
#include <cmath>

namespace pmkl {
namespace utils {

template <typename T>
bool all_close(T *input, T *target, unsigned int len,
               double atol = 1e-5, double rtol = 1e-5, int max_errors_print = 10) {
    unsigned int errors = 0;
    for (unsigned int i = 0; i < len; i++) {
        if (std::isnan(input[i]) || (std::abs(input[i] - target[i]) > atol + rtol * std::abs(target[i]))) {
#ifdef DEBUG
            if (errors < max_errors_print)
                std::cout << "Accuracy error: index[" << i << "], input: " << input[i] << ", target: " << target[i] << std::endl;
            errors++;
#else
            return false;
#endif
        }
    }
#ifdef DEBUG
    std::cout << "Total " << errors << " errors\n";
#endif
    return errors == 0;
}

}
} // namespace pmkl::utils
