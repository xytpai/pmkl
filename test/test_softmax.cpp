#include <iostream>

#include "pmkl.h"
#include "warp_softmax.h"

using namespace pmkl;
using namespace std;

int main() {
    auto l = GpuLauncher::GetInstance();
    using scalar_t = float;
    const int batch_size = 40960;
    const int eltsize = 512;
    scalar_t *input = l->malloc<scalar_t>(batch_size * eltsize);
    scalar_t *output = l->malloc<scalar_t>(batch_size * eltsize);
    l->stream_begin();
    l->set_profiling_mode(true);
    norm::warp_softmax_forward<scalar_t, scalar_t, scalar_t, false>(output, input, eltsize, eltsize, batch_size);
    l->stream_end();
}