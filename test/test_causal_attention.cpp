#include "pmkl.h"

using namespace std;
using namespace pmkl;

int main() {
    auto l = GpuLauncher::GetInstance();
    Tensor q = empty({2, 4, 1024, 64}, ScalarType::Float, 0);
    Tensor k = empty({2, 4, 1024, 64}, ScalarType::Float, 0);
    Tensor v = empty({2, 4, 1024, 64}, ScalarType::Float, 0);
    Tensor o = empty({2, 4, 1024, 64}, ScalarType::Float, 0);
    Tensor ms = empty({2, 4, 1024}, ScalarType::Float, 0);
    Tensor ls = empty({2, 4, 1024}, ScalarType::Float, 0);
    auto q_ = reinterpret_cast<float *>(q.data_ptr());
    auto k_ = reinterpret_cast<float *>(k.data_ptr());
    auto v_ = reinterpret_cast<float *>(v.data_ptr());
    auto o_ = reinterpret_cast<float *>(o.data_ptr());
    auto ms_ = reinterpret_cast<float *>(ms.data_ptr());
    auto ls_ = reinterpret_cast<float *>(ls.data_ptr());
    nlp::causal_attention_forward<float, 32, 64>(o_, q_, k_, v_, 2, 4, 1024, ms_, ls_);
}
