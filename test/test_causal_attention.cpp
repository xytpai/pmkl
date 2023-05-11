#include "pmkl.h"

#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>

using namespace std;
using namespace pmkl;

template <typename scalar_t>
void host_self_attention_forward(
    scalar_t *o, const scalar_t *q, const scalar_t *k, const scalar_t *v,
    int batch_size, int nheads, int seq_length, int hidden_size,
    bool is_causal,
    scalar_t *out_m,
    scalar_t *out_l) {
    constexpr scalar_t neg_inf = -1e20;
    batch_size = batch_size * nheads;
    for (int b = 0; b < batch_size; b++) {
        size_t offset_b = b * seq_length * hidden_size;
        auto o_b = o + offset_b;
        auto q_b = q + offset_b;
        auto k_b = k + offset_b;
        auto v_b = v + offset_b;
        auto out_m_b = out_m + b * seq_length;
        auto out_l_b = out_l + b * seq_length;
        auto seq2 = new scalar_t[seq_length * seq_length];
        for (int m = 0; m < seq_length; m++) {
            for (int n = 0; n < seq_length; n++) {
                auto q_s = q_b + m * hidden_size;
                auto k_s = k_b + n * hidden_size;
                scalar_t sum = 0;
                for (int kk = 0; kk < hidden_size; kk++)
                    sum += q_s[kk] * k_s[kk];
                seq2[m * seq_length + n] = sum * (1.0f / std::sqrt((float)hidden_size));
            }
        }
        if (is_causal) {
            for (int m = 0; m < seq_length; m++)
                for (int n = 0; n < seq_length; n++)
                    seq2[m * seq_length + n] = n >= m ? seq2[m * seq_length + n] : neg_inf;
        }
        for (int m = 0; m < seq_length; m++) {
            scalar_t max_value = neg_inf;
            for (int n = 0; n < seq_length; n++) {
                max_value = std::max(seq2[m * seq_length + n], max_value);
            }
            out_m_b[m] = max_value;
            double e2sum = 0;
            for (int n = 0; n < seq_length; n++) {
                e2sum += std::exp((double)seq2[m * seq_length + n] - max_value);
            }
            out_l_b[m] = e2sum;
            for (int n = 0; n < seq_length; n++) {
                seq2[m * seq_length + n] = std::exp(seq2[m * seq_length + n] - max_value) / e2sum;
            }
        }
        for (int m = 0; m < seq_length; m++) {
            for (int kk = 0; kk < hidden_size; kk++) {
                scalar_t sum = 0;
                for (int n = 0; n < seq_length; n++) {
                    sum += seq2[m * seq_length + n] * v_b[n * hidden_size + kk];
                }
                o_b[m * hidden_size + kk] = sum;
            }
        }
        delete[] seq2;
    }
}

template <typename scalar_t>
void host_self_attention_backward(
    scalar_t *dq, scalar_t *dk, scalar_t *dv,
    const scalar_t *dout, const scalar_t *o,
    const scalar_t *q, const scalar_t *k, const scalar_t *v,
    const scalar_t *out_m, const scalar_t *out_l,
    int batch_size, int nheads, int seq_length, int hidden_size,
    bool is_causal) {
    constexpr scalar_t neg_inf = -1e20;
    batch_size = batch_size * nheads;
    for (int b = 0; b < batch_size; b++) {
        size_t offset_b = b * seq_length * hidden_size;
        auto dq_b = dq + offset_b;
        auto dk_b = dk + offset_b;
        auto dv_b = dv + offset_b;
        auto dout_b = dout + offset_b;
        auto o_b = o + offset_b;
        auto q_b = q + offset_b;
        auto k_b = k + offset_b;
        auto v_b = v + offset_b;
        auto out_m_b = out_m + b * seq_length;
        auto out_l_b = out_l + b * seq_length;
        for (int m = 0; m < seq_length; m++) {     // k, v
            for (int n = 0; n < seq_length; n++) { // q, o, do
                scalar_t qk = 0;
                for (int kk = 0; kk < hidden_size; kk++) {
                    qk += q_b[n * hidden_size + kk] * k_b[m * hidden_size + kk];
                }
                if (is_causal) qk = m >= n ? qk : neg_inf;
                qk = std::exp(qk - out_m_b[m]) / out_l_b[m];
            }
        }
    }
}

int main() {
    auto l = GpuLauncher::GetInstance();
    l->set_profiling_mode(true);
    std::vector<std::vector<int64_t>> shapes = {
        {4, 16, 1024, 64},
    };
    for (auto shape : shapes) {
        Tensor q = empty(shape, ScalarType::Float, 0);
        Tensor k = empty(shape, ScalarType::Float, 0);
        Tensor v = empty(shape, ScalarType::Float, 0);
        Tensor o = empty(shape, ScalarType::Float, 0);
        auto s012 = shape[0] * shape[1] * shape[2];
        Tensor ms = empty({s012}, ScalarType::Float, 0);
        Tensor ls = empty({s012}, ScalarType::Float, 0);
        auto numel = q.numel();
        std::cout << numel << std::endl;

        auto q_data = new float[numel];
        auto k_data = new float[numel];
        auto v_data = new float[numel];

        auto o_cpu = new float[numel];
        auto o_gpu_c = new float[numel];

        auto m_cpu = new float[s012];
        auto m_gpu_c = new float[s012];

        auto l_cpu = new float[s012];
        auto l_gpu_c = new float[s012];

        utils::host::fill_rand<float>(q_data, numel, -1.05, 1.05);
        utils::host::fill_rand<float>(k_data, numel, -1.05, 1.05);
        utils::host::fill_rand<float>(v_data, numel, -1.05, 1.05);

        host_self_attention_forward(o_cpu, q_data, k_data, v_data,
                                    shape[0], shape[1], shape[2], shape[3], true, m_cpu, l_cpu);

        q.copy_from_cpu_ptr((void *)q_data);
        k.copy_from_cpu_ptr((void *)k_data);
        v.copy_from_cpu_ptr((void *)v_data);

        auto q_ = reinterpret_cast<float *>(q.data_ptr());
        auto k_ = reinterpret_cast<float *>(k.data_ptr());
        auto v_ = reinterpret_cast<float *>(v.data_ptr());
        auto o_ = reinterpret_cast<float *>(o.data_ptr());
        auto ms_ = reinterpret_cast<float *>(ms.data_ptr());
        auto ls_ = reinterpret_cast<float *>(ls.data_ptr());

        nlp::causal_attention_forward<float, 32, 64>(o_, q_, k_, v_, shape[0], shape[1], shape[2], ms_, ls_);
        l->memcpy((void *)o_gpu_c, (void *)o_, numel * sizeof(float), GpuLauncher::Direction::D2H);
        l->memcpy((void *)m_gpu_c, (void *)ms_, s012 * sizeof(float), GpuLauncher::Direction::D2H);
        l->memcpy((void *)l_gpu_c, (void *)ls_, s012 * sizeof(float), GpuLauncher::Direction::D2H);

        std::cout << "testing m ...\n";
        if (!utils::all_close(m_gpu_c, m_cpu, s012))
            return 1;
        std::cout << "testing l ...\n";
        if (!utils::all_close(l_gpu_c, l_cpu, s012))
            return 1;
        std::cout << "testing out ...\n";
        if (!utils::all_close(o_gpu_c, o_cpu, numel))
            return 1;
        cout << "ok";
    }
}
