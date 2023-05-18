#include <torch/extension.h>

namespace pmkl_cpu {

template <typename scalar_t>
void host_self_attention_forward(
    scalar_t *o, const scalar_t *q, const scalar_t *k, const scalar_t *v,
    int batch_size, int nheads, int seq_length, int hidden_size,
    bool is_causal,
    scalar_t *out_m,
    scalar_t *out_l) {
    scalar_t neg_inf = -std::numeric_limits<scalar_t>::infinity();
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
                    seq2[m * seq_length + n] = m >= n ? seq2[m * seq_length + n] : neg_inf;
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
    const scalar_t *dout, const scalar_t *q, const scalar_t *k, const scalar_t *v,
    const scalar_t *out_m, const scalar_t *out_l,
    int batch_size, int nheads, int seq_length, int hidden_size,
    bool is_causal) {
    scalar_t neg_inf = -std::numeric_limits<scalar_t>::infinity();
    batch_size = batch_size * nheads;
    for (int b = 0; b < batch_size; b++) {
        size_t offset_b = b * seq_length * hidden_size;
        auto dq_b = dq + offset_b;
        auto dk_b = dk + offset_b;
        auto dv_b = dv + offset_b;
        auto dout_b = dout + offset_b;
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
                qk = qk * (1.0f / std::sqrt((float)hidden_size));
                if (is_causal) qk = n >= m ? qk : neg_inf;
                scalar_t p = std::exp(qk - out_m_b[n]) / out_l_b[n];
                // dv
                for (int kk = 0; kk < hidden_size; kk++) {
                    dv_b[m * hidden_size + kk] += p * dout_b[n * hidden_size + kk];
                }

                // dq, dk
                scalar_t acc_d = 0;
                for (int inner_m = 0; inner_m < seq_length; inner_m++) { // k

                    scalar_t dqksm = 0;
                    for (int kk = 0; kk < hidden_size; kk++) {
                        dqksm += dout_b[n * hidden_size + kk] * v_b[inner_m * hidden_size + kk];
                    }

                    scalar_t d_;
                    if (inner_m == m) {
                        d_ = dqksm * p * (1 - p);
                    } else {
                        scalar_t qk_ = 0;
                        for (int kk = 0; kk < hidden_size; kk++) {
                            qk_ += q_b[n * hidden_size + kk] * k_b[inner_m * hidden_size + kk];
                        }
                        qk_ = qk_ * (1.0f / std::sqrt((float)hidden_size));
                        if (is_causal) qk_ = n >= inner_m ? qk_ : neg_inf;
                        auto p_ = std::exp(qk_ - out_m_b[n]) / out_l_b[n];
                        d_ = -dqksm * p_ * p;
                    }
                    d_ *= (1.0f / std::sqrt((float)hidden_size));
                    acc_d += d_;
                }

                for (int kk = 0; kk < hidden_size; kk++) {
                    dq_b[n * hidden_size + kk] += acc_d * k_b[m * hidden_size + kk];
                }
                for (int kk = 0; kk < hidden_size; kk++) {
                    dk_b[m * hidden_size + kk] += acc_d * q_b[n * hidden_size + kk];
                }
            }
        }
    }
}

} // namespace pmkl_cpu

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_self_attention_fw(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, bool is_causal) {
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "Input tensor shapes must match");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "Input tensor shapes must match");

    int batch_size = q.size(0);
    int num_heads = q.size(1);
    int seq_length = q.size(2);
    int hidden_size = q.size(3);

    auto out = at::empty_like(v);
    auto out_m = at::empty({batch_size, num_heads, seq_length}, v.options());
    auto out_l = at::empty({batch_size, num_heads, seq_length}, v.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        v.scalar_type(),
        "fused_self_attention_fw",
        [&]() {
            pmkl_cpu::host_self_attention_forward<scalar_t>(
                out.data<scalar_t>(),
                q.contiguous().data<scalar_t>(),
                k.contiguous().data<scalar_t>(),
                v.contiguous().data<scalar_t>(),
                batch_size,
                num_heads,
                seq_length,
                hidden_size,
                is_causal,
                out_m.data<scalar_t>(),
                out_l.data<scalar_t>());
        });

    return std::forward_as_tuple(out, out_m, out_l);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_self_attention_bw(
    const at::Tensor &dout, const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &out_m, const at::Tensor &out_l, bool is_causal) {
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "Input tensor shapes must match");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "Input tensor shapes must match");

    int batch_size = q.size(0);
    int num_heads = q.size(1);
    int seq_length = q.size(2);
    int hidden_size = q.size(3);

    auto dq = at::zeros_like(q);
    auto dk = at::zeros_like(k);
    auto dv = at::zeros_like(v);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        v.scalar_type(),
        "fused_self_attention_bw",
        [&]() {
            pmkl_cpu::host_self_attention_backward<scalar_t>(
                dq.data<scalar_t>(),
                dk.data<scalar_t>(),
                dv.data<scalar_t>(),
                dout.contiguous().data<scalar_t>(),
                q.data<scalar_t>(),
                k.data<scalar_t>(),
                v.data<scalar_t>(),
                out_m.data<scalar_t>(),
                out_l.data<scalar_t>(),
                batch_size,
                num_heads,
                seq_length,
                hidden_size,
                is_causal);
        });

    return std::forward_as_tuple(dq, dk, dv);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_self_attention_fw", &fused_self_attention_fw, "fused_self_attention_fw");
    m.def("fused_self_attention_bw", &fused_self_attention_bw, "fused_self_attention_bw");
}
