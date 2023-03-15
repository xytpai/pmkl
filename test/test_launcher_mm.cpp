#include "pmkl.h"
#include <iostream>
#include <random>

using namespace pmkl;
using namespace pmkl::utils;
using namespace std;

DEVICE inline void matmul_kernel(
    KernelInfo &info,
    const float *A, int Ah, int Aw,
    const float *B, int Bw,
    float *C) {
    int a, b, k;
    int bx = info.block_idx(0);
    int by = info.block_idx(1);
    int tx = info.thread_idx(0);
    int ty = info.thread_idx(1);
    const int BLOCK_SIZE = 16;
    int x = bx * BLOCK_SIZE + tx;
    int y = by * BLOCK_SIZE + ty;
    int ct = 0;
    int aBegin, aEnd, aStep, bBegin, bStep;
    aBegin = Aw * (by * BLOCK_SIZE);
    aStep = BLOCK_SIZE;
    aEnd = aBegin + Aw - 1;
    bBegin = (BLOCK_SIZE * bx);
    bStep = BLOCK_SIZE * Bw;
    float cSub = 0;
    for (a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        auto As = reinterpret_cast<float *>(info.shared_ptr());
        auto Bs = As + BLOCK_SIZE * BLOCK_SIZE;
        if (y < Ah && ct + tx < Aw)
            As[ty * BLOCK_SIZE + tx] = A[a + Aw * ty + tx];
        else
            As[ty * BLOCK_SIZE + tx] = 0;
        if (ct + ty < Aw && x < Bw)
            Bs[ty * BLOCK_SIZE + tx] = B[b + Bw * ty + tx];
        else
            Bs[ty * BLOCK_SIZE + tx] = 0;
        info.barrier();
        for (k = 0; k < BLOCK_SIZE; ++k)
            cSub += As[ty * BLOCK_SIZE + k] * Bs[k * BLOCK_SIZE + tx];
        ct += BLOCK_SIZE;
        info.barrier();
    }
    if (y < Ah && x < Bw)
        C[y * Bw + x] = cSub;
}

void matmul(const float *a, int ah, int aw, const float *b, int bw, float *c) {
    for (int i = 0; i < ah; i++) {
        for (int j = 0; j < bw; j++) {
            float sum = 0;
            for (int k = 0; k < aw; k++)
                sum += a[i * aw + k] * b[k * bw + j];
            c[i * bw + j] = sum;
        }
    }
}

int main() {
    auto l = GpuLauncher::GetInstance();
    const int ah = 1234;
    const int aw = 467;
    const int bw = 634;
    auto ref_a = new float[ah * aw];
    auto ref_b = new float[aw * bw];
    auto ref_c = new float[ah * bw];
    auto cc = new float[ah * bw];
    for (int i = 0; i < ah * aw; i++)
        ref_a[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    for (int i = 0; i < aw * bw; i++)
        ref_b[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    matmul(ref_a, ah, aw, ref_b, bw, ref_c);

    auto a = l->malloc<float>(ah * aw);
    auto b = l->malloc<float>(aw * bw);
    auto c = l->malloc<float>(ah * bw);
    l->memcpy((void *)a, (void *)ref_a, ah * aw * sizeof(float), GpuLauncher::Direction::H2D);
    l->memcpy((void *)b, (void *)ref_b, aw * bw * sizeof(float), GpuLauncher::Direction::H2D);
    l->stream_begin();
    l->submit(
        2 * 16 * 16 * sizeof(float), {bw / 16 + 1, ah / 16 + 1}, {16, 16},
        [=] DEVICE(KernelInfo & info) {
            matmul_kernel(info, a, ah, aw, b, bw, c);
        });
    l->stream_sync();
    l->stream_end();
    l->memcpy((void *)cc, (void *)c, ah * bw * sizeof(float), GpuLauncher::Direction::D2H);

    auto ret = all_close<float>(cc, ref_c, ah * bw);
    if (!ret) return 1;

    l->free(a);
    l->free(b);
    l->free(c);

    delete ref_a;
    delete ref_b;
    delete ref_c;
    delete cc;

    cout << "ok\n";
    return 0;
}
