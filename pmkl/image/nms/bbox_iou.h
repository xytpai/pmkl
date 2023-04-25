#pragma once

#include "core.h"

namespace pmkl {
namespace image {
namespace nms {

DEVICE_INLINE float nms_max(float a, float b) {
    return a > b ? a : b;
}

DEVICE_INLINE float nms_min(float a, float b) {
    return a < b ? a : b;
}

DEVICE_INLINE float nms_iou(float const *const a, float const *const b) {
    float left = nms_max(a[0], b[0]), right = nms_min(a[2], b[2]);
    float top = nms_max(a[1], b[1]), bottom = nms_min(a[3], b[3]);
    float width = nms_max(right - left + 1, 0.f), height = nms_max(bottom - top + 1, 0.f);
    float interS = width * height;
    float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
    float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
    return interS / (Sa + Sb - interS);
}

constexpr int threads_per_block = sizeof(unsigned long long) * 8;

int get_bbox_iou_bitlength(int n_boxes) {
    return (n_boxes + threads_per_block - 1) / threads_per_block;
}

void bbox_iou(
    const float *bboxes, const int batch_size, const int n_boxes,
    const float nms_overlap_thresh,
    const int bitlength,
    unsigned long long *intersection_bit_matrix) {
    // bboxes: batch_size * n_boxes * 4
    // -> intersection_bit_matrix: batch_size * n_boxes * bitlength
    auto l = GpuLauncher::GetInstance();
    l->submit(
        threads_per_block * 4 * sizeof(float),
        {batch_size, bitlength, bitlength}, {threads_per_block},
        [=] DEVICE(KernelInfo & info) {
            int b = info.block_idx(0);
            int lid = info.thread_idx(0);
            int b_offset = b * n_boxes * 4;
            auto dev_boxes = bboxes + b_offset;
            auto dev_mask = intersection_bit_matrix + b * n_boxes * bitlength;
            int row_start = info.block_idx(1);
            int col_start = info.block_idx(2);
            int row_size =
                nms_min(n_boxes - row_start * threads_per_block, threads_per_block);
            int col_size =
                nms_min(n_boxes - col_start * threads_per_block, threads_per_block);
            auto block_boxes = reinterpret_cast<float *>(info.shared_ptr());
            if (lid < col_size) {
                block_boxes[lid * 4 + 0] =
                    dev_boxes[(threads_per_block * col_start + lid) * 4 + 0];
                block_boxes[lid * 4 + 1] =
                    dev_boxes[(threads_per_block * col_start + lid) * 4 + 1];
                block_boxes[lid * 4 + 2] =
                    dev_boxes[(threads_per_block * col_start + lid) * 4 + 2];
                block_boxes[lid * 4 + 3] =
                    dev_boxes[(threads_per_block * col_start + lid) * 4 + 3];
            }
            info.barrer();
            if (lid < row_size) {
                int cur_box_idx = threads_per_block * row_start + lid;
                float *cur_box = dev_boxes + cur_box_idx * 4;
                int i = 0;
                unsigned long long t = 0;
                int start = 0;
                if (row_start == col_start) {
                    start = lid + 1;
                }
                for (i = start; i < col_size; i++) {
                    if (nms_iou(cur_box, block_boxes + i * 4) > nms_overlap_thresh) {
                        t |= 1ULL << i;
                    }
                }
                dev_mask[cur_box_idx * bitlength + col_start] = t;
            }
        });
    if (l->is_sync_mode()) l->stream_sync();
}

}
}
} // namespace pmkl::image::nms