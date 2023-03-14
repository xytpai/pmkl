#include "pmkl.h"
#include <iostream>

using namespace std;
using namespace pmkl;

int main() {
    cout << "testing tensor info\n";
    Tensor t;
    if (t.defined()) return 1;
    Tensor t2 = empty({1, 2, 3, 4}, ScalarType::Float, 0);
    if (!t2.defined()) return 1;
    if (t2.numel() != 1 * 2 * 3 * 4) return 1;
    if (t2.dim() != 4) return 1;
    int shape_[4] = {1, 2, 3, 4};
    int stride_[4] = {24, 12, 4, 1};
    for (int i = 0; i < t2.dim(); i++) {
        if (t2.stride(i) != stride_[i])
            return 1;
        if (t2.shape(i) != shape_[i])
            return 1;
    }
    if (t2.device() != 0) return 1;

    cout << "testing tensor storage\n";
    if (t2.storage_bytes() != 1 * 2 * 3 * 4 * element_size(ScalarType::Float)) return 1;
    if (t2.storage_bytes() != 1 * 2 * 3 * 4 * sizeof(float)) return 1;
    if (t2.data_ptr() == nullptr) return 1;
    Tensor t22 = t2;
    if (t2.storage_ref_count() != 2) return 1;
    auto st = t2.storage();
    if (t2.storage_ref_count() != t22.storage_ref_count()) return 1;
    if (t2.storage_ref_count() != 3) return 1;
    auto t3 = new Tensor(t22);
    auto t4 = new Tensor(*t3);
    if (t3->storage_ref_count() != 5) return 1;
    delete t3;
    if (t4->storage_ref_count() != 4) return 1;
    if (t4->storage().get() != t2.storage().get()) return 1;
    if (t4->storage().get() != st.get()) return 1;
    if (t2.storage_ref_count() != 4) return 1;
    {
        Tensor tt(t2);
        if (tt.storage_ref_count() != 5) return 1;
    }
    if (st.ref_count() != 4) return 1;
    Tensor temp;
    if (t2.storage_ref_count() != 4) return 1;
    temp = t2;
    if (t2.storage_ref_count() != 5) return 1;
    if (temp.storage_ref_count() != 5) return 1;

    {
        cout << "testing storage native swap\n";
        Tensor a;
        a = empty({1, 2, 3}, ScalarType::Double, 0);
        if (a.storage_ref_count() != 1) return 1;
        Tensor b = empty({1, 2, 5}, ScalarType::Double, 0);
        Tensor bb = b;
        if (bb.storage_ref_count() != 2) return 1;
        if (b.storage_ref_count() != 2) return 1;
        b = a;
        if (a.storage_ref_count() != 2) return 1;
        if (b.storage_ref_count() != 2) return 1;
        if (a.storage().get() != b.storage().get()) return 1;
        if (bb.storage_ref_count() != 1) return 1;
        if (bb.numel() != 1 * 2 * 5) return 1;
        if (b.numel() != 6) return 1;
        {
            Tensor a = empty({1, 2, 3}, ScalarType::Double, 0);
            Tensor b = empty({1, 2, 5}, ScalarType::Double, 0);
            Tensor aa = a;
            Tensor bb = b;
            {
                Tensor temp = aa;
                aa = bb;
                bb = temp;
            }
            if (aa.storage().get() != b.storage().get()) return 1;
            if (bb.storage().get() != a.storage().get()) return 1;
            if (aa.storage_ref_count() != 2) return 1;
            if (bb.storage_ref_count() != 2) return 1;
        }
        {
            Tensor a = empty({1, 2, 3}, ScalarType::Double, 0);
            Tensor b = empty({1, 2, 5}, ScalarType::Double, 0);
            Tensor aa = a;
            Tensor bb = b;
            std::swap(aa, bb);
            if (aa.storage().get() != b.storage().get()) return 1;
            if (bb.storage().get() != a.storage().get()) return 1;
            if (aa.storage_ref_count() != 2) return 1;
            if (bb.storage_ref_count() != 2) return 1;
        }
    }

    {
        cout << "testing std::move storage\n";
        Tensor a = empty({1, 2, 3}, ScalarType::Double, 0);
        Tensor aa = a;
        if (a.storage_ref_count() != 2) return 1;
        Tensor b = empty({1, 2, 3}, ScalarType::Double, 0);
        Tensor c(std::move(a));
        if (aa.storage_ref_count() != 2) return 1;
    }

    cout << "ok\n";
    return 0;
}
