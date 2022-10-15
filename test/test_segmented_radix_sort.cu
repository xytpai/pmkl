#include "pmkl.h"
#include "segmented_radix_sort.h"
#include <iostream>
#include <algorithm>
#include <stdlib.h>

using namespace pmkl;
using namespace pmkl::utils;
using namespace std;

int main() {
    auto l = GpuLauncher::GetInstance();

    for (int it = 0; it < 40; it++) {
        using key_t = float;
        using value_t = int;
        int num_segments = utils::host::randint_scalar(10, 50);
        int num_elements = utils::host::randint_scalar(10, 4096);
        bool is_descending = utils::host::randint_scalar(0, 2) > 0;
        cout << "testing sort num_segments[" << num_segments
             << "] num_elements[" << num_elements << "] is_descending[" << is_descending << "]\n";
        int total_size = num_segments * num_elements;
        auto key = new key_t[total_size];
        auto value = new value_t[total_size];
        auto key_out = new key_t[total_size];
        auto value_out = new value_t[total_size];
        utils::host::fill_rand<key_t>(key, total_size, -10000.0, 10000.0);
        for (int i = 0; i < num_segments; i++) {
            for (int j = 0; j < num_elements; j++) {
                value[i * num_elements + j] = j;
            }
        }
        auto key_dev = l->malloc<key_t>(total_size);
        auto value_dev = l->malloc<value_t>(total_size);
        l->memcpy((void *)key_dev, (void *)key, total_size * sizeof(key_t), GpuLauncher::Direction::H2D);
        l->memcpy((void *)value_dev, (void *)value, total_size * sizeof(value_t), GpuLauncher::Direction::H2D);
        l->stream_begin();
        sorting::segmented_sort_pairs(key_dev, key_dev, value_dev, value_dev, num_segments, num_elements, is_descending);
        l->stream_sync();
        l->stream_end();
        l->memcpy((void *)key_out, (void *)key_dev, total_size * sizeof(key_t), GpuLauncher::Direction::D2H);
        l->memcpy((void *)value_out, (void *)value_dev, total_size * sizeof(value_t), GpuLauncher::Direction::D2H);
        for (int i = 0; i < num_segments; i++) {
            auto key_begin = key + i * num_elements;
            auto value_begin = value + i * num_elements;
            using pair_t = std::pair<key_t, value_t>;
            vector<pair_t> v;
            for (int j = 0; j < num_elements; j++)
                v.push_back(make_pair(key_begin[j], value_begin[j]));
            if (is_descending)
                std::stable_sort(v.begin(), v.end(), [](pair_t a, pair_t b) { return a.first > b.first; });
            else
                std::stable_sort(v.begin(), v.end(), [](pair_t a, pair_t b) { return a.first < b.first; });
            for (int j = 0; j < num_elements; j++) {
                key_begin[j] = v[j].first;
                value_begin[j] = v[j].second;
            }
        }
        cout << "testing key...\n";
        if (!all_close(key_out, key, total_size))
            return 1;
        cout << "testing value...\n";
        if (!all_close(value_out, value, total_size))
            return 1;
        delete[] key;
        delete[] value;
        delete[] key_out;
        delete[] value_out;
        l->free(key_dev);
        l->free(value_dev);
    }

    cout << "ok" << endl;
    return 0;
}
