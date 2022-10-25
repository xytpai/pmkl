#include "pmkl.h"
#include "segmented_radix_select.h"
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <unordered_set>
#include <unordered_map>

using namespace pmkl;
using namespace pmkl::utils;
using namespace std;

int main() {
    auto l = GpuLauncher::GetInstance();

    cout << "testing segmented_select_pairs\n"
         << endl;
    for (int it = 0; it < 40; it++) {
        using key_t = float;
        using value_t = int;
        int num_segments = utils::host::randint_scalar(10, 50);
        int num_elements = utils::host::randint_scalar(10, 14096);
        int num_topk = utils::host::randint_scalar(10, 256);
        num_topk = num_topk > num_elements ? num_elements : num_topk;
        if (it == 10) num_topk = num_elements = 128;
        bool is_descending = utils::host::randint_scalar(0, 2) > 0;
        cout << "testing select pairs num_segments[" << num_segments
             << "] num_elements[" << num_elements << "] is_descending["
             << is_descending << "] num_topk[" << num_topk << "]\n";
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
        sorting::segmented_select_pairs(key_dev, key_dev, value_dev, value_dev, num_segments, num_elements, num_topk, is_descending);
        l->stream_sync();
        l->stream_end();
        l->memcpy((void *)key_out, (void *)key_dev, total_size * sizeof(key_t), GpuLauncher::Direction::D2H);
        l->memcpy((void *)value_out, (void *)value_dev, total_size * sizeof(value_t), GpuLauncher::Direction::D2H);

        vector<unordered_set<value_t>> indices_set_list;
        vector<unordered_map<value_t, key_t>> map_list;

        for (int i = 0; i < num_segments; i++) {
            unordered_set<value_t> indices_set;
            unordered_map<value_t, key_t> map;
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
            for (int j = 0; j < num_topk; j++) {
                indices_set.insert(v[j].second);
                map[v[j].second] = v[j].first;
            }
            indices_set_list.emplace_back(indices_set);
            map_list.emplace_back(map);
        }
        cout << "testing keys and indices...\n";
        vector<unordered_set<value_t>> out_indices_set_list;
        vector<unordered_map<value_t, key_t>> out_map_list;
        for (int i = 0; i < num_segments; i++) {
            unordered_set<value_t> out_indices_set;
            unordered_map<value_t, key_t> out_map;
            auto key_begin = key_out + i * num_elements;
            auto value_begin = value_out + i * num_elements;
            for (int m = 0; m < num_topk; m++) {
                out_indices_set.insert(value_begin[m]);
                out_map[value_begin[m]] = key_begin[m];
                if (indices_set_list[i].find(value_begin[m]) == indices_set_list[i].end()) {
                    cout << "output index " << value_begin[m] << " incorrect\n";
                    return 1;
                }
                if (fabs(map_list[i][value_begin[m]] - key_begin[m]) > 0.0001) {
                    cout << "output key " << key_begin[m] << " incorrect\n";
                    return 1;
                }
            }
            out_indices_set_list.emplace_back(out_indices_set);
            out_map_list.emplace_back(out_map);
        }
        for (int i = 0; i < num_segments; i++) {
            if (indices_set_list[i] != out_indices_set_list[i]) {
                cout << "err: indices_set_list[i] != out_indices_set_list[i]\n";
                return 1;
            }
        }
        delete[] key;
        delete[] value;
        delete[] key_out;
        delete[] value_out;
        l->free(key_dev);
        l->free(value_dev);
    }

    cout << "testing segmented_threshold_select_pairs\n"
         << endl;
    for (int it = 0; it < 40; it++) {
        using key_t = float;
        using value_t = int;
        int num_segments = utils::host::randint_scalar(10, 50);
        int num_elements = utils::host::randint_scalar(10, 14096);
        int num_topk = utils::host::randint_scalar(10, 256);
        num_topk = num_topk > num_elements ? num_elements : num_topk;
        if (it == 10) num_topk = num_elements = 128;
        bool is_descending = true;
        cout << "testing threshold select pairs num_segments[" << num_segments
             << "] num_elements[" << num_elements << "] is_descending["
             << is_descending << "] num_topk[" << num_topk << "]\n";
        int total_size = num_segments * num_elements;
        auto key = new key_t[total_size];
        auto value = new value_t[total_size];
        auto key_out = new key_t[total_size];
        auto value_out = new value_t[total_size];
        auto valid_out = new int[num_segments];
        utils::host::fill_rand<key_t>(key, total_size, -10000.0, 10000.0);
        for (int i = 0; i < num_segments; i++) {
            for (int j = 0; j < num_elements; j++) {
                value[i * num_elements + j] = j;
            }
        }
        auto key_dev = l->malloc<key_t>(total_size);
        auto value_dev = l->malloc<value_t>(total_size);
        auto valid_dev = l->malloc<int>(num_segments);
        l->memcpy((void *)key_dev, (void *)key, total_size * sizeof(key_t), GpuLauncher::Direction::H2D);
        l->memcpy((void *)value_dev, (void *)value, total_size * sizeof(value_t), GpuLauncher::Direction::H2D);
        l->stream_begin();
        sorting::segmented_threshold_select_pairs<key_t, value_t>(
            key_dev, key_dev, value_dev, value_dev, valid_dev, num_segments, num_elements, num_topk, 5000.0, is_descending);
        l->stream_sync();
        l->stream_end();
        l->memcpy((void *)key_out, (void *)key_dev, total_size * sizeof(key_t), GpuLauncher::Direction::D2H);
        l->memcpy((void *)value_out, (void *)value_dev, total_size * sizeof(value_t), GpuLauncher::Direction::D2H);
        l->memcpy((void *)valid_out, (void *)valid_dev, num_segments * sizeof(int), GpuLauncher::Direction::D2H);

        vector<unordered_set<value_t>> indices_set_list;
        vector<unordered_map<value_t, key_t>> map_list;
        int *valid_ref = new int[num_segments];
        for (int ii = 0; ii < num_segments; ii++) valid_ref[ii] = 0;

        for (int i = 0; i < num_segments; i++) {
            unordered_set<value_t> indices_set;
            unordered_map<value_t, key_t> map;
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
            for (int j = 0; j < num_topk; j++) {
                if (v[j].first >= 5000) {
                    indices_set.insert(v[j].second);
                    map[v[j].second] = v[j].first;
                    valid_ref[i]++;
                }
            }
            indices_set_list.emplace_back(indices_set);
            map_list.emplace_back(map);
        }
        cout << "testing keys and indices...\n";
        vector<unordered_set<value_t>> out_indices_set_list;
        vector<unordered_map<value_t, key_t>> out_map_list;
        for (int i = 0; i < num_segments; i++) {
            unordered_set<value_t> out_indices_set;
            unordered_map<value_t, key_t> out_map;
            auto key_begin = key_out + i * num_elements;
            auto value_begin = value_out + i * num_elements;
            if (valid_out[i] != valid_ref[i]) {
                cout << "valid count error\n";
                cout << "valid_out:" << valid_out[i] << ", valid_ref:" << valid_ref[i] << endl;
                for (int ii = 0; ii < 10; ii++)
                    cout << key_begin[ii] << ", ";
                cout << endl;
                return 1;
            }
            for (int m = 0; m < valid_out[i]; m++) {
                out_indices_set.insert(value_begin[m]);
                out_map[value_begin[m]] = key_begin[m];
                if (indices_set_list[i].find(value_begin[m]) == indices_set_list[i].end()) {
                    cout << "output index " << value_begin[m] << " incorrect\n";
                    return 1;
                }
                if (fabs(map_list[i][value_begin[m]] - key_begin[m]) > 0.0001) {
                    cout << "output key " << key_begin[m] << " incorrect\n";
                    return 1;
                }
                if (key_begin[m] < 5000) {
                    cout << "output key " << key_begin[m] << " not valid\n";
                    return 1;
                }
            }
            out_indices_set_list.emplace_back(out_indices_set);
            out_map_list.emplace_back(out_map);
        }
        for (int i = 0; i < num_segments; i++) {
            if (indices_set_list[i] != out_indices_set_list[i]) {
                cout << "err: indices_set_list[i] != out_indices_set_list[i]\n";
                return 1;
            }
        }
        delete[] key;
        delete[] value;
        delete[] key_out;
        delete[] value_out;
        delete[] valid_out;
        delete[] valid_ref;
        l->free(key_dev);
        l->free(value_dev);
        l->free(valid_dev);
    }

    cout << "ok" << endl;
    return 0;
}
