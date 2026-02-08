#include "encoder.h"
#include <chrono>
#include <iostream>

HVCache::HVCache(size_t max_entries, std::vector<int> k_mer_channels)
    : capacity(max_entries), k_mer_channels(k_mer_channels) {}

const HV_16& HVCache::get(const std::string& seq) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    // Fast path: cache hit
    auto it = map.find(seq);
    if (it != map.end()) {
        // move to front of LRU
        lru.splice(lru.begin(), lru, it->second.lru_it);
        return it->second.hv;
    }

    // Cache miss: compute
    auto t1 = high_resolution_clock::now();
    HV_16 hv = compute(seq);
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << ms_int.count() << "ms\n";

    // Evict if needed
    if (map.size() >= capacity) {
        const std::string& victim = lru.back();
        map.erase(victim);
        lru.pop_back();
    }

    // Insert new entry
    lru.push_front(seq);
    auto [new_it, inserted] = map.emplace(
        seq,
        Entry{std::move(hv), lru.begin()}
    );

    return new_it->second.hv;
}

HV_16 HVCache::compute(const std::string& seq) {
    constexpr size_t BATCH = 16;

    HV_16 hv = make_accumulator();

    HV binders[BATCH];
    size_t batch_count = 0;

    HV binder = empty_hv();

    for (const int& channel : k_mer_channels) {
        for (int index = 0; index < seq.size() - channel; ++index) {

            // Build binder for this window (unchanged logic)
            for (int it = index; it < index + channel; ++it) {
                mult(binder, get_base_hv(seq[it]), binder, it - index);
            }

            binders[batch_count++] = binder;

            // Reset binder to identity (~0)
            std::fill(binder.begin(), binder.end(), ~uint64_t(0));
            constexpr size_t excess = HV_WORDS * 64 - HV_SIZE;
            if constexpr (excess > 0)
                binder.back() &= (~uint64_t(0)) >> excess;

            // Flush batch
            if (batch_count == BATCH) {
                superpose_batch(hv, binders, batch_count);
                batch_count = 0;
            }
        }
    }

    // Final partial batch
    if (batch_count > 0)
        superpose_batch(hv, binders, batch_count);

    return hv;
}