#include "encoder.h"
#include <chrono>
#include <iostream>
#include <assert.h>


constexpr char ALPHABET[4] = {'A','C','G','T'};
template <typename F>
void for_each_nmer(int n, F&& fn) {
    const uint64_t total = 1ULL << (2 * n); // 4^n
    std::string s(n, 'A');

    for (uint64_t i = 0; i < total; ++i) {
        uint64_t x = i;
        for (int pos = n - 1; pos >= 0; --pos) {
            s[pos] = ALPHABET[x & 0b11];
            x >>= 2;
        }
        fn(s);
    }
}

HVCache::HVCache(size_t max_entries, size_t channel)
    : capacity(max_entries), channel(channel) {
        //Build k-mer map
        for_each_nmer(channel, [&, channel](const std::string& s) {
            HV binder = empty_hv();
            for (int it = 0; it < channel; ++it) {
                mult(binder, get_base_hv(s[it]), binder, it);
            }
            kmers.emplace(s, std::move(binder));
        });
    }

const HV_16& HVCache::get(const std::string& seq) {

    // Fast path: cache hit
    auto it = map.find(seq);
    if (it != map.end()) {
        // move to front of LRU
        lru.splice(lru.begin(), lru, it->second.lru_it);
        return it->second.hv;
    }

    // Cache miss: compute
    HV_16 hv = compute(seq);

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
    for (int index = 0; index < seq.size() - channel; ++index) {

        // Get binder for this window
        auto binder = kmers.find(seq.substr(index, index+channel));
        if(binder != kmers.end()) {
            binders[batch_count++] = binder->second;
        }
        // Flush batch
        if (batch_count == BATCH) {
            superpose_batch(hv, binders, batch_count);
            batch_count = 0;
        }
    }

    // Final partial batch
    if (batch_count > 0)
        superpose_batch(hv, binders, batch_count);

    return hv;
}

void HVCache::precompute(const std::string& seq) {
    HV_16 window = get(seq.substr(0,SEQUENCE_LENGTH));

    for(int index = 1; index < seq.size() - SEQUENCE_LENGTH; index++) {
        std::string key = seq.substr(index, SEQUENCE_LENGTH);
        // Fast path: cache hit
        auto it = map.find(key);
        if (it != map.end()) {
            // move to front of LRU
            lru.splice(lru.begin(), lru, it->second.lru_it);
            window = it->second.hv;
            continue;
        }

        // Cache miss: compute
        auto oldb = kmers.find(seq.substr(index-1, channel));
        assert(oldb != kmers.end());
        auto newb = kmers.find(seq.substr(index+SEQUENCE_LENGTH-(1+channel), channel));
        assert(newb != kmers.end());

        replace_hv(window, oldb->second, newb->second);

        // Evict if needed
        if (map.size() >= capacity) {
            const std::string& victim = lru.back();
            map.erase(victim);
            lru.pop_back();
        }

        // Insert new entry
        lru.push_front(key);
        auto [new_it, inserted] = map.emplace(
            key,
            Entry{HV_16(window), lru.begin()}
        );
    }
}

SIM HVCache::similarity(const std::string &l, const std::string &r) {
    SIM sim{0, -1.0f};
    for(int x = 0; x < r.size()-SEQUENCE_LENGTH; x ++ ) {
        HV_16 ref = get(r.substr(x,SEQUENCE_LENGTH));
        float new_score = cosine(get(l), ref);
        if(new_score > sim.score) {
            sim.score = new_score;
            sim.pos = x;
        }
    }
    return sim;
}