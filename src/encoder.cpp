#include "encoder.h"
#include <chrono>
#include <iostream>
#include <assert.h>

#define SIM_STEP 15

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
        auto binder = kmers.find(seq.substr(index, channel));
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

void HVCache::precompute(const std::string& seq)
{
    const size_t seq_len = seq.size();
    if (seq_len < SEQUENCE_LENGTH) return;

    std::string_view seq_view(seq);

    // Precompute all kmer pointers once
    const size_t kmer_count = seq_len - channel + 1;
    std::vector<const HV*> kmer_ptrs(kmer_count);

    for (size_t i = 0; i < kmer_count; ++i)
    {
        std::string_view kmer(seq_view.data() + i, channel);
        kmer_ptrs[i] = &kmers.find(std::string(kmer))->second;
    }

    HV_16 window = get(std::string(seq_view.substr(0, SEQUENCE_LENGTH)));

    for (size_t index = SIM_STEP;
         index < seq_len - SEQUENCE_LENGTH;
         index += SIM_STEP)
    {
        std::string_view key_view(seq_view.data() + index, SEQUENCE_LENGTH);

        auto it = map.find(std::string(key_view));
        if (it != map.end())
        {
            lru.splice(lru.begin(), lru, it->second.lru_it);
            window = it->second.hv;
            continue;
        }

        const HV* oldb[SIM_STEP];
        const HV* newb[SIM_STEP];

        for (size_t x = 0; x < SIM_STEP; ++x)
        {
            oldb[x] = kmer_ptrs[index + x - SIM_STEP];
            newb[x] = kmer_ptrs[index + SEQUENCE_LENGTH + x - (SIM_STEP + channel)];
        }

        replace_batch(window, oldb, newb, SIM_STEP);

        if (map.size() >= capacity)
        {
            const std::string& victim = lru.back();
            map.erase(victim);
            lru.pop_back();
        }

        std::string key(key_view);

        lru.push_front(key);
        map.emplace(
            key,
            Entry{HV_16(window), lru.begin()}
        );
    }
}

SIM HVCache::similarity(const std::string &l, const std::string &r) {
    SIM sim{0, -1.0f};
    for(int x = 0; x < r.size()-SEQUENCE_LENGTH; x += SIM_STEP ) {
        HV_16 ref = get(r.substr(x,SEQUENCE_LENGTH));
        float new_score = cosine(get(l), ref);
        if(new_score > sim.score) {
            sim.score = new_score;
            sim.pos = x;
        }
    }
    return sim;
}