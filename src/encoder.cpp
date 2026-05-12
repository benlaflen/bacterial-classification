#include "encoder.h"
#include <chrono>
#include <iostream>
#include <assert.h>
#include <unordered_set>
#include <random>

constexpr char ALPHABET[4] = {'A','C','G','T'};

static inline char rc_base(char c) {
    switch (c) {
        case 'A': return 'T';
        case 'C': return 'G';
        case 'G': return 'C';
        case 'T': return 'A';
        default:  return 'N';
    }
}

static std::string revcomp(const std::string& s) {
    std::string r(s.size(), 'A');
    for (size_t i = 0; i < s.size(); ++i)
        r[s.size() - 1 - i] = rc_base(s[i]);
    return r;
}

static std::string canonical(const std::string& s) {
    std::string rc = revcomp(s);
    return (rc < s) ? rc : s;
}

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
    : capacity(max_entries), channel(channel)
{
    for_each_nmer(channel, [&](const std::string& s) {

        std::string rc = revcomp(s);
        std::string canon = (rc < s) ? rc : s;
        std::cout << "\nCompiling NMER " << s << std::flush;

        // If canonical already assigned, reuse it
        auto it = kmers.find(canon);
        if (it != kmers.end()) {
            kmers.emplace(s, it->second);
            return;
        }

        // First time seeing this canonical pair
        HV binder = empty_hv();

        for (int i = 0; i < channel; ++i)
            mult(binder, get_base_hv(canon[i]), binder, i);

        kmers.emplace(canon, binder);
        kmers.emplace(s, binder);

        if (rc != s)
            kmers.emplace(rc, binder);
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
    for (int index = 0; index + channel <= seq.size(); ++index) {

        auto binder_it = kmers.find(seq.substr(index, channel));
        if (binder_it == kmers.end())
            continue;

        const HV& kmerHV = binder_it->second;
        binders[batch_count++] = kmerHV;
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
    auto empty_vec = empty_hv();

    std::string_view seq_view(seq);

    // Precompute all kmer pointers once
    const size_t kmer_count = seq_len - channel + 1;
    std::vector<const HV*> kmer_ptrs(kmer_count);

    for (size_t i = 0; i < kmer_count; ++i)
    {
        std::string_view kmer(seq_view.data() + i, channel);
        auto it = kmers.find(std::string(kmer));
        if(it != kmers.end()) kmer_ptrs[i] = &it->second;
        else kmer_ptrs[i] = &empty_vec;
    }

    HV_16 window = get(std::string(seq_view.substr(0, std::min(seq_view.size(), static_cast<size_t>(SEQUENCE_LENGTH)))));

    for (size_t index = SIM_STEP;
         index+SEQUENCE_LENGTH < seq_len;
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
    if(SEQUENCE_LENGTH >= r.size()) {
        sim.score = cosine(get(l), get(r));
        return sim;
    }

    for(int x = 0; x+SEQUENCE_LENGTH < r.size(); x += SIM_STEP ) {
        HV_16 ref = get(r.substr(x,SEQUENCE_LENGTH));
        float new_score = cosine(get(l), ref);
    //    std::cout << "\n" << std::to_string(new_score);
        if(new_score > sim.score) {
            sim.score = new_score;
            sim.pos = x;
        }
    }
    return sim;
}