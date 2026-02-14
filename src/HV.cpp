#include "HV.h"
#include <unordered_map>
#include <random>
#include <stdexcept>
#include <assert.h>

//#define NORM_COSINE

HV empty_hv() {
    HV hv(HV_WORDS, ~uint64_t(0));
    // mask off unused bits in the last word
    constexpr size_t excess = HV_WORDS * 64 - HV_SIZE;
    if constexpr (excess > 0) {
        hv.back() &= (~uint64_t(0)) >> excess;
    }

    return hv;
}

HV make_hv(int seed) {
    HV hv(HV_WORDS);

    std::mt19937_64 rng(seed);

    for (size_t i = 0; i < HV_WORDS; ++i)
        hv[i] = rng();

    // mask off unused bits in the last word
    constexpr size_t excess = HV_WORDS * 64 - HV_SIZE;
    if constexpr (excess > 0) {
        hv.back() &= (~uint64_t(0)) >> excess;
    }

    return hv;
}

const HV base_A = make_hv(0);
const HV base_C = make_hv(1);
const HV base_T = make_hv(2);
const HV base_G = make_hv(3);
const HV base_N = empty_hv();

const HV& get_base_hv(char symbol) {
    switch (symbol) {
        case 'A': return base_A;
        case 'C': return base_C;
        case 'T': return base_T;
        case 'G': return base_G;
        default: return base_N;
    }
}

/*void mult(
    const HV& l,
    const HV& r,
    HV& out,
    size_t k_bits
) {
    const size_t word_shift = k_bits >> 6;   // /64
    const size_t bit_shift  = k_bits & 63;   // %64

    for (size_t i = 0; i < HV_WORDS; ++i) {
        // r contributes bits from two adjacent words
        size_t ri = (i + word_shift) % HV_WORDS;
        size_t rj = (ri + 1) % HV_WORDS;

        uint64_t r_word;
        if (bit_shift == 0) {
            r_word = r[ri];
        } else {
            r_word =
                (r[ri] >> bit_shift) |
                (r[rj] << (64 - bit_shift));
        }

        out[i] = l[i] ^ r_word;
    }

    // mask tail bits
    constexpr size_t excess = HV_WORDS * 64 - HV_SIZE;
    if constexpr (excess > 0)
        out.back() &= (~uint64_t(0)) >> excess;
}

void superpose(HV_16& acc, const HV& r) {
    size_t bit = 0;

    for (size_t w = 0; w < HV_WORDS; ++w) {
        uint64_t word = r[w];
        for (int b = 0; b < 64 && bit < HV_SIZE; ++b, ++bit) {
            acc[bit] += (word & 1) ? 1.0f : -1.0f;
            word >>= 1;
        }
    }
}*/

// ------------------------ mult (optimized) ------------------------
void mult(const HV& l, const HV& r, HV& out, size_t k_bits) {
    const size_t word_shift = (k_bits >> 6) % HV_WORDS; // keep bounded
    const uint32_t bit_shift = static_cast<uint32_t>(k_bits & 63);

    const uint64_t* __restrict lp = l.data();
    const uint64_t* __restrict rp = r.data();
    uint64_t*       __restrict op = out.data();

    if (bit_shift == 0) {
        // Pure word rotate: out[i] = l[i] ^ r[(i+word_shift) mod HV_WORDS]
        size_t i = 0;

        // First linear span: from word_shift..end
        for (size_t j = word_shift; j < HV_WORDS; ++j, ++i) {
            op[i] = lp[i] ^ rp[j];
        }
        // Wrap once: 0..word_shift-1
        for (size_t j = 0; j < word_shift; ++j, ++i) {
            op[i] = lp[i] ^ rp[j];
        }
    } else {
        // Rotate by word_shift and then shift right by bit_shift (with wrap carry).
        // For each output word i:
        //   r_word = (r[idx] >> s) | (r[next] << (64-s))
        // where idx advances with a single wrap.
        const uint32_t s  = bit_shift;
        const uint32_t ls = 64u - s;

        size_t idx = word_shift;
        uint64_t cur = rp[idx];
        uint64_t next = rp[(idx + 1 < HV_WORDS) ? (idx + 1) : 0];

        for (size_t i = 0; i < HV_WORDS; ++i) {
            const uint64_t r_word = (cur >> s) | (next << ls);
            op[i] = lp[i] ^ r_word;

            // advance idx with at most one wrap
            idx++;
            if (idx == HV_WORDS) idx = 0;
            cur = next;
            next = rp[(idx + 1 < HV_WORDS) ? (idx + 1) : 0];
        }
    }

    // Mask tail bits in last word (same as you had).
    constexpr size_t excess = HV_WORDS * 64 - HV_SIZE;
    if constexpr (excess > 0) {
        op[HV_WORDS - 1] &= (~uint64_t(0)) >> excess;
    }
}

// ------------------------ accumulator init ------------------------
HV_16 make_accumulator() {
    return HV_16(HV_SIZE, 0);
}

// Accumulate a batch of binders into acc.
// Semantics: identical to calling superpose(acc, binders[i]) for each i.
void superpose_batch(
    HV_16& acc,
    const HV* binders,
    size_t batch_size
) {
    int16_t* __restrict ap = acc.data();

    size_t bit_base = 0;
    const size_t full_words = HV_SIZE / 64;

    for (size_t w = 0; w < full_words; ++w) {
        int16_t* block = ap + bit_base;

        // Baseline: each binder contributes -1 per bit
        const int16_t baseline = static_cast<int16_t>(-batch_size);
        for (int i = 0; i < 64; ++i)
            block[i] += baseline;

        // Count set bits across the batch
        uint8_t counts[64] = {};  // small, hot, stack-allocated

        for (size_t b = 0; b < batch_size; ++b) {
            uint64_t word = binders[b][w];
            while (word) {
                unsigned bit = __builtin_ctzll(word);
                counts[bit]++;
                word &= word - 1;
            }
        }

        // Each set bit adds +2 per occurrence
        for (int i = 0; i < 64; ++i)
            block[i] += static_cast<int16_t>(counts[i] * 2);

        bit_base += 64;
    }

    // Tail
    const size_t tail = HV_SIZE - bit_base;
    if (tail) {
        int16_t* block = ap + bit_base;

        const int16_t baseline = static_cast<int16_t>(-batch_size);
        for (size_t i = 0; i < tail; ++i)
            block[i] += baseline;

        uint8_t counts[64] = {};
        for (size_t b = 0; b < batch_size; ++b) {
            uint64_t word = binders[b][full_words];
            while (word) {
                unsigned bit = __builtin_ctzll(word);
                if (bit >= tail) break;
                counts[bit]++;
                word &= word - 1;
            }
        }

        for (size_t i = 0; i < tail; ++i)
            block[i] += static_cast<int16_t>(counts[i] * 2);
    }
}


#ifdef NORM_COSINE
float cosine(const HV_16& l, const HV_16& r) {
    int64_t dot = 0;
    int64_t norm_l = 0;
    int64_t norm_r = 0;

    const size_t n = l.size();  // should be HV_SIZE

    for (size_t i = 0; i < n; ++i) {
        int64_t li = l[i];
        int64_t ri = r[i];

        dot     += li * ri;
        norm_l  += li * li;
        norm_r  += ri * ri;
    }

    if (norm_l == 0 || norm_r == 0)
        return -1.0f;

    return static_cast<float>(
        2.0 * ((dot / (std::sqrt((double)norm_l) * std::sqrt((double)norm_r))) - 0.5f)
    );
}
#else
float cosine(const HV_16& l, const HV_16& r) {
    int64_t dot = 0;
    const size_t n = l.size();

    for (size_t i = 0; i < n; ++i)
        dot += (int64_t)(l[i] * r[i]);

    return dot;
}
#endif