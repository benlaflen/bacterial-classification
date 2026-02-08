#include "HV.h"
#include <unordered_map>
#include <random>
#include <stdexcept>

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

const HV& get_base_hv(char symbol) {
    switch (symbol) {
        case 'A': return base_A;
        case 'C': return base_C;
        case 'T': return base_T;
        case 'G': return base_G;
        default: throw std::runtime_error("Illegal base to encode: " + std::to_string(symbol));
    }
}

void mult(
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

HV_16 make_accumulator() {
    return HV_16(HV_SIZE, 0);
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
        return 0.0f;

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