#pragma once

#include <vector>
#include <cstdint>

constexpr size_t HV_SIZE = 10'000;

// 1 bit per dimension, packed
using HV = std::vector<uint64_t>;

HV empty_hv();

// number of 64-bit blocks needed
constexpr size_t HV_WORDS = (HV_SIZE + 63) / 64;

const HV& get_base_hv(char symbol);

void mult(const HV& l, const HV& r, HV& out, size_t k_bits);

using HV_16 = std::vector<int16_t>;

HV_16 make_accumulator();

void superpose_batch(HV_16& acc, const HV* binders, size_t batch_size);


inline void replace_batch(
    HV_16& acc,
    const HV* const* olds,
    const HV* const* news,
    size_t batch_size
) {
    int16_t* __restrict ap = reinterpret_cast<int16_t*>(acc.data());

    const size_t full_words = HV_SIZE / 64;
    const size_t tail_bits  = HV_SIZE - full_words * 64;

    // byte -> 8 lanes of {0,2}
    static std::array<std::array<int16_t, 8>, 256> lut;
    static bool inited = false;
    if (!inited) {
        for (int v = 0; v < 256; ++v)
            for (int i = 0; i < 8; ++i)
                lut[v][i] = (v & (1 << i)) ? 2 : 0;
        inited = true;
    }

    // Full 64-bit words
    for (size_t w = 0; w < full_words; ++w) {
        int16_t* block = ap + w * 64;

        for (size_t b = 0; b < batch_size; ++b) {
            uint64_t nw = static_cast<uint64_t>((*news[b])[w]);
            uint64_t ow = static_cast<uint64_t>((*olds[b])[w]);

            // 8 bytes in a 64-bit word
            for (int byte_i = 0; byte_i < 8; ++byte_i) {
                const uint8_t nbyte = static_cast<uint8_t>(nw);
                const uint8_t obyte = static_cast<uint8_t>(ow);
                nw >>= 8;
                ow >>= 8;

                int16_t* dst = block + byte_i * 8;
                const auto& addv = lut[nbyte];
                const auto& subv = lut[obyte];

                // unrolled 8 lanes
                dst[0] += addv[0] - subv[0];
                dst[1] += addv[1] - subv[1];
                dst[2] += addv[2] - subv[2];
                dst[3] += addv[3] - subv[3];
                dst[4] += addv[4] - subv[4];
                dst[5] += addv[5] - subv[5];
                dst[6] += addv[6] - subv[6];
                dst[7] += addv[7] - subv[7];
            }
        }
    }

    // Tail (if HV_SIZE not multiple of 64)
    if (tail_bits) {
        int16_t* block = ap + full_words * 64;
        const uint64_t mask = (tail_bits == 64) ? ~0ull : ((1ull << tail_bits) - 1ull);

        for (size_t b = 0; b < batch_size; ++b) {
            uint64_t nw = static_cast<uint64_t>((*news[b])[full_words]) & mask;
            uint64_t ow = static_cast<uint64_t>((*olds[b])[full_words]) & mask;

            for (int byte_i = 0; byte_i < 8; ++byte_i) {
                if (static_cast<size_t>(byte_i * 8) >= tail_bits) break;

                const uint8_t nbyte = static_cast<uint8_t>(nw);
                const uint8_t obyte = static_cast<uint8_t>(ow);
                nw >>= 8;
                ow >>= 8;

                int16_t* dst = block + byte_i * 8;
                const auto& addv = lut[nbyte];
                const auto& subv = lut[obyte];

                dst[0] += addv[0] - subv[0];
                dst[1] += addv[1] - subv[1];
                dst[2] += addv[2] - subv[2];
                dst[3] += addv[3] - subv[3];
                dst[4] += addv[4] - subv[4];
                dst[5] += addv[5] - subv[5];
                dst[6] += addv[6] - subv[6];
                dst[7] += addv[7] - subv[7];
            }
        }
    }
}

float cosine(const HV_16 &l, const HV_16 &r);