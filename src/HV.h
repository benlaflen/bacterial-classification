#pragma once

#include <vector>
#include <cstdint>

constexpr size_t HV_SIZE = 100'000;

// 1 bit per dimension, packed
using HV = std::vector<uint64_t>;

HV empty_hv();

// number of 64-bit blocks needed
constexpr size_t HV_WORDS = (HV_SIZE + 63) / 64;

const HV& get_base_hv(char symbol);

void mult(const HV& l, const HV& r, HV& out, size_t k_bits);

using HV_16 = std::vector<uint16_t>;

HV_16 make_accumulator();

void superpose(HV_16 &l, const HV &r);

float cosine(const HV_16 &l, const HV_16 &r);