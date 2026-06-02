#pragma once
#include "config.h"
#include <string>
#include <unordered_map>

std::vector<std::vector<HV_16>> encode(const std::vector<int16_t>& kmer_map, const std::vector<std::string>& sequences);

std::unordered_map<HV_16*, std::vector<std::pair<float, HV_16*>>> cosine_search(
    const std::vector<HV_16*>& inputs,
    const std::vector<HV_16*>& db,
    int k = 1);