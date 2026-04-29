#pragma once
#include "config.h"

std::vector<std::vector<HV_16>> encode(const std::vector<int16_t>& kmer_map, const std::vector<std::string>& sequences);
