#pragma once
#include <vector>

#define KMER_SIZE       5
#define HV_DIM          10'000
#define SEQUENCE_LENGTH 125
#define SIM_STEP        1


#define HV_WORDS        ((HV_DIM + 63) / 64)
#define KMER_BITS       (2 * KMER_SIZE)
#define KMER_COUNT      (1 << KMER_BITS)
#define KMER_MASK       (KMER_COUNT - 1)
#define K               (SEQUENCE_LENGTH - KMER_SIZE)

using HV = std::vector<uint64_t>;
using HV_16 = std::vector<int16_t>;