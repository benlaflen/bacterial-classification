#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

#include "GPU.cuh"
#include "encoder.h"
#include "HV.h"

// ---------------------------------------------------------------------------
// FASTA loader
// ---------------------------------------------------------------------------
static std::vector<std::string> load_fasta(const std::string& path) {
    std::vector<std::string> seqs;
    std::ifstream f(path);
    if (!f) {
        std::cerr << "ERROR: cannot open " << path << "\n";
        return seqs;
    }

    std::string line, current;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '>') {
            if (!current.empty()) {
                seqs.push_back(std::move(current));
                current.clear();
            }
        } else {
            current += line;
        }
    }
    if (!current.empty()) seqs.push_back(std::move(current));

    return seqs;
}

// ---------------------------------------------------------------------------
// Build the kmer_map needed by the GPU encoder (same logic as your GPU test)
// ---------------------------------------------------------------------------
static std::vector<int16_t> build_kmer_map() {
    HVCache cache(10000);
    std::vector<int16_t> kmer_map(KMER_COUNT * HV_DIM, 0);

    const char bases[] = {'A', 'C', 'T', 'G'};

    for (int kmer_idx = 0; kmer_idx < KMER_COUNT; kmer_idx++) {
        std::string kmer(KMER_SIZE, 'A');
        for (int pos = 0; pos < KMER_SIZE; pos++)
            kmer[pos] = bases[(kmer_idx >> (pos * 2)) & 0x3];

        auto it = cache.kmers.find(kmer);
        if (it == cache.kmers.end()) continue;

        const HV& hv = it->second;
        for (int d = 0; d < HV_DIM; d++)
            kmer_map[kmer_idx * HV_DIM + d] = (hv[d / 64] >> (d % 64)) & 1;
    }

    return kmer_map;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    const std::string fasta_path =
        (argc > 1) ? argv[1] : "/home/zebra/blaflen/bacteria/bacterial-classification/EXAMPLE_READS.fasta";

    std::cout << "Loading reads from: " << fasta_path << "\n";
    auto reads = load_fasta(fasta_path);
    if (reads.empty()) {
        std::cerr << "No reads loaded — aborting.\n";
        return 1;
    }
    std::cout << "Loaded " << reads.size() << " reads.\n\n";

    // -----------------------------------------------------------------------
    // Method 1: GPU batch encode
    // -----------------------------------------------------------------------
    std::cout << "=== Method 1: GPU batch encode ===\n";
    std::cout << "Building kmer_map... " << std::flush;
    auto kmer_map = build_kmer_map();
    std::cout << "done.\n";

    std::cout << "Encoding all " << reads.size() << " reads in one batch... " << std::flush;
    auto t0 = std::chrono::high_resolution_clock::now();

    auto gpu_hvs = encode(kmer_map, reads);   // pass ALL reads at once

    auto t1 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "done.\n";
    std::cout << "GPU encode time : " << gpu_ms << " ms"
              << "  (" << (gpu_ms / reads.size()) << " ms/read)\n\n";

    // -----------------------------------------------------------------------
    // Method 2: HVCache precompute + similarity (CPU path)
    // Encodes every read via precompute so each read's HV is materialised.
    // We time only the encoding (precompute) step, mirroring the GPU side.
    // -----------------------------------------------------------------------
    std::cout << "=== Method 2: HVCache precompute (CPU) ===\n";
    std::cout << "Precomputing HVs for all reads... " << std::flush;

    HVCache cache(reads.size() + 1024);

    auto t2 = std::chrono::high_resolution_clock::now();

    for (const auto& seq : reads) {
        auto t3 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        std::cout << "\nPrecomputing (current time: " << cpu_ms << "ms)";
        cache.precompute(seq);
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::cout << "done.\n";
    std::cout << "CPU precompute time : " << cpu_ms << " ms"
              << "  (" << (cpu_ms / reads.size()) << " ms/read)\n\n";

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    std::cout << "=== Summary ===\n";
    std::cout << "Reads          : " << reads.size()      << "\n";
    std::cout << "GPU total      : " << gpu_ms            << " ms\n";
    std::cout << "CPU total      : " << cpu_ms            << " ms\n";
    std::cout << "Speedup (CPU/GPU ratio): " << (cpu_ms / gpu_ms) << "x\n";

    return 0;
}