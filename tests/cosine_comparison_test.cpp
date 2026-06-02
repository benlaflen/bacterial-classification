#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <limits>
#include <ranges>
#include "GPU.cuh"
#include "encoder.h"
#include "HV.h"

// ---------------------------------------------------------------------------
// FASTA loader
// ---------------------------------------------------------------------------
static std::vector<std::string> load_fasta(const std::string& path) {
    std::vector<std::string> seqs;
    std::ifstream f(path);
    if (!f) { std::cerr << "ERROR: cannot open " << path << "\n"; return seqs; }
    std::string line, current;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '>') {
            if (!current.empty()) { seqs.push_back(std::move(current)); current.clear(); }
        } else {
            current += line;
        }
    }
    if (!current.empty()) seqs.push_back(std::move(current));
    return seqs;
}

// ---------------------------------------------------------------------------
// Build the kmer_map needed by the GPU encoder
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
// CPU cosine — raw dot product, no norm division (matches existing signature).
// Define COSINE_NORMED to switch to true cosine similarity.
// ---------------------------------------------------------------------------
static float cosine(const HV_16& l, const HV_16& r) {
#ifdef COSINE_NORMED
    int64_t dot = 0, sq_l = 0, sq_r = 0;
    for (size_t i = 0; i < l.size(); i++) {
        dot  += (int64_t)l[i] * r[i];
        sq_l += (int64_t)l[i] * l[i];
        sq_r += (int64_t)r[i] * r[i];
    }
    return (float)dot / (std::sqrt((float)sq_l) * std::sqrt((float)sq_r));
#else
    int64_t dot = 0;
    for (size_t i = 0; i < l.size(); i++)
        dot += (int64_t)l[i] * r[i];
    return (float)dot;
#endif
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    const std::string fasta_path =
        (argc > 1) ? argv[1] : "/home/zebra/blaflen/bacteria/bacterial-classification/EXAMPLE_READS.fasta";

    std::cout << "Loading reads from: " << fasta_path << "\n";
    auto reads = load_fasta(fasta_path);
    if (reads.empty()) { std::cerr << "No reads loaded — aborting.\n"; return 1; }
    std::cout << "Loaded " << reads.size() << " reads.\n\n";

    // -----------------------------------------------------------------------
    // GPU encode all reads
    // -----------------------------------------------------------------------
    std::cout << "Building kmer_map... " << std::flush;
    auto kmer_map = build_kmer_map();
    std::cout << "done.\n";

    std::cout << "Encoding all " << reads.size() << " reads... " << std::flush;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto gpu_hvs = encode(kmer_map, reads);
    auto t1 = std::chrono::high_resolution_clock::now();
    double encode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "done (" << encode_ms << " ms).\n\n";

    // -----------------------------------------------------------------------
    // Split reads in half.  Input set = first half, output set = second half.
    // -----------------------------------------------------------------------
    const int n_reads   = (int)gpu_hvs.size();
    const int half      = n_reads / 2;

    if (half == 0) { std::cerr << "Need at least 2 reads — aborting.\n"; return 1; }

    // Inputs: first window only from each read in the first half
    std::vector<HV_16*> inputs;
    inputs.reserve(half);
    for (int i : std::views::iota(0, half)) {
        if (gpu_hvs[i].empty()) { std::cerr << "Read " << i << " has no windows.\n"; return 1; }
        inputs.push_back(&gpu_hvs[i][0]);
    }

    // DB: all windows from every read in the second half, flattened
    std::vector<HV_16*> db;
    for (int i : std::views::iota(half, n_reads))
        for (HV_16& win : gpu_hvs[i])
            db.push_back(&win);

    std::cout << "Input HVs  : " << inputs.size() << " (first window of each read in first half)\n";
    std::cout << "DB HVs     : " << db.size()     << " (all windows from second-half reads)\n\n";

    // -----------------------------------------------------------------------
    // GPU k=1
    // -----------------------------------------------------------------------
    std::cout << "=== GPU cosine_search k=1 ===\n";
    auto tg1_start = std::chrono::high_resolution_clock::now();
    auto gpu1 = cosine_search(inputs, db, 1);
    auto tg1_end = std::chrono::high_resolution_clock::now();
    double gpu1_ms = std::chrono::duration<double, std::milli>(tg1_end - tg1_start).count();
    std::cout << "Time: " << gpu1_ms << " ms"
              << "  (" << gpu1_ms / inputs.size() << " ms/input)\n\n";

    // -----------------------------------------------------------------------
    // GPU k=10
    // -----------------------------------------------------------------------
    constexpr int K_TOP = 10;
    std::cout << "=== GPU cosine_search k=" << K_TOP << " ===\n";
    auto tg10_start = std::chrono::high_resolution_clock::now();
    auto gpu10 = cosine_search(inputs, db, K_TOP);
    auto tg10_end = std::chrono::high_resolution_clock::now();
    double gpu10_ms = std::chrono::duration<double, std::milli>(tg10_end - tg10_start).count();
    std::cout << "Time: " << gpu10_ms << " ms"
              << "  (" << gpu10_ms / inputs.size() << " ms/input)\n\n";

    // -----------------------------------------------------------------------
    // CPU argmax — brute force dot product over all db HVs for each input
    // -----------------------------------------------------------------------
    std::cout << "=== CPU cosine argmax ===\n";
    std::vector<int> cpu_best_idx(inputs.size(), -1);
    std::vector<float> cpu_best_score(inputs.size(), std::numeric_limits<float>::lowest());

    auto tc_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < (int)inputs.size(); i++) {
        const HV_16& q = *inputs[i];
        float best = std::numeric_limits<float>::lowest();
        int   best_j = -1;
        for (int j = 0; j < (int)db.size(); j++) {
            float s = cosine(q, *db[j]);
            if (s > best) { best = s; best_j = j; }
        }
        cpu_best_idx[i]   = best_j;
        cpu_best_score[i] = best;
    }
    auto tc_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(tc_end - tc_start).count();
    std::cout << "Time: " << cpu_ms << " ms"
              << "  (" << cpu_ms / inputs.size() << " ms/input)\n\n";

    // -----------------------------------------------------------------------
    // Agreement check (outside timing)
    //
    // For each input we compare:
    //   - GPU k=1 best db index
    //   - GPU k=10 best db index (top of the ranked list)
    //   - CPU best db index
    //
    // We recover the db index from the GPU results by scanning db[] for the
    // returned pointer.  This is O(num_inputs * db_size) but it's outside
    // timing so that's fine.  We build a reverse map for O(1) lookup instead.
    // -----------------------------------------------------------------------
    std::cout << "=== Agreement check ===\n";
    std::unordered_map<HV_16*, int> db_ptr_to_idx;
    db_ptr_to_idx.reserve(db.size());
    for (int j = 0; j < (int)db.size(); j++) db_ptr_to_idx[db[j]] = j;

    int mismatches_1_vs_cpu  = 0;
    int mismatches_10_vs_cpu = 0;
    int mismatches_1_vs_10   = 0;

    for (int i = 0; i < (int)inputs.size(); i++) {
        HV_16* ptr_gpu1  = gpu1.at(inputs[i])[0].second;
        HV_16* ptr_gpu10 = gpu10.at(inputs[i])[0].second;
        int idx_gpu1  = db_ptr_to_idx.at(ptr_gpu1);
        int idx_gpu10 = db_ptr_to_idx.at(ptr_gpu10);
        int idx_cpu   = cpu_best_idx[i];

        if (idx_gpu1  != idx_cpu)   mismatches_1_vs_cpu++;
        if (idx_gpu10 != idx_cpu)   mismatches_10_vs_cpu++;
        if (idx_gpu1  != idx_gpu10) mismatches_1_vs_10++;
    }

    auto report = [&](const char* label, int mm) {
        if (mm == 0)
            std::cout << label << ": ALL AGREE\n";
        else
            std::cout << label << ": " << mm << " / " << inputs.size() << " mismatches\n";
    };
    report("GPU k=1  vs CPU    ", mismatches_1_vs_cpu);
    report("GPU k=10 vs CPU    ", mismatches_10_vs_cpu);
    report("GPU k=1  vs GPU k=10", mismatches_1_vs_10);

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    std::cout << "\n=== Summary ===\n";
    std::cout << "Encode          : " << encode_ms << " ms\n";
    std::cout << "GPU k=1         : " << gpu1_ms   << " ms  (" << gpu1_ms  / inputs.size() << " ms/input)\n";
    std::cout << "GPU k=10        : " << gpu10_ms  << " ms  (" << gpu10_ms / inputs.size() << " ms/input)\n";
    std::cout << "CPU argmax      : " << cpu_ms    << " ms  (" << cpu_ms   / inputs.size() << " ms/input)\n";
    std::cout << "Speedup k=1  vs CPU: " << (cpu_ms / gpu1_ms)  << "x\n";
    std::cout << "Speedup k=10 vs CPU: " << (cpu_ms / gpu10_ms) << "x\n";

    return 0;
}