#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <limits>
#include <ranges>
#include <iomanip>
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
    // Split reads in half.
    // Input set  = first half,  one HV per read (first window).
    // DB set     = second half, ALL windows flattened — but we track which
    //              read each window came from so agreement is read-level.
    // -----------------------------------------------------------------------
    const int n_reads = (int)gpu_hvs.size();
    const int half    = n_reads / 2;
    if (half == 0) { std::cerr << "Need at least 2 reads — aborting.\n"; return 1; }

    // Inputs: first window of each first-half read
    std::vector<HV_16*> inputs;
    std::vector<int>    input_read_idx; // which original read each input came from
    inputs.reserve(half);
    for (int i : std::views::iota(0, half)) {
        if (gpu_hvs[i].empty()) {
            std::cerr << "Skipping read " << i << " (no windows)\n";
            continue;
        }
        inputs.push_back(&gpu_hvs[i][0]);
        input_read_idx.push_back(i);
    }

    // DB: all windows from second-half reads, with read provenance
    std::vector<HV_16*> db;
    std::vector<int>    db_read_idx; // db_read_idx[j] = original read index for db[j]
    for (int i : std::views::iota(half, n_reads))
        for (HV_16& win : gpu_hvs[i]) {
            db.push_back(&win);
            db_read_idx.push_back(i);
        }

    if (inputs.empty()) { std::cerr << "No valid input HVs.\n"; return 1; }
    if (db.empty())     { std::cerr << "No valid db HVs.\n";    return 1; }

    std::cout << "Input HVs : " << inputs.size() << " (first window of each first-half read)\n";
    std::cout << "DB HVs    : " << db.size() << " (all windows from " << (n_reads - half) << " second-half reads)\n\n";

    // -----------------------------------------------------------------------
    // GPU k=1
    // -----------------------------------------------------------------------
    std::cout << "=== GPU cosine_search k=1 ===\n";
    auto tg1_start = std::chrono::high_resolution_clock::now();
    auto gpu1 = cosine_search(inputs, db, 1);
    auto tg1_end = std::chrono::high_resolution_clock::now();
    double gpu1_ms = std::chrono::duration<double, std::milli>(tg1_end - tg1_start).count();
    std::cout << "Time: " << gpu1_ms << " ms  (" << gpu1_ms / inputs.size() << " ms/input)\n\n";

    // -----------------------------------------------------------------------
    // GPU k=10
    // -----------------------------------------------------------------------
    constexpr int K_TOP = 10;
    std::cout << "=== GPU cosine_search k=" << K_TOP << " ===\n";
    auto tg10_start = std::chrono::high_resolution_clock::now();
    auto gpu10 = cosine_search(inputs, db, K_TOP);
    auto tg10_end = std::chrono::high_resolution_clock::now();
    double gpu10_ms = std::chrono::duration<double, std::milli>(tg10_end - tg10_start).count();
    std::cout << "Time: " << gpu10_ms << " ms  (" << gpu10_ms / inputs.size() << " ms/input)\n\n";

    // -----------------------------------------------------------------------
    // CPU argmax — brute force, best window match, reported at read level
    // -----------------------------------------------------------------------
    std::cout << "=== CPU cosine argmax ===\n";
    std::vector<int>   cpu_best_db_idx(inputs.size(), -1);    // best db window index
    std::vector<float> cpu_best_score (inputs.size(), std::numeric_limits<float>::lowest());

    auto tc_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < (int)inputs.size(); i++) {
        const HV_16& q = *inputs[i];
        float best = std::numeric_limits<float>::lowest();
        int   best_j = -1;
        for (int j = 0; j < (int)db.size(); j++) {
            float s = cosine(q, *db[j]);
            if (s > best) { best = s; best_j = j; }
        }
        cpu_best_db_idx[i] = best_j;
        cpu_best_score[i]  = best;
    }
    auto tc_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(tc_end - tc_start).count();
    std::cout << "Time: " << cpu_ms << " ms  (" << cpu_ms / inputs.size() << " ms/input)\n\n";

    // -----------------------------------------------------------------------
    // Agreement check — read-level, tie-aware.
    //
    // "Agree" means: the result from method A appears somewhere in the
    // tied-top block of method B (scores within 1e-5 of the best).
    // This handles the common case where many windows score identically
    // and the two methods just happen to pick different tied winners.
    // -----------------------------------------------------------------------
    std::cout << "=== Agreement check (read-level, tie-aware) ===\n\n";

    // Reverse map: db pointer → db index
    std::unordered_map<HV_16*, int> db_ptr_to_idx;
    db_ptr_to_idx.reserve(db.size());
    for (int j = 0; j < (int)db.size(); j++) db_ptr_to_idx[db[j]] = j;

    // Returns true if `read` appears anywhere in the tied-top block of `matches`
    auto read_in_tied_top = [&](int read, const std::vector<std::pair<float, HV_16*>>& matches) -> bool {
        if (matches.empty()) return false;
        float top_score = matches[0].first;
        for (const auto& [score, ptr] : matches) {
            if (score < top_score - 1e-5f) break;
            if (db_read_idx[db_ptr_to_idx.at(ptr)] == read) return true;
        }
        return false;
    };

    int mismatches_1_vs_cpu  = 0;
    int mismatches_10_vs_cpu = 0;
    int mismatches_1_vs_10   = 0;

    std::cout << std::left
              << std::setw(6)  << "Input"
              << std::setw(10) << "GPU-k1"
              << std::setw(10) << "GPU-k10"
              << std::setw(10) << "CPU"
              << std::setw(10) << "k1==CPU"
              << std::setw(11) << "k10==CPU"
              << "  GPU-k1 score | GPU-k10 score | CPU score\n";
    std::cout << std::string(82, '-') << "\n";

    for (int i = 0; i < (int)inputs.size(); i++) {
        // GPU k=1
        HV_16* ptr_gpu1    = gpu1.at(inputs[i])[0].second;
        float  scr_gpu1    = gpu1.at(inputs[i])[0].first;
        int    read_gpu1   = db_read_idx[db_ptr_to_idx.at(ptr_gpu1)];

        // GPU k=10 top entry
        HV_16* ptr_gpu10   = gpu10.at(inputs[i])[0].second;
        float  scr_gpu10   = gpu10.at(inputs[i])[0].first;
        int    read_gpu10  = db_read_idx[db_ptr_to_idx.at(ptr_gpu10)];

        // CPU
        int    read_cpu    = db_read_idx[cpu_best_db_idx[i]];
        float  scr_cpu     = cpu_best_score[i];

        // Tie-aware agreement: k=1 agrees with k=10 if the k=1 read is
        // anywhere in the k=10 tied-top block, and vice versa.
        bool agree_1_10   = read_in_tied_top(read_gpu1,  gpu10.at(inputs[i]))
                         || read_in_tied_top(read_gpu10, gpu1.at(inputs[i]));
        // For CPU vs GPU: CPU is a single result; check if CPU's read
        // appears in GPU k=10's tied top, and if GPU k=1's read matches CPU.
        bool agree_1_cpu  = (read_gpu1 == read_cpu)
                         || read_in_tied_top(read_cpu, gpu10.at(inputs[i]));
        bool agree_10_cpu = read_in_tied_top(read_cpu, gpu10.at(inputs[i]));

        if (!agree_1_cpu)  mismatches_1_vs_cpu++;
        if (!agree_10_cpu) mismatches_10_vs_cpu++;
        if (!agree_1_10)   mismatches_1_vs_10++;

        std::cout << std::left
                  << std::setw(6)  << i
                  << std::setw(10) << read_gpu1
                  << std::setw(10) << read_gpu10
                  << std::setw(10) << read_cpu
                  << std::setw(10) << (agree_1_cpu  ? "Y" : "N")
                  << std::setw(11) << (agree_10_cpu ? "Y" : "N")
                  << "  " << std::fixed << std::setprecision(4)
                  << scr_gpu1 << " | " << scr_gpu10 << " | " << scr_cpu;

        if (!agree_1_10)
            std::cout << "  <-- genuine k=1 vs k=10 mismatch";

        std::cout << "\n";

        // On any genuine mismatch dump the ranked lists so we can diagnose
        if (!agree_1_cpu || !agree_10_cpu || !agree_1_10) {
            std::cout << "    [GPU k=10 top results for input " << i << ":]\n";
            for (int r = 0; r < (int)gpu10.at(inputs[i]).size(); r++) {
                const auto& m = gpu10.at(inputs[i])[r];
                int win_j  = db_ptr_to_idx.at(m.second);
                std::cout << "      rank " << r << ": read=" << db_read_idx[win_j]
                          << " win=" << win_j << " score=" << m.first << "\n";
            }
            std::cout << "    [CPU top-5 for input " << i << ":]\n";
            std::vector<std::pair<float,int>> cpu_ranked;
            cpu_ranked.reserve(db.size());
            for (int j = 0; j < (int)db.size(); j++)
                cpu_ranked.push_back({cosine(*inputs[i], *db[j]), j});
            std::partial_sort(cpu_ranked.begin(), cpu_ranked.begin() + 5, cpu_ranked.end(),
                              [](auto& a, auto& b){ return a.first > b.first; });
            for (int r = 0; r < 5; r++) {
                int win_j = cpu_ranked[r].second;
                std::cout << "      rank " << r << ": read=" << db_read_idx[win_j]
                          << " win=" << win_j << " score=" << cpu_ranked[r].first << "\n";
            }
        }
    }

    std::cout << "\n";
    auto report = [&](const char* label, int mm) {
        if (mm == 0) std::cout << label << ": ALL AGREE\n";
        else         std::cout << label << ": " << mm << " / " << inputs.size() << " mismatches\n";
    };
    report("GPU k=1  vs CPU      (read-level, tie-aware)", mismatches_1_vs_cpu);
    report("GPU k=10 vs CPU      (read-level, tie-aware)", mismatches_10_vs_cpu);
    report("GPU k=1  vs GPU k=10 (read-level, tie-aware)", mismatches_1_vs_10);

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    std::cout << "\n=== Summary ===\n";
    std::cout << "Encode     : " << encode_ms << " ms\n";
    std::cout << "GPU k=1    : " << gpu1_ms   << " ms  (" << gpu1_ms  / inputs.size() << " ms/input)\n";
    std::cout << "GPU k=10   : " << gpu10_ms  << " ms  (" << gpu10_ms / inputs.size() << " ms/input)\n";
    std::cout << "CPU argmax : " << cpu_ms    << " ms  (" << cpu_ms   / inputs.size() << " ms/input)\n";
    std::cout << "Speedup k=1  vs CPU: " << (cpu_ms / gpu1_ms)  << "x\n";
    std::cout << "Speedup k=10 vs CPU: " << (cpu_ms / gpu10_ms) << "x\n";

    return 0;
}