#include "GPU.cuh"
#include <stdint.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <iostream>
#include <cassert>

#define GPU_MEMORY_HEADROOM 0.8

// ---------------------------------------------------------------------------
// Device helper: extract a 2-bit-packed k-mer starting at base position `pos`
// from a packed sequence buffer.
//
// Packing convention: base at position i occupies bits [2i+1 : 2i] of the
// uint32 word at seq_packed[i / 16], i.e. 16 bases per uint32 word.
//
// Returns a KMER_BITS-wide index directly usable as kmer_map row index.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t extract_kmer(const uint32_t* __restrict__ seq_packed,
                                                  int pos)
{
    // Bit offset of the first base of this k-mer within the flat bit stream
    int bit_offset = pos * 2;

    // Which 32-bit word the k-mer starts in
    int word_idx = bit_offset / 32;

    // The k-mer may straddle two words (e.g. last 2 bits of word N, first 8 of word N+1).
    // Load both words and combine into a 64-bit value to handle straddling cleanly.
    uint32_t lo = seq_packed[word_idx];
    uint32_t hi = seq_packed[word_idx + 1];    // safe: caller ensures sequence has padding

    // Shift the combined 64-bit value right to align the k-mer at bit 0, then mask
    uint64_t combined = ((uint64_t)hi << 32) | lo;
    return (uint32_t)((combined >> bit_offset) & KMER_MASK);
}


// ---------------------------------------------------------------------------
// Main encode kernel
//
// seq_packed:      flat buffer of all sequences, 2-bit packed, concatenated.
//                  Indexed in units of uint32 words (16 bases per word).
//                  Must have at least 1 word of zero-padding at the end of
//                  each sequence to allow safe straddled k-mer reads.
//
// seq_offsets:     length num_seqs+1. seq_offsets[i] is the base-unit (not
//                  word-unit) offset of sequence i into seq_packed.
//                  seq_offsets[i+1] - seq_offsets[i] = length of sequence i.
//
// kmer_map:        shape [KMER_COUNT][HV_DIM], int16, row-major.
//                  Values are 0/1 unpacked from bit-packed binary HVs (not true -1/+1).
//                  The 2*acc - K correction in the kernel converts to true {-1,+1} sums.
//
// window_offsets:  length num_seqs+1. window_offsets[i] is the index of the
//                  first output HV for sequence i in the outputs buffer.
//                  Precomputed on CPU as:
//                    window_offsets[0] = 0
//                    window_offsets[i+1] = window_offsets[i]
//                                        + (seq_len[i] - SEQUENCE_LENGTH) / SIM_STEP + 1
//
// outputs:         flat buffer, shape [total_windows][HV_DIM], int16, row-major.
//                  Caller allocates: total_windows * HV_DIM * sizeof(int16_t).
//
// num_seqs:        number of sequences in this batch.
// ---------------------------------------------------------------------------
__global__ void hdc_encode(
    const uint32_t* __restrict__ seq_packed,
    const int*      __restrict__ seq_offsets,
    const int16_t*  __restrict__ kmer_map,
    const int*      __restrict__ window_offsets,
          int16_t*               outputs,
    int num_seqs)
{
    // --- Decode indices ---
    int seq_idx = blockIdx.x;
    int win_idx = blockIdx.y;
    int dim     = blockIdx.z * HV_BLOCK_SIZE + threadIdx.x;
    if (dim >= HV_DIM) return;

    int seq_base_offset = seq_offsets[seq_idx];         // base-unit start of this sequence
    int seq_len         = seq_offsets[seq_idx + 1] - seq_base_offset;
    int num_windows     = (seq_len - SEQUENCE_LENGTH) / SIM_STEP + 1;

    // Bounds check: grid is rectangular but sequences have variable window counts.
    // Threads beyond this sequence's actual window count exit immediately.
    // This waste is negligible when sequence lengths are similar (see design notes).
    if (win_idx >= num_windows) return;

    // --- Compute window start position in the sequence (in base units) ---
    int window_start = win_idx * SIM_STEP;

    // --- Accumulate HV for this window, for dimension `dim` ---
    // kmer_map stores 0/1 int16 values (unpacked from bit-packed binary HVs).
    // We accumulate into int32 to avoid overflow: max value is K * 1 = 120, which
    // fits in int16, but int32 keeps the arithmetic clean during the loop.
    int32_t acc = 0;

    // Pointer to the packed sequence data for this sequence.
    // seq_base_offset is in base units; convert to word units for the pointer.
    const uint32_t* seq = seq_packed + (seq_base_offset / 16);
    // Note: ensure sequences are word-aligned in your packing (seq_base_offset % 16 == 0)
    // for best memory access performance. extract_kmer is correct either way.

    for (int x = 0; x < K; x++)
    {
        // Extract the k-mer index at position (window_start + x)
        uint32_t kmer_idx = extract_kmer(seq, window_start + x);

        // Look up this thread's dimension in the k-mer's HV row.
        // kmer_map layout: row-major [KMER_COUNT][HV_DIM], so row = kmer_idx * HV_DIM
        acc += kmer_map[kmer_idx * HV_DIM + dim];
    }

    // --- Apply the {0,1} → {-1,+1} correction ---
    // acc is a sum of K values each in {0,1}. The true {-1,+1} sum is 2*acc - K.
    // K is a compile-time constant so this is a single multiply-add instruction.
    int32_t true_sum = 2 * acc - K;

    // --- Write result to output buffer ---
    // Output layout: flat row-major [total_windows][HV_DIM]
    // window_offsets[seq_idx] gives the global window index for window 0 of this sequence.
    int out_idx = (window_offsets[seq_idx] + win_idx) * HV_DIM + dim;
    outputs[out_idx] = (int16_t)true_sum;
}


// ---------------------------------------------------------------------------
// Host-side launcher (example — integrate into your pipeline as needed)
//
// Assumes all device buffers are already allocated and copied.
// Returns the CUDA error status so the caller can check.
// ---------------------------------------------------------------------------
cudaError_t launch_hdc_encode(
    const uint32_t* d_seq_packed,
    const int*      d_seq_offsets,
    const int16_t*  d_kmer_map,
    const int*      d_window_offsets,
          int16_t*  d_outputs,
    int num_seqs,
    int max_windows,        // max windows across all sequences in this batch
    cudaStream_t stream)    // pass 0 for default stream
{
    // Block: one thread per HV dimension.
    // HV_DIM must be <= 1024 (CUDA max threads per block).
    // For large HV_DIM, consider splitting into multiple threadIdx dimensions.
    dim3 block(HV_BLOCK_SIZE, 1, 1);

    // Grid: one block per (sequence, window) pair.
    // Excess window blocks for short sequences exit immediately via bounds check.
    dim3 grid(num_seqs, max_windows, HV_BLOCKS);

    hdc_encode<<<grid, block, 0, stream>>>(
        d_seq_packed,
        d_seq_offsets,
        d_kmer_map,
        d_window_offsets,
        d_outputs,
        num_seqs
    );

    return cudaGetLastError();
}



static std::vector<uint32_t> pack_sequence(const std::string& seq)
{
    int num_words = (seq.size() + 15) / 16 + 1; // +1 for safety padding
    std::vector<uint32_t> packed(num_words, 0);
    for (int i = 0; i < (int)seq.size(); i++) {
        uint32_t bits;
        switch (seq[i]) {
            case 'A': case 'a': bits = 0; break;
            case 'C': case 'c': bits = 1; break;
            case 'G': case 'g': bits = 2; break;
            case 'T': case 't': bits = 3; break;
            default:            bits = 0; break;
        }
        packed[i / 16] |= (bits << ((i % 16) * 2));
    }
    return packed;
}

static size_t get_memory_limit() {
    static size_t limit = 0;
    if (limit == 0) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, 0);
        limit = (size_t)(props.totalGlobalMem * GPU_MEMORY_HEADROOM);
    }
    return limit;
}

std::vector<std::vector<HV_16>> encode(const std::vector<int16_t>& kmer_map, const std::vector<std::string>& sequences) {
    fprintf(stderr, "[encode] START: %zu sequences\n", sequences.size());
    for (size_t i = 0; i < sequences.size(); i++)
        fprintf(stderr, "[encode]   seq[%zu] len=%zu\n", i, sequences[i].size());

    static size_t memory_limit = get_memory_limit();
    fprintf(stderr, "[encode] memory_limit = %zu bytes (%.2f GB)\n", memory_limit, memory_limit / (1024.0*1024.0*1024.0));

    std::vector<std::vector<HV_16>> output(sequences.size());
    int16_t* d_kmer_map = nullptr;
    cudaStream_t     stream            = nullptr;
    int16_t*         d_outputs         = nullptr;
    uint32_t*        d_seq_packed      = nullptr;
    int*             d_seq_offsets_dev = nullptr;
    int*             d_win_offsets_dev = nullptr;
    int              gpu_batch_start   = 0;
    std::vector<int> gpu_win_offsets;
    int              gpu_total_windows = 0;
    int              batch_start   = 0;
    int              batch_end     = 0;
    int              batch_max_len = 0;
    bool             batch_complete = false;
    int              batch_index   = 0;
    std::vector<int>      seq_offsets    = {0};
    std::vector<int>      window_offsets = {0};
    std::vector<uint32_t> seq_packed_host;

    fprintf(stderr, "[encode] Allocating d_kmer_map: KMER_COUNT=%d HV_DIM=%d => %zu bytes\n",
            KMER_COUNT, HV_DIM, (size_t)KMER_COUNT * HV_DIM * sizeof(int16_t));
    cudaError_t e = cudaMalloc(&d_kmer_map, KMER_COUNT * HV_DIM * sizeof(int16_t));
    fprintf(stderr, "[encode] cudaMalloc d_kmer_map: %s\n", cudaGetErrorString(e));
    e = cudaMemcpy(d_kmer_map, kmer_map.data(), KMER_COUNT * HV_DIM * sizeof(int16_t), cudaMemcpyHostToDevice);
    fprintf(stderr, "[encode] cudaMemcpy kmer_map H->D: %s\n", cudaGetErrorString(e));

    fprintf(stderr, "[encode] Entering main while loop (batch_start=%d)\n", batch_start);
    while (batch_start < (int)sequences.size()) {
        fprintf(stderr, "[encode] --- Top of main loop: batch_start=%d batch_end=%d batch_index=%d batch_complete=%d stream=%p ---\n",
                batch_start, batch_end, batch_index, (int)batch_complete, (void*)stream);

        // --- Assemble + pack next batch while GPU runs ---
        fprintf(stderr, "[encode]   [assemble] Entering assembly loop\n");
        while (!batch_complete && (!stream || cudaStreamQuery(stream) != cudaSuccess)) {
            if (batch_end >= (int)sequences.size()) {
                fprintf(stderr, "[encode]   [assemble] batch_end=%d >= sequences.size()=%zu => batch_complete=true\n",
                        batch_end, sequences.size());
                batch_complete = true; break;
            }
            int candidate_max     = std::max(batch_max_len, (int)sequences[batch_end].size());
            int candidate_windows = (candidate_max - SEQUENCE_LENGTH) / SIM_STEP + 1;
            size_t candidate_mem  = (size_t)(batch_end - batch_start + 1)
                                  * candidate_windows * HV_DIM * sizeof(int16_t);
            fprintf(stderr, "\r[encode]   [assemble] seq[%d] len=%zu | candidate_max=%d candidate_windows=%d candidate_mem=%zu (%.2f MB) memory_limit=%zu (%.2f MB)    ",
                    batch_end, sequences[batch_end].size(),
                    candidate_max, candidate_windows,
                    candidate_mem, candidate_mem / (1024.0*1024.0),
                    memory_limit, memory_limit / (1024.0*1024.0));
            if (candidate_mem > memory_limit) {
                fprintf(stderr, "\n[encode]   [assemble] Memory limit exceeded => batch_complete=true\n");
                batch_complete = true; break;
            }
            const std::string& seq = sequences[batch_end];
            int wins = ((int)seq.size() - SEQUENCE_LENGTH) / SIM_STEP + 1;
            seq_offsets.push_back(seq_offsets.back() + (int)seq.size());
            window_offsets.push_back(window_offsets.back() + wins);
            std::vector<uint32_t> packed = pack_sequence(seq);
            seq_packed_host.insert(seq_packed_host.end(), packed.begin(), packed.end());
            fprintf(stderr, "\n[encode]   [assemble] Added seq[%d]: wins=%d seq_offsets.back()=%d window_offsets.back()=%d packed_words=%zu seq_packed_host.size()=%zu\n",
                    batch_end, wins,
                    seq_offsets.back(), window_offsets.back(),
                    packed.size(), seq_packed_host.size());
            batch_max_len = candidate_max;
            batch_end++;
        }
        fprintf(stderr, "[encode]   [assemble] Exited assembly loop: batch_end=%d batch_complete=%d batch_max_len=%d\n",
                batch_end, (int)batch_complete, batch_max_len);
        fprintf(stderr, "[encode]   [assemble] seq_offsets(%zu): ", seq_offsets.size());
        for (int x : seq_offsets) fprintf(stderr, "%d ", x);
        fprintf(stderr, "\n");
        fprintf(stderr, "[encode]   [assemble] window_offsets(%zu): ", window_offsets.size());
        for (int x : window_offsets) fprintf(stderr, "%d ", x);
        fprintf(stderr, "\n");

        // --- Sync previous batch and collect results ---
        if (stream) {
            fprintf(stderr, "[encode]   [sync] Synchronizing stream for batch starting at gpu_batch_start=%d\n", gpu_batch_start);
            fprintf(stderr, "[encode]   [sync] gpu_total_windows=%d gpu_win_offsets.size()=%zu\n",
                    gpu_total_windows, gpu_win_offsets.size());
            fprintf(stderr, "[encode]   [sync] gpu_win_offsets: ");
            for (int x : gpu_win_offsets) fprintf(stderr, "%d ", x);
            fprintf(stderr, "\n");

            cudaError_t sync_err = cudaStreamSynchronize(stream);
            fprintf(stderr, "[encode]   [sync] cudaStreamSynchronize: %s\n", cudaGetErrorString(sync_err));

            size_t flat_bytes = (size_t)gpu_total_windows * HV_DIM * sizeof(int16_t);
            fprintf(stderr, "[encode]   [sync] Allocating flat buffer: %d windows * %d dims * 2 bytes = %zu bytes (%.2f MB)\n",
                    gpu_total_windows, HV_DIM, flat_bytes, flat_bytes / (1024.0*1024.0));
            std::vector<int16_t> flat(gpu_total_windows * HV_DIM);
            cudaError_t cpy_err = cudaMemcpy(flat.data(), d_outputs, flat_bytes, cudaMemcpyDeviceToHost);
            fprintf(stderr, "[encode]   [sync] cudaMemcpy D->H flat: %s\n", cudaGetErrorString(cpy_err));

            fprintf(stderr, "[encode]   [sync] Unpacking %zu result segments\n", gpu_win_offsets.size() - 1);
            for (int i = 0; i < (int)gpu_win_offsets.size() - 1; i++) {
                int wins = gpu_win_offsets[i + 1] - gpu_win_offsets[i];
                size_t flat_start_base = (size_t)gpu_win_offsets[i] * HV_DIM;
                fprintf(stderr, "\r[encode]   [sync] output[%d]: wins=%d flat_start_base=%zu    ",
                        gpu_batch_start + i, wins, flat_start_base);
                output[gpu_batch_start + i].resize(wins);
                for (int w = 0; w < wins; w++) {
                    size_t flat_start = ((size_t)gpu_win_offsets[i] + w) * HV_DIM;
                    output[gpu_batch_start + i][w] = HV_16(
                        flat.begin() + flat_start,
                        flat.begin() + flat_start + HV_DIM);
                }
            }
            fprintf(stderr, "\n[encode]   [sync] Unpack complete\n");

            fprintf(stderr, "[encode]   [sync] Freeing GPU resources for previous batch\n");
            fprintf(stderr, "[encode]   [sync] cudaFree(d_seq_packed):      %s\n", cudaGetErrorString(cudaFree(d_seq_packed)));
            fprintf(stderr, "[encode]   [sync] cudaFree(d_seq_offsets_dev): %s\n", cudaGetErrorString(cudaFree(d_seq_offsets_dev)));
            fprintf(stderr, "[encode]   [sync] cudaFree(d_win_offsets_dev): %s\n", cudaGetErrorString(cudaFree(d_win_offsets_dev)));
            fprintf(stderr, "[encode]   [sync] cudaFree(d_outputs):         %s\n", cudaGetErrorString(cudaFree(d_outputs)));
            fprintf(stderr, "[encode]   [sync] cudaStreamDestroy:           %s\n", cudaGetErrorString(cudaStreamDestroy(stream)));
            stream = nullptr;
            fprintf(stderr, "[encode]   [sync] Done collecting results for gpu_batch_start=%d\n", gpu_batch_start);
        }

        // --- Dispatch current batch ---
        if (batch_complete) {
            int batch_size    = batch_end - batch_start;
            int max_windows   = (batch_max_len - SEQUENCE_LENGTH) / SIM_STEP + 1;
            int total_windows = window_offsets.back();

            fprintf(stderr, "[encode]   [dispatch] Batch %d: batch_start=%d batch_end=%d batch_size=%d\n",
                    batch_index, batch_start, batch_end, batch_size);
            fprintf(stderr, "[encode]   [dispatch] batch_max_len=%d max_windows=%d total_windows=%d\n",
                    batch_max_len, max_windows, total_windows);
            fprintf(stderr, "[encode]   [dispatch] seq_packed_host.size()=%zu (%zu bytes)\n",
                    seq_packed_host.size(), seq_packed_host.size() * sizeof(uint32_t));
            fprintf(stderr, "[encode]   [dispatch] seq_offsets.size()=%zu window_offsets.size()=%zu\n",
                    seq_offsets.size(), window_offsets.size());
            fprintf(stderr, "[encode]   [dispatch] d_outputs alloc: %d * %d * 2 = %zu bytes (%.2f MB)\n",
                    total_windows, HV_DIM,
                    (size_t)total_windows * HV_DIM * sizeof(int16_t),
                    (size_t)total_windows * HV_DIM * sizeof(int16_t) / (1024.0*1024.0));

            auto cuda_check = [&](cudaError_t e, const char* what) {
                fprintf(stderr, "[encode]   [dispatch] %s: %s\n", what, cudaGetErrorString(e));
                if (e != cudaSuccess)
                    throw std::runtime_error(
                        std::string("encode batch ") + std::to_string(batch_index) +
                        " (" + std::to_string(batch_size) + " seqs): " +
                        what + ": " + cudaGetErrorString(e));
            };
            cuda_check(cudaStreamCreate(&stream),                                                        "cudaStreamCreate");
            cuda_check(cudaMalloc(&d_seq_packed,      seq_packed_host.size() * sizeof(uint32_t)),        "cudaMalloc seq_packed");
            cuda_check(cudaMalloc(&d_seq_offsets_dev, seq_offsets.size()     * sizeof(int)),             "cudaMalloc seq_offsets");
            cuda_check(cudaMalloc(&d_win_offsets_dev, window_offsets.size()  * sizeof(int)),             "cudaMalloc win_offsets");
            cuda_check(cudaMalloc(&d_outputs,         (size_t)total_windows  * HV_DIM * sizeof(int16_t)),"cudaMalloc outputs");
            cuda_check(cudaMemcpyAsync(d_seq_packed,      seq_packed_host.data(),  seq_packed_host.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), "memcpyAsync seq_packed");
            cuda_check(cudaMemcpyAsync(d_seq_offsets_dev, seq_offsets.data(),      seq_offsets.size()     * sizeof(int),      cudaMemcpyHostToDevice, stream), "memcpyAsync seq_offsets");
            cuda_check(cudaMemcpyAsync(d_win_offsets_dev, window_offsets.data(),   window_offsets.size()  * sizeof(int),      cudaMemcpyHostToDevice, stream), "memcpyAsync win_offsets");
            cuda_check(launch_hdc_encode(d_seq_packed, d_seq_offsets_dev, d_kmer_map,
                d_win_offsets_dev, d_outputs, batch_size, max_windows, stream),                          "kernel launch");

            fprintf(stderr, "[encode]   [dispatch] Kernel launched. Resetting batch state.\n");
            gpu_batch_start   = batch_start;
            gpu_win_offsets   = window_offsets;
            gpu_total_windows = total_windows;
            fprintf(stderr, "[encode]   [dispatch] gpu_batch_start=%d gpu_total_windows=%d gpu_win_offsets.size()=%zu\n",
                    gpu_batch_start, gpu_total_windows, gpu_win_offsets.size());

            batch_start   = batch_end;
            batch_max_len = 0;
            batch_complete = false;
            seq_offsets    = {0};
            window_offsets = {0};
            seq_packed_host.clear();
            batch_index++;
            fprintf(stderr, "[encode]   [dispatch] Next batch_start=%d batch_index now=%d\n", batch_start, batch_index);
        }
    }

    fprintf(stderr, "[encode] Exited main loop. batch_index=%d stream=%p\n", batch_index, (void*)stream);

    if (stream) {
        cudaStreamSynchronize(stream);
        size_t flat_bytes = (size_t)gpu_total_windows * HV_DIM * sizeof(int16_t);
        std::vector<int16_t> flat(gpu_total_windows * HV_DIM);
        cudaMemcpy(flat.data(), d_outputs, flat_bytes, cudaMemcpyDeviceToHost);
        for (int i = 0; i < (int)gpu_win_offsets.size() - 1; i++) {
            int wins = gpu_win_offsets[i + 1] - gpu_win_offsets[i];
            output[gpu_batch_start + i].resize(wins);
            for (int w = 0; w < wins; w++) {
                size_t flat_start = ((size_t)gpu_win_offsets[i] + w) * HV_DIM;
                output[gpu_batch_start + i][w] = HV_16(
                    flat.begin() + flat_start,
                    flat.begin() + flat_start + HV_DIM);
            }
        }
        cudaFree(d_seq_packed);
        cudaFree(d_seq_offsets_dev);
        cudaFree(d_win_offsets_dev);
        cudaFree(d_outputs);
        cudaStreamDestroy(stream);
    }

    cudaFree(d_kmer_map);
    fprintf(stderr, "[encode] cudaFree(d_kmer_map): done\n");

    fprintf(stderr, "[encode] DONE. Returning %zu output vectors.\n", output.size());
    return output;
}