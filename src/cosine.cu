#include "GPU.cuh"
#include <stdint.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <vector>

#define GPU_MEMORY_HEADROOM 0.8

// ---------------------------------------------------------------------------
// Device helper: extract a 2-bit-packed k-mer starting at base position `pos`
// from a packed sequence buffer.
//
// Packing convention: base at position i occupies bits [2i+1 : 2i] of the
// uint32 word at seq_packed[i / 16], i.e. 16 bases per uint32 word.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t extract_kmer(const uint32_t* __restrict__ seq_packed,
                                                  int pos)
{
    int      bit_offset = pos * 2;
    int      word_idx   = bit_offset / 32;
    uint32_t lo         = seq_packed[word_idx];
    uint32_t hi         = seq_packed[word_idx + 1];  // safe: caller ensures padding
    uint64_t combined   = ((uint64_t)hi << 32) | lo;
    return (uint32_t)((combined >> bit_offset) & KMER_MASK);
}

// ---------------------------------------------------------------------------
// HDC encode kernel
// ---------------------------------------------------------------------------
__global__ void hdc_encode(
    const uint32_t* __restrict__ seq_packed,
    const int*      __restrict__ seq_offsets,
    const int16_t*  __restrict__ kmer_map,
    const int*      __restrict__ window_offsets,
          int16_t*               outputs,
    int num_seqs)
{
    int seq_idx = blockIdx.x;
    int win_idx = blockIdx.y;
    int dim     = blockIdx.z * HV_BLOCK_SIZE + threadIdx.x;
    if (dim >= HV_DIM) return;

    int seq_base_offset = seq_offsets[seq_idx];
    int seq_len         = seq_offsets[seq_idx + 1] - seq_base_offset;
    int num_windows     = (seq_len - SEQUENCE_LENGTH) / SIM_STEP + 1;
    if (win_idx >= num_windows) return;

    int window_start = win_idx * SIM_STEP;

    int32_t acc = 0;
    const uint32_t* seq = seq_packed + (seq_base_offset / 16);

    for (int x = 0; x < K; x++) {
        uint32_t kmer_idx = extract_kmer(seq, window_start + x);
        acc += kmer_map[kmer_idx * HV_DIM + dim];
    }

    int32_t true_sum = 2 * acc - K;
    int out_idx = (window_offsets[seq_idx] + win_idx) * HV_DIM + dim;
    outputs[out_idx] = (int16_t)true_sum;
}

// ---------------------------------------------------------------------------
// Host-side encode launcher
// ---------------------------------------------------------------------------
cudaError_t launch_hdc_encode(
    const uint32_t* d_seq_packed,
    const int*      d_seq_offsets,
    const int16_t*  d_kmer_map,
    const int*      d_window_offsets,
          int16_t*  d_outputs,
    int num_seqs,
    int max_windows,
    cudaStream_t stream)
{
    dim3 block(HV_BLOCK_SIZE, 1, 1);
    dim3 grid(num_seqs, max_windows, HV_BLOCKS);
    hdc_encode<<<grid, block, 0, stream>>>(
        d_seq_packed, d_seq_offsets, d_kmer_map,
        d_window_offsets, d_outputs, num_seqs);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Sequence packing helpers
// ---------------------------------------------------------------------------
static std::vector<uint32_t> pack_sequence(const std::string& seq)
{
    int num_words = (seq.size() + 15) / 16 + 1;
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

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------
std::vector<std::vector<HV_16>> encode(const std::vector<int16_t>& kmer_map,
                                       const std::vector<std::string>& sequences)
{
    fprintf(stderr, "[encode] START: %zu sequences\n", sequences.size());
    for (size_t i = 0; i < sequences.size(); i++)
        fprintf(stderr, "[encode]   seq[%zu] len=%zu\n", i, sequences[i].size());

    static size_t memory_limit = get_memory_limit();
    fprintf(stderr, "[encode] memory_limit = %zu bytes (%.2f GB)\n",
            memory_limit, memory_limit / (1024.0*1024.0*1024.0));

    std::vector<std::vector<HV_16>> output(sequences.size());

    int16_t*     d_kmer_map        = nullptr;
    cudaStream_t stream            = nullptr;
    int16_t*     d_outputs         = nullptr;
    uint32_t*    d_seq_packed      = nullptr;
    int*         d_seq_offsets_dev = nullptr;
    int*         d_win_offsets_dev = nullptr;

    int              gpu_batch_start   = 0;
    std::vector<int> gpu_win_offsets;
    int              gpu_total_windows = 0;

    int  batch_start    = 0;
    int  batch_end      = 0;
    int  batch_max_len  = 0;
    bool batch_complete = false;
    int  batch_index    = 0;

    std::vector<int>      seq_offsets    = {0};
    std::vector<int>      window_offsets = {0};
    std::vector<uint32_t> seq_packed_host;

    fprintf(stderr, "[encode] Allocating d_kmer_map: KMER_COUNT=%d HV_DIM=%d => %zu bytes\n",
            KMER_COUNT, HV_DIM, (size_t)KMER_COUNT * HV_DIM * sizeof(int16_t));
    cudaError_t e = cudaMalloc(&d_kmer_map, KMER_COUNT * HV_DIM * sizeof(int16_t));
    fprintf(stderr, "[encode] cudaMalloc d_kmer_map: %s\n", cudaGetErrorString(e));
    e = cudaMemcpy(d_kmer_map, kmer_map.data(), KMER_COUNT * HV_DIM * sizeof(int16_t),
                   cudaMemcpyHostToDevice);
    fprintf(stderr, "[encode] cudaMemcpy kmer_map H->D: %s\n", cudaGetErrorString(e));

    fprintf(stderr, "[encode] Entering main while loop (batch_start=%d)\n", batch_start);

    while (batch_start < (int)sequences.size()) {
        fprintf(stderr,
            "[encode] --- Top of main loop: batch_start=%d batch_end=%d "
            "batch_index=%d batch_complete=%d stream=%p ---\n",
            batch_start, batch_end, batch_index, (int)batch_complete, (void*)stream);

        // --- Assemble + pack next batch while GPU runs ---
        fprintf(stderr, "[encode]   [assemble] Entering assembly loop\n");
        while (!batch_complete && (!stream || cudaStreamQuery(stream) != cudaSuccess)) {
            if (batch_end >= (int)sequences.size()) {
                fprintf(stderr,
                    "[encode]   [assemble] batch_end=%d >= sequences.size()=%zu => batch_complete=true\n",
                    batch_end, sequences.size());
                batch_complete = true; break;
            }
            int    candidate_max     = std::max(batch_max_len, (int)sequences[batch_end].size());
            int    candidate_windows = (candidate_max - SEQUENCE_LENGTH) / SIM_STEP + 1;
            size_t candidate_mem     = (size_t)(batch_end - batch_start + 1)
                                     * candidate_windows * HV_DIM * sizeof(int16_t);
            fprintf(stderr,
                "\r[encode]   [assemble] seq[%d] len=%zu | candidate_max=%d "
                "candidate_windows=%d candidate_mem=%zu (%.2f MB) memory_limit=%zu (%.2f MB)    ",
                batch_end, sequences[batch_end].size(),
                candidate_max, candidate_windows,
                candidate_mem, candidate_mem / (1024.0*1024.0),
                memory_limit,  memory_limit  / (1024.0*1024.0));

            if (candidate_mem > memory_limit) {
                fprintf(stderr,
                    "\n[encode]   [assemble] Memory limit exceeded => batch_complete=true\n");
                batch_complete = true; break;
            }

            const std::string& seq  = sequences[batch_end];
            int wins = ((int)seq.size() - SEQUENCE_LENGTH) / SIM_STEP + 1;
            seq_offsets.push_back(seq_offsets.back() + (int)seq.size());
            window_offsets.push_back(window_offsets.back() + wins);
            std::vector<uint32_t> packed = pack_sequence(seq);
            seq_packed_host.insert(seq_packed_host.end(), packed.begin(), packed.end());

            fprintf(stderr,
                "\n[encode]   [assemble] Added seq[%d]: wins=%d seq_offsets.back()=%d "
                "window_offsets.back()=%d packed_words=%zu seq_packed_host.size()=%zu\n",
                batch_end, wins,
                seq_offsets.back(), window_offsets.back(),
                packed.size(), seq_packed_host.size());

            batch_max_len = candidate_max;
            batch_end++;
        }
        fprintf(stderr,
            "[encode]   [assemble] Exited assembly loop: batch_end=%d batch_complete=%d batch_max_len=%d\n",
            batch_end, (int)batch_complete, batch_max_len);

        // --- Sync previous batch and collect results ---
        if (stream) {
            fprintf(stderr,
                "[encode]   [sync] Synchronizing stream for batch starting at gpu_batch_start=%d\n",
                gpu_batch_start);
            cudaError_t sync_err = cudaStreamSynchronize(stream);
            fprintf(stderr, "[encode]   [sync] cudaStreamSynchronize: %s\n",
                    cudaGetErrorString(sync_err));

            size_t flat_bytes = (size_t)gpu_total_windows * HV_DIM * sizeof(int16_t);
            std::vector<int16_t> flat(gpu_total_windows * HV_DIM);
            cudaError_t cpy_err = cudaMemcpy(flat.data(), d_outputs, flat_bytes,
                                             cudaMemcpyDeviceToHost);
            fprintf(stderr, "[encode]   [sync] cudaMemcpy D->H flat: %s\n",
                    cudaGetErrorString(cpy_err));

            for (int i = 0; i < (int)gpu_win_offsets.size() - 1; i++) {
                int    wins            = gpu_win_offsets[i + 1] - gpu_win_offsets[i];
                size_t flat_start_base = (size_t)gpu_win_offsets[i] * HV_DIM;
                fprintf(stderr,
                    "\r[encode]   [sync] output[%d]: wins=%d flat_start_base=%zu    ",
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

            cudaFree(d_seq_packed);
            cudaFree(d_seq_offsets_dev);
            cudaFree(d_win_offsets_dev);
            cudaFree(d_outputs);
            cudaStreamDestroy(stream);
            stream = nullptr;
        }

        // --- Dispatch current batch ---
        if (batch_complete) {
            int batch_size    = batch_end - batch_start;
            int max_windows   = (batch_max_len - SEQUENCE_LENGTH) / SIM_STEP + 1;
            int total_windows = window_offsets.back();

            fprintf(stderr,
                "[encode]   [dispatch] Batch %d: batch_start=%d batch_end=%d batch_size=%d\n",
                batch_index, batch_start, batch_end, batch_size);
            fprintf(stderr,
                "[encode]   [dispatch] batch_max_len=%d max_windows=%d total_windows=%d\n",
                batch_max_len, max_windows, total_windows);

            auto cuda_check = [&](cudaError_t err, const char* what) {
                fprintf(stderr, "[encode]   [dispatch] %s: %s\n", what, cudaGetErrorString(err));
                if (err != cudaSuccess)
                    throw std::runtime_error(
                        std::string("encode batch ") + std::to_string(batch_index) +
                        " (" + std::to_string(batch_size) + " seqs): " +
                        what + ": " + cudaGetErrorString(err));
            };

            cuda_check(cudaStreamCreate(&stream),                                                        "cudaStreamCreate");
            cuda_check(cudaMalloc(&d_seq_packed,      seq_packed_host.size() * sizeof(uint32_t)),        "cudaMalloc seq_packed");
            cuda_check(cudaMalloc(&d_seq_offsets_dev, seq_offsets.size()     * sizeof(int)),             "cudaMalloc seq_offsets");
            cuda_check(cudaMalloc(&d_win_offsets_dev, window_offsets.size()  * sizeof(int)),             "cudaMalloc win_offsets");
            cuda_check(cudaMalloc(&d_outputs,         (size_t)total_windows  * HV_DIM * sizeof(int16_t)),"cudaMalloc outputs");

            cuda_check(cudaMemcpyAsync(d_seq_packed,      seq_packed_host.data(), seq_packed_host.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), "memcpyAsync seq_packed");
            cuda_check(cudaMemcpyAsync(d_seq_offsets_dev, seq_offsets.data(),     seq_offsets.size()     * sizeof(int),      cudaMemcpyHostToDevice, stream), "memcpyAsync seq_offsets");
            cuda_check(cudaMemcpyAsync(d_win_offsets_dev, window_offsets.data(),  window_offsets.size()  * sizeof(int),      cudaMemcpyHostToDevice, stream), "memcpyAsync win_offsets");

            cuda_check(launch_hdc_encode(d_seq_packed, d_seq_offsets_dev, d_kmer_map,
                d_win_offsets_dev, d_outputs, batch_size, max_windows, stream),                          "kernel launch");

            gpu_batch_start   = batch_start;
            gpu_win_offsets   = window_offsets;
            gpu_total_windows = total_windows;

            batch_start    = batch_end;
            batch_max_len  = 0;
            batch_complete = false;
            seq_offsets    = {0};
            window_offsets = {0};
            seq_packed_host.clear();
            batch_index++;
        }
    }

    fprintf(stderr, "[encode] Exited main loop. stream=%p\n", (void*)stream);
    cudaFree(d_kmer_map);
    fprintf(stderr, "[encode] DONE. Returning %zu output vectors.\n", output.size());
    return output;
}

// ===========================================================================
// Cosine search — shared infrastructure
// ===========================================================================
//
// HVs produced by encode() are sums of K bipolar {-1,+1} rows, so their L2
// norms vary with k-mer content.  We compute true cosine similarity: both the
// query (input) norm and the db norm are needed.
//
// Query norms are precomputed once per launch (one float per input HV) and
// reused across all db comparisons, saving redundant work.
//
// Db norms are computed on-the-fly inside each cosine kernel — this avoids
// storing a separate norm array for the db and keeps the hot path to two
// memory streams (input row + db row).
//
// Layout everywhere:
//   inputs : [num_inputs][HV_DIM]  int16, row-major
//   db     : [num_db][HV_DIM]      int16, row-major
// ===========================================================================

// ---------------------------------------------------------------------------
// Precompute inverse norms for a batch of HVs. One thread per HV.
// ---------------------------------------------------------------------------
__global__ void compute_inv_norms_kernel(
    const int16_t* __restrict__ d_hvs,
          float*                d_inv_norms,
    int num_hvs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hvs) return;

    const int16_t* row = d_hvs + (size_t)idx * HV_DIM;
    int64_t sq = 0;
    for (int d = 0; d < HV_DIM; d++) {
        int32_t v = row[d];
        sq += v * v;
    }
    d_inv_norms[idx] = 1.0f / sqrtf((float)sq);
}

// ---------------------------------------------------------------------------
// Inner reduction helper: accumulate dot product and db-side squared norm
// for one (input, db) pair across HV_DIM, using warp shuffles.
//
// All threads call this; only warp-lane-0 threads hold the per-warp sums on
// return.  The caller is responsible for reducing across warps via shared mem.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void reduce_dot_norm(
    const int16_t* __restrict__ input_row,
    const int16_t* __restrict__ db_row,
    int32_t& out_partial_dot,
    int32_t& out_partial_norm)
{
    int32_t partial_dot  = 0;
    int32_t partial_norm = 0;

    for (int d = threadIdx.x; d < HV_DIM; d += blockDim.x) {
        int32_t q = input_row[d];
        int32_t v = db_row[d];
        partial_dot  += q * v;
        partial_norm += v * v;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        partial_dot  += __shfl_down_sync(0xffffffff, partial_dot,  offset);
        partial_norm += __shfl_down_sync(0xffffffff, partial_norm, offset);
    }

    out_partial_dot  = partial_dot;
    out_partial_norm = partial_norm;
}

// ===========================================================================
// Mode 1: argmax — one best (score, db_index) per input HV
//
// Strategy: grid is (num_inputs × num_db).  Each block computes one cosine
// score and races to update a packed uint64 best-so-far per input using
// atomicMax.
//
// Packing trick: cosine ∈ [-1, 1].  We store (score + 1) ∈ [0, 2] as a
// float, which is always positive, so its IEEE 754 bit pattern is ordered
// identically to its float value — unsigned integer comparison == float
// comparison.  We pack as:
//
//   packed = (uint64_t)(score+1 bits) << 32 | (uint32_t)db_idx
//
// atomicMax on the uint64 finds the max score; ties go to the lower db_idx
// (irrelevant in practice).  A second kernel unpacks the result.
//
// Internal scratch: d_packed[num_inputs] uint64, allocated inside the launcher.
// ===========================================================================

__global__ void cosine_argmax_kernel(
    const int16_t* __restrict__ d_inputs,
    const int16_t* __restrict__ d_db,
    const float*   __restrict__ d_inv_norm_input,
          uint64_t*             d_packed,   // [num_inputs], zero-initialised by launcher
    int num_db)
{
    int input_idx = blockIdx.x;
    int db_idx    = blockIdx.y;

    const int16_t* input_row = d_inputs + (size_t)input_idx * HV_DIM;
    const int16_t* db_row    = d_db     + (size_t)db_idx    * HV_DIM;

    __shared__ int64_t smem_dot;
    __shared__ int64_t smem_norm;
    if (threadIdx.x == 0) { smem_dot = 0; smem_norm = 0; }
    __syncthreads();

    int32_t partial_dot, partial_norm;
    reduce_dot_norm(input_row, db_row, partial_dot, partial_norm);

    if ((threadIdx.x & 31) == 0) {
        atomicAdd((unsigned long long*)&smem_dot,  (unsigned long long)(int64_t)partial_dot);
        atomicAdd((unsigned long long*)&smem_norm, (unsigned long long)(int64_t)partial_norm);
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    float score  = (float)smem_dot * d_inv_norm_input[input_idx]
                 / sqrtf((float)smem_norm);
    float biased = score + 1.0f;          // shift into [0,2] so bits are unsigned-comparable

    uint32_t score_bits;
    memcpy(&score_bits, &biased, sizeof(float));
    uint64_t candidate = ((uint64_t)score_bits << 32) | (uint32_t)db_idx;

    atomicMax((unsigned long long*)&d_packed[input_idx], (unsigned long long)candidate);
}

__global__ void unpack_argmax_kernel(
    const uint64_t* __restrict__ d_packed,
          float*                 d_scores_out,
          int*                   d_idx_out,
    int num_inputs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_inputs) return;

    uint64_t packed = d_packed[i];
    uint32_t score_bits = (uint32_t)(packed >> 32);
    int      db_idx     = (int)(uint32_t)(packed & 0xFFFFFFFFu);

    float biased;
    memcpy(&biased, &score_bits, sizeof(float));
    d_scores_out[i] = biased - 1.0f;
    d_idx_out[i]    = db_idx;
}

// ---------------------------------------------------------------------------
// launch_cosine_argmax
//
// d_inputs      [num_inputs][HV_DIM]   int16, device ptr, caller owns lifetime
// d_db          [num_db][HV_DIM]       int16, device ptr, caller owns lifetime
// d_scores_out  [num_inputs]           float, device ptr, caller allocates
// d_idx_out     [num_inputs]           int,   device ptr, caller allocates
// ---------------------------------------------------------------------------
cudaError_t launch_cosine_argmax(
    const int16_t*  d_inputs,
    const int16_t*  d_db,
          float*    d_scores_out,
          int*      d_idx_out,
    int             num_inputs,
    int             num_db,
    cudaStream_t    stream)
{
    cudaError_t e;

    // Inverse norms for input HVs
    float* d_inv_norms = nullptr;
    e = cudaMalloc(&d_inv_norms, num_inputs * sizeof(float));
    if (e != cudaSuccess) return e;
    {
        int blk = 256, grd = (num_inputs + blk - 1) / blk;
        compute_inv_norms_kernel<<<grd, blk, 0, stream>>>(d_inputs, d_inv_norms, num_inputs);
    }

    // Packed scratch buffer: zero-init so atomicMax starts from the floor
    // (score+1 = 0 means score = -1, the lowest possible cosine)
    uint64_t* d_packed = nullptr;
    e = cudaMalloc(&d_packed, num_inputs * sizeof(uint64_t));
    if (e != cudaSuccess) { cudaFree(d_inv_norms); return e; }
    e = cudaMemsetAsync(d_packed, 0, num_inputs * sizeof(uint64_t), stream);
    if (e != cudaSuccess) { cudaFree(d_inv_norms); cudaFree(d_packed); return e; }

    // Main kernel: (num_inputs × num_db) blocks, 256 threads each
    {
        constexpr int THREADS = 256;
        dim3 blk(THREADS);
        dim3 grd(num_inputs, num_db);
        cosine_argmax_kernel<<<grd, blk, 0, stream>>>(
            d_inputs, d_db, d_inv_norms, d_packed, num_db);
        e = cudaGetLastError();
        if (e != cudaSuccess) { cudaFree(d_inv_norms); cudaFree(d_packed); return e; }
    }

    // Unpack packed uint64 → (float score, int idx)
    {
        int blk = 256, grd = (num_inputs + blk - 1) / blk;
        unpack_argmax_kernel<<<grd, blk, 0, stream>>>(
            d_packed, d_scores_out, d_idx_out, num_inputs);
        e = cudaGetLastError();
    }

    cudaFree(d_inv_norms);
    cudaFree(d_packed);
    return e;
}

// ===========================================================================
// Mode 2: top-k — k best (score, db_index) pairs per input HV, descending
//
// Strategy:
//   1. Score matrix kernel: grid (num_inputs × num_db), writes all cosine
//      scores to a flat [num_inputs * num_db] float buffer.
//   2. Selection:
//      - num_db <= MAX_DB_SHARED: bitonic sort entirely in shared memory,
//        one block per input row.  Fast, no extra allocation.
//      - num_db >  MAX_DB_SHARED: CUB DeviceSegmentedSort on the full matrix,
//        then slice the leading k entries per row.
// ===========================================================================

#define MAX_DB_SHARED 2048   // 2048 * (4+4) bytes = 16 KB shared mem per block

// ---------------------------------------------------------------------------
// Score matrix kernel — same grid shape as argmax but writes all scores
// ---------------------------------------------------------------------------
__global__ void cosine_score_matrix_kernel(
    const int16_t* __restrict__ d_inputs,
    const int16_t* __restrict__ d_db,
    const float*   __restrict__ d_inv_norm_input,
          float*                d_scores,          // [num_inputs * num_db]
    int num_db)
{
    int input_idx = blockIdx.x;
    int db_idx    = blockIdx.y;

    const int16_t* input_row = d_inputs + (size_t)input_idx * HV_DIM;
    const int16_t* db_row    = d_db     + (size_t)db_idx    * HV_DIM;

    __shared__ int64_t smem_dot;
    __shared__ int64_t smem_norm;
    if (threadIdx.x == 0) { smem_dot = 0; smem_norm = 0; }
    __syncthreads();

    int32_t partial_dot, partial_norm;
    reduce_dot_norm(input_row, db_row, partial_dot, partial_norm);

    if ((threadIdx.x & 31) == 0) {
        atomicAdd((unsigned long long*)&smem_dot,  (unsigned long long)(int64_t)partial_dot);
        atomicAdd((unsigned long long*)&smem_norm, (unsigned long long)(int64_t)partial_norm);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float score = (float)smem_dot * d_inv_norm_input[input_idx]
                    / sqrtf((float)smem_norm);
        d_scores[input_idx * num_db + db_idx] = score;
    }
}

// ---------------------------------------------------------------------------
// Bitonic sort + top-k slice, shared memory path (num_db <= MAX_DB_SHARED)
//
// One block per input row.  Loads scores + original indices into shared mem,
// sorts descending, writes the first k to the output buffers.
// ---------------------------------------------------------------------------
__global__ void topk_bitonic_kernel(
    const float* __restrict__ d_scores,      // [num_inputs * num_db]
          float*              d_scores_out,  // [num_inputs * k]
          int*                d_idx_out,     // [num_inputs * k]
    int num_db,
    int k)
{
    int input_idx = blockIdx.x;

    __shared__ float sh_scores[MAX_DB_SHARED];
    __shared__ int   sh_idx   [MAX_DB_SHARED];

    const float* row = d_scores + (size_t)input_idx * num_db;
    for (int i = threadIdx.x; i < num_db; i += blockDim.x) {
        sh_scores[i] = row[i];
        sh_idx[i]    = i;
    }
    __syncthreads();

    // Bitonic sort (descending)
    for (int size = 2; size <= num_db; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
            for (int i = threadIdx.x; i < num_db; i += blockDim.x) {
                int  partner    = i ^ stride;
                bool descending = ((i & size) == 0);
                if (partner > i) {
                    bool do_swap = descending
                                   ? (sh_scores[i] < sh_scores[partner])
                                   : (sh_scores[i] > sh_scores[partner]);
                    if (do_swap) {
                        float fs       = sh_scores[i]; sh_scores[i] = sh_scores[partner]; sh_scores[partner] = fs;
                        int   fi       = sh_idx[i];    sh_idx[i]    = sh_idx[partner];    sh_idx[partner]    = fi;
                    }
                }
            }
        }
    }
    __syncthreads();

    float*  out_s = d_scores_out + (size_t)input_idx * k;
    int*    out_i = d_idx_out    + (size_t)input_idx * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        out_s[i] = sh_scores[i];
        out_i[i] = sh_idx[i];
    }
}

// ---------------------------------------------------------------------------
// launch_cosine_topk
//
// d_inputs      [num_inputs][HV_DIM]   int16, device ptr, caller owns lifetime
// d_db          [num_db][HV_DIM]       int16, device ptr, caller owns lifetime
// d_scores_out  [num_inputs][k]        float, device ptr, caller allocates
// d_idx_out     [num_inputs][k]        int,   device ptr, caller allocates
// k             results per input, descending by score (clamped to num_db)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Large-db top-k: k-pass partial selection sort, one block per input row.
//
// Each of the k passes does a parallel reduction over num_db scores to find
// the current maximum among unselected entries, then masks it out.
// Threads stripe over num_db; each warp finds its local max, then a
// shared-memory reduction picks the block-wide winner.
//
// Complexity: O(num_db * k) — fine for small k and the cold path.
// No external dependencies.
// ---------------------------------------------------------------------------
__global__ void topk_partial_kernel(
    const float* __restrict__ d_scores,   // [num_inputs * num_db]
          float*              d_scores_out,// [num_inputs * k]
          int*                d_idx_out,   // [num_inputs * k]
    int num_db,
    int k)
{
    int input_idx = blockIdx.x;
    const float* row = d_scores + (size_t)input_idx * num_db;

    // Track which db entries have already been selected
    // We mark them by writing -FLT_MAX into a local shadow — but we can't
    // afford num_db shared memory here (it's large by definition).
    // Instead we keep only the k selected indices in registers/shared mem
    // and re-scan each pass checking against them.
    __shared__ int   selected[/* k, but k is runtime */ 1024]; // supports k up to 1024
    __shared__ float best_val;
    __shared__ int   best_idx;

    // Only need k slots; caller guarantees k <= num_db and practically k << 1024
    // Zero-init selected sentinel
    for (int i = threadIdx.x; i < k; i += blockDim.x) selected[i] = -1;
    __syncthreads();

    float* out_s = d_scores_out + (size_t)input_idx * k;
    int*   out_i = d_idx_out    + (size_t)input_idx * k;

    for (int pass = 0; pass < k; pass++) {
        // Each thread finds its best unselected candidate
        float t_best_val = -1e38f;
        int   t_best_idx = -1;

        for (int j = threadIdx.x; j < num_db; j += blockDim.x) {
            // Check if j was already selected in a prior pass
            bool already = false;
            for (int s = 0; s < pass; s++)
                if (selected[s] == j) { already = true; break; }
            if (already) continue;

            float v = row[j];
            if (v > t_best_val) { t_best_val = v; t_best_idx = j; }
        }

        // Warp reduction
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, t_best_val, offset);
            int   oi = __shfl_down_sync(0xffffffff, t_best_idx, offset);
            if (ov > t_best_val) { t_best_val = ov; t_best_idx = oi; }
        }

        // Block reduction via shared memory
        if (threadIdx.x == 0) { best_val = -1e38f; best_idx = -1; }
        __syncthreads();

        if ((threadIdx.x & 31) == 0) {
            // One atomic CAS loop to update best — use a simple critical section
            // by serialising warp leaders. For small warp counts this is fine.
            // We abuse atomicMax on the packed (val_bits, ~idx) uint64 trick.
            // Simpler: just let warp 0 leader write; serialise with __syncthreads.
            // Since we only have blockDim.x/32 warps this is fast.
        }
        // Easier: gather into shared, let thread 0 pick winner serially.
        // With 256 threads that's 8 candidates — negligible.
        __shared__ float sh_warp_val[8];
        __shared__ int   sh_warp_idx[8];
        int warp_id = threadIdx.x / 32;
        if ((threadIdx.x & 31) == 0) {
            sh_warp_val[warp_id] = t_best_val;
            sh_warp_idx[warp_id] = t_best_idx;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            int n_warps = (blockDim.x + 31) / 32;
            float bv = -1e38f; int bi = -1;
            for (int w = 0; w < n_warps; w++)
                if (sh_warp_val[w] > bv) { bv = sh_warp_val[w]; bi = sh_warp_idx[w]; }
            best_val = bv;
            best_idx = bi;
            selected[pass] = bi;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            out_s[pass] = best_val;
            out_i[pass] = best_idx;
        }
        __syncthreads();
    }
}

cudaError_t launch_cosine_topk(
    const int16_t*  d_inputs,
    const int16_t*  d_db,
          float*    d_scores_out,
          int*      d_idx_out,
    int             num_inputs,
    int             num_db,
    int             k,
    cudaStream_t    stream)
{
    if (k > num_db) k = num_db;
    cudaError_t e;

    // Inverse norms for input HVs
    float* d_inv_norms = nullptr;
    e = cudaMalloc(&d_inv_norms, num_inputs * sizeof(float));
    if (e != cudaSuccess) return e;
    {
        int blk = 256, grd = (num_inputs + blk - 1) / blk;
        compute_inv_norms_kernel<<<grd, blk, 0, stream>>>(d_inputs, d_inv_norms, num_inputs);
    }

    // Full score matrix
    float* d_score_matrix = nullptr;
    e = cudaMalloc(&d_score_matrix, (size_t)num_inputs * num_db * sizeof(float));
    if (e != cudaSuccess) { cudaFree(d_inv_norms); return e; }

    {
        constexpr int THREADS = 256;
        dim3 blk(THREADS);
        dim3 grd(num_inputs, num_db);
        cosine_score_matrix_kernel<<<grd, blk, 0, stream>>>(
            d_inputs, d_db, d_inv_norms, d_score_matrix, num_db);
        e = cudaGetLastError();
        if (e != cudaSuccess) { cudaFree(d_inv_norms); cudaFree(d_score_matrix); return e; }
    }
    cudaFree(d_inv_norms);

    if (num_db <= MAX_DB_SHARED) {
        // Shared-memory bitonic sort path
        // Threads: round num_db up to next warp, cap at 256
        int threads = std::min(((num_db + 31) / 32) * 32, 256);
        topk_bitonic_kernel<<<num_inputs, threads, 0, stream>>>(
            d_score_matrix, d_scores_out, d_idx_out, num_db, k);
        e = cudaGetLastError();
    } else {
        // Large-db path: k-pass partial selection, no external dependencies.
        // One block per input row; O(num_db * k) per row.
        topk_partial_kernel<<<num_inputs, 256, 0, stream>>>(
            d_score_matrix, d_scores_out, d_idx_out, num_db, k);
        e = cudaGetLastError();
    }

    cudaFree(d_score_matrix);
    return e;
}

// ===========================================================================
// C++ API
// ===========================================================================

std::unordered_map<HV_16*, std::vector<std::pair<float, HV_16*>>> cosine_search(
    const std::vector<HV_16*>& inputs,
    const std::vector<HV_16*>& db,
    int k)
{
    const int num_inputs = (int)inputs.size();
    const int num_db     = (int)db.size();
    if (k > num_db) k = num_db;

    // Flatten inputs and db into contiguous int16 buffers
    std::vector<int16_t> flat_inputs((size_t)num_inputs * HV_DIM);
    std::vector<int16_t> flat_db    ((size_t)num_db     * HV_DIM);

    for (int i = 0; i < num_inputs; i++)
        std::copy(inputs[i]->begin(), inputs[i]->end(),
                  flat_inputs.begin() + (size_t)i * HV_DIM);
    for (int i = 0; i < num_db; i++)
        std::copy(db[i]->begin(), db[i]->end(),
                  flat_db.begin() + (size_t)i * HV_DIM);

    // Upload to device
    int16_t* d_inputs = nullptr;
    int16_t* d_db     = nullptr;
    cudaMalloc(&d_inputs, flat_inputs.size() * sizeof(int16_t));
    cudaMalloc(&d_db,     flat_db.size()     * sizeof(int16_t));
    cudaMemcpy(d_inputs, flat_inputs.data(), flat_inputs.size() * sizeof(int16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_db,     flat_db.data(),     flat_db.size()     * sizeof(int16_t), cudaMemcpyHostToDevice);

    float* d_scores = nullptr;
    int*   d_idx    = nullptr;
    cudaMalloc(&d_scores, (size_t)num_inputs * k * sizeof(float));
    cudaMalloc(&d_idx,    (size_t)num_inputs * k * sizeof(int));

    cudaError_t e;
    if (k == 1) {
        e = launch_cosine_argmax(d_inputs, d_db, d_scores, d_idx, num_inputs, num_db, 0);
    } else {
        e = launch_cosine_topk(d_inputs, d_db, d_scores, d_idx, num_inputs, num_db, k, 0);
    }
    if (e != cudaSuccess) {
        cudaFree(d_inputs); cudaFree(d_db); cudaFree(d_scores); cudaFree(d_idx);
        throw std::runtime_error(std::string("cosine_search: ") + cudaGetErrorString(e));
    }
    cudaDeviceSynchronize();

    // Download results
    std::vector<float> h_scores((size_t)num_inputs * k);
    std::vector<int>   h_idx   ((size_t)num_inputs * k);
    cudaMemcpy(h_scores.data(), d_scores, h_scores.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idx.data(),    d_idx,    h_idx.size()    * sizeof(int),   cudaMemcpyDeviceToHost);

    cudaFree(d_inputs); cudaFree(d_db); cudaFree(d_scores); cudaFree(d_idx);

    // Build result map: input pointer → [(score, db pointer), ...]
    std::unordered_map<HV_16*, std::vector<std::pair<float, HV_16*>>> result;
    result.reserve(num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        std::vector<std::pair<float, HV_16*>> matches(k);
        for (int j = 0; j < k; j++) {
            float  score  = h_scores[i * k + j];
            HV_16* db_ptr = db[h_idx[i * k + j]];
            matches[j] = {score, db_ptr};
        }
        result[inputs[i]] = std::move(matches);
    }
    return result;
}