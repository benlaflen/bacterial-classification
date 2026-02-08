#include <iostream>
#include <string>

#include "encoder.h"
#include "HV.h"

int main() {
    // Hardcoded test sequences
    std::string seq1 = "ACGTACGTACGTACGTACGT";
    std::string seq2 = "ACGTACGTTCGTACGTACGT";
    std::string seq3 = "TTTTTTTTTTTT";
    std::string seq4 = "AAAAAAAA";

    // Cache can hold a few entries for the test
    HVCache cache(8);

    // Compute / fetch hypervectors
    const HV_16& hv1 = cache.get(seq1);
    const HV_16& hv2 = cache.get(seq2);
    const HV_16& hv3 = cache.get(seq3);
    const HV_16& hv4 = cache.get(seq4);

    // Compute similarities
    float sim11 = cosine(hv1, hv1);
    float sim12 = cosine(hv1, hv2);
    float sim13 = cosine(hv1, hv3);
    float sim23 = cosine(hv2, hv3);
    float sim14 = cosine(hv1, hv4);
    float sim24 = cosine(hv2, hv4);
    float sim34 = cosine(hv3, hv4);

    // Print results
    std::cout << "Similarity(seq1, seq1): " << sim11 << '\n';
    std::cout << "Similarity(seq1, seq2): " << sim12 << '\n';
    std::cout << "Similarity(seq1, seq3): " << sim13 << '\n';
    std::cout << "Similarity(seq2, seq3): " << sim23 << '\n';
    std::cout << "Similarity(seq1, seq4): " << sim14 << '\n';
    std::cout << "Similarity(seq2, seq4): " << sim24 << '\n';
    std::cout << "Similarity(seq3, seq4): " << sim34 << '\n';

    return 0;
}