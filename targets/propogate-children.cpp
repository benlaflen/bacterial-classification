#include <iostream>
#include <string>
#include <stdexcept>
#include <math.h>

#include "encoder.h"
#include "HV.h"
#include "hierarchy.h"

#define THRESHOLD 0.65

const std::vector<std::string> ORDER = {
    "s__", "g__", "f__", "o__", "c__", "p__", "k__"
};

int main(int argc, char* argv[]) {
    HVCache cache = HVCache();
    std::cout << "\nCache size is " << std::to_string(CACHE_SIZE) << " entries (" << std::to_string(CACHE_BYTES) << "b)\n";
    std::vector<int> counts(101,0);

    if(argc != 3) throw std::runtime_error("Usage: propogate-children sequence-directory taxonomy-file");

    Hierarchy tree = Hierarchy(argv[1], argv[2]);

    for (size_t x = 1; x < ORDER.size(); ++x) {
        const std::string& rank = ORDER[x];
        const std::string& child_rank = ORDER[x - 1];

        for (const auto& parent: tree.get_order(ORDER[x])) {

            auto& parent_seqs = tree.get_sequences(parent);

            std::vector<std::string> child_seqs;
            for (const auto& child : tree.get_children(parent)) {
                if (!child.starts_with(child_rank)) continue;
                auto& cs = tree.get_sequences(child);
                child_seqs.insert(child_seqs.end(), cs.begin(), cs.end());
            }

            // === YOUR LOGIC HERE ===
            // parent_seqs : sequences directly at this node
            // child_seqs  : union of all immediate children
            std::vector<std::string> added_seqs;
            for(int x = 0; x < child_seqs.size(); x++) {
                const auto &child = child_seqs[x];
                std::cout << "\rAnalyzing " << parent << " " << std::to_string(x) << "/" << std::to_string(child_seqs.size()) << "        " << std::flush;
                float best_score = 0;
                auto child_hv = cache.get(child);
                for(const auto &parent_seq : parent_seqs) best_score = std::max(best_score, cosine(child_hv, cache.get(parent_seq)));
                for(const auto &parent_seq : added_seqs) best_score = std::max(best_score, cosine(child_hv, cache.get(parent_seq)));
                counts[static_cast<int>(std::floor(best_score*100))]++;
                if (best_score < THRESHOLD ) {
                    tree.append_sequence(parent, child);
                    added_seqs.push_back(child);
                }
            }
        }
    }
    std::cout << "\n\nbucket,count";
    for(int x = 0; x < 100; x++) {
        std::cout <<"\n" << std::to_string(x/100) << "," << std::to_string(counts[x]);
    }
}