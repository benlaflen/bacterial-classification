#include <iostream>
#include <string>
#include <stdexcept>
#include <math.h>

#include "encoder.h"
#include "HV.h"
#include "hierarchy.h"

#define THRESHOLD 0.75

const std::vector<std::string> ORDER = {
    "k__", "p__", "c__", "o__", "f__", "g__", "s__"
};

void collect_descendant_sequences(
    Hierarchy& tree,
    const std::string& node,
    std::vector<std::string>& out)
{

    // Recurse into children
    auto children = tree.get_children(node);
    if (!children) return;

    for (const auto& child : *children) {
        // Add sequences at this node
        auto& seqs = tree.get_sequences(child);
        out.insert(out.end(), seqs.begin(), seqs.end());
        collect_descendant_sequences(tree, child, out);
    }
    std::cout << "\n\nFound " << std::to_string(out.size()) << " sequences in " << node;
}

int main(int argc, char* argv[]) {
    HVCache cache = HVCache();
    std::cout << "\nCache size is " << std::to_string(CACHE_SIZE) << " entries (" << std::to_string(CACHE_BYTES) << "b)\n";
    std::vector<int> counts(101,0);
    int total_propogated = 0;

    if(argc != 3) throw std::runtime_error("Usage: propogate-children sequence-directory taxonomy-file");

    Hierarchy tree = Hierarchy(argv[1], argv[2]);

    for (size_t x = 1; x < ORDER.size(); ++x) {
        const std::string& rank = ORDER[x];
        const std::string& child_rank = ORDER[x - 1];

        for (const auto& parent: tree.get_order(ORDER[x])) {

            auto& parent_seqs = tree.get_sequences(parent);

            std::vector<std::string> descendant_seqs;
            collect_descendant_sequences(tree, parent, descendant_seqs);

            // === YOUR LOGIC HERE ===
            // parent_seqs : sequences directly at this node
            // child_seqs  : union of all immediate children
            std::vector<std::string> added_seqs;
            for(int x = 0; x < descendant_seqs.size(); x++) {
                const auto &child = descendant_seqs[x];
                std::cout << "\nAnalyzing " << parent << " " << std::to_string(x) << "/" << std::to_string(descendant_seqs.size()) << "        " << std::flush;
                float best_score = 0;
                auto child_hv = cache.get(child);
                for(const auto &parent_seq : parent_seqs) best_score = std::max(best_score, cosine(child_hv, cache.get(parent_seq)));
                for(const auto &parent_seq : added_seqs) best_score = std::max(best_score, cosine(child_hv, cache.get(parent_seq)));
                std::cout << "\nBest Score: " << std::to_string(best_score);
                counts[static_cast<int>(std::floor(best_score))]++;
                if (best_score < THRESHOLD ) {
                    tree.append_sequence(parent, child);
                    added_seqs.push_back(child);
                    total_propogated+=1;
                }
            }
        }
    }
    std::cout << "\n\nbucket,count";
    for(int x = 0; x < 100; x++) {
        std::cout <<"\n" << std::to_string(x/100) << "," << std::to_string(counts[x]);
    }
    std::cout << "\n\nPropogated: " << std::to_string(total_propogated) << "\n";
}