#include "encoder.h"
#include "HV.h"
#include "hierarchy.h"

#include <filesystem>
#include <fstream>
#include <unordered_set>
#include <iostream>
#include <algorithm>

#define ENABLE_LOGGING 1

#if ENABLE_LOGGING
    #define LOG(x) do { x; } while(0)
#else
    #define LOG(x) do {} while(0)
#endif

constexpr float threshold = 0.0;
constexpr int BEAM_WIDTH = 5;

struct sequence {
    std::string sequence;
    std::vector<std::string> path;
    float score;

    const std::vector<std::string>get_children(Hierarchy &tree) {
        if (path.empty()) return tree.get_order("p__");
        auto children = tree.get_children(path.back());
        if(!children) return {path.back()};
        return *children;
    }
};

int main(int argc, char* argv[]) {
    if(argc != 4)
        throw std::runtime_error("Usage: identify sequence-directory taxonomy-file input-fasta");

    Hierarchy tree = Hierarchy(argv[1], argv[2]);
    HVCache cache;

    std::unordered_map<std::string, std::vector<std::string>> outputs;

    // ---------------- FASTA LOAD ----------------
    std::filesystem::path fasta = argv[3];
    std::vector<std::string> seqs;
    std::unordered_map<std::string, std::string> seq_to_name;

    if (std::filesystem::exists(fasta)) {
        std::ifstream in(fasta);
        std::string line, seq, name;

        while (std::getline(in, line)) {
            if (line.starts_with(">")) {
                if (!seq.empty()) {
                    seqs.push_back(seq);
                    seq_to_name.emplace(seq, name);
                    seq.clear();
                }
                name = line.substr(1);
            } else {
                seq += line;
            }
        }

        if (!seq.empty()) {
            seqs.push_back(seq);
            seq_to_name.emplace(seq, name);
        }
    }

    LOG(
        std::cout << "Loaded " << seqs.size() << " sequences\n";
        for (auto &s : seqs)
            std::cout << "  len=" << s.size()
                      << " name=" << seq_to_name[s] << "\n";
    );

    // ---------------- INITIAL BEAMS ----------------
    std::unordered_map<std::string, std::vector<sequence>> sequences;
    for (auto &seq : seqs) {
        sequences.emplace(seq, std::vector<sequence>{
            sequence{seq, {}, 0.0f}
        });
    }

    // ---------------- TAXONOMY WALK ----------------
    for (int x = 0; x < 6; x++) {

        LOG(std::cout << "\n=== TAXONOMY LEVEL " << x << " ===\n");

        std::unordered_map<std::string, std::vector<sequence*>> child_categories;

        for (auto &[target, curr_sequence] : sequences) {
            LOG(
                std::cout << "Target len=" << target.size()
                          << " beam=" << curr_sequence.size() << "\n";
            );

            std::vector<sequence> next_seqs;
            size_t total = 0;
            for (auto &seq : curr_sequence) total += seq.get_children(tree).size();
            next_seqs.reserve(total);

            for (auto &seq : curr_sequence) {
                auto children = seq.get_children(tree);

                LOG(
                    std::cout << "  Expanding [";
                    for (auto &p : seq.path) std::cout << p << " ";
                    std::cout << "] -> " << children.size() << " children\n";
                );

                for (auto &child : children) {
                    child_categories[child];

                    std::vector<std::string> new_path = seq.path;
                    new_path.push_back(child);

                    next_seqs.push_back(sequence{
                        seq.sequence,
                        std::move(new_path),
                        0.0f
                    });

                    child_categories[child].push_back(&next_seqs.back());
                }
            }

            curr_sequence = std::move(next_seqs);
        }

        // ---------------- SCORING ----------------
        for (auto &[child, seqs] : child_categories) {
            LOG(
                std::cout << " Scoring category " << child
                          << " (" << seqs.size() << " candidates)\n";
            );

            for (auto *seq : seqs) {
                cache.precompute(seq->sequence);
                float best_score = -1.0f;
                for (const auto &cat_seq : tree.get_sequences(child)) {
                    float score = cache.similarity(cat_seq, seq->sequence).score;
                    best_score = std::max(
                        best_score,
                        score
                    );
        //            LOG(std::cout << "      score=" << score << "\n");
                }

                seq->score = best_score;

                LOG(
                    std::cout << "   best score=" << best_score
                              << " target_len=" << seq->sequence.size()
                              << "\n";
                );
            }
        }

        // ---------------- BEAM PRUNE ----------------
        for (auto it = sequences.begin(); it != sequences.end(); ) {
            auto &target = it->first;
            auto &seqs   = it->second;

            LOG(std::cout << "  Pre-prune: " << seqs.size() << "\n");

            std::sort(seqs.begin(), seqs.end(),
                      [](auto &a, auto &b) { return a.score > b.score; });

            if (seqs.empty() || seqs[0].score < threshold) {
                LOG(std::cout << "  EARLY EXIT (best=" 
                              << (seqs.empty() ? 0.0f : seqs[0].score)
                              << ")\n");

                outputs.emplace(
                    target,
                    seqs.empty() ? std::vector<std::string>{} : seqs[0].path
                );

                it = sequences.erase(it);
                continue;
            }

            size_t keep = 0;
            while (keep < seqs.size() &&
                   keep < BEAM_WIDTH &&
                   seqs[keep].score >= threshold)
                ++keep;

            LOG(std::cout << "  Keeping " << keep << " candidates\n");

            seqs.resize(keep);
            ++it;
        }
    }

    // ---------------- FINAL ASSIGNMENTS ----------------
    for (auto &[target, seqs] : sequences) {
        LOG(
            std::cout << "FINAL target len=" << target.size()
                      << " score=" << seqs[0].score << "\n";
        );
        outputs[target] = seqs[0].path;
    }

    // ---------------- OUTPUT ----------------
    LOG(std::cout << "\nRESULTS\n");

    for (auto &[key, output] : outputs) {
        std::cout << "\n" << seq_to_name[key] << ":";
        for (auto &piece : output)
            std::cout << " " << piece;
    }

    std::cout << "\n";
}