#include "encoder.h"
#include "HV.h"
#include "hierarchy.h"
#include <filesystem>
#include <fstream>
#include <unordered_set>
#include <iostream>

constexpr float threshold = 0.65;
constexpr int BEAM_WIDTH = 5;

const std::vector<std::string> ORDER = {
    "k__", "p__", "c__", "o__", "f__", "g__", "s__"
};

struct sequence {
    std::string sequence;
    std::vector<std::string> path;
    float score;

    std::vector<std::string> get_children(Hierarchy &tree) {
        if (path.size() == 0) return tree.get_order("k__");
        return tree.get_children(path[path.size()-1]);
    }
};

int main(int argc, char* argv[]) {
    if(argc != 4) throw std::runtime_error("Usage: identify sequence-directory taxonomy-file input-fasta");

    Hierarchy tree = Hierarchy(argv[1], argv[2]);
    HVCache cache = HVCache();
    std::unordered_map<std::string, std::vector<std::string>> outputs;

    //Create initial sequences
    std::filesystem::path fasta = argv[3];
    std::vector<std::string> seqs;
    std::unordered_map<std::string, std::string> seq_to_name;

    if (std::filesystem::exists(fasta)) {
        std::ifstream in(fasta);
        std::string line, seq, name;

        while (std::getline(in, line)) {
            if (line.starts_with(">")) {
                // flush previous sequence
                if (!seq.empty()) {
                    seqs.push_back(seq);
                    seq_to_name.emplace(seq, name);
                    seq.clear();
                }
                // store header without '>'
                name = line.substr(1);
            } else {
                seq += line;
            }
        }

        // flush last sequence
        if (!seq.empty()) {
            seqs.push_back(seq);
            seq_to_name.emplace(seq, name);
        }
    }

    std::unordered_map<std::string, std::vector<sequence>> sequences;
    for(auto seq: seqs) sequences.emplace(seq, std::vector<sequence>{sequence{
        seq,
        std::vector<std::string>(),
        0.0f
    }});
    //For each order
    for(int x = 0; x < 7; x++) {

        //Create list for children of each category
        std::unordered_map<std::string, std::vector<sequence *>> child_categories;

        //For each sequence
        for(auto &[target, curr_sequence]: sequences) {
            //For each seq in sequence
            std::vector<sequence> next_seqs;
            for(auto &seq: curr_sequence) {
                //add all children of current category of seq to list of children
                for(auto child: seq.get_children(tree)) {
                    child_categories.emplace(child, std::vector<sequence *>());

                //add a new seq to sequence for each child of current category
                    std::vector<std::string> new_path = seq.path;
                    new_path.push_back(child);
                    sequence n({
                        seq.sequence,
                        std::move(new_path),
                        0
                    });
                    next_seqs.push_back(n);
                    child_categories[child].push_back(&n);
                }

                sequences[target] = next_seqs;
            }
        }

        //For child category in children
        for(auto& [child, seqs]: child_categories) {
            //For each seq that uses this category
            for(auto& seq: seqs) {
                float best_score = 0.0f;
                //Get max score between category and seq
                for(const auto &cat_seq: tree.get_sequences(child)) best_score = std::max(best_score, cosine(cache.get(cat_seq), cache.get(seq->sequence)));

                //Update seq
                seq->score = best_score;
            }
        }
    
        //for each sequence
        for(auto &[target, seqs]: sequences) {
            //sort sequence
            std::sort(seqs.begin(), seqs.end(),
                [](const sequence& a, const sequence& b) {
                    return a.score > b.score; // higher first
                });

            //if the top seq in sequence is below threshold, sequence id is just the top seq minus last term
            if(seqs[0].score < threshold) {
                outputs.emplace(target, seqs[0].path);
                sequences.erase(target);
            }

            //Otherwise keep top n seq in each sequence
            size_t keep = 0;
            while (keep < seqs.size() &&
                keep < BEAM_WIDTH &&
                seqs[keep].score >= threshold) {
                ++keep;
            }

            seqs.resize(keep);
        }
    }
    //Add the highest candidate for each remaining target sequence
    for(auto &[target, seqs]: sequences) {
        outputs[target] = seqs[0].path;
    }
    
    for(auto &[key, output]: outputs) {
        std::cout << "\n" << seq_to_name[key] << ":";
        for(auto &piece: output) std::cout << " " << piece;
    }
    std::cout << "\n";
}