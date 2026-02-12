#include "hierarchy.h"
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <stdexcept>

using namespace std;

static inline std::string normalize_category(std::string s) {
    for (char& c : s) {
        if (c == ' ') c = '_';
    }
    return s;
}

Hierarchy::Hierarchy(std::string seq_dir, std::string taxes) : seq_dir(seq_dir), taxes(taxes) {
    unordered_map<string, unordered_set<string>> tmp;
    ifstream in(taxes);

    string line;
    getline(in, line);

    while (getline(in, line)) {
        if (line.empty()) continue;

        string seq_id, leaf, tax;
        {
            stringstream ss(line);
            getline(ss, seq_id, '\t');
            getline(ss, leaf, '\t');
            getline(ss, tax);
        }

        vector<string> ranks;
        string tok;
        stringstream ts(tax);
        while (getline(ts, tok, ';')) {
            tok.erase(0, tok.find_first_not_of(" "));
            if (tok.find("__") != string::npos && !tok.ends_with("__")) {
                ranks.push_back(normalize_category(tok));
            }
        }

        for (size_t i = 1; i < ranks.size(); ++i) {
            tmp[ranks[i - 1]].insert(ranks[i]);
        }
    }

    // convert sets → vectors
    for (auto& [p, kids] : tmp) {
        children[p] = vector<string>(kids.begin(), kids.end());
    }
}

const std::vector<std::string> &Hierarchy::get_sequences(std::string category){
    category = normalize_category(category);
    auto it = seq_cache.find(category);
    if (it != seq_cache.end())
        return it->second;

    vector<string> seqs;
    filesystem::path fasta = seq_dir / (category + ".fasta");

    if (filesystem::exists(fasta)) {
        ifstream in(fasta);
        string line, seq;
        while (getline(in, line)) {
            if (line.starts_with(">")) {
                if (!seq.empty()) {
                    seqs.push_back(seq);
                    seq.clear();
                }
            } else {
                seq += line;
            }
        }
        if (!seq.empty()) seqs.push_back(seq);
    }

    return seq_cache.emplace(category, std::move(seqs)).first->second;
}

void Hierarchy::append_sequence(string category, string seq) {
    category = normalize_category(category);
    // 1. Update in-memory cache
    auto it = seq_cache.find(category);
    if(it != seq_cache.end()) {
        auto& vec = seq_cache[category];
        vec.push_back(seq);
    }

    // 2. Append to FASTA on disk
    filesystem::path fasta = seq_dir / (category + ".fasta");

    // Ensure directory exists
    filesystem::create_directories(fasta.parent_path());

    ofstream out(fasta, ios::app);
    if (!out) {
        throw runtime_error("Failed to open FASTA for append: " + fasta.string());
    }

    string id = make_seq_id(category);
    out << ">" << id << "\n";
    out << seq << "\n";
}

vector<string> Hierarchy::get_order(string order) {
    vector<string> categories;
    for(const auto& [parent, kids] : children) {
        if (parent.starts_with(order)) categories.push_back(parent);
    }
    return categories;
}

const std::vector<std::string> * Hierarchy::get_children(std::string category) {
    auto it = children.find(category);
    if (it != children.end()) return &(it->second);
    return nullptr;
    throw std::runtime_error("No category " + category + " to get children for");
}

string Hierarchy::make_seq_id(const string& category) {
    auto norm = normalize_category(category);
    auto it = seq_cache.find(norm);
    size_t n = (it == seq_cache.end()) ? 0 : it->second.size();
    return norm + "_appended_" + to_string(n);
}