#pragma once
#include <unordered_map>
#include <vector>
#include <string>
#include <filesystem>

class Hierarchy {
public:
    Hierarchy(std::string seq_dir, std::string taxes);

    const std::vector<std::string> &get_sequences(std::string category);

    void append_sequence(std::string category, std::string seq);

    std::vector<std::string> get_order(std::string order);

    const std::vector<std::string> *get_children(std::string category);

private:
    std::string make_seq_id(const std::string& category);

    std::filesystem::path seq_dir;
    std::string taxes;

    // taxonomy tree
    std::unordered_map<std::string, std::vector<std::string>> children;

    // sequence cache (lazy)
    std::unordered_map<std::string, std::vector<std::string>> seq_cache;
};