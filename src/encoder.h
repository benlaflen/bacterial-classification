#pragma once
#include <string>
#include <unordered_map>
#include <list>
#include "HV.h"

#define DEFAULT_WINDOW 5

class HVCache {
public:
    explicit HVCache(size_t max_entries, std::vector<int> k_mer_channels = {DEFAULT_WINDOW});

    const HV_16& get(const std::string& seq);

private:
    struct Entry {
        HV_16 hv;
        std::list<std::string>::iterator lru_it;
    };

    size_t capacity;
    std::list<std::string> lru;
    std::unordered_map<std::string, Entry> map;

    std::vector<int> k_mer_channels;

    HV_16 compute(const std::string& seq);
};