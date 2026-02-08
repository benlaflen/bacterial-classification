#pragma once
#include <string>
#include <unordered_map>
#include <list>
#include "HV.h"

#define DEFAULT_WINDOW 5

constexpr size_t CACHE_BYTES = 1'000'000'000;
constexpr size_t HV_MEM = HV_SIZE * 2;
constexpr size_t CACHE_SIZE = CACHE_BYTES / HV_MEM;

class HVCache {
public:
    explicit HVCache(size_t max_entries = CACHE_SIZE, std::vector<int> k_mer_channels = {DEFAULT_WINDOW});

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