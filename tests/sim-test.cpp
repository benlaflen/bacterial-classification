#include <iostream>
#include <string>
#include <chrono>

#include "encoder.h"
#include "HV.h"

int main() {
    // Cache can hold a few entries for the test
    HVCache cache(10000);


    std::vector<std::string> queries = {
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "AAGAAGAAGAAGAAGAAGAAGAAGAAGAAG",
        "GAGAAGAGAAGAGAAGAGAAGAGAAGAGAA",
        "GAGAGAGAGAGAGAGAGAGAGAGAGAGAGA",
        "GAGGAGAGGAGAGGAGAGGAGAGGAGAGGA",
        "GGAGGAGGAGGAGGAGGAGGAGGAGGAGGA",
        "GGAGGGGAGGGGAGGGGAGGGGAGGGGAGG",
        "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
    };
    int index = 0;
    for(auto &ref: queries) {
        int i2 = 0;
        for (auto &ref2: queries) {
            auto sim = cosine(cache.get(ref2), cache.get(ref));
            std::cout << "\nRef " << std::to_string(index) << "/" << std::to_string(i2) << ": " << std::to_string(sim);
            i2++;
        }
        index++;
    }
}