#include "encoder.h"
#include "HV.h"
#include "hierarchy.h"

constexpr float threshold = 0.65;

struct sequence {
    std::string sequence;
    std::vector<std::string> path;
    float score;
};

int main(int argc, char* argv[]) {
    if(argc != 4) throw std::runtime_error("Usage: propogate-children sequence-directory taxonomy-file input-fasta");

    Hierarchy tree = Hierarchy(argv[1], argv[2]);

    //Create initial sequences

    //
}