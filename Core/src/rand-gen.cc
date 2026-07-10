#include <Core/rand-gen.h>

namespace ksk::core {
float randomFloat() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    return dis(gen);
}
Real randomReal() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<Real> dis(0.0, 1.0);
    return dis(gen);
}
}