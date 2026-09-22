#pragma once

#include <array>
#include <cstdint>
#include <random>

struct StepResult {
    std::array<float, 4> state{};
    float reward{0.0f};
    bool done{false};
};

class CartPole {
public:
    explicit CartPole(std::uint32_t seed = 42);

    std::array<float, 4> reset();
    StepResult step(int action);

private:
    std::mt19937 rng_;
    std::uniform_real_distribution<float> init_dist_{-0.05f, 0.05f};
    std::array<float, 4> state_{};
    int steps_{0};

    static constexpr float gravity_ = 9.8f;
    static constexpr float masscart_ = 1.0f;
    static constexpr float masspole_ = 0.1f;
    static constexpr float total_mass_ = masscart_ + masspole_;
    static constexpr float length_ = 0.5f;
    static constexpr float polemass_length_ = masspole_ * length_;
    static constexpr float force_mag_ = 10.0f;
    static constexpr float tau_ = 0.02f;
    static constexpr float theta_threshold_radians_ = 12.0f * 2.0f * 3.14159265358979323846f / 360.0f;
    static constexpr float x_threshold_ = 2.4f;
    static constexpr int max_steps_ = 500;
};
