#pragma once

#include <torch/torch.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <random>
#include <vector>

struct Transition {
    std::array<float, 4> state{};
    int64_t action{0};
    float reward{0.0f};
    std::array<float, 4> next_state{};
    bool done{false};
};

class ReplayBuffer {
public:
    ReplayBuffer(std::size_t capacity, std::uint32_t seed);

    void push(const Transition& transition);
    std::vector<Transition> sample(std::size_t batch_size);
    std::size_t size() const noexcept;

private:
    std::size_t capacity_;
    std::deque<Transition> data_;
    std::mt19937 rng_;
};

struct QNetworkImpl : torch::nn::Module {
    QNetworkImpl();
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear out{nullptr};
};
TORCH_MODULE(QNetwork);

struct DQNConfig {
    float gamma{0.99f};
    float learning_rate{1e-3f};
    float epsilon_start{1.0f};
    float epsilon_end{0.05f};
    float epsilon_decay_steps{20000.0f};
    std::size_t replay_capacity{50000};
    std::size_t batch_size{128};
    int target_sync_interval{500};
    float grad_clip_norm{10.0f};
};

class DQNAgent {
public:
    DQNAgent(torch::Device device, DQNConfig config = {}, std::uint32_t seed = 42);

    int select_action(const std::array<float, 4>& state, bool greedy = false);
    void remember(const Transition& transition);
    bool optimize();
    void sync_target();
    float epsilon() const;
    std::size_t replay_size() const noexcept;

private:
    torch::Tensor states_to_tensor(const std::vector<std::array<float, 4>>& states) const;

    torch::Device device_;
    DQNConfig config_;
    QNetwork policy_;
    QNetwork target_;
    torch::optim::Adam optimizer_;
    ReplayBuffer replay_;
    std::mt19937 rng_;
    std::uniform_real_distribution<float> unit_{0.0f, 1.0f};
    int64_t action_steps_{0};
    int optimize_steps_{0};
};
