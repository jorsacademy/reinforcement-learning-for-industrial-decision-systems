#include "dqn.hpp"

#include <torch/nn/utils/clip_grad.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

ReplayBuffer::ReplayBuffer(std::size_t capacity, std::uint32_t seed)
    : capacity_(capacity), rng_(seed) {
    if (capacity_ == 0) {
        throw std::invalid_argument("ReplayBuffer capacity must be positive");
    }
}

void ReplayBuffer::push(const Transition& transition) {
    if (data_.size() == capacity_) {
        data_.pop_front();
    }
    data_.push_back(transition);
}

std::vector<Transition> ReplayBuffer::sample(std::size_t batch_size) {
    if (batch_size > data_.size()) {
        throw std::invalid_argument("Cannot sample more transitions than available");
    }

    // Draw only batch_size unique indices instead of shuffling the entire replay
    // buffer on every optimization step. For the common case batch_size << size,
    // this reduces sampling work from O(N) to approximately O(batch_size).
    std::uniform_int_distribution<std::size_t> index_dist(0, data_.size() - 1);
    std::unordered_set<std::size_t> selected;
    selected.reserve(batch_size * 2);

    std::vector<Transition> batch;
    batch.reserve(batch_size);

    while (batch.size() < batch_size) {
        const auto index = index_dist(rng_);
        if (selected.insert(index).second) {
            batch.push_back(data_[index]);
        }
    }

    return batch;
}

std::size_t ReplayBuffer::size() const noexcept {
    return data_.size();
}

QNetworkImpl::QNetworkImpl()
    : fc1(register_module("fc1", torch::nn::Linear(4, 128))),
      fc2(register_module("fc2", torch::nn::Linear(128, 128))),
      out(register_module("out", torch::nn::Linear(128, 2))) {}

torch::Tensor QNetworkImpl::forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    return out->forward(x);
}

DQNAgent::DQNAgent(torch::Device device, DQNConfig config, std::uint32_t seed)
    : device_(device),
      config_(config),
      policy_(QNetwork()),
      target_(QNetwork()),
      optimizer_(policy_->parameters(), torch::optim::AdamOptions(config_.learning_rate)),
      replay_(config_.replay_capacity, seed + 1),
      rng_(seed) {
    torch::manual_seed(static_cast<int64_t>(seed));
    policy_->to(device_);
    target_->to(device_);
    sync_target();
    target_->eval();
}

float DQNAgent::epsilon() const {
    const float decay = std::exp(-static_cast<float>(action_steps_) / config_.epsilon_decay_steps);
    return config_.epsilon_end +
           (config_.epsilon_start - config_.epsilon_end) * decay;
}

int DQNAgent::select_action(const std::array<float, 4>& state, bool greedy) {
    ++action_steps_;

    if (!greedy && unit_(rng_) < epsilon()) {
        std::uniform_int_distribution<int> action_dist(0, 1);
        return action_dist(rng_);
    }

    torch::NoGradGuard no_grad;
    auto state_tensor = torch::from_blob(
        const_cast<float*>(state.data()), {1, 4}, torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .to(device_);
    const auto q_values = policy_->forward(state_tensor);
    return q_values.argmax(1).item<int>();
}

void DQNAgent::remember(const Transition& transition) {
    replay_.push(transition);
}

torch::Tensor DQNAgent::states_to_tensor(
    const std::vector<std::array<float, 4>>& states) const {
    std::vector<float> flat;
    flat.reserve(states.size() * 4);
    for (const auto& state : states) {
        flat.insert(flat.end(), state.begin(), state.end());
    }
    return torch::from_blob(
               flat.data(),
               {static_cast<long>(states.size()), 4},
               torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .to(device_);
}

bool DQNAgent::optimize() {
    if (replay_.size() < config_.batch_size) {
        return false;
    }

    const auto batch = replay_.sample(config_.batch_size);

    std::vector<std::array<float, 4>> states;
    std::vector<std::array<float, 4>> next_states;
    std::vector<int64_t> actions;
    std::vector<float> rewards;
    std::vector<float> nonterminal;

    states.reserve(batch.size());
    next_states.reserve(batch.size());
    actions.reserve(batch.size());
    rewards.reserve(batch.size());
    nonterminal.reserve(batch.size());

    for (const auto& t : batch) {
        states.push_back(t.state);
        next_states.push_back(t.next_state);
        actions.push_back(t.action);
        rewards.push_back(t.reward);
        nonterminal.push_back(t.done ? 0.0f : 1.0f);
    }

    auto state_tensor = states_to_tensor(states);
    auto next_state_tensor = states_to_tensor(next_states);
    auto action_tensor = torch::tensor(actions, torch::TensorOptions().dtype(torch::kInt64)).to(device_);
    auto reward_tensor = torch::tensor(rewards, torch::TensorOptions().dtype(torch::kFloat32)).to(device_);
    auto nonterminal_tensor = torch::tensor(nonterminal, torch::TensorOptions().dtype(torch::kFloat32)).to(device_);

    auto current_q = policy_->forward(state_tensor)
                         .gather(1, action_tensor.unsqueeze(1))
                         .squeeze(1);

    torch::Tensor target_q;
    {
        torch::NoGradGuard no_grad;
        const auto max_result = target_->forward(next_state_tensor).max(1);
        const auto next_q = std::get<0>(max_result);
        target_q = reward_tensor + config_.gamma * nonterminal_tensor * next_q;
    }

    const auto loss = torch::smooth_l1_loss(current_q, target_q);

    optimizer_.zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_norm_(policy_->parameters(), config_.grad_clip_norm);
    optimizer_.step();

    ++optimize_steps_;
    if (optimize_steps_ % config_.target_sync_interval == 0) {
        sync_target();
    }

    return true;
}

void DQNAgent::sync_target() {
    torch::NoGradGuard no_grad;
    auto source_params = policy_->named_parameters(true);
    auto target_params = target_->named_parameters(true);
    for (const auto& item : source_params) {
        target_params[item.key()].copy_(item.value());
    }

    auto source_buffers = policy_->named_buffers(true);
    auto target_buffers = target_->named_buffers(true);
    for (const auto& item : source_buffers) {
        target_buffers[item.key()].copy_(item.value());
    }
}

std::size_t DQNAgent::replay_size() const noexcept {
    return replay_.size();
}
