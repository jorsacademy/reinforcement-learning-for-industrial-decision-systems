#include "cartpole.hpp"
#include "dqn.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
struct RunConfig {
    int episodes{500};
    int warmup_steps{1000};
    std::uint32_t seed{42};
};

int parse_positive_int(const std::string& value, const char* flag, bool allow_zero = false) {
    std::size_t parsed = 0;
    const int result = std::stoi(value, &parsed);
    if (parsed != value.size() || result < (allow_zero ? 0 : 1)) {
        throw std::invalid_argument(std::string("Invalid value for ") + flag + ": " + value);
    }
    return result;
}

RunConfig parse_args(int argc, char** argv) {
    RunConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("Missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--episodes") {
            config.episodes = parse_positive_int(require_value("--episodes"), "--episodes");
        } else if (arg == "--warmup-steps") {
            config.warmup_steps = parse_positive_int(
                require_value("--warmup-steps"), "--warmup-steps", true);
        } else if (arg == "--seed") {
            const int seed = parse_positive_int(require_value("--seed"), "--seed", true);
            config.seed = static_cast<std::uint32_t>(seed);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: train_dqn [--episodes N] [--warmup-steps N] [--seed N]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }
    return config;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const auto run = parse_args(argc, argv);

        torch::set_num_threads(std::max(1, static_cast<int>(std::thread::hardware_concurrency())));

        const torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Device: " << (device.is_cuda() ? "CUDA" : "CPU") << '\n';
        std::cout << "Episodes: " << run.episodes
                  << " | warmup steps: " << run.warmup_steps
                  << " | seed: " << run.seed << '\n';

        CartPole env(run.seed);
        DQNAgent agent(device, DQNConfig{}, run.seed);

        std::vector<float> returns;
        returns.reserve(static_cast<std::size_t>(run.episodes));
        int total_steps = 0;

        const auto started = std::chrono::steady_clock::now();

        for (int episode = 1; episode <= run.episodes; ++episode) {
            auto state = env.reset();
            float episode_return = 0.0f;

            for (;;) {
                const int action = agent.select_action(state);
                const auto result = env.step(action);

                agent.remember(Transition{
                    state,
                    static_cast<int64_t>(action),
                    result.reward,
                    result.state,
                    result.done,
                });

                ++total_steps;
                if (total_steps >= run.warmup_steps) {
                    agent.optimize();
                }

                state = result.state;
                episode_return += result.reward;

                if (result.done) {
                    break;
                }
            }

            returns.push_back(episode_return);

            if (episode % 10 == 0 || episode == run.episodes) {
                const std::size_t window = std::min<std::size_t>(20, returns.size());
                const float average = std::accumulate(returns.end() - window, returns.end(), 0.0f) /
                                      static_cast<float>(window);

                std::cout << "Episode " << std::setw(3) << episode
                          << " | return " << std::setw(6) << episode_return
                          << " | avg20 " << std::fixed << std::setprecision(1) << std::setw(6) << average
                          << " | epsilon " << std::setprecision(3) << agent.epsilon()
                          << " | replay " << agent.replay_size() << '\n';

                if (average >= 475.0f && window == 20) {
                    std::cout << "Solved: average return over the last 20 episodes >= 475.\n";
                    break;
                }
            }
        }

        const auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();

        std::cout << "Training steps: " << total_steps << '\n';
        std::cout << "Elapsed seconds: " << std::fixed << std::setprecision(2) << elapsed << '\n';
        if (elapsed > 0.0) {
            std::cout << "Training throughput: "
                      << static_cast<double>(total_steps) / elapsed
                      << " env steps/s\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
