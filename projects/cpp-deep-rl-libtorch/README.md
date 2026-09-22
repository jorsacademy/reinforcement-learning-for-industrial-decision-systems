# C++ Deep Reinforcement Learning with LibTorch

A compact, native C++17 Deep Q-Network (DQN) implementation using LibTorch and a custom CartPole environment.

The project intentionally avoids Python bindings in the training loop. Environment dynamics, replay buffer, action selection and optimization are implemented in C++.

## Features

- Native C++17 CartPole environment
- LibTorch neural network and optimizer
- DQN with target network
- Experience replay
- Epsilon-greedy exploration
- Huber loss (`smooth_l1_loss`)
- Gradient clipping
- Deterministic seeding
- Automatic CUDA use when available
- CMake build

## Network

```text
state (4)
  -> Linear(4, 128)
  -> ReLU
  -> Linear(128, 128)
  -> ReLU
  -> Linear(128, 2)
  -> Q(left), Q(right)
```

## Requirements

- CMake 3.18+
- C++17 compiler
- LibTorch

Download a LibTorch distribution matching your platform from the official PyTorch website.

## Build

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release -j
```

On Linux/macOS, run:

```bash
./train_dqn
```

On multi-config generators such as Visual Studio:

```powershell
.\Release\train_dqn.exe
```

## Training behavior

The executable trains DQN on a native CartPole implementation. It prints episode return, rolling 20-episode mean, epsilon and replay-buffer size.

The run stops early when the mean return over the last 20 episodes reaches 475, or after 500 episodes.

Example output format:

```text
Device: CPU
Episode  10 | return   31.0 | avg20   24.8 | epsilon 0.978 | replay 248
...
Solved: average return over the last 20 episodes >= 475.
Training steps: ...
Elapsed seconds: ...
Training throughput: ... env steps/s
```

Numbers above are illustrative. Actual convergence and throughput depend on hardware, LibTorch build and compiler settings.

## Project structure

```text
.
├── CMakeLists.txt
├── include
│   ├── cartpole.hpp
│   └── dqn.hpp
└── src
    ├── cartpole.cpp
    ├── dqn.cpp
    └── main.cpp
```

## Notes on performance

The environment is native C++ and therefore avoids Python/Gym step-call overhead. Release builds enable compiler optimization (`-O3` on GCC/Clang). LibTorch can use CUDA automatically when `torch::cuda::is_available()` returns true.

For small networks such as CartPole DQN, CPU execution may be faster than GPU execution because kernel-launch and host/device synchronization overhead can dominate computation. Benchmark both on your own machine before assuming CUDA is faster.

## Possible extensions

- Double DQN
- Dueling DQN
- Prioritized experience replay
- Vectorized native environments
- PPO in LibTorch
- Arcade Learning Environment (ALE) integration
- SUMO traffic-control environments

## License

This repository is licensed under the **JORS Academy Non-Commercial Source License 1.0**. Commercial use is prohibited without a separate prior written commercial license. See [`LICENSE`](LICENSE) for the complete terms.
