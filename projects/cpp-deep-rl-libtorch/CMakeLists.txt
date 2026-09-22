cmake_minimum_required(VERSION 3.18)
project(cpp_deep_rl_libtorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

add_executable(train_dqn
    src/main.cpp
    src/cartpole.cpp
    src/dqn.cpp
)

target_include_directories(train_dqn PRIVATE include)
target_link_libraries(train_dqn PRIVATE "${TORCH_LIBRARIES}")

if (MSVC)
    target_compile_options(train_dqn PRIVATE /W4 /permissive-)
else()
    target_compile_options(train_dqn PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()

set_property(TARGET train_dqn PROPERTY CXX_STANDARD 17)
