#include "cartpole.hpp"

#include <cmath>
#include <stdexcept>

CartPole::CartPole(std::uint32_t seed) : rng_(seed) {
    reset();
}

std::array<float, 4> CartPole::reset() {
    for (auto& value : state_) {
        value = init_dist_(rng_);
    }
    steps_ = 0;
    return state_;
}

StepResult CartPole::step(int action) {
    if (action != 0 && action != 1) {
        throw std::invalid_argument("CartPole action must be 0 or 1");
    }

    float x = state_[0];
    float x_dot = state_[1];
    float theta = state_[2];
    float theta_dot = state_[3];

    const float force = action == 1 ? force_mag_ : -force_mag_;
    const float costheta = std::cos(theta);
    const float sintheta = std::sin(theta);

    const float temp =
        (force + polemass_length_ * theta_dot * theta_dot * sintheta) / total_mass_;

    const float thetaacc =
        (gravity_ * sintheta - costheta * temp) /
        (length_ * (4.0f / 3.0f - masspole_ * costheta * costheta / total_mass_));

    const float xacc = temp - polemass_length_ * thetaacc * costheta / total_mass_;

    x += tau_ * x_dot;
    x_dot += tau_ * xacc;
    theta += tau_ * theta_dot;
    theta_dot += tau_ * thetaacc;

    state_ = {x, x_dot, theta, theta_dot};
    ++steps_;

    const bool done =
        x < -x_threshold_ || x > x_threshold_ ||
        theta < -theta_threshold_radians_ || theta > theta_threshold_radians_ ||
        steps_ >= max_steps_;

    return StepResult{state_, 1.0f, done};
}
