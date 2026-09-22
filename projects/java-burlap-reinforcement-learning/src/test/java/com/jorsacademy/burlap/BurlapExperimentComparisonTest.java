package com.jorsacademy.burlap;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class BurlapExperimentComparisonTest {

    @Test
    void valueIterationFindsShortPolicy() {
        int steps = BurlapExperimentComparison.valueIterationSteps();
        assertTrue(steps > 0);
        assertTrue(steps < 100, "Value Iteration should solve Four Rooms efficiently");
    }
}
