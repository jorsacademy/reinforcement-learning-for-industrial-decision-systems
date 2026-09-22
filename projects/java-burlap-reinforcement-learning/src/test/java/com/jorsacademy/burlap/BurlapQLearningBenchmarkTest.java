package com.jorsacademy.burlap;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BurlapQLearningBenchmarkTest {

    @Test
    void trainingProducesBoundedEpisodeStatistics() {
        int episodes = 20;
        int maxSteps = 100;
        BurlapQLearningBenchmark.TrainingResult result =
                BurlapQLearningBenchmark.train(episodes, maxSteps, 42L);

        assertEquals(episodes, result.episodeSteps().size());
        assertTrue(result.bestEpisodeSteps() >= 0);
        assertTrue(result.bestEpisodeSteps() <= maxSteps);
        assertFalse(Double.isNaN(result.firstWindowAverage()));
        assertFalse(Double.isNaN(result.lastWindowAverage()));
        assertTrue(result.episodeSteps().stream().allMatch(s -> s >= 0 && s <= maxSteps));
    }

    @Test
    void sameSeedReproducesEpisodeSequence() {
        var first = BurlapQLearningBenchmark.train(25, 150, 123456L);
        var second = BurlapQLearningBenchmark.train(25, 150, 123456L);
        assertEquals(first.episodeSteps(), second.episodeSteps());
        assertEquals(first.firstWindowAverage(), second.firstWindowAverage());
        assertEquals(first.lastWindowAverage(), second.lastWindowAverage());
    }

    @Test
    void invalidEpisodeCountIsRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> BurlapQLearningBenchmark.train(1, 100));
    }

    @Test
    void invalidStepLimitIsRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> BurlapQLearningBenchmark.train(10, 0));
    }
}
