package com.jorsacademy.burlap;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.singleagent.SADomain;
import burlap.statehashing.simple.SimpleHashableStateFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** Multi-seed Q-learning benchmark with Value Iteration reference and uncertainty reporting. */
public final class BurlapExperimentComparison {

    public static final int SEED_COUNT = 20;
    public static final long BASE_SEED = 202600L;
    public static final int MOVING_AVERAGE_WINDOW = 25;
    private static final double Z_95 = 1.96;

    private static final Path RESULTS_DIR = Path.of("results");
    private static final Path CURVE_CSV = RESULTS_DIR.resolve("learning-curve.csv");
    private static final Path SEED_CSV = RESULTS_DIR.resolve("seed-summary.csv");
    private static final Path SVG_PATH = RESULTS_DIR.resolve("learning-curve.svg");

    private BurlapExperimentComparison() {}

    public static void main(String[] args) throws IOException {
        MultiSeedResult result = runMultiSeedBenchmark(SEED_COUNT);
        int viSteps = valueIterationSteps();

        Files.createDirectories(RESULTS_DIR);
        writeCurveCsv(result.curve(), viSteps);
        writeSeedCsv(result.runs());
        writeSvg(result.curve(), viSteps);

        System.out.println("BURLAP experiment: Q-Learning vs Value Iteration");
        System.out.println("Independent Q-learning seeds : " + result.runs().size());
        System.out.println("Episodes per seed            : " + BurlapQLearningBenchmark.TRAINING_EPISODES);
        System.out.printf(Locale.US, "Mean first-50 actions        : %.2f%n", result.meanFirstWindow());
        System.out.printf(Locale.US, "Mean last-50 actions         : %.2f%n", result.meanLastWindow());
        System.out.printf(Locale.US, "Mean improvement             : %.2f%%%n", result.improvementPercent());
        System.out.printf(Locale.US, "Last-50 across-seed SD       : %.2f%n", result.lastWindowSd());
        System.out.printf(Locale.US, "Last-50 95%% CI              : [%.2f, %.2f]%n",
                result.lastWindowCiLow(), result.lastWindowCiHigh());
        System.out.println("Value Iteration rollout      : " + viSteps + " actions");
        System.out.println("Aggregate CSV                : " + CURVE_CSV);
        System.out.println("Per-seed CSV                 : " + SEED_CSV);
        System.out.println("Learning curve               : " + SVG_PATH);
    }

    public static MultiSeedResult runMultiSeedBenchmark(int seedCount) {
        if (seedCount < 2) {
            throw new IllegalArgumentException("seedCount must be at least 2");
        }

        List<SeedRun> runs = new ArrayList<>(seedCount);
        for (int i = 0; i < seedCount; i++) {
            long seed = BASE_SEED + i;
            BurlapQLearningBenchmark.TrainingResult training = BurlapQLearningBenchmark.train(
                    BurlapQLearningBenchmark.TRAINING_EPISODES,
                    BurlapQLearningBenchmark.MAX_STEPS_PER_EPISODE,
                    seed);
            runs.add(new SeedRun(seed, training));
        }

        List<PointStats> curve = aggregateMovingAverageCurve(runs);
        double meanFirst = mean(runs.stream().map(r -> r.training().firstWindowAverage()).toList());
        List<Double> lastValues = runs.stream().map(r -> r.training().lastWindowAverage()).toList();
        double meanLast = mean(lastValues);
        double lastSd = sampleSd(lastValues, meanLast);
        double margin = Z_95 * lastSd / Math.sqrt(seedCount);

        return new MultiSeedResult(
                List.copyOf(runs), List.copyOf(curve), meanFirst, meanLast, lastSd,
                Math.max(0.0, meanLast - margin), meanLast + margin);
    }

    static List<PointStats> aggregateMovingAverageCurve(List<SeedRun> runs) {
        int episodes = runs.get(0).training().episodeSteps().size();
        List<PointStats> curve = new ArrayList<>(episodes);
        for (int episode = 0; episode < episodes; episode++) {
            List<Double> values = new ArrayList<>(runs.size());
            for (SeedRun run : runs) {
                List<Integer> steps = run.training().episodeSteps();
                int from = Math.max(0, episode - MOVING_AVERAGE_WINDOW + 1);
                values.add(steps.subList(from, episode + 1).stream()
                        .mapToInt(Integer::intValue).average().orElse(0.0));
            }
            double avg = mean(values);
            double sd = sampleSd(values, avg);
            double margin = Z_95 * sd / Math.sqrt(values.size());
            curve.add(new PointStats(episode + 1, avg, sd,
                    Math.max(0.0, avg - margin), avg + margin));
        }
        return curve;
    }

    public static int valueIterationSteps() {
        GridWorldDomain gridWorld = new GridWorldDomain(
                BurlapQLearningBenchmark.GRID_SIZE, BurlapQLearningBenchmark.GRID_SIZE);
        gridWorld.setMapToFourRooms();
        gridWorld.setTf(new GridWorldTerminalFunction(
                BurlapQLearningBenchmark.GOAL_X, BurlapQLearningBenchmark.GOAL_Y));
        SADomain domain = gridWorld.generateDomain();
        GridWorldState initial = new GridWorldState(0, 0);

        ValueIteration planner = new ValueIteration(
                domain,
                BurlapQLearningBenchmark.DISCOUNT,
                new SimpleHashableStateFactory(),
                1e-5,
                1000);
        planner.toggleReachabiltiyTerminalStatePruning(true);
        GreedyQPolicy policy = planner.planFromState(initial);
        Episode rollout = PolicyUtils.rollout(
                policy, initial, domain.getModel(), BurlapQLearningBenchmark.MAX_STEPS_PER_EPISODE);
        return rollout.numActions();
    }

    static void writeCurveCsv(List<PointStats> curve, int viSteps) throws IOException {
        StringBuilder out = new StringBuilder("episode,mean_moving_average_25,std_dev,ci95_low,ci95_high,value_iteration_actions\n");
        for (PointStats p : curve) {
            out.append(p.episode()).append(',')
                    .append(format(p.mean())).append(',')
                    .append(format(p.sd())).append(',')
                    .append(format(p.ciLow())).append(',')
                    .append(format(p.ciHigh())).append(',')
                    .append(viSteps).append('\n');
        }
        Files.writeString(CURVE_CSV, out.toString(), StandardCharsets.UTF_8);
    }

    static void writeSeedCsv(List<SeedRun> runs) throws IOException {
        StringBuilder out = new StringBuilder("seed,first_50_avg,last_50_avg,improvement_percent,best_episode_actions\n");
        for (SeedRun run : runs) {
            var t = run.training();
            out.append(run.seed()).append(',')
                    .append(format(t.firstWindowAverage())).append(',')
                    .append(format(t.lastWindowAverage())).append(',')
                    .append(format(t.improvementPercent())).append(',')
                    .append(t.bestEpisodeSteps()).append('\n');
        }
        Files.writeString(SEED_CSV, out.toString(), StandardCharsets.UTF_8);
    }

    static void writeSvg(List<PointStats> curve, int viSteps) throws IOException {
        int width = 1000, height = 560;
        int left = 70, right = 25, top = 35, bottom = 65;
        int plotW = width - left - right, plotH = height - top - bottom;
        double maxY = Math.max(50.0, curve.stream().mapToDouble(PointStats::ciHigh).max().orElse(50.0));

        StringBuilder meanLine = new StringBuilder();
        StringBuilder band = new StringBuilder();
        for (int i = 0; i < curve.size(); i++) {
            PointStats p = curve.get(i);
            double x = x(i, curve.size(), left, plotW);
            meanLine.append(point(x, y(p.mean(), maxY, top, plotH)));
            band.append(point(x, y(p.ciHigh(), maxY, top, plotH)));
        }
        for (int i = curve.size() - 1; i >= 0; i--) {
            PointStats p = curve.get(i);
            band.append(point(x(i, curve.size(), left, plotW), y(p.ciLow(), maxY, top, plotH)));
        }
        double viY = y(viSteps, maxY, top, plotH);

        String svg = """
                <svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560" viewBox="0 0 1000 560">
                  <rect width="1000" height="560" fill="white"/>
                  <text x="500" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">BURLAP Four Rooms: Q-Learning Across 20 Seeds</text>
                  <line x1="70" y1="35" x2="70" y2="495" stroke="black"/>
                  <line x1="70" y1="495" x2="975" y2="495" stroke="black"/>
                  <text x="520" y="545" text-anchor="middle" font-family="sans-serif" font-size="14">Training episode</text>
                  <text x="18" y="265" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 18 265)">Actions per episode</text>
                  <polygon points="%s" fill="#1f77b4" fill-opacity="0.18" stroke="none"/>
                  <polyline points="%s" fill="none" stroke="#1f77b4" stroke-width="2.5"/>
                  <line x1="70" y1="%.2f" x2="975" y2="%.2f" stroke="#d62728" stroke-width="2" stroke-dasharray="7 5"/>
                  <text x="965" y="%.2f" text-anchor="end" font-family="sans-serif" font-size="13">Value Iteration: %d actions</text>
                  <text x="80" y="55" font-family="sans-serif" font-size="13">Mean 25-episode moving average; shaded area = 95%% CI</text>
                </svg>
                """.formatted(band, meanLine, viY, viY, Math.max(50, viY - 7), viSteps);
        Files.writeString(SVG_PATH, svg, StandardCharsets.UTF_8);
    }

    private static double mean(List<Double> values) {
        return values.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
    }

    private static double sampleSd(List<Double> values, double mean) {
        if (values.size() < 2) return 0.0;
        double sum = values.stream().mapToDouble(v -> (v - mean) * (v - mean)).sum();
        return Math.sqrt(sum / (values.size() - 1));
    }

    private static String format(double v) {
        return String.format(Locale.US, "%.4f", v);
    }

    private static double x(int index, int count, int left, int plotW) {
        return left + plotW * (index / (double) Math.max(1, count - 1));
    }

    private static double y(double value, double maxY, int top, int plotH) {
        return top + plotH * (1.0 - Math.min(value, maxY) / maxY);
    }

    private static String point(double x, double y) {
        return String.format(Locale.US, "%.2f,%.2f ", x, y);
    }

    public record SeedRun(long seed, BurlapQLearningBenchmark.TrainingResult training) {}
    public record PointStats(int episode, double mean, double sd, double ciLow, double ciHigh) {}

    public record MultiSeedResult(
            List<SeedRun> runs,
            List<PointStats> curve,
            double meanFirstWindow,
            double meanLastWindow,
            double lastWindowSd,
            double lastWindowCiLow,
            double lastWindowCiHigh
    ) {
        public double improvementPercent() {
            return meanFirstWindow == 0.0
                    ? 0.0
                    : 100.0 * (meanFirstWindow - meanLastWindow) / meanFirstWindow;
        }
    }
}
