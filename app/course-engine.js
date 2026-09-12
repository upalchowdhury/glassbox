/* Glassbox course simulations. Deliberately small, deterministic, and browser-safe. */
(function attachCourseEngine(root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.GlassboxCourse = api;
}(typeof window !== 'undefined' ? window : globalThis, function courseEngine() {
  'use strict';

  const round = (value, places = 4) => Number(Number(value).toFixed(places));
  const softmax = values => {
    const peak = Math.max(...values);
    const exp = values.map(v => Math.exp(v - peak));
    const total = exp.reduce((a, b) => a + b, 0);
    return exp.map(v => v / total);
  };
  const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
  const matmul = (left, right) => left.map(row => right[0].map((_, col) =>
    row.reduce((sum, value, i) => sum + value * right[i][col], 0)));

  const fixtures = Object.freeze({
    tinyGPT: {
      prompt: 'pip has ', tokenIds: [35, 29, 35, 5, 28, 21, 38, 5],
      target: 't', targetId: 39,
      checkpoints: {
        0: { loss: 3.8074, targetProbability: 0.0219, adamStep: 0, weight: 0.018, gradient: 0.116 },
        10: { loss: 2.4361, targetProbability: 0.0874, adamStep: 10, weight: 0.012, gradient: 0.042 },
        100: { loss: 1.484, targetProbability: 0.2271, adamStep: 100, weight: -0.004, gradient: 0.009 },
      },
    },
    moe: {
      residual: [0.6, -0.2, 0.8],
      normalLogits: [1.7, 0.9, 0.2, -0.3],
      collapseLogits: [4.8, 0.3, 0.2, 0.1],
      experts: [[0.72, -0.11, 0.31], [0.12, 0.48, -0.08], [-0.34, 0.22, 0.51], [0.41, 0.05, -0.27]],
      normalDispatch: [0, 1, 0, 2, 1, 3, 0, 2],
      collapsedDispatch: [0, 0, 0, 0, 0, 0, 0, 0],
    },
    lora: {
      input: 4, output: 4,
      A: [[0.2, -0.1, 0.3, 0.1], [-0.2, 0.4, 0.1, -0.3], [0.1, 0.2, -0.2, 0.3], [0.3, -0.2, 0.2, 0.1]],
      B: [[0.3, -0.1, 0.2, 0.1], [-0.2, 0.4, -0.1, 0.2], [0.1, 0.2, 0.3, -0.2], [0.2, -0.3, 0.1, 0.4]],
    },
    environment: {
      structured: { train: ['set-status-green', 'set-status-blue'], validation: ['set-status-amber'], heldout: ['set-status-red'], schema: '{"status":"green","note":"short"}' },
      code: { train: ['fix_add'], validation: ['fix_subtract'], heldout: ['fix_clamp'], files: ['math.js', 'math.test.js', 'README.md'] },
    },
  });

  function moeRoute({ topK = 2, collapse = false, capacity = 3 } = {}) {
    const logits = collapse ? fixtures.moe.collapseLogits : fixtures.moe.normalLogits;
    const probabilities = softmax(logits);
    const selected = probabilities.map((probability, expert) => ({ expert, probability }))
      .sort((a, b) => b.probability - a.probability).slice(0, Math.max(1, Math.min(4, topK)));
    const selectedMass = selected.reduce((sum, item) => sum + item.probability, 0);
    const masked = probabilities.map((probability, expert) => selected.some(item => item.expert === expert)
      ? probability / selectedMass : 0);
    const output = fixtures.moe.experts[0].map((_, channel) => round(masked.reduce(
      (sum, probability, expert) => sum + probability * fixtures.moe.experts[expert][channel], 0)));
    const dispatch = collapse ? fixtures.moe.collapsedDispatch : fixtures.moe.normalDispatch;
    const counts = [0, 1, 2, 3].map(expert => dispatch.filter(item => item === expert).length);
    const dropped = counts.map(count => Math.max(0, count - capacity));
    const dispatchFraction = counts.map(count => count / dispatch.length);
    const meanProbability = collapse ? probabilities : [0.31, 0.27, 0.24, 0.18];
    const auxiliaryLoss = round(4 * dot(meanProbability, dispatchFraction));
    return { logits, probabilities: probabilities.map(v => round(v)), selected, masked: masked.map(v => round(v)), output,
      counts, dropped, auxiliaryLoss, collapse, capacity, selectedMass: round(selectedMass) };
  }

  function loraUpdate(rank = 2) {
    const r = Math.max(1, Math.min(4, Number(rank)));
    const A = fixtures.lora.A.slice(0, r);
    const B = fixtures.lora.B.map(row => row.slice(0, r));
    const delta = matmul(B, A).map(row => row.map(value => round(value)));
    return { rank: r, A, B, delta, shape: [fixtures.lora.output, fixtures.lora.input],
      trainableParameters: r * (fixtures.lora.input + fixtures.lora.output), fullParameters: fixtures.lora.input * fixtures.lora.output,
      frozenBase: true };
  }

  function verifyStructuredOutput(text) {
    try {
      const value = JSON.parse(text);
      const ok = typeof value === 'object' && value !== null && value.status === 'green' && typeof value.note === 'string';
      return { ok, evidence: ok ? 'JSON parsed; status is exactly "green" and note is a string.' : 'Expected JSON with status "green" and a string note.' };
    } catch (_) { return { ok: false, evidence: 'Not valid JSON; prose and partial JSON are rejected.' }; }
  }

  function verifyCodeRepair(source) {
    const hasFix = /return\s+a\s*\+\s*b\s*;?/.test(source);
    const hardCodes = /return\s+2\s*;?/.test(source);
    const ok = hasFix && !hardCodes;
    return { ok, tests: [{ name: 'adds positive values', pass: ok }, { name: 'adds negative values', pass: ok },
      { name: 'does not hard-code one fixture', pass: !hardCodes }], evidence: ok ? 'All pinned tests pass.' : 'The pinned tests reject this patch.' };
  }

  function splitIsolated(environment) {
    const groups = [environment.train, environment.validation, environment.heldout];
    return groups.every((group, index) => group.every(item => groups.every((other, otherIndex) =>
      otherIndex === index || !other.includes(item))));
  }

  function environmentValid(spec) {
    return Boolean(spec && spec.reset && spec.termination && spec.verifier && spec.heldout && spec.antiHack);
  }

  function groupAdvantages(rewards) {
    const mean = rewards.reduce((sum, reward) => sum + reward, 0) / rewards.length;
    const variance = rewards.reduce((sum, reward) => sum + (reward - mean) ** 2, 0) / rewards.length;
    const std = Math.sqrt(variance);
    return { mean: round(mean), std: round(std), advantages: std === 0 ? rewards.map(() => 0) : rewards.map(reward => round((reward - mean) / std)) };
  }

  function grpoTrace({ brokenVerifier = false, groupSize = 8, rewardWeight = 1, sparse = true } = {}) {
    const base = [1, 0, 1, 0, 1, 0, 1, 0];
    const completions = ['{"status":"green","note":"fixed"}', '{"status":"green"}', '{"status":"green","note":"ok"}', 'green',
      '{"status":"green","note":"done"}', '{"status":"red","note":"wrong"}', '{"status":"green","note":"fixed"}', 'answer: green'];
    const count = Math.max(2, Math.min(8, Number(groupSize)));
    const rewards = base.slice(0, count).map((reward, index) => round((brokenVerifier && index === 3 ? 1 : reward) * rewardWeight));
    const stats = groupAdvantages(rewards);
    const rollouts = completions.slice(0, count).map((text, index) => ({ text, reward: rewards[index], advantage: stats.advantages[index],
      tokens: 7 + index * 2, termination: rewards[index] ? 'submit_solution()' : 'schema failure',
      evidence: brokenVerifier && index === 3 ? 'Broken substring verifier accepted “green”.' : rewards[index] ? 'Strict verifier passed exact JSON.' : 'Strict verifier rejected the output.' }));
    return { rollouts, ...stats, clippedRatio: 1.18, clipEpsilon: 0.2, sparse, brokenVerifier,
      policyChange: 'Positive-advantage trajectories increase in probability; negative-advantage trajectories decrease.' };
  }

  function scaleEstimate({ dataWorkers = 4, expertWorkers = 2, rolloutWorkers = 4, episodeSeconds = 18 } = {}) {
    const rolloutRate = round(rolloutWorkers * 60 / episodeSeconds, 2);
    const trainRate = round(dataWorkers * 3.2, 2);
    const queue = round(Math.max(0, rolloutRate - trainRate), 2);
    const utilization = round(Math.min(0.96, 0.32 + dataWorkers * 0.07 + expertWorkers * 0.08 + rolloutWorkers * 0.025), 2);
    return { dataWorkers, expertWorkers, rolloutWorkers, episodeSeconds, rolloutRate, trainRate, queue, utilization,
      checkpointMinutes: round(8 + (dataWorkers + expertWorkers) * 0.7, 1), costPerHour: round(dataWorkers * 1.4 + expertWorkers * 1.8 + rolloutWorkers * 0.6, 2) };
  }

  return { fixtures, softmax, moeRoute, loraUpdate, verifyStructuredOutput, verifyCodeRepair, splitIsolated,
    environmentValid, groupAdvantages, grpoTrace, scaleEstimate };
}));
