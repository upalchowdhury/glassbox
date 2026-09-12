/* Transparent worked-example arithmetic, not a training simulator or benchmark. */
(function(root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.GlassboxCourse = api;
})(typeof window === 'undefined' ? globalThis : window, function() {
  'use strict';
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  function requireValues(values) {
    if (!Array.isArray(values) || !values.length || !values.every(finite)) throw new RangeError('Expected a nonempty array of finite numbers.');
  }
  function softmax(logits) {
    requireValues(logits);
    const peak = Math.max(...logits);
    const values = logits.map(x => Math.exp(x - peak));
    const sum = values.reduce((a, b) => a + b, 0);
    return values.map(x => x / sum);
  }
  function lossTrace(logits = [Math.log(2), 0, 0], target = 1) {
    requireValues(logits);
    if (!Number.isInteger(target) || target < 0 || target >= logits.length) throw new RangeError('Invalid target.');
    const probabilities = softmax(logits);
    const peak = Math.max(...logits);
    const loss = peak - logits[target] + Math.log(logits.reduce((s, z) => s + Math.exp(z - peak), 0));
    return { logits, probabilities, loss, gradients: probabilities.map((p, j) => p - Number(j === target)) };
  }
  function adamFirstStep({weight = 0.4, gradient = 0.2, rate = 0.1, decay = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8} = {}) {
    if (![weight, gradient, rate, decay, beta1, beta2, epsilon].every(finite) ||
        rate < 0 || decay < 0 || beta1 < 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1 || epsilon <= 0) throw new RangeError('Invalid optimizer inputs.');
    const m = (1 - beta1) * gradient, v = (1 - beta2) * gradient ** 2;
    const mHat = m / (1 - beta1), vHat = v / (1 - beta2);
    return { weight, gradient, rate, decay, beta1, beta2, epsilon, m, v, mHat, vHat,
      updated: (1 - rate * decay) * weight - rate * mHat / (Math.sqrt(vHat) + epsilon) };
  }
  function matmul(a, b) {
    if (!Array.isArray(a) || !a.length || !Array.isArray(b) || !b.length) throw new RangeError('Empty matrix.');
    a.forEach(requireValues); b.forEach(requireValues);
    if (a.some(row => row.length !== b.length) || b.some(row => row.length !== b[0].length)) throw new RangeError('Incompatible matrix shapes.');
    return a.map(row => b[0].map((_, j) => row.reduce((sum, x, i) => sum + x * b[i][j], 0)));
  }
  function loraExample() {
    const W = [[1, 0], [0, 1]], A = [[1, 2]], B = [[0.1], [0.2]], x = [[1], [0]];
    const delta = matmul(B, A);
    const merged = W.map((row, i) => row.map((v, j) => v + delta[i][j]));
    return { W, A, B, delta, merged, baseOutput: matmul(W, x), adaptedOutput: matmul(merged, x),
      adapterParameters: 4, baseParameters: 4 };
  }
  function moeRoute({topK = 2, capacity = 2} = {}) {
    if (!Number.isInteger(topK) || topK < 1 || topK > 4 || !Number.isInteger(capacity) || capacity < 0) throw new RangeError('Invalid routing settings.');
    const batch = [[0.6, 0.3, 0.08, 0.02], [0.6, 0.3, 0.08, 0.02], [0.6, 0.08, 0.3, 0.02], [0.6, 0.08, 0.02, 0.3]];
    const experts = [[2, 0], [0, 3], [1, 1], [-1, 1]];
    const selections = batch.map(ps => ps.map((p, id) => ({p, id})).sort((a, b) => b.p - a.p || a.id - b.id).slice(0, topK));
    const probabilities = batch[0];
    const mass = selections[0].reduce((s, item) => s + item.p, 0);
    const weights = probabilities.map((p, id) => selections[0].some(x => x.id === id) ? p / mass : 0);
    const output = experts[0].map((_, j) => weights.reduce((s, w, i) => s + w * experts[i][j], 0));
    const counts = experts.map((_, id) => selections.reduce((s, row) => s + Number(row.some(x => x.id === id)), 0));
    return {probabilities, weights, output, counts, accepted: counts.map(n => Math.min(n, capacity)), overflow: counts.map(n => Math.max(0, n - capacity)),
      assignments: counts.reduce((s, n) => s + n, 0), capacity,
      // Output above is the first token's weighted merge BEFORE capacity dispatch.
      outputStage: 'before capacity'};
  }
  function verifyStructuredOutput(text, requestedStatus) {
    if (typeof requestedStatus !== 'string' || !requestedStatus.trim()) throw new RangeError('The task must specify a status.');
    let value;
    try { value = JSON.parse(text); }
    catch (_) { return {syntax: false, schema: false, correct: false, ok: false, evidence: 'Rejected: not valid JSON.'}; }
    const schema = value !== null && !Array.isArray(value) && typeof value === 'object' &&
      Object.keys(value).length === 2 && Object.hasOwn(value, 'status') && Object.hasOwn(value, 'note') &&
      typeof value.status === 'string' && typeof value.note === 'string' && value.note.trim().length > 0 && value.note.length <= 80;
    const correct = Boolean(schema && value.status === requestedStatus);
    return {syntax: true, schema: Boolean(schema), correct, ok: correct,
      evidence: !schema ? 'Rejected: expected exactly status and a nonempty note of at most 80 characters.' :
        correct ? 'Accepted: schema and requested status match. Note helpfulness is not assessed.' : 'Rejected: valid schema, wrong requested status.'};
  }
  function groupAdvantages(rewards) {
    requireValues(rewards);
    const mean = rewards.reduce((s, r) => s + r, 0) / rewards.length;
    const std = Math.sqrt(rewards.reduce((s, r) => s + (r - mean) ** 2, 0) / rewards.length);
    return {mean, std, advantages: std === 0 ? rewards.map(() => 0) : rewards.map(r => (r - mean) / std)};
  }
  function clippedSurrogate(ratio, advantage, epsilon = 0.2) {
    if (![ratio, advantage, epsilon].every(finite) || ratio < 0 || epsilon < 0 || epsilon >= 1) throw new RangeError('Invalid surrogate inputs.');
    const clipped = Math.min(1 + epsilon, Math.max(1 - epsilon, ratio));
    return {ratio, advantage, clipped, surrogate: Math.min(ratio * advantage, clipped * advantage)};
  }
  function queueEstimate({workers = 4, episodeSeconds = 20, learnerPerMinute = 8, minutes = 5, initialQueue = 0} = {}) {
    if (![workers, episodeSeconds, learnerPerMinute, minutes, initialQueue].every(finite) ||
        !Number.isInteger(workers) || workers < 0 || episodeSeconds <= 0 || learnerPerMinute < 0 || minutes < 0 || initialQueue < 0) throw new RangeError('Invalid queue inputs.');
    const producedPerMinute = workers * 60 / episodeSeconds;
    const netPerMinute = producedPerMinute - learnerPerMinute;
    return {producedPerMinute, learnerPerMinute, netPerMinute, finalQueue: Math.max(0, initialQueue + netPerMinute * minutes)};
  }
  return {softmax, lossTrace, adamFirstStep, matmul, loraExample, moeRoute, verifyStructuredOutput, groupAdvantages, clippedSurrogate, queueEstimate};
});
