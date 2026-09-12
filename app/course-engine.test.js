const test = require('node:test');
const assert = require('node:assert/strict');
const course = require('./course-engine.js');

test('MoE top-k masks all but the selected experts and reports imbalance', () => {
  const trace = course.moeRoute({ topK: 2, collapse: true, capacity: 2 });
  assert.equal(trace.selected.length, 2);
  assert.equal(trace.masked.filter(value => value > 0).length, 2);
  assert.deepEqual(trace.dropped, [6, 0, 0, 0]);
  assert.ok(trace.auxiliaryLoss > 1);
});

test('LoRA update dimensions and parameter count follow B @ A', () => {
  const update = course.loraUpdate(2);
  assert.deepEqual(update.shape, [4, 4]);
  assert.equal(update.delta.length, 4);
  assert.equal(update.delta[0].length, 4);
  assert.equal(update.trainableParameters, 16);
});

test('structured and code verifiers reject reward-hacking attempts', () => {
  assert.equal(course.verifyStructuredOutput('{"status":"green","note":"done"}').ok, true);
  assert.equal(course.verifyStructuredOutput('{"status":"green"}').ok, false);
  assert.equal(course.verifyCodeRepair('function add(a,b) { return a + b; }').ok, true);
  assert.equal(course.verifyCodeRepair('function add(a,b) { return 2; }').ok, false);
});

test('GRPO advantages are centered and zero when every reward is equal', () => {
  const mixed = course.groupAdvantages([0, 1, 1, 0]);
  assert.deepEqual(mixed.advantages, [-1, 1, 1, -1]);
  assert.deepEqual(course.groupAdvantages([1, 1, 1]).advantages, [0, 0, 0]);
});

test('course splits are isolated', () => {
  assert.equal(course.splitIsolated(course.fixtures.environment.structured), true);
  assert.equal(course.splitIsolated(course.fixtures.environment.code), true);
});

test('environment gate unlocks the deterministic GRPO main flow', () => {
  const spec = { reset: true, termination: true, verifier: true, heldout: true, antiHack: false };
  assert.equal(course.environmentValid(spec), false);
  spec.antiHack = true;
  assert.equal(course.environmentValid(spec), true);
  const trace = course.grpoTrace({ groupSize: 8 });
  assert.equal(trace.rollouts.length, 8);
  assert.equal(trace.advantages.reduce((sum, value) => sum + value, 0), 0);
});
