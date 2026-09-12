const test = require('node:test');
const assert = require('node:assert/strict');
const math = require('./course-engine.js');
const book = require('./reading-chapters.js');
const reader = require('./reading.js');
require('./course-content.js');
const near = (actual, expected, epsilon = 1e-10) => assert.ok(Math.abs(actual - expected) < epsilon, `${actual} != ${expected}`);
const state = {depth:'explain', completed:[], bookmark:null};

test('cross-entropy, softmax, and analytic logit gradients agree', () => {
  const r = math.lossTrace();
  assert.deepEqual(r.probabilities, [0.5, 0.25, 0.25]);
  near(r.loss, Math.log(4));
  assert.deepEqual(r.gradients, [0.5, -0.75, 0.25]);
  r.logits.forEach((z, i) => {
    const plus = [...r.logits], minus = [...r.logits];
    plus[i] = z + 1e-5; minus[i] = z - 1e-5;
    near((math.lossTrace(plus).loss - math.lossTrace(minus).loss) / 2e-5, r.gradients[i], 1e-8);
  });
  near(math.lossTrace([1000, -1000], 1).loss, 2000);
  assert.throws(() => math.lossTrace([], 0), RangeError);
  assert.throws(() => math.lossTrace([1], 3), RangeError);
});

test('first AdamW step includes moments, bias correction, and decoupled decay', () => {
  const r = math.adamFirstStep();
  near(r.m, 0.02); near(r.v, 0.00004); near(r.mHat, 0.2); near(r.vHat, 0.04);
  near(r.updated, 0.2996000049999998);
  near(math.adamFirstStep({rate:0}).updated, 0.4);
  near(math.adamFirstStep({gradient:0}).updated, 0.3996);
  assert.throws(() => math.adamFirstStep({beta1:1}), RangeError);
});

test('MoE top-k controls both merge and batch assignments; capacity counts contributions', () => {
  const two = math.moeRoute();
  assert.deepEqual(two.counts, [4, 2, 1, 1]);
  assert.deepEqual(two.accepted, [2, 2, 1, 1]);
  assert.deepEqual(two.overflow, [2, 0, 0, 0]);
  assert.equal(two.assignments, 8);
  near(two.weights[0], 2/3); near(two.weights[1], 1/3);
  near(two.output[0], 4/3); near(two.output[1], 1);
  const one = math.moeRoute({topK:1});
  assert.equal(one.assignments, 4); assert.deepEqual(one.counts, [4, 0, 0, 0]);
  assert.deepEqual(one.output, [2, 0]);
  assert.deepEqual(math.moeRoute({capacity:0}).accepted, [0, 0, 0, 0]);
  assert.equal(math.moeRoute({topK:4}).assignments, 16);
  assert.throws(() => math.moeRoute({topK:1.5}), RangeError);
});

test('LoRA merge and separate branches produce the same output', () => {
  const r = math.loraExample();
  assert.deepEqual(r.delta, [[0.1, 0.2], [0.2, 0.4]]);
  const branch = math.matmul(r.B, math.matmul(r.A, [[1], [0]]));
  r.adaptedOutput.forEach((row, i) => near(row[0], r.baseOutput[i][0] + branch[i][0]));
  assert.deepEqual(r.W, [[1, 0], [0, 1]]);
  assert.equal(r.adapterParameters, r.baseParameters);
  assert.throws(() => math.matmul([[1, 2]], [[1]]), RangeError);
});

test('JSON verifier is task-aware and separates syntax, schema, and correctness', () => {
  const green = '{"status":"green","note":"done"}';
  assert.equal(math.verifyStructuredOutput(green, 'green').ok, true);
  const wrong = math.verifyStructuredOutput(green, 'amber');
  assert.equal(wrong.syntax, true); assert.equal(wrong.schema, true); assert.equal(wrong.ok, false);
  const missing = math.verifyStructuredOutput('{"status":"amber"}', 'amber');
  assert.equal(missing.syntax, true); assert.equal(missing.schema, false);
  for (const text of ['null', '[]', '"amber"', 'not amber', '{"status":"amber","note":""}',
    '{"status":"amber","note":"ok","unexpected":true}', '{"status":"amber","note":' + JSON.stringify('x'.repeat(81)) + '}']) {
    assert.equal(math.verifyStructuredOutput(text, 'amber').ok, false, text);
  }
  assert.throws(() => math.verifyStructuredOutput(green), RangeError);
});

test('no source-text test matcher or fabricated training/benchmark API remains', () => {
  for (const key of ['verifyCodeRepair', 'grpoTrace', 'scaleEstimate', 'environmentValid', 'fixtures']) assert.equal(math[key], undefined);
});

test('advantages use the documented population convention and handle equal groups', () => {
  const r = math.groupAdvantages([1, 0, 1, 0]);
  near(r.mean, 0.5); near(r.std, 0.5);
  assert.deepEqual(r.advantages, [1, -1, 1, -1]);
  assert.deepEqual(math.groupAdvantages([10, 0, 10, 0]).advantages, r.advantages);
  assert.deepEqual(math.groupAdvantages([12, 2, 12, 2]).advantages, r.advantages);
  for (const rewards of [[0, 0], [1, 1], [5]]) assert.ok(math.groupAdvantages(rewards).advantages.every(x => x === 0));
  assert.throws(() => math.groupAdvantages([]), RangeError);
});

test('clipping is sign-dependent, not a universal clamp', () => {
  near(math.clippedSurrogate(1.3, 1).surrogate, 1.2);
  near(math.clippedSurrogate(1.3, -1).surrogate, -1.3);
  near(math.clippedSurrogate(0.7, 1).surrogate, 0.7);
  near(math.clippedSurrogate(0.7, -1).surrogate, -0.8);
  assert.throws(() => math.clippedSurrogate(-1, 1), RangeError);
});

test('queue estimates use stated rates and cannot invent utilization or negative pending work', () => {
  assert.equal(math.queueEstimate().finalQueue, 20);
  assert.equal(math.queueEstimate({workers:8}).finalQueue, 80);
  assert.equal(math.queueEstimate({workers:0, initialQueue:2}).finalQueue, 0);
  assert.equal(math.queueEstimate().utilization, undefined);
  assert.throws(() => math.queueEstimate({episodeSeconds:0}), RangeError);
});

test('every chapter has a complete self-contained reading path with explained questions', () => {
  assert.equal(book.chapters.length, 8);
  assert.deepEqual(book.chapters.map(c => c.id), globalThis.GlassboxCourseContent.modules.map(m => m[0]));
  assert.equal(new Set(book.chapters.map(c => c.id)).size, 8);
  book.chapters.forEach(c => {
    assert.ok(c.sections.length >= 4, c.id);
    assert.ok(reader.wordCount(c) >= 600, c.id);
    assert.ok(c.checks.length >= 3, c.id);
    c.checks.forEach(([q, a]) => { assert.ok(q.length > 20); assert.ok(a.length > 60); });
    const html = reader.render(c.id, state);
    for (const s of c.sections) {
      assert.ok(html.includes(reader.escape(s.title)), s.title);
      if (s.example) assert.ok(reader.example(s.example).includes('Worked example'));
    }
    for (const [, a] of c.checks) assert.ok(html.includes(reader.escape(a)));
    assert.ok(!html.includes('<textarea') && !html.includes('<input'));
  });
});

test('summary is explicitly incomplete; book always exposes every derivation and explanation', () => {
  const summary = reader.render('tinygpt', {...state, depth:'see'});
  assert.ok(summary.includes('not the full chapter'));
  assert.ok(!summary.includes('data-act="markRead"'));
  const full = reader.render('book', {...state, depth:'see'});
  for (const c of book.chapters) for (const s of c.sections) if (s.derivation) assert.ok(full.includes(reader.escape(s.derivation)));
  assert.ok(full.includes('Print / save as PDF'));
  assert.ok(full.includes('Glossary'));
  assert.ok(book.glossary.length >= 35);
  for (const [, , id] of book.glossary) assert.ok(book.chapters.some(c => c.id === id));
});

test('content rendering escapes HTML and does not imply completion from a visit', () => {
  assert.equal(reader.escape('<script>"&'), '&lt;script&gt;&quot;&amp;');
  assert.ok(reader.render('start', state).includes('0 of 8'));
  assert.deepEqual(state.completed, []);
  assert.ok(reader.render('start', {...state, completed:['tinygpt']}).includes('1 of 8'));
});
