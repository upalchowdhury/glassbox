/* Reading views: original prose and calculated examples, independent of run-data. */
(function(root, factory) {
  const api = factory(root.GlassboxBook, root.GlassboxCourse);
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.GlassboxReader = api;
})(typeof window === 'undefined' ? globalThis : window, function(book, math) {
  'use strict';
  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
  const n = value => Number(value.toFixed(4)).toString();
  const vec = values => '[' + values.map(n).join(', ') + ']';
  const paras = values => values.map(p => `<p>${esc(p)}</p>`).join('');
  const calc = text => `<pre class="worked-calculation">${esc(text)}</pre>`;
  const link = (page, text) => `<a href="#${esc(page)}">${esc(text)}</a>`;
  const table = (caption, headers, rows) => `<div class="reading-table" tabindex="0" role="region" aria-label="${esc(caption)}"><table><caption>${esc(caption)}</caption><thead><tr>${headers.map(h => `<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map((v, i) => i ? `<td>${esc(v)}</td>` : `<th scope="row">${esc(v)}</th>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  function example(id) {
    if (!math) return '<p class="callout">The calculation script is unavailable. The surrounding prose includes the worked values. Keep course-engine.js beside index.html to restore the tables.</p>';
    let html;
    if (id === 'loss') {
      const r = math.lossTrace();
      html = table('One next-token prediction', ['Candidate', 'Probability', 'Logit gradient'], r.probabilities.map((p, i) => [i === 1 ? 'Second (target)' : i === 0 ? 'First' : 'Third', n(p), n(r.gradients[i])])) + calc(`Target loss = −ln(${n(r.probabilities[1])}) = ${n(r.loss)} nats`);
    } else if (id === 'adam') {
      const r = math.adamFirstStep();
      html = calc(`First step, m_old = v_old = 0\nw = ${r.weight}, g = ${r.gradient}, η = ${r.rate}, λ = ${r.decay}\nβ1 = ${r.beta1}, β2 = ${r.beta2}, ε = ${r.epsilon}\nm = ${r.m.toFixed(2)}, v = ${r.v.toFixed(5)}\nm̂ = ${n(r.mHat)}, v̂ = ${n(r.vHat)}\nw_new = 0.999 × 0.4 − 0.1 × 0.2 / (√0.04 + ε)\n      ≈ ${n(r.updated)}`);
    } else if (id === 'moe') {
      const r = math.moeRoute();
      html = table('First token: routing before capacity limits', ['Expert', 'Router probability', 'Top-2 weight'], r.probabilities.map((p, i) => [i, n(p), n(r.weights[i])])) + calc(`Weighted output before capacity = ${vec(r.output)}\nTop-1 comparison = ${vec(math.moeRoute({topK:1}).output)}`) + table('Four-token batch: capacity of two assignments per expert', ['Expert', 'Assigned', 'Accepted', 'Overflow'], r.counts.map((v, i) => [i, v, r.accepted[i], r.overflow[i]]));
    } else if (id === 'lora') {
      const r = math.loraExample();
      html = table('Rank-1 update, scale = 1', ['Output row', 'B × A row', 'Merged W + B × A row'], r.delta.map((row, i) => [i, vec(row), vec(r.merged[i])])) + calc(`Input x = [1, 0]\nBase Wx = ${vec(r.baseOutput.flat())}\nAdapted (W + BA)x = ${vec(r.adaptedOutput.flat())}\nAdapter parameters: ${r.adapterParameters}; full 2×2 matrix: ${r.baseParameters}`);
    } else if (id === 'verifier') {
      const cases = ['{"status":"amber","note":"done"}', '{"status":"green","note":"done"}', '{"status":"amber"}', 'The status is not amber'];
      html = table('Task: set status to amber', ['Response', 'JSON syntax', 'Schema', 'Task check'], cases.map(text => {
        const r = math.verifyStructuredOutput(text, 'amber');
        return [text, r.syntax ? 'Pass' : 'Fail', r.syntax ? (r.schema ? 'Pass' : 'Fail') : 'Not reached', r.schema ? (r.correct ? 'Pass' : 'Fail') : 'Not reached'];
      }));
    } else if (id === 'advantages') {
      const rewards = [1, 0, 1, 0], r = math.groupAdvantages(rewards);
      html = table('One group, population standard deviation', ['Attempt', 'Reward', 'Reward − mean', 'Advantage'], rewards.map((reward, i) => [i + 1, reward, n(reward - r.mean), n(r.advantages[i])])) + calc(`Mean = ${r.mean}; standard deviation = ${r.std}`);
    } else if (id === 'clip') {
      html = table('Clipped surrogate to maximize, ε = 0.2', ['Advantage A', 'Ratio ρ', 'Unclipped ρA', 'Surrogate'], [1, -1].flatMap(a => [0.7, 1.3].map(r => [a, r, n(a*r), n(math.clippedSurrogate(r, a).surrogate)])));
    } else if (id === 'queue') {
      html = table('Constant-rate illustration, empty initial queue, five minutes', ['Rollout workers', 'Produced / min', 'Learner capacity / min', 'Pending after 5 min'], [4, 8].map(workers => {
        const r = math.queueEstimate({workers});
        return [workers, r.producedPerMinute, r.learnerPerMinute, r.finalQueue];
      }));
    } else throw new Error('Unknown worked example: ' + id);
    return `<figure class="worked-example"><figcaption>Worked example · calculated from the stated inputs, rounded for display</figcaption>${html}</figure>`;
  }
  function wordCount(chapter) {
    return [chapter.question, chapter.prerequisite, chapter.outcome, chapter.intuition,
      ...chapter.sections.flatMap(s => [s.title, ...s.paragraphs, s.calculation || '', s.derivation || '']),
      ...chapter.checks.flat(), chapter.takeaway, chapter.bridge].join(' ').trim().split(/\s+/).length;
  }
  function chapterView(chapter, state, fullBook = false) {
    const index = book.chapters.indexOf(chapter), summary = state.depth === 'see' && !fullBook;
    const prefix = fullBook ? chapter.id + '-' : '';
    const route = fullBook ? 'book' : chapter.id;
    const heading = fullBook ? 'h2' : 'h1', sectionHeading = fullBook ? 'h3' : 'h2';
    const done = state.completed.includes(chapter.id);
    return `<article class="reader chapter" id="${chapter.id}" aria-labelledby="${chapter.id}-title">
      <header class="chapter-header"><p class="eyebrow">Chapter ${index + 1} of ${book.chapters.length} · ${wordCount(chapter).toLocaleString()} words</p>
      <${heading} id="${chapter.id}-title">${esc(chapter.title)}</${heading}><p class="chapter-question">${esc(chapter.question)}</p>
      <p class="reading-prerequisite"><strong>Bring with you:</strong> ${esc(chapter.prerequisite)}</p>
      <p><strong>By the end:</strong> ${esc(chapter.outcome)}</p></header>
      <aside class="mental-model"><h2>The idea to hold onto</h2><p>${esc(chapter.intuition)}</p></aside>
      ${summary ? `<div class="summary-notice"><p>You are reading a summary, not the full chapter. The complete explanation includes worked examples and explained review questions.</p><button data-act="setDepth" data-arg="explain">Read the full explanation</button></div>` : `
      <nav class="chapter-toc" aria-label="In this chapter"><h2>In this chapter</h2><ol>${chapter.sections.map((s, i) => `<li>${link(route + '/' + prefix + 'section-' + (i + 1), s.title)}</li>`).join('')}</ol></nav>
      ${chapter.sections.map((s, i) => `<section class="book-section" id="${prefix}section-${i + 1}"><${sectionHeading}><span class="section-number">${i + 1}.</span> ${esc(s.title)}</${sectionHeading}>${paras(s.paragraphs)}${s.calculation ? calc(s.calculation) : ''}${s.example ? example(s.example) : ''}${s.derivation ? `<details class="derivation" ${state.depth === 'derive' || fullBook ? 'open' : ''}><summary>Go deeper · notation and derivation</summary><p>${esc(s.derivation)}</p></details>` : ''}</section>`).join('')}
      <section class="reading-review book-section" id="${prefix}review"><${sectionHeading}>Pause and explain it to yourself</${sectionHeading}><p>No typing, score, or gate. Form an answer in your head, then read the explanation. If it surprises you, revisit the relevant section.</p>
      ${chapter.checks.map(([q, a], i) => `<div class="review-pair"><h3>${i + 1}. ${esc(q)}</h3><p><strong>Explanation.</strong> ${esc(a)}</p></div>`).join('')}</section>`}
      <aside class="chapter-takeaway"><h2>Keep this</h2><p>${esc(chapter.takeaway)}</p></aside>
      <p class="chapter-bridge">${esc(chapter.bridge)}</p>
      ${!fullBook && !summary ? `<div class="reading-actions"><button data-act="markRead" data-focus="read-${chapter.id}" data-arg="${chapter.id}" aria-pressed="${done}">${done ? 'Marked as read · undo' : 'Mark chapter as read'}</button><span>Reading progress only—not a mastery score.</span></div>` : ''}
      <aside class="reading-further"><h2>Optional further reading</h2><p>The chapter is self-contained. Recorded traces are separate measurements, not the source of the hypothetical examples above.</p>
      <p>${chapter.references.map(id => link(id, referenceTitle(id))).join(' · ')}</p>${chapter.sources.length ? `<ul>${chapter.sources.map(([title, url]) => `<li><a href="${esc(url)}" rel="noopener noreferrer" target="_blank">${esc(title)}</a> <span class="small muted">(external source)</span></li>`).join('')}</ul>` : ''}</aside>
      </article>`;
  }
  function referenceTitle(id) {
    return ({forward:'Recorded tensor inspector', update:'Recorded gradient update', pretrain:'Recorded pretraining', generate:'Recorded generation', mlp:'MLP reference', sft:'Recorded SFT', lora:'Recorded LoRA', preference:'Preference reference', rlvr:'Verifier and RL reference', evaluate:'Recorded evaluation', checks:'Numerical checks', map:'Recorded-run scope'})[id] || id;
  }
  function home(state) {
    const count = state.completed.length;
    return `<div class="reader book-home"><p class="eyebrow">The readable guide to language-model systems</p><h1>Understand the whole journey.<br>One idea at a time.</h1>
      <p class="chapter-question">From predicting the next token to learning from rewards and acting with tools.</p>
      <p>Follow Pip through eight connected chapters. Read the explanations, follow small calculations, and use the fully explained questions to check your reasoning. No code to write. No setup to complete. No quizzes to unlock the next page.</p>
      <div class="home-actions">${link('tinygpt', 'Start with how a model learns →')} ${state.bookmark ? link(state.bookmark, 'Resume your bookmark') : ''}</div>
      <div class="reading-contract"><h2>What this book is for</h2><p>A connected conceptual foundation: explain the mechanisms, predict what changes, and diagnose misleading results. Reading alone does not certify practical expertise, and opening a page never marks it as understood.</p>
      <p>Default: <strong>Read</strong> shows the complete chapter. <strong>Deep dive</strong> also opens optional derivations. <strong>Summary</strong> is for orientation and review. ${link('book', 'Read or print the complete book')} · ${link('glossary', 'Open the glossary')}.</p></div>
      <p class="reading-status">${count} of ${book.chapters.length} chapters marked as read · saved in this browser when storage is available.</p>
      <ol class="reading-path">${book.chapters.map((c, i) => `<li><a href="#${c.id}"><span class="chapter-index">${String(i + 1).padStart(2, '0')}</span><span><strong>${esc(c.title)}</strong><span>${esc(c.question)}</span></span>${state.completed.includes(c.id) ? '<span class="read-label">Read</span>' : ''}</a></li>`).join('')}</ol>
      <aside class="reading-further"><h2>Three kinds of evidence, clearly separated</h2><p>Chapter calculations use explicitly stated teaching inputs. Case studies are hypothetical, not benchmarks. The optional reference lessons inspect a small Python model’s recorded tensors and results. None of the reading pages trains a model or executes model-written code.</p><button data-act="toggleAdvanced">Show or hide recorded references</button></aside></div>`;
  }
  function glossaryView() {
    return `<article class="reader"><p class="eyebrow">Return to the mechanism</p><h1>Glossary</h1><p>A quick reminder, with a link to the chapter that explains each term. Definitions support the full reading path; they do not replace it.</p><dl class="book-glossary">${book.glossary.map(([term, definition, page]) => `<div><dt>${esc(term)}</dt><dd>${esc(definition)} ${link(page, 'Read the explanation →')}</dd></div>`).join('')}</dl></article>`;
  }
  function render(page, state) {
    if (!book || !book.chapters || !book.chapters.length) return '<section class="reader"><h1>The reading chapters could not load</h1><p>Keep reading-chapters.js, reading.js, reading.css, and course-engine.js beside index.html, then reload. Recorded references remain separate from the reading content.</p></section>';
    if (page === 'start') return home(state);
    if (page === 'glossary') return glossaryView();
    if (page === 'book') return `<header class="reader book-intro"><p class="eyebrow">Glassbox · complete reading edition</p><h1>The whole journey</h1><p>All eight chapters, calculations, review explanations, and derivations. No interactions are required to reveal essential content.</p><button data-act="printBook">Print / save as PDF</button><p class="small muted">Uses your browser’s print dialog. Printing does not mark chapters as read.</p><ol>${book.chapters.map(c => `<li>${link('book/' + c.id, c.title)}</li>`).join('')}</ol></header>` + book.chapters.map(c => chapterView(c, state, true)).join('') + glossaryView();
    const chapter = book.chapters.find(c => c.id === page);
    return chapter ? chapterView(chapter, state) : '';
  }
  return {render, example, wordCount, escape:esc};
});
