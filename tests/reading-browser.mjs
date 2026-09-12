/* No npm dependencies. Node 22+ and Chrome/Chromium. Tests use a disposable profile. */
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createServer} from 'node:http';
import {readFile, mkdtemp, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, resolve, dirname} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const app = resolve(dirname(fileURLToPath(import.meta.url)), '../app');
const chromePath = process.env.CHROME_BIN || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'chromium');
const html = await readFile(join(app, 'index.html'), 'utf8');
const allowed = new Set(['index.html', 'course-engine.js', 'course-content.js', 'reading-chapters.js', 'reading.js', 'reading.css']);
const server = createServer(async (req, res) => {
  const name = new URL(req.url, 'http://localhost').pathname.slice(1) || 'index.html';
  try {
    if (name === 'bad-data.html') {
      res.setHeader('Content-Type', 'text/html');
      res.end(html.replace(/(<script id="run-data" type="application\/json">)[\s\S]*?(<\/script>)/, '$1{}$2'));
    } else if (name === 'missing-content.html') {
      res.setHeader('Content-Type', 'text/html');
      res.end(html.replace('<script src="reading-chapters.js"></script>', ''));
    } else if (allowed.has(name)) {
      res.setHeader('Content-Type', name.endsWith('.js') ? 'text/javascript' : name.endsWith('.css') ? 'text/css' : 'text/html');
      res.end(await readFile(join(app, name)));
    } else { res.writeHead(404); res.end(); }
  } catch (err) { res.writeHead(500); res.end(String(err)); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const base = `http://127.0.0.1:${server.address().port}`;
const profile = await mkdtemp(join(tmpdir(), 'glassbox-reading-browser-'));
let chrome, socket, seq = 0;
const pending = new Map(), errors = [];
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
try {
  chrome = spawn(chromePath, ['--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
    '--disable-background-networking', '--disable-extensions', '--disable-component-update', '--remote-debugging-port=0',
    `--user-data-dir=${profile}`, 'about:blank'], {stdio:['ignore', 'ignore', 'pipe']});
  const debug = await new Promise((resolve, reject) => {
    let output = '';
    const timer = setTimeout(() => reject(new Error('Chrome debugger did not start: ' + output.slice(-1000))), 20000);
    chrome.once('error', err => {clearTimeout(timer); reject(err);});
    chrome.stderr.on('data', data => {
      output += data;
      const match = output.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (match) { clearTimeout(timer); resolve(new URL(match[1]).origin.replace('ws:', 'http:')); }
    });
    chrome.once('exit', code => {clearTimeout(timer); reject(new Error('Chrome exited: ' + code));});
  });
  const target = await (await fetch(debug + '/json/new?about:blank', {method:'PUT'})).json();
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  socket.onmessage = event => {
    const msg = JSON.parse(event.data);
    if (msg.id && pending.has(msg.id)) {
      const {resolve, reject, timer} = pending.get(msg.id);
      clearTimeout(timer); pending.delete(msg.id);
      if (msg.error) reject(new Error(JSON.stringify(msg.error))); else resolve(msg.result);
    } else if (msg.method === 'Runtime.exceptionThrown') errors.push(msg.params.exceptionDetails);
  };
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++seq;
    const timer = setTimeout(() => {pending.delete(id); reject(new Error('CDP timed out: ' + method));}, 10000);
    pending.set(id, {resolve, reject, timer});
    socket.send(JSON.stringify({id, method, params}));
  });
  const evaluate = async expression => {
    const r = await send('Runtime.evaluate', {expression, returnByValue:true, awaitPromise:true});
    if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails));
    return r.result.value;
  };
  const waitFor = async (expression, label) => {
    for (let i = 0; i < 100; i++) { if (await evaluate(expression)) return; await delay(50); }
    throw new Error('Not ready: ' + label);
  };
  const navigate = async (url, condition) => {
    await send('Page.navigate', {url});
    await waitFor(`location.href === ${JSON.stringify(url)} && document.readyState === 'complete' && (${condition})`, url);
  };
  const route = async id => {
    await evaluate(`location.hash = ${JSON.stringify(id)}`);
    await waitFor(`state.page === ${JSON.stringify(id.split('/')[0])} && document.getElementById('main').dataset.route === ${JSON.stringify(id)}`, id);
  };
  await send('Runtime.enable'); await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride', {width:1280, height:900, deviceScaleFactor:1, mobile:false});
  await navigate(base + '/index.html', `document.querySelector('.book-home')`);
  assert.equal(await evaluate('state.depth'), 'explain');
  assert.deepEqual(await evaluate('state.completed'), []);
  const chapters = await evaluate('GlassboxBook.chapters.map(c => c.id)');
  for (const id of chapters) {
    await route(id);
    assert.ok(await evaluate('document.querySelectorAll(".book-section").length >= 5'), id);
    assert.ok(await evaluate('document.querySelectorAll(".review-pair").length >= 3'), id);
    assert.equal(await evaluate('document.querySelector("#main textarea, #main input") === null'), true);
    assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true, id + ' desktop overflow');
  }
  assert.deepEqual(await evaluate('state.completed'), [], 'visits must not count as read');
  await route('tinygpt');
  await evaluate(`document.getElementById('depth').value = 'see'; document.getElementById('depth').dispatchEvent(new Event('change', {bubbles:true}));`);
  assert.equal(await evaluate('document.querySelectorAll(".book-section").length'), 0);
  await evaluate(`document.querySelector('[data-act="setDepth"]').click()`);
  assert.ok(await evaluate('document.querySelectorAll(".book-section").length > 0'));
  await evaluate(`document.getElementById('depth').value = 'derive'; document.getElementById('depth').dispatchEvent(new Event('change', {bubbles:true}));`);
  assert.equal(await evaluate('document.querySelectorAll(".derivation:not([open])").length'), 0);
  await evaluate(`document.querySelector('[data-act="markRead"]').click()`);
  assert.deepEqual(await evaluate('state.completed'), ['tinygpt']);
  await evaluate(`document.querySelectorAll('.chapter-toc a')[2].focus()`);
  await send('Input.dispatchKeyEvent', {type:'keyDown', key:'Enter', code:'Enter', windowsVirtualKeyCode:13});
  await send('Input.dispatchKeyEvent', {type:'keyUp', key:'Enter', code:'Enter', windowsVirtualKeyCode:13});
  await waitFor(`document.getElementById('main').dataset.route === 'tinygpt/section-3'`, 'keyboard section link');
  await route('tinygpt/section-3');
  await evaluate(`document.getElementById('bookmark-course').click()`);
  assert.equal(await evaluate('state.bookmark'), 'tinygpt/section-3');
  await navigate(base + '/index.html?reload=1#tinygpt/section-3', `state.page === 'tinygpt' && document.querySelector('.chapter')`);
  assert.deepEqual(await evaluate('state.completed'), ['tinygpt']);
  assert.equal(await evaluate('state.depth'), 'derive');
  await route('moe');
  await evaluate(`document.getElementById('resume-course').click()`);
  await waitFor(`location.hash === '#tinygpt/section-3' && state.page === 'tinygpt'`, 'resume bookmark');
  assert.ok(await evaluate('document.getElementById("section-3").getBoundingClientRect().top >= 0'));
  await evaluate(`document.querySelector('[data-act="markRead"]').focus(); document.querySelector('[data-act="markRead"]').click()`);
  assert.deepEqual(await evaluate('state.completed'), []);
  assert.equal(await evaluate('document.activeElement.dataset.focus'), 'read-tinygpt');
  await route('grpo'); // no environment gate
  await route('forward');
  assert.ok(await evaluate('document.querySelectorAll(".cell").length > 0'), 'recorded tensor grid');
  await evaluate(`document.getElementById('checkpoint').value = 'base'; document.getElementById('checkpoint').dispatchEvent(new Event('change', {bubbles:true}));`);
  assert.equal(await evaluate('state.checkpoint'), 'base');
  await route('tinygpt');
  assert.equal(await evaluate(`document.querySelector('.lesson-nav .next').dataset.goto`), 'moe', 'references must not alter reading sequence');
  for (const id of ['tokenize', 'update', 'mlp', 'pretrain', 'generate', 'sft', 'lora', 'preference', 'rlvr', 'evaluate', 'checks', 'map']) {
    await route(id); assert.ok(await evaluate('document.getElementById("main").innerText.length > 100'), id);
  }
  await route('book');
  assert.equal(await evaluate('document.querySelectorAll("article.chapter").length'), 8);
  assert.equal(await evaluate('document.querySelectorAll(".derivation:not([open])").length'), 0);
  assert.equal(await evaluate(`document.querySelectorAll('[data-act="markRead"]').length`), 0);
  assert.equal(await evaluate(`document.querySelectorAll('[id]').length === new Set([...document.querySelectorAll('[id]')].map(e => e.id)).size`), true, 'unique full-book anchors');
  await send('Emulation.setEmulatedMedia', {media:'print'});
  assert.equal(await evaluate(`getComputedStyle(document.querySelector('.sidebar')).display`), 'none');
  const pdf = await send('Page.printToPDF', {printBackground:true});
  assert.ok(pdf.data.length > 100000, 'substantial printable book');
  await send('Emulation.setEmulatedMedia', {media:''});
  await route('glossary');
  assert.ok(await evaluate('document.querySelectorAll("dt").length >= 35'));
  await send('Emulation.setDeviceMetricsOverride', {width:390, height:844, deviceScaleFactor:1, mobile:true});
  for (const id of ['start', ...chapters, 'book', 'glossary']) {
    await route(id);
    assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true, id + ' mobile overflow');
  }
  await route('tinygpt');
  const screenshotArg = process.argv.find(x => x.startsWith('--screenshot='));
  if (screenshotArg) {
    const screenshot = await send('Page.captureScreenshot', {format:'png'});
    await writeFile(screenshotArg.slice('--screenshot='.length), Buffer.from(screenshot.data, 'base64'));
  }
  await evaluate(`localStorage.setItem('glassbox.reading.v2', '{broken json')`);
  await navigate(base + '/index.html?storage=corrupt#tinygpt', 'state.page === "tinygpt"');
  assert.equal(await evaluate('state.depth'), 'explain');
  assert.deepEqual(await evaluate('state.completed'), []);
  const injected = await send('Page.addScriptToEvaluateOnNewDocument', {source:`Storage.prototype.setItem = function() { throw new Error('blocked storage'); }; Storage.prototype.getItem = function() { throw new Error('blocked storage'); };`});
  await navigate(base + '/index.html?storage=blocked#moe', 'state.page === "moe"');
  await evaluate(`document.querySelector('[data-act="markRead"]').click()`);
  assert.ok(await evaluate('document.getElementById("live").innerText.includes("Storage unavailable")'));
  await send('Page.removeScriptToEvaluateOnNewDocument', {identifier:injected.identifier});
  await navigate(base + '/bad-data.html#grpo', 'state.page === "grpo" && document.querySelector(".chapter")');
  assert.equal(await evaluate('document.getElementById("boot-error").hidden'), false);
  await navigate(base + '/missing-content.html', 'document.getElementById("main").innerText.includes("chapters could not load")');
  await navigate(pathToFileURL(join(app, 'index.html')).href + '#environment', 'state.page === "environment" && document.querySelector(".chapter")');
  assert.equal(errors.length, 0, JSON.stringify(errors));
  console.log('PASS: eight chapters, all references, depths, ungated reading, progress/undo, bookmark/reload, glossary, printable PDF, mobile layout, storage failures, missing content/data, and offline file access.');
} finally {
  socket?.close();
  if (chrome && chrome.exitCode === null) {
    chrome.kill('SIGTERM');
    await Promise.race([new Promise(resolve => chrome.once('exit', resolve)), delay(3000)]);
    if (chrome.exitCode === null) chrome.kill('SIGKILL');
  }
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
  // Only the exact disposable Chrome profile created above is removed.
  await rm(profile, {recursive:true, force:true, maxRetries:3});
}
