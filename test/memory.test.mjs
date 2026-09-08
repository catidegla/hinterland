import test from 'node:test';
import assert from 'node:assert/strict';

import { Memory } from '../src/memory.mjs';
import { NullEmbedder, cosine } from '../src/embed.mjs';
import { toMatchQuery, fuse } from '../src/search.mjs';

/**
 * A deterministic embedder. Each dimension counts a keyword, so two texts
 * about the same subject land near each other without needing a model.
 */
class FakeEmbedder {
  constructor({ available = true, model = 'fake-embed' } = {}) {
    this.model = model;
    this.isAvailable = available;
    this.calls = 0;
    this.lastError = null;
    this.vocabulary = ['payment', 'invoice', 'deploy', 'server', 'benin', 'lagos', 'cat', 'dog'];
  }

  async available() {
    return this.isAvailable;
  }

  async embed(texts) {
    this.calls += 1;
    if (!this.isAvailable) {
      this.lastError = 'offline';
      return null;
    }

    return texts.map((text) => {
      const lower = text.toLowerCase();
      return new Float32Array(this.vocabulary.map((word) => (lower.includes(word) ? 1 : 0)));
    });
  }
}

const fresh = (embedder) => Memory.open(':memory:', embedder ? { embedder } : {});

test('it works with no embedder at all', async (t) => {
  const memory = fresh();
  t.after(() => memory.close());

  await memory.remember('The payment gateway for Benin uses MTN MoMo');
  await memory.remember('Deploy the server with pm2 on the staging box');

  const hits = await memory.recall('payment gateway');

  // The whole offline-first claim. No model, no network, still retrieves.
  assert.equal(hits.length, 1);
  assert.match(hits[0].content, /payment gateway/);
  assert.ok(hits[0].ranks.lexical, 'the match came from full text search');
  assert.equal(hits[0].ranks.semantic, undefined);
});

test('a memory is written even when the embedder is down', async (t) => {
  const embedder = new FakeEmbedder({ available: false });
  const memory = fresh(embedder);
  t.after(() => memory.close());

  const written = await memory.remember('The invoice service retries three times');

  // A write that depends on a reachable model is a write that loses data on a
  // connection that drops.
  assert.ok(written.id);
  assert.equal(memory.stats().memories, 1);
  assert.equal(memory.stats().embedded, 0);
  assert.equal(memory.pending(), 1);

  const hits = await memory.recall('invoice');
  assert.equal(hits.length, 1, 'and it is still retrievable');
});

test('backfill embeds what was written offline, once the model returns', async (t) => {
  const embedder = new FakeEmbedder({ available: false });
  const memory = fresh(embedder);
  t.after(() => memory.close());

  await memory.remember('The payment gateway for Benin uses MTN MoMo');
  await memory.remember('Deploy the server with pm2');
  assert.equal(memory.pending(), 2);

  // Connectivity returns.
  embedder.isAvailable = true;
  const result = await memory.backfill();

  assert.equal(result.embedded, 2);
  assert.equal(result.remaining, 0);
  assert.equal(memory.stats().embedded, 2);
});

test('backfill reports why it did nothing rather than pretending it worked', async (t) => {
  const embedder = new FakeEmbedder({ available: false });
  const memory = fresh(embedder);
  t.after(() => memory.close());

  await memory.remember('anything at all');
  const result = await memory.backfill();

  assert.equal(result.embedded, 0);
  assert.equal(result.remaining, 1);
  assert.ok(result.reason, 'a silent no-op would look identical to success');
});

test('semantic recall finds a memory that shares no words with the query', async (t) => {
  const embedder = new FakeEmbedder();
  const memory = fresh(embedder);
  t.after(() => memory.close());

  await memory.remember('Mobile money settlement in Benin');
  await memory.remember('Feeding schedule for the dog');

  // "payment" appears in neither memory, so lexical search cannot help. The
  // fake embedder puts it in the same dimension as the first one.
  const hits = await memory.recall('payment');

  assert.ok(hits.length >= 1);
  assert.match(hits[0].content, /Mobile money/);
  assert.ok(hits[0].ranks.semantic, 'this can only have come from the vector side');
});

test('both rankers contribute when both can answer', async (t) => {
  const embedder = new FakeEmbedder();
  const memory = fresh(embedder);
  t.after(() => memory.close());

  await memory.remember('The payment gateway in Benin settles overnight');
  await memory.remember('Unrelated note about a cat');

  const hits = await memory.recall('payment benin');
  const top = hits[0];

  assert.ok(top.ranks.lexical, 'lexical found it');
  assert.ok(top.ranks.semantic, 'so did semantic');
});

test('vectors from a different model are never compared', async (t) => {
  const oldModel = new FakeEmbedder({ model: 'old-model' });
  const memory = fresh(oldModel);
  t.after(() => memory.close());

  await memory.remember('Mobile money settlement in Benin');
  assert.equal(memory.stats().embedded, 1);

  // Swapping the model leaves vectors that describe a different space.
  // Comparing across them produces confident nonsense.
  memory.embedder = new FakeEmbedder({ model: 'new-model' });

  const hits = await memory.recall('payment');
  assert.equal(hits.length, 0, 'no lexical match and no comparable vector');
  assert.equal(memory.pending(), 1, 'and it is queued for re-embedding');
});

test('rewriting content invalidates the old vector', async (t) => {
  const embedder = new FakeEmbedder();
  const memory = fresh(embedder);
  t.after(() => memory.close());

  const first = await memory.remember('Notes about the dog');
  assert.equal(memory.stats().embedded, 1);

  await memory.remember('Notes about the deploy server', { id: first.id });

  // The vector described text that no longer exists.
  const hits = await memory.recall('deploy');
  assert.equal(hits.length, 1);
  assert.match(hits[0].content, /deploy server/);
  assert.equal(memory.stats().memories, 1);
});

test('scope keeps separate memories apart', async (t) => {
  const memory = fresh();
  t.after(() => memory.close());

  await memory.remember('The invoice service retries', { scope: 'work' });
  await memory.remember('The invoice from the garage', { scope: 'personal' });

  assert.equal((await memory.recall('invoice')).length, 2);
  assert.equal((await memory.recall('invoice', { scope: 'work' })).length, 1);
  assert.match((await memory.recall('invoice', { scope: 'personal' }))[0].content, /garage/);
});

test('punctuation in a query does not become an FTS syntax error', async (t) => {
  const memory = fresh();
  t.after(() => memory.close());

  await memory.remember('The payment gateway returned an error');

  // Each of these is valid FTS5 syntax that means something other than what
  // the user typed, or is simply a parse error.
  for (const query of ['payment AND', '"payment', 'payment OR NOT', 'payment*', "payment's", 'payment -gateway']) {
    const hits = await memory.recall(query);
    assert.ok(Array.isArray(hits), `query "${query}" should not throw`);
  }

  assert.equal((await memory.recall('"payment gateway"')).length, 1);
});

test('forget removes the memory and its vector', async (t) => {
  const embedder = new FakeEmbedder();
  const memory = fresh(embedder);
  t.after(() => memory.close());

  const written = await memory.remember('Mobile money settlement in Benin');
  assert.equal(memory.stats().embedded, 1);

  assert.equal(memory.forget(written.id), true);
  assert.equal(memory.forget(written.id), false, 'forgetting twice is not an error');

  assert.equal(memory.stats().memories, 0);
  assert.equal(memory.stats().embedded, 0, 'the vector went with it');
  assert.equal((await memory.recall('benin')).length, 0);
});

test('a batch write costs one embedding call, not one per item', async (t) => {
  const embedder = new FakeEmbedder();
  const memory = fresh(embedder);
  t.after(() => memory.close());

  await memory.rememberMany([
    { content: 'first note about payment' },
    { content: 'second note about deploy' },
    { content: 'third note about server' },
  ]);

  assert.equal(memory.stats().memories, 3);
  assert.equal(memory.stats().embedded, 3);
  // On a slow link, three round trips instead of one is the difference
  // between usable and not.
  assert.equal(embedder.calls, 1);
});

test('recall counts are tracked for later pruning', async (t) => {
  const memory = fresh();
  t.after(() => memory.close());

  const written = await memory.remember('The payment gateway settles overnight');

  await memory.recall('payment');
  await memory.recall('gateway');

  assert.equal(memory.get(written.id).recalled, 2);
  assert.ok(memory.get(written.id).lastRecalledAt);
});

test('an empty or whitespace query returns nothing rather than everything', async (t) => {
  const memory = fresh();
  t.after(() => memory.close());

  await memory.remember('something');

  assert.deepEqual(await memory.recall(''), []);
  assert.deepEqual(await memory.recall('   '), []);
  // Single characters are dropped as terms, so this has nothing to match on.
  assert.deepEqual(await memory.recall('a'), []);
});

test('remember rejects an empty memory instead of storing a blank row', async (t) => {
  const memory = fresh();
  t.after(() => memory.close());

  await assert.rejects(() => memory.remember(''), TypeError);
  await assert.rejects(() => memory.remember('   '), TypeError);
  await assert.rejects(() => memory.remember(null), TypeError);
});

test('tags and metadata survive a round trip', async (t) => {
  const memory = fresh();
  t.after(() => memory.close());

  const written = await memory.remember('Deploy notes', {
    tags: ['ops', 'staging'],
    metadata: { author: 'chris', priority: 2 },
    source: 'runbook.md',
  });

  const read = memory.get(written.id);
  assert.deepEqual(read.tags, ['ops', 'staging']);
  assert.deepEqual(read.metadata, { author: 'chris', priority: 2 });
  assert.equal(read.source, 'runbook.md');
});

test('the match query builder quotes terms and drops noise', () => {
  assert.equal(toMatchQuery('payment gateway'), '"payment" OR "gateway"');
  assert.equal(toMatchQuery('payment AND gateway'), '"payment" OR "and" OR "gateway"');
  assert.equal(toMatchQuery('a b'), null, 'single characters carry no signal');
  assert.equal(toMatchQuery('!!!'), null);
  assert.match(toMatchQuery('say "hi"'), /"say"/);
});

test('fusion ranks by position, not by score scale', () => {
  // BM25 returns large negative numbers and cosine returns 0 to 1. Any
  // weighted sum of the two is a fudge factor that needs retuning forever.
  const lexical = [{ id: 'a' }, { id: 'b' }, { id: 'c' }];
  const semantic = [{ id: 'c' }, { id: 'a' }, { id: 'd' }];

  const fused = fuse([
    { name: 'lexical', rows: lexical },
    { name: 'semantic', rows: semantic },
  ], { limit: 4 });

  // "a" is first and second; "c" is third and first. Both beat the singles.
  assert.equal(fused[0].id, 'a');
  assert.ok(['b', 'd'].includes(fused[2].id));
  assert.deepEqual(fused.find((f) => f.id === 'a').ranks, { lexical: 1, semantic: 2 });
});

test('cosine handles the degenerate cases without dividing by zero', () => {
  assert.equal(cosine(new Float32Array([0, 0]), new Float32Array([1, 1])), 0);
  assert.equal(cosine(new Float32Array([1, 0]), new Float32Array([1, 0, 0])), 0, 'mismatched lengths');
  assert.equal(Math.round(cosine(new Float32Array([1, 1]), new Float32Array([1, 1]))), 1);
});

test('the null embedder never reaches the network', async () => {
  const embedder = new NullEmbedder();

  assert.equal(await embedder.available(), false);
  assert.equal(await embedder.embed(['anything']), null);
  assert.equal(embedder.model, null);
});
