<div align="center">

# hinterland

**Agent memory that works where the network does not.**

One SQLite file. Full text recall with no model at all, semantic recall when one is reachable, and backfill for everything written offline.

[![CI](https://github.com/catidegla/hinterland/actions/workflows/ci.yml/badge.svg)](https://github.com/catidegla/hinterland/actions/workflows/ci.yml)
[![Zero dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)](package.json)
[![Node](https://img.shields.io/badge/node-%E2%89%A522.5-339933)](package.json)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

```js
import { Memory } from 'hinterland';

const memory = Memory.open('./memory.db');

await memory.remember('The Benin gateway settles overnight, so refunds lag by a day');
const hits = await memory.recall('why are refunds slow');
```

No API key. No server. No embedding model. That works, right now, on a machine that has never been online.

## The problem this is shaped around

Every agent memory library assumes the network. `remember()` calls an embedding API, so a write fails when the connection drops. On a link that drops for hours at a time, a memory store that refuses to write until it can reach a model is not a memory store, it is an outage.

hinterland inverts that. **The write is durable first and embedded second**, and the embedding is allowed to fail:

```js
await memory.remember('...');   // returns as soon as it is on disk
memory.pending();               // 1, no vector yet
await memory.recall('...');     // still finds it, on full text search
```

Later, when the model is reachable again:

```js
await memory.backfill();        // { embedded: 47, remaining: 0 }
```

Nothing was lost while it was offline, and nothing had to be retried by hand.

## Retrieval degrades, it does not break

| What you have | What recall uses |
| :--- | :--- |
| Nothing | BM25 full text search, built into SQLite |
| An embedding model | BM25 **and** vector search, fused |
| A model that was there and is now gone | BM25, plus every vector already stored |

The two rankers are combined with **reciprocal rank fusion**, not a weighted sum. BM25 scores are large negatives and cosine similarities sit between zero and one; any weighted blend of the two is a fudge factor that needs retuning every time either side changes. RRF only compares positions, so scale never enters into it.

Results say which ranker found them:

```js
[{ content: 'The Benin gateway...', score: 0.031, ranks: { lexical: 1, semantic: 3 } }]
```

## Command line

```bash
npx hinterland remember "The Benin gateway settles overnight" --tags ops
npx hinterland recall "why are refunds slow"
npx hinterland stats
npx hinterland backfill --embedder ollama
```

```
$ hinterland stats

  file     ./hinterland.db
  memories 412
  embedded 389  (23 pending)
  embedder nomic-embed-text
  scopes   work 380, personal 32
```

When recall finds nothing and no embedder is configured, it says so rather than shrugging:

```
  Nothing matched.
  Only full text search ran. An embedder would also match on meaning.
```

That line is the difference between "this tool is bad" and "I should turn on embeddings".

## Moving memories between machines

Export is JSON lines, not one big array, so a transfer cut halfway still yields whole records:

```bash
hinterland export memories.jsonl
# carry the file however you like
hinterland import memories.jsonl
```

Ids are preserved, so importing the same file twice **updates rather than duplicates**. On an intermittent link the same file gets carried across more than once, and that has to be safe.

A corrupt line is skipped with a count rather than aborting the import.

## API

Four verbs. That is the whole surface.

```js
const memory = Memory.open('./memory.db', { embedder });

await memory.remember(content, { scope, tags, metadata, source });
await memory.rememberMany([...]);          // one transaction, one embedding call
await memory.recall(query, { limit, scope, mode });
memory.forget(id);
await memory.backfill();

memory.list({ scope, limit });
memory.stats();
```

An agent memory library whose API takes an afternoon to learn loses to one that takes a minute, regardless of which retrieves better.

### Embedders

Default is `none`, so nothing reaches the network unless you ask.

```js
import { OllamaEmbedder } from 'hinterland/embed';

const memory = Memory.open('./memory.db', {
  embedder: new OllamaEmbedder({ model: 'nomic-embed-text' }),
});
```

Every method on an embedder can fail, and none of that failure reaches your code as an exception. An embedding model is an upgrade, not a dependency.

Writing a custom one is two methods:

```js
class MyEmbedder {
  model = 'my-model';
  async available() { return true; }
  async embed(texts) { return texts.map(t => new Float32Array(...)); }  // null when unavailable
}
```

## Things it does on purpose

**Vectors from different models are never compared.** Swap your embedding model and the old vectors stop being used and get queued for re-embedding, rather than being silently compared across incompatible spaces to produce confident nonsense.

**Rewriting a memory drops its vector**, because the old one described text that no longer exists.

**Search is a linear scan, not an approximate index.** For a personal or single-device memory in the low tens of thousands of rows, scanning Float32 arrays takes a few milliseconds. An ANN index would add a dependency, a build step and a class of silent recall failures in exchange for a speedup nobody at this scale would notice.

**Queries are sanitised before they reach FTS5.** A user typing `payment AND` or an apostrophe should get a search, not a syntax error.

## What has and has not been verified

32 tests on Linux, macOS and Windows. The offline path, the backfill path, model swapping, query sanitisation and the export round trip all have tests.

**The Ollama embedder has not been run against a live Ollama.** It is written against the documented `/api/embed` and `/api/tags` endpoints and tested with a deterministic fake, which is the right way to test fusion and ranking, but it is not the same as a real model server. If you run it against one, an issue describing what broke is the most useful thing you could send.

## Requirements

Node 22.5 or newer, for the built-in `node:sqlite`. Nothing else. No native build, no Python, no `node-gyp`.

## License

[MIT](LICENSE)
