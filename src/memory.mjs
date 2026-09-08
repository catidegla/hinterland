/**
 * The public API. Four verbs and a file path.
 *
 * remember, recall, forget, backfill. Everything else on this class exists to
 * support those, and the surface stays small on purpose: an agent memory
 * library whose API takes an afternoon to learn will lose to one that takes a
 * minute, regardless of which retrieves better.
 */

import { randomUUID } from 'node:crypto';

import { open, packVector, rowToMemory } from './store.mjs';
import { NullEmbedder } from './embed.mjs';
import { lexicalSearch, vectorSearch, fuse } from './search.mjs';

export class Memory {
  /**
   * @param {string} path      SQLite file, or ':memory:'
   * @param {object} options
   * @param {object} options.embedder  optional, see src/embed.mjs
   */
  constructor(path = ':memory:', { embedder = new NullEmbedder() } = {}) {
    this.path = path;
    this.db = open(path);
    this.embedder = embedder;
    this.pendingEmbedding = new Set();
  }

  static open(path, options) {
    return new Memory(path, options);
  }

  /**
   * Write a memory. Returns as soon as it is durable.
   *
   * The embedding is attempted afterwards and its failure is not the caller's
   * problem, because on an intermittent connection it will fail routinely and
   * a write that depends on it is a write that loses data.
   */
  async remember(content, { scope = 'default', source = null, tags = [], metadata = {}, id = null } = {}) {
    if (typeof content !== 'string' || !content.trim()) {
      throw new TypeError('remember() needs a non-empty string');
    }

    const now = new Date().toISOString();
    const memoryId = id ?? randomUUID();

    this.db
      .prepare(
        `INSERT INTO memories (id, content, scope, source, tags, metadata, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           content = excluded.content,
           scope = excluded.scope,
           source = excluded.source,
           tags = excluded.tags,
           metadata = excluded.metadata,
           updated_at = excluded.updated_at`,
      )
      .run(memoryId, content, scope, source, JSON.stringify(tags), JSON.stringify(metadata), now, now);

    // Content changed, so any existing vector describes text that is gone.
    this.db.prepare('DELETE FROM embeddings WHERE memory_id = ?').run(memoryId);

    await this.#tryEmbed([{ id: memoryId, content }]);

    return this.get(memoryId);
  }

  /** Write many at once. One transaction, one embedding call. */
  async rememberMany(items) {
    const now = new Date().toISOString();
    const written = [];

    const insert = this.db.prepare(
      `INSERT INTO memories (id, content, scope, source, tags, metadata, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at`,
    );

    this.db.exec('BEGIN');
    try {
      for (const item of items) {
        const id = item.id ?? randomUUID();
        insert.run(
          id,
          item.content,
          item.scope ?? 'default',
          item.source ?? null,
          JSON.stringify(item.tags ?? []),
          JSON.stringify(item.metadata ?? {}),
          now,
          now,
        );
        written.push({ id, content: item.content });
      }
      this.db.exec('COMMIT');
    } catch (error) {
      this.db.exec('ROLLBACK');
      throw error;
    }

    await this.#tryEmbed(written);
    return written.map((w) => this.get(w.id));
  }

  /**
   * Retrieve. Lexical always, vector when it can be.
   *
   * @returns {Promise<Array>} memories with a score and the ranks that produced it
   */
  async recall(query, { limit = 10, scope = null, mode = 'auto' } = {}) {
    if (typeof query !== 'string' || !query.trim()) return [];

    const overFetch = Math.max(limit * 3, 30);
    const rankings = [];

    if (mode !== 'vector') {
      const lexical = lexicalSearch(this.db, query, { limit: overFetch, scope });
      if (lexical.length) rankings.push({ name: 'lexical', rows: lexical });
    }

    if (mode !== 'lexical') {
      const vectors = await this.embedder.embed([query]);
      if (vectors && vectors[0]) {
        const semantic = vectorSearch(this.db, vectors[0], {
          limit: overFetch,
          scope,
          model: this.embedder.model,
        });
        if (semantic.length) rankings.push({ name: 'semantic', rows: semantic });
      }
    }

    if (!rankings.length) return [];

    const results = fuse(rankings, { limit });

    // Recall counts are useful for pruning later, and are not worth failing a
    // read over.
    try {
      const bump = this.db.prepare(
        'UPDATE memories SET recalled = recalled + 1, last_recalled_at = ? WHERE id = ?',
      );
      const now = new Date().toISOString();
      for (const result of results) bump.run(now, result.id);
    } catch {
      // ignore
    }

    return results.map((row) => ({ ...rowToMemory(row), score: row.score, ranks: row.ranks }));
  }

  get(id) {
    return rowToMemory(this.db.prepare('SELECT * FROM memories WHERE id = ?').get(id));
  }

  forget(id) {
    const result = this.db.prepare('DELETE FROM memories WHERE id = ?').run(id);
    return result.changes > 0;
  }

  /** Everything in a scope, newest first. */
  list({ scope = null, limit = 100, offset = 0 } = {}) {
    const rows = scope
      ? this.db
          .prepare('SELECT * FROM memories WHERE scope = ? ORDER BY created_at DESC LIMIT ? OFFSET ?')
          .all(scope, limit, offset)
      : this.db.prepare('SELECT * FROM memories ORDER BY created_at DESC LIMIT ? OFFSET ?').all(limit, offset);

    return rows.map(rowToMemory);
  }

  /**
   * Embed everything that has no vector yet.
   *
   * This is the other half of writing without a model. Call it when
   * connectivity returns, from a cron, or never: recall works either way, it
   * just works better once this has run.
   */
  async backfill({ batchSize = 32, onProgress = null } = {}) {
    if (!(await this.embedder.available())) {
      return { embedded: 0, remaining: this.pending(), reason: this.embedder.lastError ?? 'no embedder' };
    }

    let embedded = 0;

    for (;;) {
      const rows = this.db
        .prepare(
          `SELECT m.id, m.content FROM memories m
           LEFT JOIN embeddings e ON e.memory_id = m.id AND e.model = ?
           WHERE e.memory_id IS NULL
           LIMIT ?`,
        )
        .all(this.embedder.model, batchSize);

      if (!rows.length) break;

      const ok = await this.#tryEmbed(rows);
      if (!ok) break;

      embedded += rows.length;
      onProgress?.({ embedded, remaining: this.pending() });
    }

    return { embedded, remaining: this.pending() };
  }

  /** How many memories have no vector for the current model. */
  pending() {
    if (!this.embedder.model) return 0;

    const row = this.db
      .prepare(
        `SELECT COUNT(*) AS n FROM memories m
         LEFT JOIN embeddings e ON e.memory_id = m.id AND e.model = ?
         WHERE e.memory_id IS NULL`,
      )
      .get(this.embedder.model);

    return row.n;
  }

  stats() {
    const total = this.db.prepare('SELECT COUNT(*) AS n FROM memories').get().n;
    const withVectors = this.db.prepare('SELECT COUNT(*) AS n FROM embeddings').get().n;
    const scopes = this.db.prepare('SELECT scope, COUNT(*) AS n FROM memories GROUP BY scope').all();

    return {
      path: this.path,
      memories: total,
      embedded: withVectors,
      pending: this.pending(),
      embedder: this.embedder.model ?? 'none',
      scopes: Object.fromEntries(scopes.map((s) => [s.scope, s.n])),
    };
  }

  close() {
    this.db.close();
  }

  /** @returns {Promise<boolean>} whether vectors were written */
  async #tryEmbed(rows) {
    if (!rows.length) return true;

    const vectors = await this.embedder.embed(rows.map((r) => r.content));
    if (!vectors) return false;

    const now = new Date().toISOString();
    const insert = this.db.prepare(
      `INSERT INTO embeddings (memory_id, model, dimensions, vector, created_at)
       VALUES (?, ?, ?, ?, ?)
       ON CONFLICT(memory_id) DO UPDATE SET
         model = excluded.model, dimensions = excluded.dimensions,
         vector = excluded.vector, created_at = excluded.created_at`,
    );

    this.db.exec('BEGIN');
    try {
      rows.forEach((row, i) => {
        insert.run(row.id, this.embedder.model, vectors[i].length, packVector(vectors[i]), now);
      });
      this.db.exec('COMMIT');
    } catch (error) {
      this.db.exec('ROLLBACK');
      throw error;
    }

    return true;
  }
}
