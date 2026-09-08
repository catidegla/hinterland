/**
 * Retrieval.
 *
 * Two independent rankers, fused. Full text search always runs, because it
 * needs nothing but the database file. Vector search runs when an embedding
 * model has been reachable at some point and the query itself can be embedded
 * now.
 *
 * They are combined with reciprocal rank fusion rather than by mixing scores.
 * BM25 scores and cosine similarities are not on the same scale and never will
 * be, so any weighted sum of them is a fudge factor that has to be retuned
 * every time either side changes. RRF only looks at positions, so it does not
 * care about scale at all.
 */

import { unpackVector } from './store.mjs';
import { cosine } from './embed.mjs';

/**
 * The constant damps the influence of the top few positions. 60 is the value
 * from the original paper and is not worth inventing a new one for.
 */
const RRF_K = 60;

/**
 * FTS5 treats a lot of punctuation as syntax, so a user query containing a
 * quote or a bare AND becomes a syntax error rather than a search. Everything
 * is quoted as a phrase term instead.
 */
export function toMatchQuery(query) {
  const terms = query
    .toLowerCase()
    .split(/[^\p{L}\p{N}_]+/u)
    .filter((t) => t.length > 1);

  if (!terms.length) return null;

  return terms.map((t) => `"${t.replace(/"/g, '""')}"`).join(' OR ');
}

export function lexicalSearch(db, query, { limit = 20, scope = null } = {}) {
  const match = toMatchQuery(query);
  if (!match) return [];

  const params = [match];
  let sql = `
    SELECT m.*, bm25(memories_fts) AS score
    FROM memories_fts
    JOIN memories m ON m.rowid = memories_fts.rowid
    WHERE memories_fts MATCH ?
  `;

  if (scope) {
    sql += ' AND m.scope = ?';
    params.push(scope);
  }

  // bm25 returns a negative number where more negative is a better match.
  sql += ' ORDER BY score ASC LIMIT ?';
  params.push(limit);

  try {
    return db.prepare(sql).all(...params);
  } catch {
    // A malformed match expression is a bad query, not a broken store.
    return [];
  }
}

/**
 * Brute force cosine over stored vectors.
 *
 * Deliberately not an ANN index. On the scale this is built for, a personal or
 * single-device memory in the low tens of thousands of rows, a linear scan of
 * Float32 arrays is a few milliseconds, and an approximate index would add a
 * dependency, a build step and a class of silent recall failures in exchange
 * for a speedup nobody at this scale would notice.
 */
export function vectorSearch(db, queryVector, { limit = 20, scope = null, model = null } = {}) {
  if (!queryVector) return [];

  const params = [];
  let sql = `
    SELECT m.*, e.vector, e.dimensions
    FROM embeddings e
    JOIN memories m ON m.id = e.memory_id
    WHERE 1 = 1
  `;

  if (model) {
    // Vectors from different models are not comparable, so mixing them
    // produces confident nonsense.
    sql += ' AND e.model = ?';
    params.push(model);
  }

  if (scope) {
    sql += ' AND m.scope = ?';
    params.push(scope);
  }

  const rows = db.prepare(sql).all(...params);
  const scored = [];

  for (const row of rows) {
    if (row.dimensions !== queryVector.length) continue;
    scored.push({ ...row, score: cosine(queryVector, unpackVector(row.vector)) });
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, limit);
}

/**
 * Reciprocal rank fusion.
 *
 * @param {Array<{rows: Array, weight?: number, name: string}>} rankings
 */
export function fuse(rankings, { limit = 10, k = RRF_K } = {}) {
  const combined = new Map();

  for (const { rows, weight = 1, name } of rankings) {
    rows.forEach((row, index) => {
      const entry = combined.get(row.id) ?? { row, score: 0, ranks: {} };
      entry.score += weight * (1 / (k + index + 1));
      entry.ranks[name] = index + 1;
      combined.set(row.id, entry);
    });
  }

  return [...combined.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((entry) => ({ ...entry.row, score: entry.score, ranks: entry.ranks }));
}
