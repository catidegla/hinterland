/**
 * Storage. One SQLite file, no server, no dependencies.
 *
 * The schema separates the memory from its embedding on purpose. A memory is
 * written the moment it is made, whether or not an embedding model is
 * reachable, and vectors are backfilled later. That ordering is the whole
 * offline-first argument: on a connection that drops for hours at a time, a
 * memory store that refuses to write until it can call an embedding API is not
 * a memory store, it is an outage.
 */

import { DatabaseSync } from 'node:sqlite';
import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

const SCHEMA_VERSION = 1;

const SCHEMA = `
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
  id         TEXT PRIMARY KEY,
  content    TEXT NOT NULL,
  scope      TEXT NOT NULL DEFAULT 'default',
  source     TEXT,
  tags       TEXT NOT NULL DEFAULT '[]',
  metadata   TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  recalled   INTEGER NOT NULL DEFAULT 0,
  last_recalled_at TEXT
);

CREATE INDEX IF NOT EXISTS memories_scope     ON memories(scope);
CREATE INDEX IF NOT EXISTS memories_created   ON memories(created_at);

-- Full text search, which is what makes retrieval work with no model at all.
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
  content,
  content='memories',
  content_rowid='rowid',
  tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.rowid, old.content);
  INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;

-- Separate table, so a memory exists whether or not a vector does.
CREATE TABLE IF NOT EXISTS embeddings (
  memory_id  TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
  model      TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  vector     BLOB NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS embeddings_model ON embeddings(model);
`;

export function open(path = ':memory:') {
  if (path !== ':memory:') {
    mkdirSync(dirname(path), { recursive: true });
  }

  const db = new DatabaseSync(path);

  // WAL keeps reads working while a write is in flight, which matters when a
  // background backfill is running against the same file.
  if (path !== ':memory:') db.exec('PRAGMA journal_mode = WAL');
  db.exec('PRAGMA foreign_keys = ON');
  db.exec(SCHEMA);

  const version = db.prepare('SELECT value FROM meta WHERE key = ?').get('schema_version');
  if (!version) {
    db.prepare('INSERT INTO meta (key, value) VALUES (?, ?)').run('schema_version', String(SCHEMA_VERSION));
  } else if (Number(version.value) > SCHEMA_VERSION) {
    db.close();
    throw new Error(
      `${path} was written by a newer version of hinterland (schema ${version.value}, this build understands ${SCHEMA_VERSION}). ` +
        'Upgrade rather than letting an older build write to it.',
    );
  }

  return db;
}

/** Float32 round trips through a BLOB without JSON inflating it fourfold. */
export function packVector(vector) {
  return Buffer.from(new Float32Array(vector).buffer);
}

export function unpackVector(blob) {
  const copy = Buffer.from(blob);
  return new Float32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4);
}

export function rowToMemory(row) {
  if (!row) return null;

  return {
    id: row.id,
    content: row.content,
    scope: row.scope,
    source: row.source ?? null,
    tags: JSON.parse(row.tags),
    metadata: JSON.parse(row.metadata),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    recalled: row.recalled,
    lastRecalledAt: row.last_recalled_at ?? null,
  };
}
