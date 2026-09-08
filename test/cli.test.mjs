/**
 * The CLI, against a real database file.
 *
 * Every assertion here is about behaviour a user sees on their first run,
 * which is the run most likely to be broken and least likely to be tested.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdtemp, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const run = promisify(execFile);
const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const CLI = join(ROOT, 'bin', 'hinterland.mjs');

async function cli(args, cwd) {
  try {
    const { stdout, stderr } = await run(process.execPath, [CLI, ...args], { cwd });
    return { code: 0, stdout, stderr };
  } catch (error) {
    return { code: error.code ?? 1, stdout: error.stdout ?? '', stderr: error.stderr ?? '' };
  }
}

async function workspace(t) {
  const dir = await mkdtemp(join(tmpdir(), 'hinterland-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  return dir;
}

test('remember then recall, with no model anywhere', async (t) => {
  const dir = await workspace(t);

  const stored = await cli(['remember', 'The Benin gateway settles overnight', '--tags', 'ops,payments'], dir);
  assert.equal(stored.code, 0);

  const found = await cli(['recall', 'gateway settlement'], dir);
  assert.equal(found.code, 0);
  assert.match(found.stdout, /Benin gateway settles overnight/);
  assert.match(found.stdout, /lexical/);
});

test('recall exits non-zero when nothing matches, so a script can tell', async (t) => {
  const dir = await workspace(t);
  await cli(['remember', 'something about deployments'], dir);

  const missed = await cli(['recall', 'zzzzz nonexistent'], dir);

  assert.equal(missed.code, 1);
  assert.match(missed.stdout, /Nothing matched/);
});

test('it says why recall is weaker without an embedder', async (t) => {
  const dir = await workspace(t);
  await cli(['remember', 'a note about deployments'], dir);

  const missed = await cli(['recall', 'zzzzz'], dir);

  // Telling someone the search was lexical only is the difference between
  // "this tool is bad" and "I should turn on embeddings".
  assert.match(missed.stdout, /full text search/i);
});

test('json output is machine readable for every command', async (t) => {
  const dir = await workspace(t);

  const stored = await cli(['remember', 'a memory about invoices', '--json'], dir);
  const parsed = JSON.parse(stored.stdout);
  assert.ok(parsed.id);
  assert.equal(parsed.content, 'a memory about invoices');

  const stats = JSON.parse((await cli(['stats', '--json'], dir)).stdout);
  assert.equal(stats.memories, 1);
  assert.equal(stats.embedder, 'none');

  const list = JSON.parse((await cli(['list', '--json'], dir)).stdout);
  assert.equal(list.length, 1);

  const recall = JSON.parse((await cli(['recall', 'invoices', '--json'], dir)).stdout);
  assert.equal(recall.length, 1);
});

test('scopes keep memories apart on the command line too', async (t) => {
  const dir = await workspace(t);

  await cli(['remember', 'invoice from the garage', '--scope', 'personal'], dir);
  await cli(['remember', 'invoice service retries', '--scope', 'work'], dir);

  const work = JSON.parse((await cli(['recall', 'invoice', '--scope', 'work', '--json'], dir)).stdout);
  assert.equal(work.length, 1);
  assert.match(work[0].content, /retries/);
});

test('forget accepts the short id that the other commands print', async (t) => {
  const dir = await workspace(t);

  const stored = JSON.parse((await cli(['remember', 'a forgettable note', '--json'], dir)).stdout);
  const shortId = stored.id.slice(0, 8);

  const forgotten = await cli(['forget', shortId], dir);
  assert.equal(forgotten.code, 0);
  assert.match(forgotten.stdout, /Forgot/);

  const stats = JSON.parse((await cli(['stats', '--json'], dir)).stdout);
  assert.equal(stats.memories, 0);
});

test('forgetting something that is not there exits non-zero', async (t) => {
  const dir = await workspace(t);
  const missing = await cli(['forget', 'deadbeef'], dir);

  assert.equal(missing.code, 1);
  assert.match(missing.stdout, /No memory matching/);
});

test('backfill without an embedder explains itself instead of doing nothing', async (t) => {
  const dir = await workspace(t);
  await cli(['remember', 'anything'], dir);

  const result = await cli(['backfill'], dir);

  assert.equal(result.code, 2);
  assert.match(result.stderr, /--embedder ollama/);
});

test('export and import round trip without duplicating', async (t) => {
  const dir = await workspace(t);

  await cli(['remember', 'first memory about payments'], dir);
  await cli(['remember', 'second memory about deploys'], dir);

  const exported = await cli(['export', 'out.jsonl'], dir);
  assert.match(exported.stdout, /Wrote 2 memories/);

  // JSON lines, so a transfer cut halfway still yields whole records.
  const raw = await readFile(join(dir, 'out.jsonl'), 'utf8');
  assert.equal(raw.trim().split('\n').length, 2);
  for (const line of raw.trim().split('\n')) JSON.parse(line);

  // Importing into the same store must update rather than duplicate, because
  // on an intermittent link the same file gets carried across more than once.
  const reimported = await cli(['import', 'out.jsonl'], dir);
  assert.match(reimported.stdout, /Imported 2/);

  const stats = JSON.parse((await cli(['stats', '--json'], dir)).stdout);
  assert.equal(stats.memories, 2, 'ids are preserved, so nothing was duplicated');
});

test('import survives a corrupt line rather than losing the whole file', async (t) => {
  const dir = await workspace(t);
  const { writeFile } = await import('node:fs/promises');

  await writeFile(
    join(dir, 'mixed.jsonl'),
    [
      JSON.stringify({ id: 'a', content: 'good record one', scope: 'default', tags: [] }),
      '{ truncated halfway',
      JSON.stringify({ id: 'b', content: 'good record two', scope: 'default', tags: [] }),
    ].join('\n'),
  );

  const result = await cli(['import', 'mixed.jsonl'], dir);

  assert.match(result.stdout, /Imported 2/);
  assert.match(result.stdout, /skipped 1/);
});

test('the database is created on demand, not required up front', async (t) => {
  const dir = await workspace(t);

  const stats = await cli(['stats', '--json'], dir);
  assert.equal(stats.code, 0);
  assert.equal(JSON.parse(stats.stdout).memories, 0);
});

test('help and version behave', async (t) => {
  const dir = await workspace(t);

  const version = await cli(['--version'], dir);
  assert.match(version.stdout.trim(), /^\d+\.\d+\.\d+$/);

  const help = await cli([], dir);
  assert.match(help.stdout, /remember/);
  assert.match(help.stdout, /recall/);

  const unknown = await cli(['nonsense'], dir);
  assert.equal(unknown.code, 1);
});
