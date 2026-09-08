#!/usr/bin/env node
/**
 * hinterland
 *
 * Agent memory that works where the network does not.
 */

import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { Memory } from '../src/memory.mjs';
import { embedderFrom } from '../src/embed.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const pkg = JSON.parse(await readFile(join(ROOT, 'package.json'), 'utf8'));

const argv = process.argv.slice(2);
const command = argv[0] && !argv[0].startsWith('--') ? argv[0] : null;
const has = (name) => argv.includes(`--${name}`);
const value = (name, fallback = null) => {
  const i = argv.indexOf(`--${name}`);
  return i === -1 ? fallback : argv[i + 1];
};
const positional = (n) => argv.slice(1).filter((a, i, all) => !a.startsWith('--') && !all[i - 1]?.startsWith('--'))[n];

const tty = process.stdout.isTTY && !process.env.NO_COLOR;
const paint = (code, s) => (tty ? `\x1b[${code}m${s}\x1b[0m` : s);
const c = {
  bold: (s) => paint('1', s),
  dim: (s) => paint('2', s),
  green: (s) => paint('32', s),
  yellow: (s) => paint('33', s),
  cyan: (s) => paint('36', s),
};

function usage() {
  console.log(`
${c.bold('hinterland')} ${pkg.version}
Agent memory that works where the network does not.

  ${c.bold('remember')} <text>      store a memory
  ${c.bold('recall')} <query>       retrieve, lexically and semantically
  ${c.bold('forget')} <id>          delete one
  ${c.bold('list')}                 most recent first
  ${c.bold('backfill')}             embed everything written while offline
  ${c.bold('stats')}                what is stored and what is pending
  ${c.bold('export')} <file>        write every memory as JSON lines
  ${c.bold('import')} <file>        read them back

Options
  --db <path>          database file (default: ./hinterland.db)
  --scope <name>       partition memories, for example work or personal
  --tags a,b           comma separated, for remember
  --limit <n>          how many results (default: 10)
  --embedder <name>    ollama, or none (default: none, nothing leaves the machine)
  --model <name>       embedding model (default: nomic-embed-text)
  --json               machine readable output

Examples
  hinterland remember "The Benin gateway settles overnight" --tags ops
  hinterland recall "how does settlement work"
  hinterland backfill --embedder ollama
`);
}

function openMemory() {
  const embedder = embedderFrom({
    embedder: value('embedder', 'none'),
    model: value('model', 'nomic-embed-text'),
    baseUrl: value('ollama-url', 'http://127.0.0.1:11434'),
  });

  return Memory.open(value('db', './hinterland.db'), { embedder });
}

function printMemory(m, { showScore = false } = {}) {
  const when = m.createdAt.slice(0, 16).replace('T', ' ');
  const score = showScore && m.score !== undefined ? c.dim(` ${m.score.toFixed(4)}`) : '';
  const via = m.ranks
    ? c.dim(` [${Object.keys(m.ranks).join('+')}]`)
    : '';

  console.log(`  ${c.cyan(m.id.slice(0, 8))} ${c.dim(when)}${score}${via}`);
  console.log(`    ${m.content}`);
  if (m.tags.length || m.scope !== 'default') {
    const bits = [m.scope !== 'default' ? m.scope : null, ...m.tags].filter(Boolean);
    console.log(`    ${c.dim(bits.join(', '))}`);
  }
  console.log('');
}

const commands = {
  async remember() {
    const text = positional(0);
    if (!text) {
      console.error('Give something to remember: hinterland remember "..."');
      process.exit(2);
    }

    const memory = openMemory();
    const tags = value('tags') ? value('tags').split(',').map((t) => t.trim()) : [];

    const written = await memory.remember(text, {
      scope: value('scope', 'default'),
      source: value('source'),
      tags,
    });

    if (has('json')) {
      console.log(JSON.stringify(written, null, 2));
    } else {
      console.log('');
      printMemory(written);
      if (memory.pending() > 0) {
        console.log(c.dim(`  Stored without an embedding. Run "hinterland backfill --embedder ollama" when you can.`));
        console.log('');
      }
    }

    memory.close();
  },

  async recall() {
    const query = positional(0);
    if (!query) {
      console.error('Give something to recall: hinterland recall "..."');
      process.exit(2);
    }

    const memory = openMemory();
    const hits = await memory.recall(query, {
      limit: Number(value('limit', 10)),
      scope: value('scope'),
    });

    if (has('json')) {
      console.log(JSON.stringify(hits, null, 2));
      memory.close();
      process.exit(hits.length ? 0 : 1);
    }

    console.log('');
    if (!hits.length) {
      console.log(c.yellow('  Nothing matched.'));
      const stats = memory.stats();
      if (stats.memories > 0 && stats.embedder === 'none') {
        console.log(c.dim('  Only full text search ran. An embedder would also match on meaning.'));
      }
      console.log('');
      memory.close();
      process.exit(1);
    }

    for (const hit of hits) printMemory(hit, { showScore: true });
    memory.close();
  },

  async forget() {
    const id = positional(0);
    if (!id) {
      console.error('Give an id to forget.');
      process.exit(2);
    }

    const memory = openMemory();

    // Short ids are what the other commands print, so accept them.
    const target = id.length < 36
      ? memory.list({ limit: 10000 }).find((m) => m.id.startsWith(id))?.id
      : id;

    const removed = target ? memory.forget(target) : false;
    memory.close();

    console.log(removed ? `Forgot ${target}` : `No memory matching ${id}`);
    process.exit(removed ? 0 : 1);
  },

  async list() {
    const memory = openMemory();
    const all = memory.list({ scope: value('scope'), limit: Number(value('limit', 20)) });

    if (has('json')) {
      console.log(JSON.stringify(all, null, 2));
    } else {
      console.log('');
      if (!all.length) console.log(c.dim('  Nothing stored yet.\n'));
      for (const m of all) printMemory(m);
    }

    memory.close();
  },

  async backfill() {
    const memory = openMemory();

    if (memory.embedder.model === null) {
      console.error('Backfill needs an embedder. Try: hinterland backfill --embedder ollama');
      memory.close();
      process.exit(2);
    }

    const before = memory.pending();
    if (before === 0) {
      console.log('Nothing pending.');
      memory.close();
      return;
    }

    console.log(`Embedding ${before} memories with ${memory.embedder.model}...`);
    const result = await memory.backfill({
      onProgress: ({ embedded, remaining }) => {
        process.stdout.write(`\r  ${embedded} done, ${remaining} to go   `);
      },
    });

    process.stdout.write('\r');

    if (result.embedded === 0) {
      console.log(c.yellow(`Could not embed: ${result.reason}`));
      memory.close();
      process.exit(1);
    }

    console.log(c.green(`Embedded ${result.embedded}. ${result.remaining} still pending.`));
    memory.close();
  },

  async stats() {
    const memory = openMemory();
    const stats = memory.stats();
    memory.close();

    if (has('json')) return console.log(JSON.stringify(stats, null, 2));

    console.log('');
    console.log(`  ${c.dim('file    ')} ${stats.path}`);
    console.log(`  ${c.dim('memories')} ${stats.memories}`);
    console.log(`  ${c.dim('embedded')} ${stats.embedded}${stats.pending ? c.yellow(`  (${stats.pending} pending)`) : ''}`);
    console.log(`  ${c.dim('embedder')} ${stats.embedder}`);
    if (Object.keys(stats.scopes).length) {
      console.log(`  ${c.dim('scopes  ')} ${Object.entries(stats.scopes).map(([k, v]) => `${k} ${v}`).join(', ')}`);
    }
    console.log('');
  },

  async export() {
    const file = positional(0);
    if (!file) {
      console.error('Give a file to write: hinterland export memories.jsonl');
      process.exit(2);
    }

    const memory = openMemory();
    const all = memory.list({ limit: 1_000_000 });
    memory.close();

    // JSON lines rather than one array, so a partial transfer over a bad link
    // still yields whole records.
    await writeFile(file, all.map((m) => JSON.stringify(m)).join('\n') + '\n');
    console.log(`Wrote ${all.length} memories to ${file}`);
  },

  async import() {
    const file = positional(0);
    if (!file) {
      console.error('Give a file to read: hinterland import memories.jsonl');
      process.exit(2);
    }

    const raw = await readFile(file, 'utf8');
    const lines = raw.split('\n').filter((l) => l.trim());

    const memory = openMemory();
    let imported = 0;
    let skipped = 0;

    for (const line of lines) {
      try {
        const record = JSON.parse(line);
        // Ids are preserved, so importing the same file twice updates rather
        // than duplicating.
        await memory.remember(record.content, {
          id: record.id,
          scope: record.scope,
          source: record.source,
          tags: record.tags ?? [],
          metadata: record.metadata ?? {},
        });
        imported += 1;
      } catch {
        skipped += 1;
      }
    }

    memory.close();
    console.log(`Imported ${imported}${skipped ? `, skipped ${skipped} unreadable line(s)` : ''}`);
  },
};

if (has('version')) {
  console.log(pkg.version);
} else if (!command || has('help') || command === 'help') {
  usage();
} else if (commands[command]) {
  try {
    await commands[command]();
  } catch (error) {
    console.error(`hinterland: ${error.message}`);
    process.exit(1);
  }
} else {
  console.error(`Unknown command: ${command}`);
  usage();
  process.exit(1);
}
