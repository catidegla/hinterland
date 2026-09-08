/**
 * Optional embeddings.
 *
 * Every method here can fail, and none of that failure is allowed to reach the
 * caller as an exception. An embedding model is an upgrade, not a dependency:
 * if one is reachable, recall gets semantic matching; if not, recall still
 * works on full text search and the vectors get backfilled whenever the model
 * next answers.
 *
 * That is the difference between offline-first and offline-tolerant. A store
 * that throws when the model is unreachable has made the model mandatory in
 * everything but name.
 */

const DEFAULT_OLLAMA = 'http://127.0.0.1:11434';

/** Never available. The default, so nothing reaches the network unasked. */
export class NullEmbedder {
  constructor() {
    this.model = null;
    this.dimensions = 0;
  }

  async available() {
    return false;
  }

  async embed() {
    return null;
  }
}

export class OllamaEmbedder {
  /**
   * @param {object} options
   * @param {string} options.baseUrl
   * @param {string} options.model     an embedding model, not a chat model
   * @param {number} options.timeoutMs
   */
  constructor({ baseUrl = DEFAULT_OLLAMA, model = 'nomic-embed-text', timeoutMs = 10000, fetchImpl = fetch } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.model = model;
    this.timeoutMs = timeoutMs;
    this.fetch = fetchImpl;
    this.dimensions = 0;
    this.lastError = null;
  }

  async available() {
    try {
      const response = await this.fetch(`${this.baseUrl}/api/tags`, {
        signal: AbortSignal.timeout(this.timeoutMs),
      });
      if (!response.ok) {
        this.lastError = `HTTP ${response.status}`;
        return false;
      }

      const body = await response.json();
      const names = (body.models ?? []).map((m) => m.name ?? m.model);

      // Ollama tags carry a version suffix, so a configured "nomic-embed-text"
      // should match an installed "nomic-embed-text:latest".
      const found = names.some((n) => n === this.model || n.split(':')[0] === this.model.split(':')[0]);

      if (!found) {
        this.lastError = `${this.model} is not pulled. Run: ollama pull ${this.model}`;
        return false;
      }

      this.lastError = null;
      return true;
    } catch (error) {
      this.lastError = error.message;
      return false;
    }
  }

  /**
   * @param {string[]} texts
   * @returns {Promise<Float32Array[]|null>} null when unavailable, never throws
   */
  async embed(texts) {
    if (!texts.length) return [];

    try {
      const response = await this.fetch(`${this.baseUrl}/api/embed`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: this.model, input: texts }),
        signal: AbortSignal.timeout(this.timeoutMs * Math.max(1, Math.ceil(texts.length / 8))),
      });

      if (!response.ok) {
        this.lastError = `HTTP ${response.status}`;
        return null;
      }

      const body = await response.json();
      const vectors = body.embeddings ?? (body.embedding ? [body.embedding] : null);

      if (!Array.isArray(vectors) || vectors.length !== texts.length) {
        this.lastError = 'unexpected embedding response shape';
        return null;
      }

      this.dimensions = vectors[0]?.length ?? 0;
      this.lastError = null;

      return vectors.map((v) => new Float32Array(v));
    } catch (error) {
      this.lastError = error.message;
      return null;
    }
  }
}

/** Cosine similarity. Vectors are normalised on the fly rather than at write. */
export function cosine(a, b) {
  if (a.length !== b.length) return 0;

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export function embedderFrom(config = {}) {
  if (!config.embedder || config.embedder === 'none') return new NullEmbedder();
  if (config.embedder === 'ollama') return new OllamaEmbedder(config);

  throw new Error(`Unknown embedder "${config.embedder}". Use "ollama" or "none".`);
}
