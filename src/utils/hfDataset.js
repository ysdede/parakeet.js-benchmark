const HF_DATASET_API = 'https://datasets-server.huggingface.co';
const MIN_REQUEST_GAP_MS = 500;

let lastRequestAt = 0;
let requestQueue = Promise.resolve();

function buildUrl(path, params) {
  const url = new URL(`${HF_DATASET_API}${path}`);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      url.searchParams.set(key, String(value));
    }
  });
  return url.toString();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function enqueueRequest(task) {
  const run = requestQueue
    .catch(() => {})
    .then(async () => {
      const now = Date.now();
      const elapsed = now - lastRequestAt;
      if (elapsed < MIN_REQUEST_GAP_MS) {
        await sleep(MIN_REQUEST_GAP_MS - elapsed);
      }
      const result = await task();
      lastRequestAt = Date.now();
      return result;
    });
  requestQueue = run.catch(() => {});
  return run;
}

function parseRetryAfterMs(response, fallbackMs) {
  const retryAfter = response.headers?.get('retry-after');
  if (!retryAfter) return fallbackMs;
  const asSec = Number(retryAfter);
  if (Number.isFinite(asSec)) return Math.max(fallbackMs, asSec * 1000);
  const asDate = Date.parse(retryAfter);
  if (Number.isFinite(asDate)) {
    const diff = asDate - Date.now();
    return Number.isFinite(diff) ? Math.max(fallbackMs, diff) : fallbackMs;
  }
  return fallbackMs;
}

async function fetchJson(url, options = {}) {
  const { retries = 2, retryDelayMs = 700 } = options;
  let lastError = null;

  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      const response = await enqueueRequest(() => fetch(url));
      if (!response.ok) {
        const bodyText = await response.text();
        const shouldRetry = response.status >= 500 || response.status === 429;
        if (shouldRetry && attempt < retries) {
          const backoff = parseRetryAfterMs(response, retryDelayMs * (attempt + 1));
          await sleep(backoff);
          continue;
        }
        throw new Error(`Request failed (${response.status}): ${bodyText}`);
      }
      return response.json();
    } catch (error) {
      lastError = error;
      if (attempt >= retries) break;
      const isNetwork = /Failed to fetch|NetworkError|Load failed|CORS|ERR_FAILED/i.test(String(error?.message || error));
      const backoff = isNetwork
        ? Math.max(1200, retryDelayMs * (attempt + 1) * 2)
        : retryDelayMs * (attempt + 1);
      await sleep(backoff);
    }
  }

  throw lastError || new Error('Request failed');
}

export async function fetchDatasetSplits(dataset) {
  const url = buildUrl('/splits', { dataset });
  const data = await fetchJson(url);
  return data.splits || [];
}

export async function fetchDatasetInfo(dataset, config) {
  const url = buildUrl('/info', { dataset, config });
  return fetchJson(url);
}

export async function fetchDatasetRows({ dataset, config, split, offset = 0, length = 100 }) {
  const safeLength = Math.max(1, Math.min(100, Number(length) || 1));
  const safeOffset = Math.max(0, Number(offset) || 0);
  const url = buildUrl('/rows', {
    dataset,
    config,
    split,
    offset: safeOffset,
    length: safeLength,
  });

  return fetchJson(url);
}

export function extractAudioUrl(audioField) {
  if (!audioField) return null;

  if (typeof audioField === 'string') {
    return audioField;
  }

  if (Array.isArray(audioField)) {
    for (const item of audioField) {
      if (!item) continue;
      if (typeof item === 'string') return item;
      if (typeof item === 'object') {
        if (item.src) return item.src;
        if (item.url) return item.url;
        if (item.path) return item.path;
      }
    }
  }

  if (typeof audioField === 'object') {
    if (audioField.src) return audioField.src;
    if (audioField.url) return audioField.url;
    if (audioField.path) return audioField.path;
  }

  return null;
}

export function normalizeReferenceText(value) {
  return String(value || '')
    .replace(/PARAGRAPH/g, '\n\n')
    .replace(/NEWLINE/g, '\n')
    .replace(/\s+\n/g, '\n')
    .replace(/\n\s+/g, '\n')
    .trim();
}

export function normalizeDatasetRow(rowWrapper, fallbackIndex = 0) {
  const row = rowWrapper?.row || rowWrapper || {};
  const rowIndex = rowWrapper?.row_idx ?? fallbackIndex;
  const audioUrl = extractAudioUrl(row.audio);

  return {
    rowIndex,
    audioUrl,
    referenceText: normalizeReferenceText(row.transcription || row.text || row.transcript || ''),
    speaker: row.speaker || '',
    gender: row.gender || '',
    speed: row.speed,
    volume: row.volume,
    sampleRate: Number(row.sample_rate) || 16000,
    raw: row,
  };
}

export async function fetchSequentialRows({ dataset, config, split, startOffset = 0, limit = 10 }) {
  const rows = [];
  let cursor = Math.max(0, Number(startOffset) || 0);
  let totalRows = null;

  while (rows.length < limit) {
    const pageLength = Math.min(100, limit - rows.length);
    const page = await fetchDatasetRows({
      dataset,
      config,
      split,
      offset: cursor,
      length: pageLength,
    });

    const pageRows = page.rows || [];
    totalRows = page.num_rows_total ?? totalRows;
    if (!pageRows.length) break;

    rows.push(...pageRows);
    cursor += pageRows.length;

    if (pageRows.length < pageLength) {
      break;
    }
  }

  return {
    rows,
    totalRows,
  };
}

function normalizeSeed(seed) {
  if (seed === undefined || seed === null || seed === '') return null;
  const text = String(seed);
  let h = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function createMulberry32(seed) {
  let t = seed >>> 0;
  return function next() {
    t += 0x6D2B79F5;
    let x = Math.imul(t ^ (t >>> 15), t | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

export async function fetchRandomRows({ dataset, config, split, totalRows, sampleCount, seed }) {
  const targetCount = Math.max(1, Number(sampleCount) || 1);
  const maxRows = Math.max(1, Number(totalRows) || 1);
  const wanted = Math.min(targetCount, maxRows);
  const seedValue = normalizeSeed(seed);
  const rand = seedValue === null ? Math.random : createMulberry32(seedValue);
  const requestedOffsets = [];
  const selectedOffsets = new Set();
  const successfulOffsets = [];
  const failedOffsets = [];
  const rows = [];

  const maxOffsetAttempts = Math.min(maxRows * 3, maxRows + wanted * 20);
  let offsetAttempts = 0;

  while (selectedOffsets.size < wanted && offsetAttempts < maxOffsetAttempts && selectedOffsets.size < maxRows) {
    offsetAttempts += 1;
    const offset = Math.floor(rand() * maxRows);
    if (selectedOffsets.has(offset)) continue;
    selectedOffsets.add(offset);
    requestedOffsets.push(offset);
  }

  const pageMap = new Map();
  requestedOffsets.forEach((offset) => {
    const pageStart = Math.floor(offset / 100) * 100;
    const list = pageMap.get(pageStart) || [];
    list.push(offset);
    pageMap.set(pageStart, list);
  });

  const pageStarts = Array.from(pageMap.keys()).sort((a, b) => a - b);
  for (const pageStart of pageStarts) {
    if (rows.length >= wanted) break;

    try {
      const page = await fetchDatasetRows({ dataset, config, split, offset: pageStart, length: 100 });
      const pageRows = page.rows || [];
      const offsetsInPage = pageMap.get(pageStart) || [];
      for (const absoluteOffset of offsetsInPage) {
        if (rows.length >= wanted) break;
        const row = pageRows[absoluteOffset - pageStart];
        if (!row) {
          failedOffsets.push(absoluteOffset);
          continue;
        }
        rows.push(row);
        successfulOffsets.push(absoluteOffset);
      }
    } catch {
      const offsetsInPage = pageMap.get(pageStart) || [];
      failedOffsets.push(...offsetsInPage);
    }
  }

  return {
    rows,
    offsets: successfulOffsets,
    failedOffsets,
    totalRows: maxRows,
    seedUsed: seedValue,
    requestedCount: wanted,
  };
}

export function getConfigsAndSplits(splitsResponse) {
  const byConfig = new Map();

  (splitsResponse || []).forEach((item) => {
    if (!item?.config || !item?.split) return;
    const list = byConfig.get(item.config) || [];
    if (!list.includes(item.split)) list.push(item.split);
    byConfig.set(item.config, list);
  });

  return byConfig;
}

export { HF_DATASET_API };
