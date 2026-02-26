export function normalizeText(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function levenshteinDistance(a, b) {
  const left = a || '';
  const right = b || '';
  if (left === right) return 0;
  if (!left.length) return right.length;
  if (!right.length) return left.length;

  const prev = new Array(right.length + 1);
  const curr = new Array(right.length + 1);

  for (let j = 0; j <= right.length; j += 1) {
    prev[j] = j;
  }

  for (let i = 1; i <= left.length; i += 1) {
    curr[0] = i;
    for (let j = 1; j <= right.length; j += 1) {
      const cost = left[i - 1] === right[j - 1] ? 0 : 1;
      curr[j] = Math.min(
        prev[j] + 1,
        curr[j - 1] + 1,
        prev[j - 1] + cost
      );
    }

    for (let j = 0; j <= right.length; j += 1) {
      prev[j] = curr[j];
    }
  }

  return prev[right.length];
}

export function textSimilarity(a, b) {
  const left = normalizeText(a);
  const right = normalizeText(b);
  const maxLen = Math.max(left.length, right.length);
  if (!maxLen) return 1;
  return 1 - levenshteinDistance(left, right) / maxLen;
}

export function mean(values) {
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function percentile(values, p) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[index];
}

export function stddev(values) {
  if (values.length < 2) return 0;
  const avg = mean(values);
  const variance = values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}

export function summarize(values) {
  const numeric = values.filter((value) => Number.isFinite(value));
  if (!numeric.length) {
    return {
      count: 0,
      min: null,
      max: null,
      mean: null,
      median: null,
      p90: null,
      stddev: null,
    };
  }

  return {
    count: numeric.length,
    min: Math.min(...numeric),
    max: Math.max(...numeric),
    mean: mean(numeric),
    median: median(numeric),
    p90: percentile(numeric, 90),
    stddev: stddev(numeric),
  };
}

export function safeNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function csvEscape(value) {
  if (value === null || value === undefined) return '';
  const text = String(value);
  if (/[",\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

export function toCsv(rows, columns) {
  const header = columns.join(',');
  const lines = rows.map((row) => columns.map((col) => csvEscape(row[col])).join(','));
  return [header, ...lines].join('\n');
}

export function flattenRunRecord(run) {
  const metrics = run.metrics || {};

  return {
    batch_id: run.batchId,
    started_at: run.startedAt,
    finished_at: run.finishedAt,
    run_id: run.id,
    sample_key: run.sampleKey,
    sample_order: run.sampleOrder,
    sample_row_index: run.rowIndex,
    repeat_index: run.repeatIndex,
    audio_duration_sec: run.audioDurationSec,
    speaker: run.speaker,
    gender: run.gender,
    speed: run.speed,
    volume: run.volume,
    transcription: run.transcription,
    reference_text: run.referenceText,
    exact_match_first: run.exactMatchToFirst,
    similarity_first: run.similarityToFirst,
    preprocess_ms: metrics.preprocess_ms,
    encode_ms: metrics.encode_ms,
    decode_ms: metrics.decode_ms,
    tokenize_ms: metrics.tokenize_ms,
    total_ms: metrics.total_ms,
    rtf: metrics.rtf,
    preprocessor_backend: metrics.preprocessor_backend,
    error: run.error || '',
    model_key: run.modelKey,
    backend: run.backend,
    encoder_quant: run.encoderQuant,
    decoder_quant: run.decoderQuant,
    preprocessor: run.preprocessor,
    preprocessor_backend_setting: run.preprocessorBackend,
    hardware_cpu: run.hardwareCpu,
    hardware_gpu: run.hardwareGpu,
    hardware_gpu_model: run.hardwareGpuModel,
    hardware_gpu_cores: run.hardwareGpuCores,
    hardware_vram: run.hardwareVram,
    hardware_memory: run.hardwareMemory,
    hardware_webgpu: run.hardwareWebgpu,
  };
}

export const RUN_CSV_COLUMNS = [
  'batch_id',
  'started_at',
  'finished_at',
  'run_id',
  'sample_key',
  'sample_order',
  'sample_row_index',
  'repeat_index',
  'audio_duration_sec',
  'speaker',
  'gender',
  'speed',
  'volume',
  'transcription',
  'reference_text',
  'exact_match_first',
  'similarity_first',
  'preprocess_ms',
  'encode_ms',
  'decode_ms',
  'tokenize_ms',
  'total_ms',
  'rtf',
  'preprocessor_backend',
  'error',
  'model_key',
  'backend',
  'encoder_quant',
  'decoder_quant',
  'preprocessor',
  'preprocessor_backend_setting',
  'hardware_cpu',
  'hardware_gpu',
  'hardware_gpu_model',
  'hardware_gpu_cores',
  'hardware_vram',
  'hardware_memory',
  'hardware_webgpu',
];
