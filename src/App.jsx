import React, { useEffect, useMemo, useRef, useState } from 'react';
import Chart from 'chart.js/auto';
import { MODELS, getParakeetModel, ParakeetModel } from 'parakeet.js';
import {
  fetchDatasetInfo,
  fetchDatasetRows,
  fetchDatasetSplits,
  getConfigsAndSplits,
  normalizeDatasetRow,
} from './utils/hfDataset';
import {
  RUN_CSV_COLUMNS,
  flattenRunRecord,
  mean,
  median,
  normalizeText,
  percentile,
  stddev,
  summarize,
  textSimilarity,
  toCsv,
} from './utils/benchmarkStats';
import {
  fetchModelFiles,
  getAvailableQuantModes,
  pickPreferredQuant,
} from './utils/modelSelection';
import './App.css';

const SETTINGS_KEY = 'parakeet.benchmark.settings.v1';
const SNAPSHOTS_KEY = 'parakeet.benchmark.snapshots.v1';
const DATASET_SPLITS_CACHE_PREFIX = 'parakeet.dataset.splits.v1:';
const DATASET_INFO_CACHE_PREFIX = 'parakeet.dataset.info.v1:';
const DATASET_ROWS_CACHE_PREFIX = 'parakeet.dataset.rows.v1:';
const DATASET_META_CACHE_TTL_MS = 12 * 60 * 60 * 1000;
const DATASET_ROWS_CACHE_TTL_MS = 24 * 60 * 60 * 1000;
const MAX_SAMPLE_COUNT = 10000;
const MAX_SNAPSHOTS = 20;

const MODEL_OPTIONS = Object.entries(MODELS).map(([key, config]) => ({
  key,
  label: config.displayName || key,
}));

const BACKENDS = ['webgpu-hybrid', 'webgpu', 'wasm'];
const QUANTS = ['fp32', 'int8', 'fp16'];
const PREPROCESSOR_MODEL = 'nemo128';
const WARMUP_AUDIO_FALLBACK_URL = 'https://raw.githubusercontent.com/ysdede/parakeet.js/master/examples/demo/public/assets/life_Jim.wav';
const FP16_REVISION_BY_MODEL = {
  'parakeet-tdt-0.6b-v2': 'feat/fp16-canonical-v2',
  'parakeet-tdt-0.6b-v3': 'feat/fp16-canonical-v3',
};
const SNAPSHOT_PARAM_OPTIONS = [
  { key: 'total', label: 'Total', summaryKey: 'totalMean', runField: 'total_ms', color: 'rgba(154, 170, 209, 0.85)' },
  { key: 'preprocess', label: 'Preprocess', summaryKey: 'preprocessMean', runField: 'preprocess_ms', color: 'rgba(217, 179, 122, 0.85)' },
  { key: 'encode', label: 'Encode', summaryKey: 'encodeMean', runField: 'encode_ms', color: 'rgba(124, 166, 220, 0.85)' },
  { key: 'decode', label: 'Decode', summaryKey: 'decodeMean', runField: 'decode_ms', color: 'rgba(121, 194, 159, 0.85)' },
];

function clamp(value, fallback, min, max) {
  const num = Number(value);
  if (!Number.isFinite(num)) return fallback;
  return Math.max(min, Math.min(max, num));
}

function pct(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '-';
}

function ms(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)} ms` : '-';
}

function rtfx(value) {
  return Number.isFinite(value) ? `${value.toFixed(2)}x` : '-';
}

function calcRtfx(audioDurationSec, stageMs) {
  const duration = Number(audioDurationSec);
  const latencyMs = Number(stageMs);
  if (!Number.isFinite(duration) || !Number.isFinite(latencyMs) || duration <= 0 || latencyMs <= 0) {
    return null;
  }
  return (duration * 1000) / latencyMs;
}

function getFp16Revision(modelKey) {
  return FP16_REVISION_BY_MODEL[modelKey] || 'main';
}

function deltaPercent(base, next, lowerIsBetter = true) {
  if (!Number.isFinite(base) || !Number.isFinite(next) || base === 0) return '-';
  const raw = ((next - base) / base) * 100;
  const better = lowerIsBetter ? raw < 0 : raw > 0;
  const sign = raw > 0 ? '+' : '';
  return `${sign}${raw.toFixed(1)}% ${better ? 'better' : 'worse'}`;
}

function saveText(fileName, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  a.click();
  URL.revokeObjectURL(url);
}

function getWarmupAudioCandidates() {
  const base = import.meta.env.BASE_URL || '/';
  return [
    `${base}assets/life_Jim.wav`,
    WARMUP_AUDIO_FALLBACK_URL,
  ];
}

const AUDIO_CACHE_DB = 'parakeet-benchmark-cache';
const AUDIO_CACHE_STORE = 'audio-files';

function openAudioCacheDb() {
  if (typeof indexedDB === 'undefined') {
    return Promise.reject(new Error('IndexedDB is unavailable'));
  }
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(AUDIO_CACHE_DB, 1);
    request.onerror = () => reject(new Error('Failed to open IndexedDB'));
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(AUDIO_CACHE_STORE)) {
        db.createObjectStore(AUDIO_CACHE_STORE, { keyPath: 'key' });
      }
    };
    request.onsuccess = () => resolve(request.result);
  });
}

async function getCachedAudioBlob(cacheKey) {
  try {
    const db = await openAudioCacheDb();
    return await new Promise((resolve) => {
      const tx = db.transaction(AUDIO_CACHE_STORE, 'readonly');
      const store = tx.objectStore(AUDIO_CACHE_STORE);
      const req = store.get(cacheKey);
      req.onsuccess = () => resolve(req.result?.blob || null);
      req.onerror = () => resolve(null);
    });
  } catch {
    return null;
  }
}

async function putCachedAudioBlob(cacheKey, blob) {
  try {
    const db = await openAudioCacheDb();
    await new Promise((resolve) => {
      const tx = db.transaction(AUDIO_CACHE_STORE, 'readwrite');
      const store = tx.objectStore(AUDIO_CACHE_STORE);
      store.put({ key: cacheKey, blob, timestamp: Date.now() });
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    });
  } catch {
    // Ignore caching failures and continue normally.
  }
}

async function clearCachedAudioBlobs() {
  try {
    const db = await openAudioCacheDb();
    await new Promise((resolve) => {
      const tx = db.transaction(AUDIO_CACHE_STORE, 'readwrite');
      const store = tx.objectStore(AUDIO_CACHE_STORE);
      store.clear();
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    });
  } catch {
    // Ignore IndexedDB cache clear failures.
  }
}

function linearFit(xArr, yArr) {
  const n = xArr.length;
  if (n < 2) return { a: 0, b: 0, r2: 0 };
  const mx = xArr.reduce((s, v) => s + v, 0) / n;
  const my = yArr.reduce((s, v) => s + v, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (xArr[i] - mx) * (yArr[i] - my); den += (xArr[i] - mx) ** 2; }
  const a = den === 0 ? 0 : num / den;
  const b = my - a * mx;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < n; i++) { ssRes += (yArr[i] - (a * xArr[i] + b)) ** 2; ssTot += (yArr[i] - my) ** 2; }
  const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  return { a, b, r2 };
}

function histogram(values, binWidth) {
  if (!values.length) return { labels: [], counts: [] };
  const min = Math.floor(Math.min(...values) / binWidth) * binWidth;
  const max = Math.ceil(Math.max(...values) / binWidth) * binWidth;
  const labels = [];
  const counts = [];
  for (let lo = min; lo < max; lo += binWidth) {
    labels.push(`${lo}–${lo + binWidth}`);
    counts.push(values.filter((v) => v >= lo && v < lo + binWidth).length);
  }
  return { labels, counts };
}

function chartBase(yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'nearest', intersect: false },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#b0bdd0',
          font: { family: 'Inter', size: 10 },
          boxWidth: 8,
          padding: 10,
        },
      },
      tooltip: {
        backgroundColor: '#181c28',
        titleColor: '#eaf0f9',
        bodyColor: '#b0bdd0',
        borderColor: 'rgba(255,255,255,0.08)',
        borderWidth: 1,
        padding: 8,
        bodyFont: { family: 'JetBrains Mono', size: 10 },
        titleFont: { family: 'Inter', size: 11 },
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(255,255,255,0.03)' },
        ticks: { color: '#6b7a90', font: { family: 'JetBrains Mono', size: 10 } },
      },
      y: {
        grid: { color: 'rgba(255,255,255,0.03)' },
        ticks: { color: '#6b7a90', font: { family: 'JetBrains Mono', size: 10 } },
        title: {
          display: true,
          text: yLabel,
          color: '#b0bdd0',
          font: { family: 'Inter', size: 11, weight: '600' },
        },
      },
    },
  };
}

function ChartCard({ title, badge, config }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !config) return undefined;
    if (chartRef.current) chartRef.current.destroy();
    chartRef.current = new Chart(canvasRef.current, config);
    return () => {
      if (chartRef.current) chartRef.current.destroy();
    };
  }, [config]);

  return (
    <section className="chart-panel">
      <h3>{title} <span className="badge">{badge}</span></h3>
      <div className="chart-wrap"><canvas ref={canvasRef} /></div>
    </section>
  );
}

function loadSettings() {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}');
  } catch {
    return {};
  }
}

function loadSnapshots() {
  try {
    const parsed = JSON.parse(localStorage.getItem(SNAPSHOTS_KEY) || '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function toCacheKey(prefix, ...parts) {
  return `${prefix}${parts.map((part) => encodeURIComponent(String(part))).join('::')}`;
}

function normalizeSeedText(seed) {
  if (seed === undefined || seed === null || seed === '') return null;
  return String(seed);
}

function createSeededRng(seedText) {
  if (!seedText) return Math.random;
  let h = 2166136261;
  for (let i = 0; i < seedText.length; i += 1) {
    h ^= seedText.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  let t = h >>> 0;
  return function next() {
    t += 0x6D2B79F5;
    let x = Math.imul(t ^ (t >>> 15), t | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function pickUniqueRandomIndices(total, count, seedText) {
  const maxTotal = Math.max(0, Number(total) || 0);
  const wanted = Math.max(0, Math.min(maxTotal, Number(count) || 0));
  if (!wanted) return [];
  const rng = createSeededRng(seedText);
  const picked = new Set();
  while (picked.size < wanted) {
    picked.add(Math.floor(rng() * maxTotal));
  }
  return Array.from(picked);
}

function readCacheEntry(key) {
  try {
    const parsed = JSON.parse(localStorage.getItem(key) || 'null');
    if (!parsed || typeof parsed !== 'object') return null;
    if (typeof parsed.savedAt !== 'number') return null;
    return parsed;
  } catch {
    return null;
  }
}

function writeCacheEntry(key, data) {
  try {
    localStorage.setItem(key, JSON.stringify({
      savedAt: Date.now(),
      data,
    }));
  } catch {
    // Ignore storage quota / serialization errors.
  }
}

function clearLocalStorageByPrefix(prefix) {
  try {
    const toDelete = [];
    for (let i = 0; i < localStorage.length; i += 1) {
      const key = localStorage.key(i);
      if (key && key.startsWith(prefix)) toDelete.push(key);
    }
    toDelete.forEach((key) => localStorage.removeItem(key));
    return toDelete.length;
  } catch {
    return 0;
  }
}

function isFresh(entry, ttlMs = DATASET_META_CACHE_TTL_MS) {
  return !!entry && (Date.now() - entry.savedAt) <= ttlMs;
}

async function readThroughCache({ key, fetcher, ttlMs = DATASET_META_CACHE_TTL_MS, forceRefresh = false }) {
  const cached = readCacheEntry(key);
  if (!forceRefresh && isFresh(cached, ttlMs)) {
    return { data: cached.data, fromCache: true, stale: false };
  }

  try {
    const data = await fetcher();
    writeCacheEntry(key, data);
    return { data, fromCache: false, stale: false };
  } catch (error) {
    if (cached?.data) {
      return { data: cached.data, fromCache: true, stale: true, error };
    }
    throw error;
  }
}

function compactRunsForStorage(runs) {
  return runs.map((run) => ({
    id: run.id,
    sampleKey: run.sampleKey,
    repeatIndex: run.repeatIndex,
    exactMatchToFirst: run.exactMatchToFirst,
    similarityToFirst: run.similarityToFirst,
    audioDurationSec: run.audioDurationSec,
    metrics: run.metrics ? {
      preprocess_ms: run.metrics.preprocess_ms,
      encode_ms: run.metrics.encode_ms,
      decode_ms: run.metrics.decode_ms,
      tokenize_ms: run.metrics.tokenize_ms,
      total_ms: run.metrics.total_ms,
      rtf: run.metrics.rtf,
      encode_rtfx: calcRtfx(run.audioDurationSec, run.metrics.encode_ms),
      decode_rtfx: calcRtfx(run.audioDurationSec, run.metrics.decode_ms),
      preprocessor_backend: run.metrics.preprocessor_backend,
    } : null,
    error: run.error || null,
  }));
}

function formatGpuLabel(profile) {
  if (!profile) return '-';
  const gpuDesc = profile.webgpu?.info?.description || profile.webgpu?.info?.device;
  if (gpuDesc) return gpuDesc;
  if (profile.webgl?.renderer) return profile.webgl.renderer;
  return profile.browser?.webgpuSupported ? 'WebGPU available' : 'No GPU info';
}

function parseGpuModel(profile) {
  const text = [
    profile?.webgpu?.info?.description,
    profile?.webgpu?.info?.device,
    profile?.webgl?.renderer,
  ].filter(Boolean).join(' | ');
  if (!text) return '-';

  const candidates = [
    /NVIDIA\s+GeForce\s+([A-Za-z0-9 .\-]+)/i,
    /(GeForce\s+[A-Za-z0-9 .\-]+)/i,
    /(Radeon\s+[A-Za-z0-9 .\-]+)/i,
    /(RX\s*\d{3,4}[A-Za-z0-9 ]*)/i,
    /(Arc\s+[A-Za-z0-9 .\-]+)/i,
    /(Intel\(R\)\s+[A-Za-z0-9 .\-]+Graphics)/i,
    /(Apple\s+[A-Za-z0-9 .\-]+)/i,
  ];

  for (const regex of candidates) {
    const match = text.match(regex);
    if (match?.[1]) {
      return match[1].replace(/\s+Direct3D.*$/i, '').replace(/\s*\(0x[0-9A-F]+\).*$/i, '').trim();
    }
  }
  return '-';
}

function formatCpuLabel(profile) {
  if (!profile) return '-';
  const cores = profile.browser?.hardwareConcurrency;
  if (!Number.isFinite(cores)) return 'Unknown CPU';
  return `${cores} logical cores`;
}

function summarizeHardwareProfile(profile) {
  if (!profile) {
    return {
      cpuLabel: '-',
      gpuLabel: '-',
      gpuModelLabel: '-',
      gpuCoresLabel: 'Unavailable in browser',
      vramLabel: 'Unavailable in browser',
      systemMemoryLabel: '-',
      webgpuLabel: 'No',
    };
  }
  return {
    cpuLabel: formatCpuLabel(profile),
    gpuLabel: formatGpuLabel(profile),
    gpuModelLabel: parseGpuModel(profile),
    gpuCoresLabel: 'Unavailable in browser',
    vramLabel: 'Unavailable in browser',
    systemMemoryLabel: Number.isFinite(profile.browser?.deviceMemory) ? `${profile.browser.deviceMemory} GB (hint)` : '-',
    webgpuLabel: profile.webgpu?.adapterFound ? 'Yes' : 'No',
  };
}

async function collectHardwareProfile() {
  const profile = {
    capturedAt: new Date().toISOString(),
    browser: {
      userAgent: navigator.userAgent || null,
      language: navigator.language || null,
      platform: navigator.userAgentData?.platform || navigator.platform || null,
      hardwareConcurrency: navigator.hardwareConcurrency ?? null,
      deviceMemory: navigator.deviceMemory ?? null,
      crossOriginIsolated: globalThis.crossOriginIsolated ?? false,
      webgpuSupported: typeof navigator !== 'undefined' && 'gpu' in navigator,
    },
    webgpu: {
      adapterFound: false,
      isFallbackAdapter: null,
      info: null,
      limits: null,
      features: [],
      error: null,
    },
    webgl: null,
  };

  if ('gpu' in navigator) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        const info = adapter.info || {};
        profile.webgpu = {
          adapterFound: true,
          isFallbackAdapter: adapter.isFallbackAdapter ?? null,
          info: {
            vendor: info.vendor || null,
            architecture: info.architecture || null,
            device: info.device || null,
            description: info.description || null,
          },
          limits: {
            maxBufferSize: adapter.limits?.maxBufferSize ?? null,
            maxStorageBufferBindingSize: adapter.limits?.maxStorageBufferBindingSize ?? null,
            maxComputeWorkgroupStorageSize: adapter.limits?.maxComputeWorkgroupStorageSize ?? null,
          },
          features: Array.from(adapter.features || []),
          error: null,
        };
      }
    } catch (error) {
      profile.webgpu.error = String(error);
    }
  }

  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (gl) {
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      profile.webgl = ext
        ? {
          vendor: gl.getParameter(ext.UNMASKED_VENDOR_WEBGL),
          renderer: gl.getParameter(ext.UNMASKED_RENDERER_WEBGL),
        }
        : { available: true, debugRendererInfo: false };
    }
  } catch (error) {
    profile.webgl = { error: String(error) };
  }

  return profile;
}

export default function App() {
  const saved = loadSettings();
  const [modelKey, setModelKey] = useState(saved.modelKey || MODEL_OPTIONS[0]?.key || 'parakeet-tdt-0.6b-v2');
  const [backend, setBackend] = useState(saved.backend || 'webgpu-hybrid');
  const [encoderQuant, setEncoderQuant] = useState(saved.encoderQuant || 'fp32');
  const [decoderQuant, setDecoderQuant] = useState(saved.decoderQuant || 'int8');
  const [preprocessorBackend, setPreprocessorBackend] = useState(saved.preprocessorBackend || 'onnx');
  const [cpuThreads, setCpuThreads] = useState(clamp(saved.cpuThreads, Math.max(1, (navigator.hardwareConcurrency || 4) - 1), 1, 64));
  const [enableProfiling, setEnableProfiling] = useState(saved.enableProfiling !== false);

  const [datasetId, setDatasetId] = useState(saved.datasetId || 'ysdede/parrot-radiology-asr-en');
  const [datasetConfig, setDatasetConfig] = useState(saved.datasetConfig || 'default');
  const [datasetSplit, setDatasetSplit] = useState(saved.datasetSplit || 'train');
  const [offset, setOffset] = useState(clamp(saved.offset, 0, 0, 1_000_000));
  const [sampleCount, setSampleCount] = useState(clamp(saved.sampleCount, 6, 1, MAX_SAMPLE_COUNT));
  const [repeatCount, setRepeatCount] = useState(clamp(saved.repeatCount, 5, 1, 100));
  const [warmups, setWarmups] = useState(clamp(saved.warmups, 1, 0, 10));
  const [randomize, setRandomize] = useState(saved.randomize !== false);
  const [randomSeed, setRandomSeed] = useState(saved.randomSeed ?? '42');

  const [configs, setConfigs] = useState(['default']);
  const [splits, setSplits] = useState(['train', 'validation', 'test']);
  const [splitCounts, setSplitCounts] = useState({});
  const [features, setFeatures] = useState([]);
  const [preparedSamples, setPreparedSamples] = useState([]);

  const [modelStatus, setModelStatus] = useState('Model not loaded');
  const [modelProgress, setModelProgress] = useState('');
  const [resolvedModelInfo, setResolvedModelInfo] = useState('');
  const [datasetStatus, setDatasetStatus] = useState('Idle');
  const [benchStatus, setBenchStatus] = useState('No runs yet');

  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [isLoadingDataset, setIsLoadingDataset] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [isModelReady, setIsModelReady] = useState(false);

  const [progress, setProgress] = useState({ current: 0, total: 0, stage: '' });
  const [runs, setRuns] = useState([]);
  const [snapshotName, setSnapshotName] = useState('');
  const [snapshots, setSnapshots] = useState(loadSnapshots());
  const [compareAId, setCompareAId] = useState('');
  const [compareBId, setCompareBId] = useState('');
  const [selectedSnapshotIds, setSelectedSnapshotIds] = useState([]);
  const [selectedCompareParams, setSelectedCompareParams] = useState(['decode']);
  const [showPreparedSamples, setShowPreparedSamples] = useState(false);
  const [activeTab, setActiveTab] = useState('benchmark');
  const [theme, setTheme] = useState(() => localStorage.getItem('parakeet-theme') || 'dark');
  const [pivotGroupBy, setPivotGroupBy] = useState('quant');
  const [pivotMetrics, setPivotMetrics] = useState(['totalMean', 'encodeMean', 'decodeMean', 'rtfMedian']);
  const fileInputRef = useRef(null);
  const [hardwareProfile, setHardwareProfile] = useState(null);
  const [hardwareStatus, setHardwareStatus] = useState('');
  const [isLoadingHardware, setIsLoadingHardware] = useState(false);
  const [encoderQuantOptions, setEncoderQuantOptions] = useState(QUANTS);
  const [decoderQuantOptions, setDecoderQuantOptions] = useState(QUANTS);

  const modelRef = useRef(null);
  const releaseQueueRef = useRef(Promise.resolve());
  const stopRef = useRef(false);
  const audioCacheRef = useRef(new Map());

  // Theme
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('parakeet-theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'));

  async function releaseModelResources(model) {
    if (!model) return;
    try { model.stopProfiling?.(); } catch { }
    const releasables = [
      model.encoderSession,
      model.joinerSession,
      model.onnxPreprocessor?.session,
    ];
    await Promise.all(releasables.map(async (session) => {
      if (!session) return;
      try {
        if (typeof session.release === 'function') {
          await session.release();
        } else if (typeof session.dispose === 'function') {
          session.dispose();
        }
      } catch { }
    }));
  }

  function queueModelRelease(model) {
    if (!model) return releaseQueueRef.current;
    const releaseTask = releaseQueueRef.current
      .catch(() => { })
      .then(() => releaseModelResources(model));
    releaseQueueRef.current = releaseTask.catch(() => { });
    return releaseTask;
  }

  useEffect(() => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify({
      modelKey,
      backend,
      encoderQuant,
      decoderQuant,
      preprocessorBackend,
      cpuThreads,
      enableProfiling,
      datasetId,
      datasetConfig,
      datasetSplit,
      offset,
      sampleCount,
      repeatCount,
      warmups,
      randomize,
      randomSeed,
    }));
  }, [modelKey, backend, encoderQuant, decoderQuant, preprocessorBackend, cpuThreads, enableProfiling, datasetId, datasetConfig, datasetSplit, offset, sampleCount, repeatCount, warmups, randomize, randomSeed]);

  useEffect(() => {
    // Changing model/runtime parameters invalidates the previously loaded model.
    const previousModel = modelRef.current;
    modelRef.current = null;
    void queueModelRelease(previousModel);
    setIsModelReady(false);
    setResolvedModelInfo('');
  }, [modelKey, backend, encoderQuant, decoderQuant, preprocessorBackend, cpuThreads]);

  useEffect(() => () => {
    const model = modelRef.current;
    modelRef.current = null;
    void queueModelRelease(model);
  }, []);

  useEffect(() => {
    setEncoderQuant((current) => (
      encoderQuantOptions.includes(current)
        ? current
        : pickPreferredQuant(encoderQuantOptions, backend, 'encoder')
    ));
    setDecoderQuant((current) => (
      decoderQuantOptions.includes(current)
        ? current
        : pickPreferredQuant(decoderQuantOptions, backend, 'decoder')
    ));
  }, [backend, encoderQuantOptions, decoderQuantOptions]);

  useEffect(() => {
    let cancelled = false;
    const repoId = MODELS[modelKey]?.repoId || modelKey;
    const revision = getFp16Revision(modelKey);

    (async () => {
      const files = await fetchModelFiles(repoId, revision);
      if (cancelled) return;

      const encOptions = getAvailableQuantModes(files, 'encoder-model');
      const decOptions = getAvailableQuantModes(files, 'decoder_joint-model');

      setEncoderQuantOptions(encOptions);
      setDecoderQuantOptions(decOptions);
    })();

    return () => {
      cancelled = true;
    };
  }, [modelKey]);

  useEffect(() => {
    localStorage.setItem(SNAPSHOTS_KEY, JSON.stringify(snapshots));
  }, [snapshots]);

  useEffect(() => {
    if (!snapshots.length) {
      setCompareAId('');
      setCompareBId('');
      return;
    }
    if (!compareAId) setCompareAId(snapshots[0].id);
    if (!compareBId) setCompareBId(snapshots[1]?.id || snapshots[0].id);
  }, [snapshots, compareAId, compareBId]);

  useEffect(() => {
    setSelectedSnapshotIds((prev) => prev.filter((id) => snapshots.some((s) => s.id === id)));
  }, [snapshots]);

  async function refreshDatasetMeta(nextDataset = datasetId, nextConfig = datasetConfig, nextSplit = datasetSplit, forceRefresh = false) {
    setIsLoadingDataset(true);
    setDatasetStatus('Loading dataset metadata...');
    try {
      const splitsCacheKey = toCacheKey(DATASET_SPLITS_CACHE_PREFIX, nextDataset);
      const splitsResult = await readThroughCache({
        key: splitsCacheKey,
        fetcher: () => fetchDatasetSplits(nextDataset),
        forceRefresh,
      });
      const splitItems = splitsResult.data || [];
      const map = getConfigsAndSplits(splitItems);
      const allConfigs = Array.from(map.keys());
      const safeConfig = allConfigs.includes(nextConfig) ? nextConfig : (allConfigs[0] || 'default');
      const allSplits = map.get(safeConfig) || ['train', 'validation', 'test'];
      const safeSplit = allSplits.includes(nextSplit) ? nextSplit : (allSplits[0] || 'train');

      setConfigs(allConfigs.length ? allConfigs : ['default']);
      setSplits(allSplits);
      setDatasetConfig(safeConfig);
      setDatasetSplit(safeSplit);

      const infoCacheKey = toCacheKey(DATASET_INFO_CACHE_PREFIX, nextDataset, safeConfig);
      const infoResult = await readThroughCache({
        key: infoCacheKey,
        fetcher: () => fetchDatasetInfo(nextDataset, safeConfig),
        forceRefresh,
      });
      const info = infoResult.data || {};
      setFeatures(Object.keys(info?.dataset_info?.features || {}));
      setSplitCounts(info?.dataset_info?.splits || {});
      const usedCache = splitsResult.fromCache || infoResult.fromCache;
      const staleCache = splitsResult.stale || infoResult.stale;
      const mode = staleCache ? 'cache-stale' : (usedCache ? 'cache' : 'live');
      setDatasetStatus(`Ready: ${safeConfig}/${safeSplit} (${mode})`);
    } catch (error) {
      console.error(error);
      setDatasetStatus(`Metadata failed: ${error.message}`);
    } finally {
      setIsLoadingDataset(false);
    }
  }

  useEffect(() => {
    refreshDatasetMeta();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function refreshHardwareProfile() {
    setIsLoadingHardware(true);
    setHardwareStatus('Reading browser hardware profile...');
    try {
      const profile = await collectHardwareProfile();
      setHardwareProfile(profile);
      setHardwareStatus('Ready');
    } catch (error) {
      console.error(error);
      setHardwareStatus(`Hardware probe failed: ${error.message}`);
    } finally {
      setIsLoadingHardware(false);
    }
  }

  useEffect(() => {
    refreshHardwareProfile();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function prepareSampleRows() {
    const requested = clamp(sampleCount, 6, 1, MAX_SAMPLE_COUNT);
    setDatasetStatus('Preparing sample rows (user pipeline simulation)...');

    const fetchDatasetRowsCached = ({ dataset, config, split, offset: rowOffset = 0, length = 100 }) => {
      const key = toCacheKey(DATASET_ROWS_CACHE_PREFIX, dataset, config, split, rowOffset, length);
      return readThroughCache({
        key,
        ttlMs: DATASET_ROWS_CACHE_TTL_MS,
        fetcher: () => fetchDatasetRows({
          dataset,
          config,
          split,
          offset: rowOffset,
          length,
        }),
      });
    };

    let rawRows = [];
    let randomMeta = null;
    if (randomize) {
      // Match regular demo behavior: fetch one small row pool, then randomly pick from it.
      const poolLength = Math.max(requested, Math.min(100, requested * 4));
      const poolResult = await fetchDatasetRowsCached({
        dataset: datasetId,
        config: datasetConfig,
        split: datasetSplit,
        offset,
        length: poolLength,
      });
      const poolRows = poolResult.data?.rows || [];
      const seedText = normalizeSeedText(randomSeed);
      const seedMaterial = seedText
        ? `${seedText}|${datasetId}|${datasetConfig}|${datasetSplit}|${offset}|${requested}`
        : null;
      const chosenIndices = pickUniqueRandomIndices(poolRows.length, requested, seedMaterial);
      rawRows = chosenIndices.map((idx) => poolRows[idx]).filter(Boolean);
      randomMeta = {
        requestedCount: requested,
        poolCount: poolRows.length,
        fromCache: poolResult.fromCache,
      };
    } else {
      let cursor = Math.max(0, Number(offset) || 0);
      let rowsFromCache = false;

      while (rawRows.length < requested) {
        const pageLength = Math.min(100, requested - rawRows.length);
        const pageResult = await fetchDatasetRowsCached({
          dataset: datasetId,
          config: datasetConfig,
          split: datasetSplit,
          offset: cursor,
          length: pageLength,
        });

        const pageRows = pageResult.data?.rows || [];
        rowsFromCache = rowsFromCache || !!pageResult.fromCache;
        if (!pageRows.length) break;

        rawRows.push(...pageRows);
        cursor += pageRows.length;
        if (pageRows.length < pageLength) break;
      }

      randomMeta = {
        fromCache: rowsFromCache,
      };
    }

    const normalized = rawRows
      .map((item, idx) => normalizeDatasetRow(item, idx))
      .filter((item) => item.audioUrl);

    if (!normalized.length) {
      throw new Error('No playable rows found in the selected slice.');
    }

    setPreparedSamples(normalized);
    if (randomize) {
      const reqCount = randomMeta?.requestedCount || requested;
      const shortfall = reqCount - normalized.length;
      const poolCount = randomMeta?.poolCount || normalized.length;
      const warning = shortfall > 0 ? `, shortfall ${shortfall}` : '';
      const mode = randomMeta?.fromCache ? 'cache' : 'live';
      setDatasetStatus(`Prepared ${normalized.length}/${reqCount} samples from ${poolCount} rows (${mode}, seed: ${String(randomSeed) || 'random'}${warning})`);
    } else {
      const mode = randomMeta?.fromCache ? 'cache' : 'live';
      setDatasetStatus(`Prepared ${normalized.length} samples (${mode})`);
    }
    return normalized;
  }

  async function clearDatasetCache() {
    setDatasetStatus('Clearing dataset cache...');
    try {
      const removedMeta = clearLocalStorageByPrefix(DATASET_SPLITS_CACHE_PREFIX)
        + clearLocalStorageByPrefix(DATASET_INFO_CACHE_PREFIX)
        + clearLocalStorageByPrefix(DATASET_ROWS_CACHE_PREFIX);
      await clearCachedAudioBlobs();
      audioCacheRef.current.clear();
      setPreparedSamples([]);
      setDatasetStatus(`Dataset cache cleared (removed ${removedMeta} metadata/row entries + audio blobs). Model/ONNX cache untouched.`);
    } catch (error) {
      setDatasetStatus(`Failed to clear dataset cache: ${error.message}`);
    }
  }

  async function decodeAudio(url) {
    const key = `${url}::16000`;
    if (audioCacheRef.current.has(key)) {
      return audioCacheRef.current.get(key);
    }

    let audioData = null;
    const cachedBlob = await getCachedAudioBlob(key);
    if (cachedBlob) {
      audioData = await cachedBlob.arrayBuffer();
    } else {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Audio fetch failed (${res.status})`);
      const blob = await res.blob();
      audioData = await blob.arrayBuffer();
      await putCachedAudioBlob(key, blob);
    }

    const ctx = new AudioContext({ sampleRate: 16000 });
    try {
      const decoded = await ctx.decodeAudioData(audioData.slice(0));
      const channel = decoded.getChannelData(0);
      const pcm = new Float32Array(channel.length);
      pcm.set(channel);
      const result = {
        pcm,
        sampleRate: 16000,
        durationSec: pcm.length / 16000,
      };
      audioCacheRef.current.set(key, result);
      return result;
    } finally {
      await ctx.close();
    }
  }

  async function verifyModel(model) {
    const expectedText = 'it is not life as we know or understand it';
    let sample = null;
    let lastError = null;
    const warmupSources = getWarmupAudioCandidates();
    for (const src of warmupSources) {
      try {
        sample = await decodeAudio(src);
        break;
      } catch (error) {
        lastError = error;
      }
    }
    if (!sample) {
      throw lastError || new Error('Warmup audio could not be loaded');
    }
    const result = await model.transcribe(sample.pcm, sample.sampleRate, {
      enableProfiling: false,
      returnConfidences: false,
      returnTimestamps: false,
    });

    const got = normalizeText(result?.utterance_text || '');
    const expected = normalizeText(expectedText);
    if (!got.includes(expected)) {
      throw new Error(`Verification mismatch. Expected phrase not found. Got: "${result?.utterance_text || ''}"`);
    }
  }

  async function loadModel() {
    setIsLoadingModel(true);
    setModelStatus('Loading model...');
    setIsModelReady(false);
    setModelProgress('');
    setResolvedModelInfo('');

    try {
      const previousModel = modelRef.current;
      modelRef.current = null;
      await queueModelRelease(previousModel);

      const baseOptions = {
        backend,
        encoderQuant,
        decoderQuant,
        revision: getFp16Revision(modelKey),
        preprocessor: PREPROCESSOR_MODEL,
        preprocessorBackend,
        cpuThreads,
        verbose: false,
        progress: ({ file, loaded, total }) => {
          if (!total) {
            setModelProgress(file || 'Downloading...');
            return;
          }
          setModelProgress(`${file}: ${Math.round((loaded / total) * 100)}%`);
        },
      };

      const hub = await getParakeetModel(modelKey, baseOptions);
      modelRef.current = await ParakeetModel.fromUrls({
        ...hub.urls,
        filenames: hub.filenames,
        preprocessorBackend: hub.preprocessorBackend,
        backend,
        cpuThreads,
        verbose: false,
      });

      const resolvedQuant = hub?.quantisation
        ? `resolved e:${hub.quantisation.encoder} d:${hub.quantisation.decoder}`
        : '';
      const resolvedRevision = baseOptions.revision ? `revision:${baseOptions.revision}` : '';
      const loadedFiles = hub?.filenames
        ? `${hub.filenames.encoder}, ${hub.filenames.decoder}`
        : '';
      const backendHint = backend.startsWith('webgpu')
        ? 'decoder executes on WASM in webgpu modes'
        : '';
      setResolvedModelInfo([resolvedQuant, resolvedRevision, loadedFiles, backendHint].filter(Boolean).join(' | '));

      setModelStatus('Verifying model...');
      setModelProgress('Running reference transcription');
      await verifyModel(modelRef.current);
      setModelStatus('Model ready (verified)');
      setIsModelReady(true);
      setModelProgress('');
    } catch (error) {
      console.error(error);
      const failedModel = modelRef.current;
      modelRef.current = null;
      await queueModelRelease(failedModel);
      setModelStatus(`Load failed: ${error.message}`);
    } finally {
      setIsLoadingModel(false);
    }
  }

  function stopRun() {
    stopRef.current = true;
    setBenchStatus('Stopping after current run...');
  }

  async function runBenchmark() {
    if (!modelRef.current || !isModelReady) {
      setBenchStatus('Load and verify model first');
      return;
    }

    setIsRunning(true);
    stopRef.current = false;
    setBenchStatus('Starting benchmark batch...');

    try {
      const samples = preparedSamples.length ? preparedSamples : await prepareSampleRows();
      const total = samples.length * repeatCount;
      const batchId = `batch-${Date.now()}`;
      const out = [];
      let done = 0;

      for (let s = 0; s < samples.length; s += 1) {
        if (stopRef.current) break;
        const sample = samples[s];
        const sampleKey = `${datasetSplit}:${sample.rowIndex}`;

        setProgress({ current: done, total, stage: `Preparing ${sampleKey} (download + decode)` });

        let decoded;
        try {
          decoded = await decodeAudio(sample.audioUrl);
        } catch (error) {
          out.push({
            id: `${batchId}-${sampleKey}-decode-error`,
            batchId,
            sampleKey,
            rowIndex: sample.rowIndex,
            repeatIndex: 0,
            audioDurationSec: null,
            referenceText: sample.referenceText,
            transcription: '',
            exactMatchToFirst: null,
            similarityToFirst: null,
            metrics: null,
            error: `Decode error: ${error.message}`,
            modelKey,
            backend,
            encoderQuant,
            decoderQuant,
            preprocessor: PREPROCESSOR_MODEL,
            preprocessorBackend,
            hardwareCpu: hardwareSummary.cpuLabel,
            hardwareGpu: hardwareSummary.gpuLabel,
            hardwareGpuModel: hardwareSummary.gpuModelLabel,
            hardwareGpuCores: hardwareSummary.gpuCoresLabel,
            hardwareVram: hardwareSummary.vramLabel,
            hardwareMemory: hardwareSummary.systemMemoryLabel,
            hardwareWebgpu: hardwareSummary.webgpuLabel,
            startedAt: new Date().toISOString(),
            finishedAt: new Date().toISOString(),
          });
          continue;
        }

        for (let w = 0; w < warmups; w += 1) {
          if (stopRef.current) break;
          setProgress({ current: done, total, stage: `Transcribing warmup ${w + 1}/${warmups} for ${sampleKey}` });
          await modelRef.current.transcribe(decoded.pcm, decoded.sampleRate, {
            enableProfiling,
            returnConfidences: false,
            returnTimestamps: false,
          });
        }

        let baseline = null;
        for (let r = 1; r <= repeatCount; r += 1) {
          if (stopRef.current) break;

          const startedAt = new Date().toISOString();
          setProgress({ current: done, total, stage: `Transcribing run ${r}/${repeatCount} for ${sampleKey}` });

          try {
            const result = await modelRef.current.transcribe(decoded.pcm, decoded.sampleRate, {
              enableProfiling,
              returnConfidences: true,
              returnTimestamps: true,
            });

            const normalized = normalizeText(result.utterance_text || '');
            if (baseline === null) baseline = normalized;

            out.push({
              id: `${batchId}-${sampleKey}-run-${r}`,
              batchId,
              sampleKey,
              rowIndex: sample.rowIndex,
              repeatIndex: r,
              audioDurationSec: decoded.durationSec,
              referenceText: sample.referenceText,
              transcription: result.utterance_text || '',
              exactMatchToFirst: baseline === normalized,
              similarityToFirst: textSimilarity(baseline, normalized),
              metrics: result.metrics,
              error: null,
              modelKey,
              backend,
              encoderQuant,
              decoderQuant,
              preprocessor: PREPROCESSOR_MODEL,
              preprocessorBackend,
              hardwareCpu: hardwareSummary.cpuLabel,
              hardwareGpu: hardwareSummary.gpuLabel,
              hardwareGpuModel: hardwareSummary.gpuModelLabel,
              hardwareGpuCores: hardwareSummary.gpuCoresLabel,
              hardwareVram: hardwareSummary.vramLabel,
              hardwareMemory: hardwareSummary.systemMemoryLabel,
              hardwareWebgpu: hardwareSummary.webgpuLabel,
              startedAt,
              finishedAt: new Date().toISOString(),
            });
          } catch (error) {
            out.push({
              id: `${batchId}-${sampleKey}-run-${r}-error`,
              batchId,
              sampleKey,
              rowIndex: sample.rowIndex,
              repeatIndex: r,
              audioDurationSec: decoded.durationSec,
              referenceText: sample.referenceText,
              transcription: '',
              exactMatchToFirst: null,
              similarityToFirst: null,
              metrics: null,
              error: `Transcribe error: ${error.message}`,
              modelKey,
              backend,
              encoderQuant,
              decoderQuant,
              preprocessor: PREPROCESSOR_MODEL,
              preprocessorBackend,
              hardwareCpu: hardwareSummary.cpuLabel,
              hardwareGpu: hardwareSummary.gpuLabel,
              hardwareGpuModel: hardwareSummary.gpuModelLabel,
              hardwareGpuCores: hardwareSummary.gpuCoresLabel,
              hardwareVram: hardwareSummary.vramLabel,
              hardwareMemory: hardwareSummary.systemMemoryLabel,
              hardwareWebgpu: hardwareSummary.webgpuLabel,
              startedAt,
              finishedAt: new Date().toISOString(),
            });
          }

          done += 1;
        }
      }

      setRuns((prev) => [...prev, ...out]);
      setBenchStatus(stopRef.current ? `Stopped. Added ${out.length} rows.` : `Completed. Added ${out.length} rows.`);
      setProgress({ current: done, total, stage: stopRef.current ? 'Stopped' : 'Complete' });
    } catch (error) {
      console.error(error);
      setBenchStatus(`Benchmark failed: ${error.message}`);
    } finally {
      stopRef.current = false;
      setIsRunning(false);
    }
  }

  const okRuns = useMemo(
    () => runs.filter((r) => !r.error && r.metrics && Number.isFinite(r.metrics.total_ms)),
    [runs]
  );

  const metrics = useMemo(() => ({
    total: summarize(okRuns.map((r) => r.metrics?.total_ms)),
    preprocess: summarize(okRuns.map((r) => r.metrics?.preprocess_ms)),
    encode: summarize(okRuns.map((r) => r.metrics?.encode_ms)),
    decode: summarize(okRuns.map((r) => r.metrics?.decode_ms)),
    tokenize: summarize(okRuns.map((r) => r.metrics?.tokenize_ms)),
    rtf: summarize(okRuns.map((r) => r.metrics?.rtf)),
    encodeRtfx: summarize(okRuns.map((r) => calcRtfx(r.audioDurationSec, r.metrics?.encode_ms))),
    decodeRtfx: summarize(okRuns.map((r) => calcRtfx(r.audioDurationSec, r.metrics?.decode_ms))),
  }), [okRuns]);

  const repeatability = useMemo(() => {
    const exact = okRuns.map((r) => r.exactMatchToFirst).filter((v) => typeof v === 'boolean');
    const similarity = okRuns.map((r) => r.similarityToFirst).filter(Number.isFinite);
    return {
      exactRate: exact.length ? exact.filter(Boolean).length / exact.length : null,
      similarityMean: similarity.length ? mean(similarity) : null,
      similarityStd: similarity.length ? stddev(similarity) : null,
    };
  }, [okRuns]);

  const sampleStats = useMemo(() => {
    const map = new Map();
    okRuns.forEach((run) => {
      const entry = map.get(run.sampleKey) || {
        sampleKey: run.sampleKey,
        preprocess: [],
        encode: [],
        decode: [],
        encodeRtfx: [],
        decodeRtfx: [],
        total: [],
        exact: [],
        sim: [],
        texts: [],
      };
      if (Number.isFinite(run.metrics?.preprocess_ms)) entry.preprocess.push(run.metrics.preprocess_ms);
      if (Number.isFinite(run.metrics?.encode_ms)) entry.encode.push(run.metrics.encode_ms);
      if (Number.isFinite(run.metrics?.decode_ms)) entry.decode.push(run.metrics.decode_ms);
      const encodeRtfx = calcRtfx(run.audioDurationSec, run.metrics?.encode_ms);
      const decodeRtfx = calcRtfx(run.audioDurationSec, run.metrics?.decode_ms);
      if (Number.isFinite(encodeRtfx)) entry.encodeRtfx.push(encodeRtfx);
      if (Number.isFinite(decodeRtfx)) entry.decodeRtfx.push(decodeRtfx);
      if (Number.isFinite(run.metrics?.total_ms)) entry.total.push(run.metrics.total_ms);
      if (typeof run.exactMatchToFirst === 'boolean') entry.exact.push(run.exactMatchToFirst ? 1 : 0);
      if (Number.isFinite(run.similarityToFirst)) entry.sim.push(run.similarityToFirst);
      entry.texts.push(normalizeText(run.transcription || ''));
      map.set(run.sampleKey, entry);
    });

    return Array.from(map.values()).map((entry) => ({
      sampleKey: entry.sampleKey,
      runs: entry.total.length,
      uniqueOutputs: new Set(entry.texts).size,
      exactRate: entry.exact.length ? mean(entry.exact) : null,
      similarity: entry.sim.length ? mean(entry.sim) : null,
      preprocessMean: entry.preprocess.length ? mean(entry.preprocess) : null,
      encodeMean: entry.encode.length ? mean(entry.encode) : null,
      decodeMean: entry.decode.length ? mean(entry.decode) : null,
      encodeRtfxMean: entry.encodeRtfx.length ? mean(entry.encodeRtfx) : null,
      decodeRtfxMean: entry.decodeRtfx.length ? mean(entry.decodeRtfx) : null,
      encodeRtfxStd: entry.encodeRtfx.length ? stddev(entry.encodeRtfx) : null,
      decodeRtfxStd: entry.decodeRtfx.length ? stddev(entry.decodeRtfx) : null,
      decodeStd: entry.decode.length ? stddev(entry.decode) : null,
      totalMean: entry.total.length ? mean(entry.total) : null,
    })).sort((a, b) => (a.exactRate ?? 1) - (b.exactRate ?? 1));
  }, [okRuns]);

  const configStats = useMemo(() => {
    const map = new Map();
    okRuns.forEach((run) => {
      const key = `${run.preprocessorBackend} | ${run.backend}`;
      const item = map.get(key) || {
        key,
        runs: 0,
        preprocess: [],
        encode: [],
        decode: [],
        tokenize: [],
        total: [],
      };
      item.runs += 1;
      if (Number.isFinite(run.metrics?.preprocess_ms)) item.preprocess.push(run.metrics.preprocess_ms);
      if (Number.isFinite(run.metrics?.encode_ms)) item.encode.push(run.metrics.encode_ms);
      if (Number.isFinite(run.metrics?.decode_ms)) item.decode.push(run.metrics.decode_ms);
      if (Number.isFinite(run.metrics?.tokenize_ms)) item.tokenize.push(run.metrics.tokenize_ms);
      if (Number.isFinite(run.metrics?.total_ms)) item.total.push(run.metrics.total_ms);
      map.set(key, item);
    });

    return Array.from(map.values())
      .map((item) => {
        const preprocessMean = item.preprocess.length ? mean(item.preprocess) : null;
        const encodeMean = item.encode.length ? mean(item.encode) : null;
        const decodeMean = item.decode.length ? mean(item.decode) : null;
        const tokenizeMean = item.tokenize.length ? mean(item.tokenize) : null;
        const totalMean = item.total.length ? mean(item.total) : null;
        return {
          key: item.key,
          runs: item.runs,
          preprocessMean,
          encodeMean,
          decodeMean,
          tokenizeMean,
          totalMean,
          preprocessShare: Number.isFinite(preprocessMean) && Number.isFinite(totalMean) && totalMean > 0
            ? preprocessMean / totalMean
            : null,
          decodeShare: Number.isFinite(decodeMean) && Number.isFinite(totalMean) && totalMean > 0
            ? decodeMean / totalMean
            : null,
        };
      })
      .sort((a, b) => (a.totalMean ?? Number.POSITIVE_INFINITY) - (b.totalMean ?? Number.POSITIVE_INFINITY));
  }, [okRuns]);

  const chartConfigs = useMemo(() => {
    if (!okRuns.length) return null;

    const encDecBase = chartBase('Decode (ms)');
    const encDec = {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Encoder vs decoder',
          backgroundColor: 'rgba(124, 166, 220, 0.72)',
          pointRadius: 4,
          data: okRuns
            .filter((r) => Number.isFinite(r.metrics?.encode_ms) && Number.isFinite(r.metrics?.decode_ms))
            .map((r) => ({ x: r.metrics.encode_ms, y: r.metrics.decode_ms, sampleKey: r.sampleKey, repeatIndex: r.repeatIndex })),
        }],
      },
      options: {
        ...encDecBase,
        plugins: {
          ...encDecBase.plugins,
          tooltip: {
            ...encDecBase.plugins.tooltip,
            callbacks: {
              label: (ctx) => {
                const p = ctx.raw;
                return `${p.sampleKey} run ${p.repeatIndex}: enc ${p.x.toFixed(1)} ms, dec ${p.y.toFixed(1)} ms`;
              },
            },
          },
        },
        scales: {
          ...encDecBase.scales,
          x: {
            ...encDecBase.scales.x,
            title: { display: true, text: 'Encode (ms)', color: '#b0bdd0', font: { family: 'Inter', size: 11, weight: '600' } },
          },
        },
      },
    };

    const rtfxRunOrderBase = chartBase('RTFx');
    const rtfxRunOrderPoints = okRuns.map((run, idx) => ({
      runOrder: idx + 1,
      sampleKey: run.sampleKey,
      repeatIndex: run.repeatIndex,
      encoderRtfx: calcRtfx(run.audioDurationSec, run.metrics?.encode_ms),
      decoderRtfx: calcRtfx(run.audioDurationSec, run.metrics?.decode_ms),
    }));
    const rtfxRunOrder = {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Encoder RTFx',
            backgroundColor: 'rgba(124, 166, 220, 0.78)',
            pointRadius: 3,
            data: rtfxRunOrderPoints
              .filter((point) => Number.isFinite(point.encoderRtfx))
              .map((point) => ({
                x: point.runOrder,
                y: point.encoderRtfx,
                sampleKey: point.sampleKey,
                repeatIndex: point.repeatIndex,
              })),
          },
          {
            label: 'Decoder RTFx',
            backgroundColor: 'rgba(121, 194, 159, 0.78)',
            pointRadius: 3,
            data: rtfxRunOrderPoints
              .filter((point) => Number.isFinite(point.decoderRtfx))
              .map((point) => ({
                x: point.runOrder,
                y: point.decoderRtfx,
                sampleKey: point.sampleKey,
                repeatIndex: point.repeatIndex,
              })),
          },
        ],
      },
      options: {
        ...rtfxRunOrderBase,
        plugins: {
          ...rtfxRunOrderBase.plugins,
          tooltip: {
            ...rtfxRunOrderBase.plugins.tooltip,
            callbacks: {
              label: (ctx) => {
                const point = ctx.raw;
                return `${ctx.dataset.label}: run #${point.x} (${point.sampleKey}, repeat ${point.repeatIndex}) => ${point.y.toFixed(2)}x`;
              },
            },
          },
        },
        scales: {
          ...rtfxRunOrderBase.scales,
          x: {
            ...rtfxRunOrderBase.scales.x,
            title: { display: true, text: 'Run order', color: '#b0bdd0', font: { family: 'Inter', size: 11, weight: '600' } },
          },
        },
      },
    };

    const rtfxDurationBase = chartBase('RTFx');
    const rtfxDuration = {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Encoder RTFx',
            backgroundColor: 'rgba(124, 166, 220, 0.72)',
            pointRadius: 4,
            data: okRuns
              .map((run) => ({
                x: run.audioDurationSec,
                y: calcRtfx(run.audioDurationSec, run.metrics?.encode_ms),
                sampleKey: run.sampleKey,
                repeatIndex: run.repeatIndex,
              }))
              .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y)),
          },
          {
            label: 'Decoder RTFx',
            backgroundColor: 'rgba(121, 194, 159, 0.72)',
            pointRadius: 4,
            data: okRuns
              .map((run) => ({
                x: run.audioDurationSec,
                y: calcRtfx(run.audioDurationSec, run.metrics?.decode_ms),
                sampleKey: run.sampleKey,
                repeatIndex: run.repeatIndex,
              }))
              .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y)),
          },
        ],
      },
      options: {
        ...rtfxDurationBase,
        plugins: {
          ...rtfxDurationBase.plugins,
          tooltip: {
            ...rtfxDurationBase.plugins.tooltip,
            callbacks: {
              label: (ctx) => {
                const point = ctx.raw;
                return `${ctx.dataset.label}: ${point.sampleKey} run ${point.repeatIndex}, dur ${point.x.toFixed(2)} s => ${point.y.toFixed(2)}x`;
              },
            },
          },
        },
        scales: {
          ...rtfxDurationBase.scales,
          x: {
            ...rtfxDurationBase.scales.x,
            title: { display: true, text: 'Audio duration (s)', color: '#b0bdd0', font: { family: 'Inter', size: 11, weight: '600' } },
          },
        },
      },
    };

    const durPreBase = chartBase('Preprocess (ms)');
    const durPre = {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Duration vs preprocess',
          backgroundColor: 'rgba(217, 179, 122, 0.72)',
          pointRadius: 4,
          data: okRuns
            .filter((r) => Number.isFinite(r.audioDurationSec) && Number.isFinite(r.metrics?.preprocess_ms))
            .map((r) => ({ x: r.audioDurationSec, y: r.metrics.preprocess_ms, sampleKey: r.sampleKey, repeatIndex: r.repeatIndex })),
        }],
      },
      options: {
        ...durPreBase,
        scales: {
          ...durPreBase.scales,
          x: {
            ...durPreBase.scales.x,
            title: { display: true, text: 'Audio duration (s)', color: '#b0bdd0', font: { family: 'Inter', size: 11, weight: '600' } },
          },
        },
      },
    };

    const repeatIds = Array.from(new Set(okRuns.map((r) => r.repeatIndex))).sort((a, b) => a - b);
    const trend = {
      type: 'line',
      data: {
        labels: repeatIds.map((id) => `Run ${id}`),
        datasets: [
          {
            label: 'Decode mean (ms)',
            borderColor: 'rgba(217, 179, 122, 0.95)',
            backgroundColor: 'rgba(217, 179, 122, 0.2)',
            tension: 0.35,
            pointRadius: 3,
            data: repeatIds.map((id) => mean(okRuns.filter((r) => r.repeatIndex === id).map((r) => r.metrics?.decode_ms).filter(Number.isFinite))),
          },
          {
            label: 'Total mean (ms)',
            borderColor: 'rgba(154,170,209,0.95)',
            backgroundColor: 'rgba(154,170,209,0.2)',
            tension: 0.35,
            pointRadius: 3,
            data: repeatIds.map((id) => mean(okRuns.filter((r) => r.repeatIndex === id).map((r) => r.metrics?.total_ms).filter(Number.isFinite))),
          },
        ],
      },
      options: chartBase('Milliseconds'),
    };

    const stageMean = {
      preprocess: mean(okRuns.map((r) => r.metrics?.preprocess_ms).filter(Number.isFinite)),
      encode: mean(okRuns.map((r) => r.metrics?.encode_ms).filter(Number.isFinite)),
      decode: mean(okRuns.map((r) => r.metrics?.decode_ms).filter(Number.isFinite)),
      tokenize: mean(okRuns.map((r) => r.metrics?.tokenize_ms).filter(Number.isFinite)),
    };
    const bottleneck = {
      type: 'doughnut',
      data: {
        labels: ['Preprocess', 'Encode', 'Decode', 'Tokenize'],
        datasets: [{
          data: [stageMean.preprocess, stageMean.encode, stageMean.decode, stageMean.tokenize],
          backgroundColor: ['#d9b37a', '#7ca6dc', '#79c29f', '#9aaad1'],
          borderColor: '#0e1117',
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#b0bdd0', font: { family: 'Inter', size: 11 } },
          },
        },
      },
    };

    const compareLabels = configStats.map((c) => c.key);
    const compareStages = {
      type: 'bar',
      data: {
        labels: compareLabels,
        datasets: [
          { label: 'Preprocess', data: configStats.map((c) => c.preprocessMean), backgroundColor: 'rgba(217, 179, 122, 0.82)' },
          { label: 'Encode', data: configStats.map((c) => c.encodeMean), backgroundColor: 'rgba(124, 166, 220, 0.82)' },
          { label: 'Decode', data: configStats.map((c) => c.decodeMean), backgroundColor: 'rgba(121, 194, 159, 0.82)' },
        ],
      },
      options: {
        ...chartBase('Mean ms'),
        scales: {
          ...chartBase('Mean ms').scales,
          x: {
            ...chartBase('Mean ms').scales.x,
            stacked: true,
            ticks: { color: '#6b7a90', maxRotation: 30, minRotation: 30, font: { family: 'JetBrains Mono', size: 9 } },
          },
          y: {
            ...chartBase('Mean ms').scales.y,
            stacked: true,
          },
        },
      },
    };

    // ═══ NEW CHART 1: Duration vs Total + Regression Line ═══
    const durTotalPoints = okRuns
      .filter((r) => Number.isFinite(r.audioDurationSec) && Number.isFinite(r.metrics?.total_ms))
      .map((r) => ({ x: r.audioDurationSec, y: r.metrics.total_ms, sampleKey: r.sampleKey }));
    const dtX = durTotalPoints.map((p) => p.x);
    const dtY = durTotalPoints.map((p) => p.y);
    const dtFit = linearFit(dtX, dtY);
    const dtLineData = dtX.length ? [
      { x: Math.min(...dtX), y: dtFit.a * Math.min(...dtX) + dtFit.b },
      { x: Math.max(...dtX), y: dtFit.a * Math.max(...dtX) + dtFit.b },
    ] : [];
    const durTotalBase = chartBase('Total (ms)');
    const durTotal = {
      type: 'scatter',
      data: {
        datasets: [
          { label: 'Total ms', backgroundColor: 'rgba(154, 170, 209, 0.72)', pointRadius: 4, data: durTotalPoints },
          { label: `Fit: ${dtFit.a.toFixed(1)}×dur + ${dtFit.b.toFixed(0)} (R²=${dtFit.r2.toFixed(3)})`, type: 'line', borderColor: 'rgba(224, 169, 78, 0.9)', borderWidth: 2, borderDash: [6, 3], pointRadius: 0, data: dtLineData, fill: false },
        ],
      },
      options: { ...durTotalBase, scales: { ...durTotalBase.scales, x: { ...durTotalBase.scales.x, title: { display: true, text: 'Audio duration (s)', color: '#b0bdd0', font: { family: 'Inter', size: 11, weight: '600' } } } } },
    };

    // ═══ NEW CHART 2: Transcription Length vs Decode ═══
    const txDecPoints = okRuns
      .filter((r) => r.transcription && Number.isFinite(r.metrics?.decode_ms))
      .map((r) => ({ x: r.transcription.length, y: r.metrics.decode_ms, sampleKey: r.sampleKey }));
    const txDecBase = chartBase('Decode (ms)');
    const txDecode = {
      type: 'scatter',
      data: { datasets: [{ label: 'Transcription length vs decode', backgroundColor: 'rgba(121, 194, 159, 0.72)', pointRadius: 4, data: txDecPoints }] },
      options: { ...txDecBase, scales: { ...txDecBase.scales, x: { ...txDecBase.scales.x, title: { display: true, text: 'Transcription length (chars)', color: '#b0bdd0', font: { family: 'Inter', size: 11, weight: '600' } } } } },
    };

    // ═══ NEW CHART 3: Phase Timing Box Plots (bar with error bars visualized as ranges) ═══
    const phaseNames = ['Preprocess', 'Encode', 'Decode', 'Tokenize'];
    const phaseFields = ['preprocess_ms', 'encode_ms', 'decode_ms', 'tokenize_ms'];
    const phaseColors = ['rgba(217, 179, 122, 0.82)', 'rgba(124, 166, 220, 0.82)', 'rgba(121, 194, 159, 0.82)', 'rgba(154, 170, 209, 0.82)'];
    const phaseData = phaseFields.map((f) => okRuns.map((r) => r.metrics?.[f]).filter(Number.isFinite));
    const phaseBoxBase = chartBase('Time (ms)');
    const phaseBox = {
      type: 'bar',
      data: {
        labels: phaseNames,
        datasets: [
          { label: 'p25', data: phaseData.map((d) => percentile(d, 25)), backgroundColor: 'transparent', borderWidth: 0, barPercentage: 0.6 },
          { label: 'Median', data: phaseData.map((d) => median(d) - (percentile(d, 25) || 0)), backgroundColor: phaseColors, borderWidth: 0, barPercentage: 0.6 },
          { label: 'p75', data: phaseData.map((d) => (percentile(d, 75) || 0) - (median(d) || 0)), backgroundColor: phaseColors.map((c) => c.replace('0.82', '0.45')), borderWidth: 0, barPercentage: 0.6 },
        ],
      },
      options: {
        ...phaseBoxBase,
        scales: {
          ...phaseBoxBase.scales,
          x: { ...phaseBoxBase.scales.x, stacked: true },
          y: { ...phaseBoxBase.scales.y, stacked: true },
        },
        plugins: {
          ...phaseBoxBase.plugins,
          tooltip: {
            ...phaseBoxBase.plugins.tooltip,
            callbacks: {
              label: (ctx) => {
                const idx = ctx.dataIndex;
                const d = phaseData[idx];
                if (!d.length) return '-';
                return `${phaseNames[idx]}: p25=${percentile(d, 25)?.toFixed(1)} med=${median(d)?.toFixed(1)} p75=${percentile(d, 75)?.toFixed(1)} min=${Math.min(...d).toFixed(1)} max=${Math.max(...d).toFixed(1)} std=${stddev(d).toFixed(1)}`;
              },
            },
          },
          legend: { display: false },
        },
      },
    };

    // ═══ NEW CHART 4: Duration Bucket Breakdown ═══
    const bucketWidth = 4;
    const bucketMap = {};
    okRuns.forEach((r) => {
      if (!Number.isFinite(r.audioDurationSec)) return;
      const b = Math.floor(r.audioDurationSec / bucketWidth) * bucketWidth;
      if (!bucketMap[b]) bucketMap[b] = { preprocess: [], encode: [], decode: [] };
      if (Number.isFinite(r.metrics?.preprocess_ms)) bucketMap[b].preprocess.push(r.metrics.preprocess_ms);
      if (Number.isFinite(r.metrics?.encode_ms)) bucketMap[b].encode.push(r.metrics.encode_ms);
      if (Number.isFinite(r.metrics?.decode_ms)) bucketMap[b].decode.push(r.metrics.decode_ms);
    });
    const bucketKeys = Object.keys(bucketMap).map(Number).sort((a, b) => a - b);
    const bucketLabels = bucketKeys.map((k) => `${k}-${k + bucketWidth}s`);
    const durationBucketBase = chartBase('Mean (ms)');
    const durationBucket = {
      type: 'bar',
      data: {
        labels: bucketLabels,
        datasets: [
          { label: 'Preprocess', data: bucketKeys.map((k) => mean(bucketMap[k].preprocess)), backgroundColor: 'rgba(217, 179, 122, 0.82)' },
          { label: 'Encode', data: bucketKeys.map((k) => mean(bucketMap[k].encode)), backgroundColor: 'rgba(124, 166, 220, 0.82)' },
          { label: 'Decode', data: bucketKeys.map((k) => mean(bucketMap[k].decode)), backgroundColor: 'rgba(121, 194, 159, 0.82)' },
        ],
      },
      options: {
        ...durationBucketBase,
        scales: {
          ...durationBucketBase.scales,
          x: { ...durationBucketBase.scales.x, stacked: true, ticks: { color: '#6b7a90', font: { family: 'JetBrains Mono', size: 9 } } },
          y: { ...durationBucketBase.scales.y, stacked: true },
        },
      },
    };

    // ═══ NEW CHART 5: RTF Histogram ═══
    const rtfValues = okRuns.map((r) => r.metrics?.rtf).filter(Number.isFinite);
    const rtfRange = rtfValues.length ? Math.max(...rtfValues) - Math.min(...rtfValues) : 0;
    const rtfBinWidth = rtfRange > 200 ? 20 : rtfRange > 100 ? 10 : 5;
    const rtfHist = histogram(rtfValues, rtfBinWidth);
    const rtfHistBase = chartBase('Count');
    const rtfHistogram = {
      type: 'bar',
      data: {
        labels: rtfHist.labels,
        datasets: [{ label: 'RTF distribution', data: rtfHist.counts, backgroundColor: 'rgba(124, 166, 220, 0.72)', borderRadius: 2 }],
      },
      options: {
        ...rtfHistBase,
        scales: {
          ...rtfHistBase.scales,
          x: { ...rtfHistBase.scales.x, title: { display: true, text: 'RTF (realtime factor)', color: '#b0bdd0', font: { family: 'Inter', size: 11, weight: '600' } }, ticks: { color: '#6b7a90', maxRotation: 45, font: { family: 'JetBrains Mono', size: 8 } } },
        },
        plugins: { ...rtfHistBase.plugins, legend: { display: false } },
      },
    };

    // ═══ NEW CHART 6: Similarity Distribution ═══
    const simPoints = okRuns
      .filter((r) => Number.isFinite(r.audioDurationSec) && Number.isFinite(r.similarityToFirst))
      .map((r) => ({
        x: r.audioDurationSec,
        y: r.similarityToFirst,
        sampleKey: r.sampleKey,
        exact: r.exactMatchToFirst,
      }));
    const simBase = chartBase('Similarity');
    const simDistribution = {
      type: 'scatter',
      data: {
        datasets: [
          { label: 'Exact match', backgroundColor: 'rgba(93, 186, 130, 0.8)', pointRadius: 5, data: simPoints.filter((p) => p.exact === true) },
          { label: 'Partial match', backgroundColor: 'rgba(224, 169, 78, 0.8)', pointRadius: 5, data: simPoints.filter((p) => p.exact === false && p.y >= 0.9) },
          { label: 'Low similarity', backgroundColor: 'rgba(224, 107, 127, 0.8)', pointRadius: 5, data: simPoints.filter((p) => p.exact === false && p.y < 0.9) },
        ],
      },
      options: {
        ...simBase,
        scales: {
          ...simBase.scales,
          x: { ...simBase.scales.x, title: { display: true, text: 'Audio duration (s)', color: '#b0bdd0', font: { family: 'Inter', size: 11, weight: '600' } } },
          y: { ...simBase.scales.y, min: 0, max: 1.05, title: { display: true, text: 'Similarity to first', color: '#b0bdd0', font: { family: 'Inter', size: 11, weight: '600' } } },
        },
        plugins: {
          ...simBase.plugins,
          tooltip: { ...simBase.plugins.tooltip, callbacks: { label: (ctx) => { const p = ctx.raw; return `${p.sampleKey}: sim=${(p.y * 100).toFixed(1)}% dur=${p.x.toFixed(2)}s`; } } },
        },
      },
    };

    // ═══ NEW CHART 7: Encode/Total Ratio vs Duration ═══
    const ratioPoints = okRuns
      .filter((r) => Number.isFinite(r.audioDurationSec) && Number.isFinite(r.metrics?.encode_ms) && Number.isFinite(r.metrics?.total_ms) && r.metrics.total_ms > 0)
      .map((r) => ({ x: r.audioDurationSec, encRatio: r.metrics.encode_ms / r.metrics.total_ms, decRatio: r.metrics.decode_ms / r.metrics.total_ms, sampleKey: r.sampleKey }));
    const ratioBase = chartBase('Fraction of total');
    const encodeRatio = {
      type: 'scatter',
      data: {
        datasets: [
          { label: 'Encode / Total', backgroundColor: 'rgba(124, 166, 220, 0.72)', pointRadius: 4, data: ratioPoints.map((p) => ({ x: p.x, y: p.encRatio })) },
          { label: 'Decode / Total', backgroundColor: 'rgba(121, 194, 159, 0.72)', pointRadius: 4, data: ratioPoints.map((p) => ({ x: p.x, y: p.decRatio })) },
        ],
      },
      options: {
        ...ratioBase,
        scales: {
          ...ratioBase.scales,
          x: { ...ratioBase.scales.x, title: { display: true, text: 'Audio duration (s)', color: '#b0bdd0', font: { family: 'Inter', size: 11, weight: '600' } } },
          y: { ...ratioBase.scales.y, min: 0, max: 1, title: { display: true, text: 'Fraction of total time', color: '#b0bdd0', font: { family: 'Inter', size: 11, weight: '600' } } },
        },
      },
    };

    // ═══ NEW CHART 8: Throughput Over Time ═══
    const timePoints = okRuns
      .filter((r) => r.startedAt && Number.isFinite(r.metrics?.total_ms))
      .map((r) => ({ x: new Date(r.startedAt).getTime(), y: r.metrics.total_ms, sampleKey: r.sampleKey }))
      .sort((a, b) => a.x - b.x);
    const t0 = timePoints.length ? timePoints[0].x : 0;
    const timeRelative = timePoints.map((p) => ({ x: ((p.x - t0) / 1000).toFixed(1), y: p.y, sampleKey: p.sampleKey }));
    const throughputBase = chartBase('Total (ms)');
    const throughput = {
      type: 'line',
      data: {
        labels: timeRelative.map((p) => `${p.x}s`),
        datasets: [{
          label: 'Total ms over time',
          borderColor: 'rgba(154, 170, 209, 0.9)',
          backgroundColor: 'rgba(154, 170, 209, 0.15)',
          tension: 0.3,
          pointRadius: 2,
          fill: true,
          data: timeRelative.map((p) => p.y),
        }],
      },
      options: {
        ...throughputBase,
        scales: {
          ...throughputBase.scales,
          x: { ...throughputBase.scales.x, title: { display: true, text: 'Elapsed (s)', color: '#b0bdd0', font: { family: 'Inter', size: 11, weight: '600' } }, ticks: { color: '#6b7a90', maxTicksLimit: 12, font: { family: 'JetBrains Mono', size: 8 } } },
        },
      },
    };

    // ═══ NEW CHART 9: Per-Sample Variance ═══
    const sampleGroups = {};
    okRuns.forEach((r) => {
      if (!Number.isFinite(r.metrics?.decode_ms)) return;
      if (!sampleGroups[r.sampleKey]) sampleGroups[r.sampleKey] = [];
      sampleGroups[r.sampleKey].push(r.metrics.decode_ms);
    });
    const sampleVariance = Object.entries(sampleGroups)
      .filter(([, vals]) => vals.length >= 2)
      .map(([key, vals]) => ({ key, std: stddev(vals), range: Math.max(...vals) - Math.min(...vals), count: vals.length }))
      .sort((a, b) => b.std - a.std)
      .slice(0, 25);
    const sampleVarBase = chartBase('Decode σ (ms)');
    const sampleVar = {
      type: 'bar',
      data: {
        labels: sampleVariance.map((s) => s.key),
        datasets: [{ label: 'Decode std', data: sampleVariance.map((s) => s.std), backgroundColor: 'rgba(224, 107, 127, 0.72)', borderRadius: 2 }],
      },
      options: {
        ...sampleVarBase,
        indexAxis: 'y',
        scales: {
          x: { ...sampleVarBase.scales.x, title: { display: true, text: 'Std deviation (ms)', color: '#b0bdd0', font: { family: 'Inter', size: 11, weight: '600' } } },
          y: { ...sampleVarBase.scales.y, title: { display: false }, ticks: { color: '#6b7a90', font: { family: 'JetBrains Mono', size: 9 } } },
        },
        plugins: {
          ...sampleVarBase.plugins,
          legend: { display: false },
          tooltip: { ...sampleVarBase.plugins.tooltip, callbacks: { label: (ctx) => { const s = sampleVariance[ctx.dataIndex]; return `${s.key}: σ=${s.std.toFixed(1)} range=${s.range.toFixed(1)} (${s.count} runs)`; } } },
        },
      },
    };

    // ═══ NEW CHART 10: Stacked Area — Phase Timeline ═══
    const areaRuns = okRuns.filter((r) => Number.isFinite(r.metrics?.preprocess_ms) && Number.isFinite(r.metrics?.encode_ms) && Number.isFinite(r.metrics?.decode_ms));
    const areaLabels = areaRuns.map((_, i) => `#${i + 1}`);
    const stackedAreaBase = chartBase('Time (ms)');
    const stackedArea = {
      type: 'line',
      data: {
        labels: areaLabels,
        datasets: [
          { label: 'Preprocess', data: areaRuns.map((r) => r.metrics.preprocess_ms), borderColor: 'rgba(217, 179, 122, 0.9)', backgroundColor: 'rgba(217, 179, 122, 0.3)', fill: true, tension: 0.3, pointRadius: 0 },
          { label: 'Encode', data: areaRuns.map((r) => r.metrics.encode_ms), borderColor: 'rgba(124, 166, 220, 0.9)', backgroundColor: 'rgba(124, 166, 220, 0.3)', fill: true, tension: 0.3, pointRadius: 0 },
          { label: 'Decode', data: areaRuns.map((r) => r.metrics.decode_ms), borderColor: 'rgba(121, 194, 159, 0.9)', backgroundColor: 'rgba(121, 194, 159, 0.3)', fill: true, tension: 0.3, pointRadius: 0 },
        ],
      },
      options: {
        ...stackedAreaBase,
        scales: {
          ...stackedAreaBase.scales,
          x: { ...stackedAreaBase.scales.x, title: { display: true, text: 'Run index', color: '#b0bdd0', font: { family: 'Inter', size: 11, weight: '600' } }, ticks: { maxTicksLimit: 15, color: '#6b7a90', font: { family: 'JetBrains Mono', size: 8 } } },
          y: { ...stackedAreaBase.scales.y, stacked: true },
        },
      },
    };

    return { encDec, rtfxRunOrder, rtfxDuration, durPre, trend, bottleneck, compareStages, durTotal, txDecode, phaseBox, durationBucket, rtfHistogram, simDistribution, encodeRatio, throughput, sampleVar, stackedArea };
  }, [okRuns, configStats]);

  const recentRuns = useMemo(() => [...runs].reverse().slice(0, 50), [runs]);

  const runErrors = runs.filter((r) => r.error).length;

  const currentSummary = useMemo(() => {
    const good = runs.filter((r) => !r.error && r.metrics && Number.isFinite(r.metrics.total_ms));
    const exactValues = good.map((r) => r.exactMatchToFirst).filter((v) => typeof v === 'boolean');
    const simValues = good.map((r) => r.similarityToFirst).filter(Number.isFinite);
    const preprocessMean = mean(good.map((r) => r.metrics?.preprocess_ms).filter(Number.isFinite));
    const encodeMean = mean(good.map((r) => r.metrics?.encode_ms).filter(Number.isFinite));
    const decodeMean = mean(good.map((r) => r.metrics?.decode_ms).filter(Number.isFinite));
    const tokenizeMean = mean(good.map((r) => r.metrics?.tokenize_ms).filter(Number.isFinite));
    const totalMean = mean(good.map((r) => r.metrics?.total_ms).filter(Number.isFinite));
    const rtfMedian = summarize(good.map((r) => r.metrics?.rtf).filter(Number.isFinite)).median;
    const encodeRtfxSummary = summarize(good.map((r) => calcRtfx(r.audioDurationSec, r.metrics?.encode_ms)).filter(Number.isFinite));
    const decodeRtfxSummary = summarize(good.map((r) => calcRtfx(r.audioDurationSec, r.metrics?.decode_ms)).filter(Number.isFinite));
    return {
      runCount: good.length,
      errorCount: runs.length - good.length,
      preprocessMean,
      encodeMean,
      decodeMean,
      tokenizeMean,
      totalMean,
      rtfMedian,
      encodeRtfxMedian: encodeRtfxSummary.median,
      decodeRtfxMedian: decodeRtfxSummary.median,
      encodeRtfxStd: encodeRtfxSummary.stddev,
      decodeRtfxStd: decodeRtfxSummary.stddev,
      exactRate: exactValues.length ? exactValues.filter(Boolean).length / exactValues.length : null,
      similarityMean: simValues.length ? mean(simValues) : null,
      preprocessShare: Number.isFinite(preprocessMean) && Number.isFinite(totalMean) && totalMean > 0 ? preprocessMean / totalMean : null,
      decodeShare: Number.isFinite(decodeMean) && Number.isFinite(totalMean) && totalMean > 0 ? decodeMean / totalMean : null,
    };
  }, [runs]);

  const hardwareSummary = useMemo(() => summarizeHardwareProfile(hardwareProfile), [hardwareProfile]);

  const compareA = useMemo(() => snapshots.find((s) => s.id === compareAId) || null, [snapshots, compareAId]);
  const compareB = useMemo(() => snapshots.find((s) => s.id === compareBId) || null, [snapshots, compareBId]);
  const selectedParamDefs = useMemo(() => SNAPSHOT_PARAM_OPTIONS.filter((item) => selectedCompareParams.includes(item.key)), [selectedCompareParams]);
  const selectedSnapshots = useMemo(() => {
    if (!selectedSnapshotIds.length) return [];
    const order = new Map(selectedSnapshotIds.map((id, idx) => [id, idx]));
    return snapshots
      .filter((snapshot) => order.has(snapshot.id))
      .sort((a, b) => order.get(a.id) - order.get(b.id));
  }, [snapshots, selectedSnapshotIds]);

  const snapshotMultiCompareCharts = useMemo(() => {
    if (!selectedSnapshots.length || !selectedParamDefs.length) return null;

    const palette = [
      'rgba(121, 194, 159, 0.95)',
      'rgba(124, 166, 220, 0.95)',
      'rgba(217, 179, 122, 0.95)',
      'rgba(154, 170, 209, 0.95)',
      'rgba(226, 151, 169, 0.95)',
      'rgba(148, 218, 217, 0.95)',
    ];

    const summaryDatasets = [];
    const byRepeatCharts = [];
    const labelsBySnapshot = selectedSnapshots.map((snapshot) => snapshot.label);

    selectedParamDefs.forEach((paramDef) => {
      const repeatSet = new Set();
      const runData = selectedSnapshots.map((snapshot) => {
        const goodRuns = (snapshot.runs || []).filter((run) => (
          !run?.error
          && Number.isFinite(run?.repeatIndex)
          && Number.isFinite(run?.metrics?.[paramDef.runField])
        ));
        goodRuns.forEach((run) => repeatSet.add(run.repeatIndex));
        return {
          snapshot,
          goodRuns,
          metricVals: goodRuns.map((run) => run.metrics[paramDef.runField]).filter(Number.isFinite),
        };
      });

      summaryDatasets.push({
        label: `${paramDef.label} mean (ms)`,
        data: runData.map((entry) => mean(entry.metricVals)),
        backgroundColor: paramDef.color,
      });

      const repeatIds = Array.from(repeatSet).sort((a, b) => a - b);
      if (!repeatIds.length) return;

      byRepeatCharts.push({
        key: paramDef.key,
        title: `${paramDef.label} Mean by Repeat`,
        config: {
          type: 'line',
          data: {
            labels: repeatIds.map((id) => `Run ${id}`),
            datasets: runData.map((entry, idx) => ({
              label: entry.snapshot.label,
              borderColor: palette[idx % palette.length],
              backgroundColor: palette[idx % palette.length].replace('0.95', '0.2'),
              tension: 0.3,
              pointRadius: 3,
              data: repeatIds.map((repeatId) => mean(
                entry.goodRuns
                  .filter((run) => run.repeatIndex === repeatId)
                  .map((run) => run.metrics[paramDef.runField])
                  .filter(Number.isFinite),
              )),
            })),
          },
          options: chartBase(`${paramDef.label} mean (ms)`),
        },
      });
    });

    if (!summaryDatasets.length && !byRepeatCharts.length) return null;

    const summary = {
      type: 'bar',
      data: {
        labels: labelsBySnapshot,
        datasets: summaryDatasets,
      },
      options: {
        ...chartBase('Milliseconds'),
        scales: {
          ...chartBase('Milliseconds').scales,
          x: {
            ...chartBase('Milliseconds').scales.x,
            ticks: {
              color: '#6b7a90',
              maxRotation: 25,
              minRotation: 25,
              font: { family: 'JetBrains Mono', size: 9 },
            },
          },
        },
      },
    };

    return { summary, byRepeatCharts };
  }, [selectedSnapshots, selectedParamDefs]);

  function defaultSnapshotLabel() {
    const now = new Date();
    const stamp = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
    return `${modelKey} | ${backend} | e:${encoderQuant} d:${decoderQuant} | preproc:${preprocessorBackend} | seed:${randomize ? String(randomSeed) : 'off'} | ${stamp}`;
  }

  function saveCurrentSnapshot() {
    if (!runs.length) {
      setBenchStatus('No runs to snapshot');
      return;
    }
    const label = snapshotName.trim() || defaultSnapshotLabel();
    const snapshot = {
      id: `snap-${Date.now()}`,
      createdAt: new Date().toISOString(),
      label,
      settings: {
        modelKey,
        backend,
        encoderQuant,
        decoderQuant,
        preprocessorBackend,
        preprocessor: PREPROCESSOR_MODEL,
        cpuThreads,
        datasetId,
        datasetConfig,
        datasetSplit,
        offset,
        sampleCount,
        repeatCount,
        warmups,
        randomize,
        randomSeed,
      },
      summary: currentSummary,
      hardwareProfile,
      hardwareSummary,
      runs: compactRunsForStorage(runs),
    };

    setSnapshots((prev) => [snapshot, ...prev].slice(0, MAX_SNAPSHOTS));
    setSnapshotName('');
    setBenchStatus(`Saved snapshot: ${label}`);
  }

  function deleteSnapshot(id) {
    setSnapshots((prev) => prev.filter((item) => item.id !== id));
  }

  function toggleSnapshotSelection(id) {
    setSelectedSnapshotIds((prev) => (
      prev.includes(id)
        ? prev.filter((item) => item !== id)
        : [...prev, id]
    ));
  }

  function toggleCompareParam(key) {
    setSelectedCompareParams((prev) => {
      if (prev.includes(key)) {
        if (prev.length === 1) return prev;
        return prev.filter((item) => item !== key);
      }
      return [...prev, key];
    });
  }

  // ═══ Pivot Table Config ═══
  const PIVOT_DIMENSIONS = [
    { key: 'model', label: 'Model', extract: (s) => s.settings?.modelKey || '-' },
    { key: 'backend', label: 'Backend', extract: (s) => s.settings?.backend || '-' },
    { key: 'quant', label: 'Quantization', extract: (s) => `e:${s.settings?.encoderQuant || '-'} d:${s.settings?.decoderQuant || '-'}` },
    { key: 'encoderQuant', label: 'Encoder Quant', extract: (s) => s.settings?.encoderQuant || '-' },
    { key: 'decoderQuant', label: 'Decoder Quant', extract: (s) => s.settings?.decoderQuant || '-' },
    { key: 'preprocessor', label: 'Preprocessor', extract: (s) => s.settings?.preprocessorBackend || '-' },
    { key: 'gpu', label: 'GPU', extract: (s) => s.hardwareSummary?.gpuModelLabel || s.hardwareSummary?.gpuLabel || '-' },
    { key: 'cpu', label: 'CPU', extract: (s) => s.hardwareSummary?.cpuLabel || '-' },
    { key: 'dataset', label: 'Dataset', extract: (s) => `${s.settings?.datasetId || '-'}/${s.settings?.datasetConfig || '-'}` },
  ];

  const PIVOT_METRICS = [
    { key: 'totalMean', label: 'Total (ms)', format: ms, lowerBetter: true },
    { key: 'preprocessMean', label: 'Preprocess (ms)', format: ms, lowerBetter: true },
    { key: 'encodeMean', label: 'Encode (ms)', format: ms, lowerBetter: true },
    { key: 'decodeMean', label: 'Decode (ms)', format: ms, lowerBetter: true },
    { key: 'tokenizeMean', label: 'Tokenize (ms)', format: ms, lowerBetter: true },
    { key: 'rtfMedian', label: 'RTF Median', format: (v) => Number.isFinite(v) ? `${v.toFixed(2)}×` : '-', lowerBetter: false },
    { key: 'encodeRtfxMedian', label: 'Enc RTFx', format: rtfx, lowerBetter: false },
    { key: 'decodeRtfxMedian', label: 'Dec RTFx', format: rtfx, lowerBetter: false },
    { key: 'exactRate', label: 'Exact %', format: pct, lowerBetter: false },
    { key: 'similarityMean', label: 'Sim %', format: pct, lowerBetter: false },
    { key: 'runCount', label: 'Runs', format: (v) => v ?? '-', lowerBetter: false },
  ];

  function togglePivotMetric(key) {
    setPivotMetrics((prev) => prev.includes(key) ? (prev.length > 1 ? prev.filter((k) => k !== key) : prev) : [...prev, key]);
  }

  function importJsonFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        const runs = data.runs || [];
        if (!runs.length) { setBenchStatus('No runs found in JSON'); return; }
        const good = runs.filter((r) => !r.error && r.metrics && Number.isFinite(r.metrics.total_ms));
        const preprocessMean = mean(good.map((r) => r.metrics?.preprocess_ms).filter(Number.isFinite));
        const encodeMean = mean(good.map((r) => r.metrics?.encode_ms).filter(Number.isFinite));
        const decodeMean = mean(good.map((r) => r.metrics?.decode_ms).filter(Number.isFinite));
        const tokenizeMean = mean(good.map((r) => r.metrics?.tokenize_ms).filter(Number.isFinite));
        const totalMean = mean(good.map((r) => r.metrics?.total_ms).filter(Number.isFinite));
        const rtfMedian = median(good.map((r) => r.metrics?.rtf).filter(Number.isFinite));
        const encRtfx = good.map((r) => calcRtfx(r.audioDurationSec, r.metrics?.encode_ms)).filter(Number.isFinite);
        const decRtfx = good.map((r) => calcRtfx(r.audioDurationSec, r.metrics?.decode_ms)).filter(Number.isFinite);
        const exactValues = good.map((r) => r.exactMatchToFirst).filter((v) => typeof v === 'boolean');
        const simValues = good.map((r) => r.similarityToFirst).filter(Number.isFinite);
        const summary = {
          runCount: good.length,
          errorCount: runs.length - good.length,
          preprocessMean,
          encodeMean,
          decodeMean,
          tokenizeMean,
          totalMean,
          rtfMedian,
          encodeRtfxMedian: median(encRtfx),
          decodeRtfxMedian: median(decRtfx),
          exactRate: exactValues.length ? exactValues.filter(Boolean).length / exactValues.length : null,
          similarityMean: simValues.length ? mean(simValues) : null,
        };
        const snapshot = {
          id: `import-${Date.now()}`,
          createdAt: data.generatedAt || new Date().toISOString(),
          label: file.name.replace(/\.json$/i, ''),
          settings: data.settings || {},
          summary,
          hardwareProfile: data.hardwareProfile || null,
          hardwareSummary: data.hardwareSummary || {},
          runs: runs.slice(0, 200),
          imported: true,
        };
        setSnapshots((prev) => [snapshot, ...prev].slice(0, MAX_SNAPSHOTS));
        setBenchStatus(`Imported ${good.length} runs from ${file.name}`);
      } catch (err) {
        setBenchStatus(`Import failed: ${err.message}`);
      }
    };
    reader.readAsText(file);
    event.target.value = '';
  }

  const pivotData = useMemo(() => {
    if (!snapshots.length) return null;
    const dim = PIVOT_DIMENSIONS.find((d) => d.key === pivotGroupBy) || PIVOT_DIMENSIONS[0];
    const groups = {};
    snapshots.forEach((s) => {
      const key = dim.extract(s);
      if (!groups[key]) groups[key] = [];
      groups[key].push(s);
    });
    const rows = Object.entries(groups).map(([groupKey, items]) => {
      const row = { groupKey, count: items.length, snapshots: items };
      PIVOT_METRICS.forEach((m) => {
        const values = items.map((s) => s.summary?.[m.key]).filter(Number.isFinite);
        row[m.key] = values.length ? mean(values) : null;
        row[`${m.key}_min`] = values.length ? Math.min(...values) : null;
        row[`${m.key}_max`] = values.length ? Math.max(...values) : null;
        row[`${m.key}_all`] = values;
      });
      return row;
    });
    return { dim, rows, groups };
  }, [snapshots, pivotGroupBy]);

  const pivotChartConfig = useMemo(() => {
    if (!pivotData || !pivotData.rows.length) return null;
    const labels = pivotData.rows.map((r) => r.groupKey);
    const activeMetrics = PIVOT_METRICS.filter((m) => pivotMetrics.includes(m.key));
    const colors = ['rgba(124, 166, 220, 0.82)', 'rgba(121, 194, 159, 0.82)', 'rgba(217, 179, 122, 0.82)', 'rgba(155, 159, 223, 0.82)', 'rgba(224, 107, 127, 0.82)', 'rgba(93, 186, 130, 0.82)'];
    const datasets = activeMetrics.map((m, i) => ({
      label: m.label,
      data: pivotData.rows.map((r) => r[m.key]),
      backgroundColor: colors[i % colors.length],
    }));
    const base = chartBase('Value');
    return {
      type: 'bar',
      data: { labels, datasets },
      options: {
        ...base,
        scales: {
          ...base.scales,
          x: { ...base.scales.x, ticks: { color: '#6b7a90', maxRotation: 30, font: { family: 'JetBrains Mono', size: 9 } } },
        },
      },
    };
  }, [pivotData, pivotMetrics]);

  function exportJson() {
    if (!runs.length) return;
    const payload = {
      generatedAt: new Date().toISOString(),
      settings: { modelKey, backend, encoderQuant, decoderQuant, preprocessorBackend, preprocessor: PREPROCESSOR_MODEL, cpuThreads, datasetId, datasetConfig, datasetSplit, offset, sampleCount, repeatCount, warmups, randomize, randomSeed },
      hardwareProfile,
      hardwareSummary,
      runs,
    };
    saveText(`parakeet-benchmark-${Date.now()}.json`, JSON.stringify(payload, null, 2), 'application/json;charset=utf-8');
  }

  function exportCsv() {
    if (!runs.length) return;
    const csv = toCsv(runs.map(flattenRunRecord), RUN_CSV_COLUMNS);
    saveText(`parakeet-benchmark-${Date.now()}.csv`, csv, 'text/csv;charset=utf-8');
  }

  const progressPct = progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0;

  return (
    <div className="app-shell">
      {/* ═══ HEADER ═══ */}
      <header className="header">
        <div className="header-left">
          <h1>Parakeet.js Benchmark</h1>
          <div className="header-pills">
            <span className="pill">
              <span className={`pill-dot ${hardwareSummary.webgpuLabel === 'Yes' ? 'green' : 'red'}`} />
              {hardwareSummary.gpuModelLabel !== '-' ? hardwareSummary.gpuModelLabel : 'No GPU'}
            </span>
            <span className="pill">
              <span className={`pill-dot ${isModelReady ? 'green' : isLoadingModel ? 'orange' : 'muted'}`} />
              {isModelReady ? 'Model ready' : isLoadingModel ? 'Loading...' : 'No model'}
            </span>
            <span className="pill">
              <span className={`pill-dot ${isRunning ? 'orange' : okRuns.length > 0 ? 'green' : 'muted'}`} />
              {isRunning ? `Running ${progress.current}/${progress.total}` : `${okRuns.length} runs`}
            </span>
          </div>
        </div>
        <div className="header-actions">
          <button className="theme-toggle" onClick={toggleTheme} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
        </div>
      </header>

      {/* ═══ TABS ═══ */}
      <nav className="tabs">
        {[
          { id: 'benchmark', label: '⚙ Benchmark' },
          { id: 'overview', label: 'Overview' },
          { id: 'charts', label: 'Charts', count: chartConfigs ? 17 : 0 },
          { id: 'compare', label: 'Compare', count: snapshots.length },
          { id: 'data', label: 'Data', count: runs.length },
        ].map((t) => (
          <button
            key={t.id}
            className={`tab ${activeTab === t.id ? 'active' : ''}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
            {t.count > 0 ? <span className="tab-count">{t.count}</span> : null}
          </button>
        ))}
      </nav>

      {/* ═══ TAB CONTENT ═══ */}
      <div className="tab-content">
        {/* ── BENCHMARK TAB ── */}
        {activeTab === 'benchmark' && (
          <div className="fade-in">
            <div className="config-panel">
              {/* ── Model Card ── */}
              <div className="config-card">
                <h3>Model &amp; Runtime</h3>
                <div className="form-gap">
                  <label>Model<select value={modelKey} onChange={(e) => setModelKey(e.target.value)} disabled={isLoadingModel || isRunning}>{MODEL_OPTIONS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}</select></label>
                  <label>Backend<select value={backend} onChange={(e) => setBackend(e.target.value)} disabled={isLoadingModel || isRunning}>{BACKENDS.map((v) => <option key={v} value={v}>{v}</option>)}</select></label>
                  <div className="row-2">
                    <label>Encoder<select value={encoderQuant} onChange={(e) => setEncoderQuant(e.target.value)} disabled={isLoadingModel || isRunning}>{encoderQuantOptions.map((v) => <option key={v} value={v}>{v}</option>)}</select></label>
                    <label>Decoder<select value={decoderQuant} onChange={(e) => setDecoderQuant(e.target.value)} disabled={isLoadingModel || isRunning}>{decoderQuantOptions.map((v) => <option key={v} value={v}>{v}</option>)}</select></label>
                  </div>
                  <label>Preprocessor<select value={preprocessorBackend} onChange={(e) => setPreprocessorBackend(e.target.value)} disabled={isLoadingModel || isRunning}><option value="js">JS (meljs)</option><option value="onnx">ONNX (nemo128.onnx)</option></select></label>
                  <div className="row-2">
                    <label>CPU threads<input type="number" min="1" max="64" value={cpuThreads} onChange={(e) => setCpuThreads(clamp(e.target.value, cpuThreads, 1, 64))} disabled={isLoadingModel || isRunning} /></label>
                    <label className="check"><input type="checkbox" checked={enableProfiling} onChange={(e) => setEnableProfiling(e.target.checked)} disabled={isLoadingModel || isRunning} />Profiling</label>
                  </div>
                  <button className="btn btn-primary" onClick={loadModel} disabled={isLoadingModel || isRunning || isModelReady}>{isLoadingModel ? 'Loading...' : isModelReady ? 'Model ready ✓' : 'Load model'}</button>
                  <p className="status-text">{modelStatus}</p>
                  {modelProgress ? <p className="subtle">{modelProgress}</p> : null}
                  {resolvedModelInfo ? <p className="subtle">{resolvedModelInfo}</p> : null}
                </div>
              </div>

              {/* ── Dataset Card ── */}
              <div className="config-card">
                <h3>Dataset</h3>
                <div className="form-gap">
                  <label>Source<input value={datasetId} onChange={(e) => setDatasetId(e.target.value)} disabled={isLoadingDataset || isRunning} placeholder="HuggingFace dataset ID" /></label>
                  <div className="row-2">
                    <label>Config<select value={datasetConfig} onChange={(e) => refreshDatasetMeta(datasetId, e.target.value, datasetSplit)} disabled={isLoadingDataset || isRunning}>{configs.map((c) => <option key={c} value={c}>{c}</option>)}</select></label>
                    <label>Split<select value={datasetSplit} onChange={(e) => setDatasetSplit(e.target.value)} disabled={isLoadingDataset || isRunning}>{splits.map((s) => <option key={s} value={s}>{s}</option>)}</select></label>
                  </div>
                  <div className="row-2">
                    <label>Offset<input type="number" min="0" value={offset} onChange={(e) => setOffset(clamp(e.target.value, offset, 0, 1_000_000))} disabled={isRunning} /></label>
                    <label>Samples<input type="number" min="1" value={sampleCount} onChange={(e) => setSampleCount(clamp(e.target.value, sampleCount, 1, MAX_SAMPLE_COUNT))} disabled={isRunning} /></label>
                  </div>
                  <div className="btn-group">
                    <button className="btn" onClick={() => refreshDatasetMeta(datasetId, datasetConfig, datasetSplit, true)} disabled={isLoadingDataset || isRunning}>{isLoadingDataset ? 'Refreshing...' : 'Refresh'}</button>
                    <button className="btn" onClick={clearDatasetCache} disabled={isLoadingDataset || isRunning}>Clear cache</button>
                    <button className="btn" onClick={prepareSampleRows} disabled={isRunning || isLoadingDataset}>Preview</button>
                  </div>
                  <p className="status-text">{datasetStatus}</p>
                  {splitCounts?.[datasetSplit]?.num_examples ? <p className="subtle">Rows: {splitCounts[datasetSplit].num_examples}</p> : null}
                </div>
              </div>

              {/* ── Run Plan Card ── */}
              <div className="config-card">
                <h3>Benchmark Run</h3>
                <div className="form-gap">
                  <div className="row-3">
                    <label>Repeats<input type="number" min="1" max="100" value={repeatCount} onChange={(e) => setRepeatCount(clamp(e.target.value, repeatCount, 1, 100))} disabled={isRunning} /></label>
                    <label>Warmups<input type="number" min="0" max="10" value={warmups} onChange={(e) => setWarmups(clamp(e.target.value, warmups, 0, 10))} disabled={isRunning} /></label>
                    <label>Seed<input value={randomSeed} onChange={(e) => setRandomSeed(e.target.value)} placeholder="42" disabled={isRunning || !randomize} /></label>
                  </div>
                  <label className="check"><input type="checkbox" checked={randomize} onChange={(e) => setRandomize(e.target.checked)} disabled={isRunning} />Randomize samples</label>
                  <div className="btn-group">
                    <button className="btn btn-primary" style={{ flex: 1 }} onClick={runBenchmark} disabled={isRunning || isLoadingModel || !isModelReady}>{isRunning ? 'Running...' : 'Start benchmark'}</button>
                    <button className="btn btn-danger" onClick={stopRun} disabled={!isRunning}>Stop</button>
                  </div>
                  <div className="btn-group">
                    <button className="btn btn-sm" onClick={exportJson} disabled={!runs.length || isRunning}>JSON</button>
                    <button className="btn btn-sm" onClick={exportCsv} disabled={!runs.length || isRunning}>CSV</button>
                    <button className="btn btn-sm" onClick={() => setRuns([])} disabled={!runs.length || isRunning}>Clear</button>
                  </div>
                  <label>Snapshot label<input value={snapshotName} onChange={(e) => setSnapshotName(e.target.value)} placeholder="auto-generated if empty" disabled={isRunning} /></label>
                  <button className="btn" onClick={saveCurrentSnapshot} disabled={!runs.length || isRunning}>Save snapshot</button>
                  <p className="status-text">{benchStatus}</p>
                </div>
              </div>
            </div>

            {/* Progress bar inside benchmark tab */}
            {(isRunning || progress.total > 0) && (
              <div className="progress-bar-wrap fade-in">
                <div className="progress-bar-track">
                  <div className="progress-bar-fill" style={{ width: `${progressPct}%` }} />
                </div>
                <div className="progress-info">
                  <span>{progress.stage}</span>
                  <span className="meta-mono">{progress.current}/{progress.total} ({progressPct}%)</span>
                </div>
              </div>
            )}

            {/* Hardware info in benchmark tab */}
            <section className="table-panel" style={{ marginTop: 12 }}>
              <div className="table-header">
                <h3>Hardware Profile</h3>
                <button className="btn btn-sm" onClick={refreshHardwareProfile} disabled={isLoadingHardware || isRunning}>{isLoadingHardware ? 'Refreshing...' : 'Refresh'}</button>
              </div>
              <div style={{ padding: '12px 14px' }}>
                <div className="kv-grid">
                  <div className="kv-row"><span>CPU</span><strong>{hardwareSummary.cpuLabel}</strong></div>
                  <div className="kv-row"><span>GPU model</span><strong>{hardwareSummary.gpuModelLabel}</strong></div>
                  <div className="kv-row"><span>GPU (raw)</span><strong>{hardwareSummary.gpuLabel}</strong></div>
                  <div className="kv-row"><span>WebGPU</span><strong>{hardwareSummary.webgpuLabel}</strong></div>
                  <div className="kv-row"><span>System RAM</span><strong>{hardwareSummary.systemMemoryLabel}</strong></div>
                </div>
                {hardwareProfile?.webgpu?.features?.length ? <p className="subtle" style={{ marginTop: 8 }}>WebGPU features: {hardwareProfile.webgpu.features.slice(0, 12).join(', ')}{hardwareProfile.webgpu.features.length > 12 ? ' ...' : ''}</p> : null}
              </div>
            </section>

            {/* Prepared samples preview */}
            <section className="table-panel" style={{ marginTop: 12 }}>
              <div className="table-header">
                <h3>Prepared Samples ({preparedSamples.length})</h3>
                <button className="btn btn-sm" onClick={() => setShowPreparedSamples((v) => !v)}>{showPreparedSamples ? 'Collapse' : 'Expand'}</button>
              </div>
              {showPreparedSamples ? (
                <div className="table-wrap">
                  {preparedSamples.length ? (
                    <table>
                      <thead><tr><th>Sample</th><th>Speaker</th><th>Gender</th><th>Speed</th><th>Volume</th><th>Reference</th></tr></thead>
                      <tbody>
                        {preparedSamples.map((s) => (
                          <tr key={`${s.rowIndex}-${s.audioUrl}`}><td>{datasetSplit}:{s.rowIndex}</td><td>{s.speaker || '-'}</td><td>{s.gender || '-'}</td><td>{Number.isFinite(s.speed) ? s.speed.toFixed(2) : '-'}</td><td>{Number.isFinite(s.volume) ? s.volume.toFixed(2) : '-'}</td><td className="text-cell">{s.referenceText || '-'}</td></tr>
                        ))}
                      </tbody>
                    </table>
                  ) : <div className="empty-row">No prepared samples yet. Click Preview above.</div>}
                </div>
              ) : <div className="empty-row">Click Expand to inspect sampled rows.</div>}
            </section>
          </div>
        )}

        {/* ── OVERVIEW TAB ── */}
        {activeTab === 'overview' && (
          <div className="fade-in">
            <div className="kpi-row">
              <div className="kpi-card teal"><div className="kpi-label">Total</div><div className="kpi-value">{ms(metrics.total.mean)}</div><div className="kpi-sub">p90 {ms(metrics.total.p90)}</div></div>
              <div className="kpi-card green"><div className="kpi-label">Preprocess</div><div className="kpi-value">{ms(metrics.preprocess.mean)}</div><div className="kpi-sub">share {pct(Number.isFinite(metrics.preprocess.mean) && Number.isFinite(metrics.total.mean) && metrics.total.mean > 0 ? metrics.preprocess.mean / metrics.total.mean : null)}</div></div>
              <div className="kpi-card orange"><div className="kpi-label">Decode</div><div className="kpi-value">{ms(metrics.decode.mean)}</div><div className="kpi-sub">std {ms(metrics.decode.stddev)}</div></div>
              <div className="kpi-card teal"><div className="kpi-label">Encode</div><div className="kpi-value">{ms(metrics.encode.mean)}</div><div className="kpi-sub">Tokenize {ms(metrics.tokenize.mean)}</div></div>
              <div className="kpi-card orange"><div className="kpi-label">Repeatability</div><div className="kpi-value">{pct(repeatability.exactRate)}</div><div className="kpi-sub">sim {pct(repeatability.similarityMean)}</div></div>
              <div className="kpi-card purple"><div className="kpi-label">RTF Median</div><div className="kpi-value">{Number.isFinite(metrics.rtf.median) ? `${metrics.rtf.median.toFixed(2)}×` : '-'}</div><div className="kpi-sub">avg dur {Number.isFinite(mean(okRuns.map((r) => r.audioDurationSec).filter(Number.isFinite))) ? mean(okRuns.map((r) => r.audioDurationSec).filter(Number.isFinite)).toFixed(2) : '-'} s</div></div>
              <div className="kpi-card teal"><div className="kpi-label">Encoder RTFx</div><div className="kpi-value">{rtfx(metrics.encodeRtfx.median)}</div><div className="kpi-sub">std {rtfx(metrics.encodeRtfx.stddev)}</div></div>
              <div className="kpi-card green"><div className="kpi-label">Decoder RTFx</div><div className="kpi-value">{rtfx(metrics.decodeRtfx.median)}</div><div className="kpi-sub">std {rtfx(metrics.decodeRtfx.stddev)}</div></div>
            </div>

            <section className="table-panel">
              <div className="table-header"><h3>Config Bottleneck</h3></div>
              <div className="table-wrap">
                {configStats.length ? (
                  <table>
                    <thead><tr><th>Config</th><th>Runs</th><th>Preproc</th><th>Encode</th><th>Decode</th><th>Total</th><th>Preproc %</th><th>Decode %</th></tr></thead>
                    <tbody>
                      {configStats.map((c) => (
                        <tr key={c.key}><td>{c.key}</td><td>{c.runs}</td><td>{ms(c.preprocessMean)}</td><td>{ms(c.encodeMean)}</td><td>{ms(c.decodeMean)}</td><td>{ms(c.totalMean)}</td><td>{pct(c.preprocessShare)}</td><td>{pct(c.decodeShare)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                ) : <div className="empty-row">Run at least one benchmark batch.</div>}
              </div>
            </section>

            <section className="table-panel">
              <div className="table-header"><h3>Per-Sample Repeatability</h3></div>
              <div className="table-wrap">
                {sampleStats.length ? (
                  <table>
                    <thead><tr><th>Sample</th><th>Runs</th><th>Unique</th><th>Exact</th><th>Sim</th><th>Preproc</th><th>Encode</th><th>Decode</th><th>Decode σ</th><th>Enc RTFx</th><th>Dec RTFx</th><th>Total</th></tr></thead>
                    <tbody>
                      {sampleStats.map((s) => (
                        <tr key={s.sampleKey}><td>{s.sampleKey}</td><td>{s.runs}</td><td>{s.uniqueOutputs}</td><td>{pct(s.exactRate)}</td><td>{pct(s.similarity)}</td><td>{ms(s.preprocessMean)}</td><td>{ms(s.encodeMean)}</td><td>{ms(s.decodeMean)}</td><td>{ms(s.decodeStd)}</td><td>{rtfx(s.encodeRtfxMean)}</td><td>{rtfx(s.decodeRtfxMean)}</td><td>{ms(s.totalMean)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                ) : <div className="empty-row">No sample stats yet.</div>}
              </div>
            </section>
          </div>
        )}

        {/* ── CHARTS TAB ── */}
        {activeTab === 'charts' && (
          <div className="fade-in">
            {chartConfigs ? (
              <>
                <h3 className="section-title">Timing & Scaling</h3>
                <div className="chart-grid">
                  <ChartCard title="Encoder vs Decoder" badge="scatter" config={chartConfigs.encDec} />
                  <ChartCard title="Duration vs Total + Fit" badge="regression" config={chartConfigs.durTotal} />
                  <ChartCard title="Duration vs Preprocess" badge="scatter" config={chartConfigs.durPre} />
                  <ChartCard title="Transcript Length vs Decode" badge="scatter" config={chartConfigs.txDecode} />
                  <ChartCard title="Phase Timing Ranges" badge="box" config={chartConfigs.phaseBox} />
                  <ChartCard title="Duration Bucket Breakdown" badge="stacked" config={chartConfigs.durationBucket} />
                </div>

                <h3 className="section-title" style={{ marginTop: 20 }}>Performance & RTF</h3>
                <div className="chart-grid">
                  <ChartCard title="Run Order vs RTFx" badge="scatter" config={chartConfigs.rtfxRunOrder} />
                  <ChartCard title="Duration vs RTFx" badge="scatter" config={chartConfigs.rtfxDuration} />
                  <ChartCard title="RTF Distribution" badge="histogram" config={chartConfigs.rtfHistogram} />
                  <ChartCard title="Encode/Decode Ratio" badge="scatter" config={chartConfigs.encodeRatio} />
                  <ChartCard title="Stage Bottleneck" badge="doughnut" config={chartConfigs.bottleneck} />
                  <ChartCard title="Config Compare" badge="stacked" config={chartConfigs.compareStages} />
                </div>

                <h3 className="section-title" style={{ marginTop: 20 }}>Stability & Consistency</h3>
                <div className="chart-grid">
                  <ChartCard title="Repeat Trend" badge="line" config={chartConfigs.trend} />
                  <ChartCard title="Throughput Over Time" badge="line" config={chartConfigs.throughput} />
                  <ChartCard title="Similarity Distribution" badge="scatter" config={chartConfigs.simDistribution} />
                  <ChartCard title="Per-Sample Variance" badge="bar" config={chartConfigs.sampleVar} />
                  <ChartCard title="Phase Timeline" badge="area" config={chartConfigs.stackedArea} />
                </div>
              </>
            ) : (
              <div className="empty-state"><p>Run a benchmark batch to generate charts.</p></div>
            )}
          </div>
        )}

        {/* ── COMPARE TAB ── */}
        {activeTab === 'compare' && (
          <div className="fade-in">
            {/* ═══ PIVOT TABLE ═══ */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <h3 className="section-title" style={{ margin: 0 }}>Pivot Analysis</h3>
              <div className="btn-group">
                <input ref={fileInputRef} type="file" accept=".json" style={{ display: 'none' }} onChange={importJsonFile} />
                <button className="btn btn-sm" onClick={() => fileInputRef.current?.click()}>Import JSON</button>
              </div>
            </div>
            <div className="pivot-controls">
              <label className="pivot-group-label">
                Group by
                <select value={pivotGroupBy} onChange={(e) => setPivotGroupBy(e.target.value)}>
                  {PIVOT_DIMENSIONS.map((d) => <option key={d.key} value={d.key}>{d.label}</option>)}
                </select>
              </label>
              <div className="pivot-metric-chips">
                {PIVOT_METRICS.map((m) => (
                  <button key={m.key} className={`param-chip ${pivotMetrics.includes(m.key) ? 'active' : ''}`} onClick={() => togglePivotMetric(m.key)}>{m.label}</button>
                ))}
              </div>
            </div>

            {pivotData && pivotData.rows.length > 0 ? (
              <>
                <section className="table-panel" style={{ marginBottom: 16 }}>
                  <div className="table-header">
                    <h3>Grouped by: {pivotData.dim.label} ({pivotData.rows.length} groups, {snapshots.length} snapshots)</h3>
                  </div>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th style={{ textAlign: 'left' }}>{pivotData.dim.label}</th>
                          <th>Snaps</th>
                          {PIVOT_METRICS.filter((m) => pivotMetrics.includes(m.key)).map((m) => (
                            <th key={m.key}>{m.label}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {pivotData.rows.map((row) => {
                          const activeM = PIVOT_METRICS.filter((m) => pivotMetrics.includes(m.key));
                          return (
                            <tr key={row.groupKey}>
                              <td style={{ textAlign: 'left', fontWeight: 600 }}>{row.groupKey}</td>
                              <td>{row.count}</td>
                              {activeM.map((m) => {
                                const allValues = pivotData.rows.map((r) => r[m.key]).filter(Number.isFinite);
                                const best = m.lowerBetter ? Math.min(...allValues) : Math.max(...allValues);
                                const isBest = Number.isFinite(row[m.key]) && row[m.key] === best && allValues.length > 1;
                                return (
                                  <td key={m.key} style={isBest ? { color: 'var(--green)', fontWeight: 700 } : {}}>
                                    {m.format(row[m.key])}
                                    {row.count > 1 && Number.isFinite(row[`${m.key}_min`]) && Number.isFinite(row[`${m.key}_max`]) && row[`${m.key}_min`] !== row[`${m.key}_max`] ? (
                                      <span style={{ fontSize: '9px', opacity: 0.6, display: 'block' }}>{m.format(row[`${m.key}_min`])} – {m.format(row[`${m.key}_max`])}</span>
                                    ) : null}
                                  </td>
                                );
                              })}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </section>
                {pivotChartConfig && <div style={{ marginBottom: 24 }}><ChartCard title={`${pivotData.dim.label} Comparison`} badge="grouped" config={pivotChartConfig} /></div>}
              </>
            ) : (
              <div className="empty-state" style={{ marginBottom: 24 }}>
                <p>Import benchmark JSON files or save snapshots to build pivot comparisons.</p>
                <p style={{ fontSize: 11, marginTop: 6 }}>Use "Import JSON" to load files from the metrics/ folder, or "Save snapshot" in the Benchmark tab.</p>
              </div>
            )}

            <h3 className="section-title" style={{ marginTop: 16 }}>A/B Snapshot Comparison</h3>
            <div className="compare-selectors">
              <label>Snapshot A<select value={compareAId} onChange={(e) => setCompareAId(e.target.value)} disabled={!snapshots.length}>{snapshots.length ? snapshots.map((s) => <option key={s.id} value={s.id}>{s.label}</option>) : <option value="">No snapshots</option>}</select></label>
              <label>Snapshot B<select value={compareBId} onChange={(e) => setCompareBId(e.target.value)} disabled={!snapshots.length}>{snapshots.length ? snapshots.map((s) => <option key={s.id} value={s.id}>{s.label}</option>) : <option value="">No snapshots</option>}</select></label>
            </div>

            <section className="table-panel">
              <div className="table-wrap">
                {compareA && compareB ? (
                  <table>
                    <thead><tr><th>Metric</th><th>A</th><th>B</th><th>Δ B vs A</th></tr></thead>
                    <tbody>
                      <tr><td>Model</td><td>{compareA.settings?.modelKey || '-'}</td><td>{compareB.settings?.modelKey || '-'}</td><td>-</td></tr>
                      <tr><td>Backend</td><td>{compareA.settings?.backend || '-'}</td><td>{compareB.settings?.backend || '-'}</td><td>-</td></tr>
                      <tr><td>Quant</td><td>e:{compareA.settings?.encoderQuant || '-'} d:{compareA.settings?.decoderQuant || '-'}</td><td>e:{compareB.settings?.encoderQuant || '-'} d:{compareB.settings?.decoderQuant || '-'}</td><td>-</td></tr>
                      <tr><td>Preprocessor</td><td>{compareA.settings?.preprocessorBackend || '-'}</td><td>{compareB.settings?.preprocessorBackend || '-'}</td><td>-</td></tr>
                      <tr><td>CPU</td><td>{compareA.hardwareSummary?.cpuLabel || '-'}</td><td>{compareB.hardwareSummary?.cpuLabel || '-'}</td><td>-</td></tr>
                      <tr><td>GPU</td><td>{compareA.hardwareSummary?.gpuModelLabel || '-'}</td><td>{compareB.hardwareSummary?.gpuModelLabel || '-'}</td><td>-</td></tr>
                      <tr><td>Seed</td><td>{compareA.settings?.randomize ? (compareA.settings?.randomSeed ?? 'random') : 'off'}</td><td>{compareB.settings?.randomize ? (compareB.settings?.randomSeed ?? 'random') : 'off'}</td><td>-</td></tr>
                      <tr className={`row-selectable ${selectedCompareParams.includes('total') ? 'row-selected' : ''}`} onClick={() => toggleCompareParam('total')}><td>Total mean</td><td>{ms(compareA.summary?.totalMean)}</td><td>{ms(compareB.summary?.totalMean)}</td><td>{deltaPercent(compareA.summary?.totalMean, compareB.summary?.totalMean, true)}</td></tr>
                      <tr className={`row-selectable ${selectedCompareParams.includes('preprocess') ? 'row-selected' : ''}`} onClick={() => toggleCompareParam('preprocess')}><td>Preprocess mean</td><td>{ms(compareA.summary?.preprocessMean)}</td><td>{ms(compareB.summary?.preprocessMean)}</td><td>{deltaPercent(compareA.summary?.preprocessMean, compareB.summary?.preprocessMean, true)}</td></tr>
                      <tr className={`row-selectable ${selectedCompareParams.includes('encode') ? 'row-selected' : ''}`} onClick={() => toggleCompareParam('encode')}><td>Encode mean</td><td>{ms(compareA.summary?.encodeMean)}</td><td>{ms(compareB.summary?.encodeMean)}</td><td>{deltaPercent(compareA.summary?.encodeMean, compareB.summary?.encodeMean, true)}</td></tr>
                      <tr className={`row-selectable ${selectedCompareParams.includes('decode') ? 'row-selected' : ''}`} onClick={() => toggleCompareParam('decode')}><td>Decode mean</td><td>{ms(compareA.summary?.decodeMean)}</td><td>{ms(compareB.summary?.decodeMean)}</td><td>{deltaPercent(compareA.summary?.decodeMean, compareB.summary?.decodeMean, true)}</td></tr>
                      <tr><td>RTF median</td><td>{Number.isFinite(compareA.summary?.rtfMedian) ? `${compareA.summary.rtfMedian.toFixed(2)}×` : '-'}</td><td>{Number.isFinite(compareB.summary?.rtfMedian) ? `${compareB.summary.rtfMedian.toFixed(2)}×` : '-'}</td><td>{deltaPercent(compareA.summary?.rtfMedian, compareB.summary?.rtfMedian, false)}</td></tr>
                      <tr><td>Encoder RTFx</td><td>{rtfx(compareA.summary?.encodeRtfxMedian)}</td><td>{rtfx(compareB.summary?.encodeRtfxMedian)}</td><td>{deltaPercent(compareA.summary?.encodeRtfxMedian, compareB.summary?.encodeRtfxMedian, false)}</td></tr>
                      <tr><td>Decoder RTFx</td><td>{rtfx(compareA.summary?.decodeRtfxMedian)}</td><td>{rtfx(compareB.summary?.decodeRtfxMedian)}</td><td>{deltaPercent(compareA.summary?.decodeRtfxMedian, compareB.summary?.decodeRtfxMedian, false)}</td></tr>
                      <tr><td>Exact repeat</td><td>{pct(compareA.summary?.exactRate)}</td><td>{pct(compareB.summary?.exactRate)}</td><td>{deltaPercent(compareA.summary?.exactRate, compareB.summary?.exactRate, false)}</td></tr>
                      <tr><td>Similarity</td><td>{pct(compareA.summary?.similarityMean)}</td><td>{pct(compareB.summary?.similarityMean)}</td><td>{deltaPercent(compareA.summary?.similarityMean, compareB.summary?.similarityMean, false)}</td></tr>
                      <tr><td>Runs</td><td>{compareA.summary?.runCount ?? '-'}</td><td>{compareB.summary?.runCount ?? '-'}</td><td>-</td></tr>
                    </tbody>
                  </table>
                ) : <div className="empty-row">Save snapshots, then select A and B above to compare.</div>}
              </div>
            </section>

            <h3 className="section-title" style={{ marginTop: 24 }}>Multi-Snapshot Comparison</h3>
            <div className="param-chips">
              {SNAPSHOT_PARAM_OPTIONS.map((opt) => (
                <button key={opt.key} className={`param-chip ${selectedCompareParams.includes(opt.key) ? 'active' : ''}`} onClick={() => toggleCompareParam(opt.key)}>{opt.label}</button>
              ))}
            </div>
            {snapshotMultiCompareCharts ? (
              <div className="chart-grid">
                <ChartCard title="Parameter Means" badge="bar" config={snapshotMultiCompareCharts.summary} />
                {snapshotMultiCompareCharts.byRepeatCharts.map((item) => (
                  <ChartCard key={item.key} title={item.title} badge="line" config={item.config} />
                ))}
              </div>
            ) : <div className="empty-state"><p>Select snapshots from the table below, then click parameter chips above to compare.</p></div>}

            <section className="table-panel" style={{ marginTop: 16 }}>
              <div className="table-header"><h3>Stored Snapshots ({snapshots.length})</h3></div>
              <div className="table-wrap">
                {snapshots.length ? (
                  <table>
                    <thead><tr><th>✓</th><th>Label</th><th>Model</th><th>Backend</th><th>Quant</th><th>Preproc</th><th>GPU</th><th>Seed</th><th>Total</th><th>Runs</th><th></th></tr></thead>
                    <tbody>
                      {snapshots.map((s) => (
                        <tr
                          key={s.id}
                          className={`row-selectable ${selectedSnapshotIds.includes(s.id) ? 'row-selected' : ''}`}
                          onClick={() => toggleSnapshotSelection(s.id)}
                        >
                          <td>{selectedSnapshotIds.includes(s.id) ? '●' : '○'}</td>
                          <td>{s.label}</td>
                          <td>{s.settings?.modelKey || '-'}</td>
                          <td>{s.settings?.backend || '-'}</td>
                          <td>{`e:${s.settings?.encoderQuant || '-'} d:${s.settings?.decoderQuant || '-'}`}</td>
                          <td>{s.settings?.preprocessorBackend || '-'}</td>
                          <td>{s.hardwareSummary?.gpuModelLabel || s.hardwareSummary?.gpuLabel || '-'}</td>
                          <td>{s.settings?.randomize ? (s.settings?.randomSeed ?? 'rnd') : 'off'}</td>
                          <td>{ms(s.summary?.totalMean)}</td>
                          <td>{s.summary?.runCount ?? '-'}</td>
                          <td><button className="btn btn-sm btn-danger" onClick={(e) => { e.stopPropagation(); deleteSnapshot(s.id); }}>×</button></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : <div className="empty-row">No snapshots yet. Run a benchmark and save a snapshot.</div>}
              </div>
            </section>
          </div>
        )}

        {/* ── DATA TAB ── */}
        {activeTab === 'data' && (
          <div className="fade-in">
            <section className="table-panel">
              <div className="table-header">
                <h3>Recent Runs ({recentRuns.length})</h3>
                <span className="meta-mono">{okRuns.length} ok / {runErrors} errors</span>
              </div>
              <div className="table-wrap">
                {recentRuns.length ? (
                  <table>
                    <thead><tr><th>#</th><th>Sample</th><th>Rep</th><th>Dur</th><th>Preproc</th><th>Encode</th><th>Decode</th><th>Token</th><th>Total</th><th>RTF</th><th>Enc RTFx</th><th>Dec RTFx</th><th>Exact</th><th>Sim</th><th>Error</th></tr></thead>
                    <tbody>
                      {recentRuns.map((r) => (
                        <tr key={r.id} className={r.error ? 'row-error' : ''}><td>{r.id}</td><td>{r.sampleKey}</td><td>{r.repeatIndex}</td><td>{Number.isFinite(r.audioDurationSec) ? `${r.audioDurationSec.toFixed(2)}s` : '-'}</td><td>{ms(r.metrics?.preprocess_ms)}</td><td>{ms(r.metrics?.encode_ms)}</td><td>{ms(r.metrics?.decode_ms)}</td><td>{ms(r.metrics?.tokenize_ms)}</td><td>{ms(r.metrics?.total_ms)}</td><td>{Number.isFinite(r.metrics?.rtf) ? `${r.metrics.rtf.toFixed(1)}×` : '-'}</td><td>{rtfx(calcRtfx(r.audioDurationSec, r.metrics?.encode_ms))}</td><td>{rtfx(calcRtfx(r.audioDurationSec, r.metrics?.decode_ms))}</td><td>{typeof r.exactMatchToFirst === 'boolean' ? (r.exactMatchToFirst ? '✓' : '✗') : '-'}</td><td>{Number.isFinite(r.similarityToFirst) ? `${(r.similarityToFirst * 100).toFixed(1)}%` : '-'}</td><td className="text-cell">{r.error || '-'}</td></tr>
                      ))}
                    </tbody>
                  </table>
                ) : <div className="empty-row">No runs yet.</div>}
              </div>
            </section>
          </div>
        )}
      </div>
    </div>
  );
}
