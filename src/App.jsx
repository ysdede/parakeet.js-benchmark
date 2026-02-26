import React, { useEffect, useMemo, useRef, useState } from 'react';
import Chart from 'chart.js/auto';
import { MODELS, getParakeetModel, ParakeetModel } from 'parakeet.js';
import {
  fetchDatasetInfo,
  fetchDatasetRows,
  fetchDatasetSplits,
  fetchRandomRows,
  fetchSequentialRows,
  getConfigsAndSplits,
  normalizeDatasetRow,
} from './utils/hfDataset';
import {
  RUN_CSV_COLUMNS,
  flattenRunRecord,
  mean,
  normalizeText,
  stddev,
  summarize,
  textSimilarity,
  toCsv,
} from './utils/benchmarkStats';
import './App.css';

const SETTINGS_KEY = 'parakeet.benchmark.settings.v1';
const SNAPSHOTS_KEY = 'parakeet.benchmark.snapshots.v1';
const DATASET_SPLITS_CACHE_PREFIX = 'parakeet.dataset.splits.v1:';
const DATASET_INFO_CACHE_PREFIX = 'parakeet.dataset.info.v1:';
const DATASET_META_CACHE_TTL_MS = 12 * 60 * 60 * 1000;
const MAX_SNAPSHOTS = 20;

const MODEL_OPTIONS = Object.entries(MODELS).map(([key, config]) => ({
  key,
  label: config.displayName || key,
}));

const BACKENDS = ['webgpu-hybrid', 'webgpu', 'wasm'];
const QUANTS = ['fp32', 'int8', 'fp16'];
const PREPROCESSOR_MODEL = 'nemo128';
const WARMUP_AUDIO_FALLBACK_URL = 'https://raw.githubusercontent.com/ysdede/parakeet.js/master/examples/demo/public/assets/life_Jim.wav';
const QUANTIZATION_ORDER = ['fp16', 'int8', 'fp32'];
const MODEL_FILES_CACHE = new Map();

function formatRepoPath(repoId) {
  return String(repoId || '')
    .split('/')
    .map((part) => encodeURIComponent(part))
    .join('/');
}

function normalizeRepoPath(path) {
  return String(path || '').replace(/^\.\/+/, '').replace(/\\/g, '/');
}

function parseModelFiles(payload) {
  if (Array.isArray(payload)) {
    return payload
      .filter((entry) => entry?.type === 'file' && typeof entry?.path === 'string')
      .map((entry) => normalizeRepoPath(entry.path));
  }

  if (payload && typeof payload === 'object' && Array.isArray(payload.siblings)) {
    return payload.siblings
      .map((entry) => normalizeRepoPath(entry?.rfilename))
      .filter(Boolean);
  }

  return [];
}

function hasModelFile(files, filename) {
  const target = normalizeRepoPath(filename);
  return files.some((path) => path === target || path.endsWith(`/${target}`));
}

async function fetchModelFiles(repoId, revision = 'main') {
  if (!repoId) return [];
  const cacheKey = `${repoId}@${revision}`;
  if (MODEL_FILES_CACHE.has(cacheKey)) return MODEL_FILES_CACHE.get(cacheKey);

  const repoPath = formatRepoPath(repoId);
  const encodedRevision = encodeURIComponent(revision);
  const treeUrl = `https://huggingface.co/api/models/${repoPath}/tree/${encodedRevision}?recursive=1`;
  const metadataUrl = `https://huggingface.co/api/models/${repoPath}?revision=${encodedRevision}`;

  try {
    const response = await fetch(treeUrl);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const files = parseModelFiles(await response.json());
    MODEL_FILES_CACHE.set(cacheKey, files);
    return files;
  } catch (treeError) {
    console.warn(`[modelSelection] Tree listing failed for ${repoId}@${revision}; trying metadata`, treeError);
  }

  try {
    const response = await fetch(metadataUrl);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const files = parseModelFiles(await response.json());
    MODEL_FILES_CACHE.set(cacheKey, files);
    return files;
  } catch (metadataError) {
    console.warn(`[modelSelection] Metadata listing failed for ${repoId}@${revision}`, metadataError);
    return [];
  }
}

function getAvailableQuantModes(files, baseName) {
  const options = QUANTIZATION_ORDER.filter((quant) => {
    if (quant === 'fp32') return hasModelFile(files, `${baseName}.onnx`);
    if (quant === 'fp16') return hasModelFile(files, `${baseName}.fp16.onnx`);
    return hasModelFile(files, `${baseName}.int8.onnx`);
  });
  return options.length ? options : ['fp32'];
}

function pickPreferredQuant(available, currentBackend, component = 'encoder') {
  let preferred;
  if (component === 'decoder') {
    preferred = ['int8', 'fp32', 'fp16'];
  } else {
    preferred = String(currentBackend || '').startsWith('webgpu')
      ? ['fp16', 'fp32', 'int8']
      : ['int8', 'fp32', 'fp16'];
  }
  return preferred.find((quant) => available.includes(quant)) || available[0] || 'fp32';
}

function revokeBlobUrls(urls) {
  for (const value of Object.values(urls || {})) {
    if (typeof value === 'string' && value.startsWith('blob:')) {
      URL.revokeObjectURL(value);
    }
  }
}

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

function chartBase(yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'nearest', intersect: false },
    plugins: {
      legend: {
        labels: {
          color: '#dce9fb',
          font: { family: 'IBM Plex Sans', size: 11 },
        },
      },
      tooltip: {
        backgroundColor: '#0f1a2d',
        borderColor: '#4f74ac',
        borderWidth: 1,
        titleColor: '#f4f9ff',
        bodyColor: '#e4eeff',
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(255,255,255,0.05)' },
        ticks: { color: '#c6d5ed', font: { family: 'IBM Plex Mono', size: 10 } },
      },
      y: {
        grid: { color: 'rgba(255,255,255,0.05)' },
        ticks: { color: '#c6d5ed', font: { family: 'IBM Plex Mono', size: 10 } },
        title: {
          display: true,
          text: yLabel,
          color: '#deebff',
          font: { family: 'IBM Plex Sans', size: 11, weight: '600' },
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
  const [sampleCount, setSampleCount] = useState(clamp(saved.sampleCount, 6, 1, 100));
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
  const [showPreparedSamples, setShowPreparedSamples] = useState(false);
  const [hardwareProfile, setHardwareProfile] = useState(null);
  const [hardwareStatus, setHardwareStatus] = useState('Not collected');
  const [isLoadingHardware, setIsLoadingHardware] = useState(false);
  const [encoderQuantOptions, setEncoderQuantOptions] = useState(QUANTS);
  const [decoderQuantOptions, setDecoderQuantOptions] = useState(QUANTS);

  const modelRef = useRef(null);
  const releaseQueueRef = useRef(Promise.resolve());
  const stopRef = useRef(false);
  const audioCacheRef = useRef(new Map());

  async function releaseModelResources(model) {
    if (!model) return;
    try { model.stopProfiling?.(); } catch {}
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
      } catch {}
    }));
  }

  function queueModelRelease(model) {
    if (!model) return releaseQueueRef.current;
    const releaseTask = releaseQueueRef.current
      .catch(() => {})
      .then(() => releaseModelResources(model));
    releaseQueueRef.current = releaseTask.catch(() => {});
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

    (async () => {
      const files = await fetchModelFiles(repoId, 'main');
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
      const summary = summarizeHardwareProfile(profile);
      setHardwareProfile(profile);
      setHardwareStatus(`CPU ${summary.cpuLabel}, GPU ${summary.gpuLabel}`);
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
    const requested = clamp(sampleCount, 6, 1, 100);
    setDatasetStatus('Preparing sample rows...');

    let rawRows = [];
    let randomMeta = null;
    if (randomize) {
      let totalRows = splitCounts?.[datasetSplit]?.num_examples;
      if (!Number.isFinite(totalRows)) {
        const probe = await fetchDatasetRows({ dataset: datasetId, config: datasetConfig, split: datasetSplit, offset: 0, length: 1 });
        totalRows = probe.num_rows_total || 1;
      }
      const sampled = await fetchRandomRows({
        dataset: datasetId,
        config: datasetConfig,
        split: datasetSplit,
        totalRows,
        sampleCount: requested,
        seed: randomSeed,
      });
      rawRows = sampled.rows;
      randomMeta = sampled;
    } else {
      const sequential = await fetchSequentialRows({
        dataset: datasetId,
        config: datasetConfig,
        split: datasetSplit,
        startOffset: offset,
        limit: requested,
      });
      rawRows = sequential.rows;
    }

    const normalized = rawRows
      .map((item, idx) => normalizeDatasetRow(item, idx))
      .filter((item) => item.audioUrl);

    if (!normalized.length) {
      throw new Error('No playable rows found in the selected slice.');
    }

    setPreparedSamples(normalized);
    if (randomize) {
      const failedCount = randomMeta?.failedOffsets?.length || 0;
      const reqCount = randomMeta?.requestedCount || requested;
      const shortfall = reqCount - normalized.length;
      const warning = shortfall > 0 ? `, shortfall ${shortfall}` : '';
      const failed = failedCount > 0 ? `, skipped ${failedCount} failing rows` : '';
      setDatasetStatus(`Prepared ${normalized.length}/${reqCount} samples (seed: ${String(randomSeed) || 'random'}${warning}${failed})`);
    } else {
      setDatasetStatus(`Prepared ${normalized.length} samples`);
    }
    return normalized;
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

      const compileFromHub = async (hubResult, requestedEncoderQuant, requestedDecoderQuant) => (
        ParakeetModel.fromUrls({
          ...hubResult.urls,
          filenames: hubResult.filenames,
          preprocessorBackend: hubResult.preprocessorBackend,
          backend,
          encoderQuant: requestedEncoderQuant,
          decoderQuant: requestedDecoderQuant,
          cpuThreads,
          verbose: false,
        })
      );

      let hub;
      let effectiveEncoderQuant = encoderQuant;
      let effectiveDecoderQuant = decoderQuant;
      const loadNotes = [];

      try {
        hub = await getParakeetModel(modelKey, baseOptions);
      } catch (firstDownloadError) {
        const selectedFp16 = encoderQuant === 'fp16' || decoderQuant === 'fp16';
        if (!selectedFp16) throw firstDownloadError;

        effectiveEncoderQuant = encoderQuant === 'fp16' ? 'fp32' : encoderQuant;
        effectiveDecoderQuant = decoderQuant === 'fp16' ? 'fp32' : decoderQuant;
        setModelProgress('FP16 assets unavailable, retrying with fp32...');
        hub = await getParakeetModel(modelKey, {
          ...baseOptions,
          encoderQuant: effectiveEncoderQuant,
          decoderQuant: effectiveDecoderQuant,
        });
        loadNotes.push('fp16 assets unavailable, retried with fp32');
      }

      try {
        modelRef.current = await compileFromHub(hub, effectiveEncoderQuant, effectiveDecoderQuant);
      } catch (firstCompileError) {
        const canRetry = hub?.quantisation?.encoder === 'fp16' || hub?.quantisation?.decoder === 'fp16';
        if (!canRetry) throw firstCompileError;

        revokeBlobUrls(hub?.urls);
        effectiveEncoderQuant = hub?.quantisation?.encoder === 'fp16' ? 'fp32' : effectiveEncoderQuant;
        effectiveDecoderQuant = hub?.quantisation?.decoder === 'fp16' ? 'fp32' : effectiveDecoderQuant;
        setModelProgress('FP16 compile failed, retrying with fp32...');

        let retryHub;
        try {
          retryHub = await getParakeetModel(modelKey, {
            ...baseOptions,
            encoderQuant: effectiveEncoderQuant,
            decoderQuant: effectiveDecoderQuant,
          });
        } catch (retryDownloadError) {
          throw new Error(
            `Initial compile failed (${firstCompileError?.message || firstCompileError}). ` +
            `FP32 retry download failed (${retryDownloadError?.message || retryDownloadError}).`
          );
        }

        try {
          modelRef.current = await compileFromHub(retryHub, effectiveEncoderQuant, effectiveDecoderQuant);
        } catch (retryCompileError) {
          throw new Error(
            `Initial compile failed (${firstCompileError?.message || firstCompileError}). ` +
            `FP32 retry compile failed (${retryCompileError?.message || retryCompileError}).`
          );
        }
        hub = retryHub;
        loadNotes.push(`fp16 compile retry -> e:${effectiveEncoderQuant} d:${effectiveDecoderQuant}`);
      }

      const resolvedQuant = hub?.quantisation
        ? `resolved e:${hub.quantisation.encoder} d:${hub.quantisation.decoder}`
        : '';
      const loadedFiles = hub?.filenames
        ? `${hub.filenames.encoder}, ${hub.filenames.decoder}`
        : '';
      const backendHint = backend.startsWith('webgpu')
        ? 'decoder executes on WASM in webgpu modes'
        : '';
      setResolvedModelInfo([resolvedQuant, loadedFiles, backendHint, ...loadNotes].filter(Boolean).join(' | '));

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

        setProgress({ current: done, total, stage: `Decoding ${sampleKey}` });

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
          setProgress({ current: done, total, stage: `Warmup ${w + 1}/${warmups} for ${sampleKey}` });
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
          setProgress({ current: done, total, stage: `Run ${r}/${repeatCount} for ${sampleKey}` });

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
            title: { display: true, text: 'Encode (ms)', color: '#deebff', font: { family: 'IBM Plex Sans', size: 11, weight: '600' } },
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
            title: { display: true, text: 'Run order', color: '#deebff', font: { family: 'IBM Plex Sans', size: 11, weight: '600' } },
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
            title: { display: true, text: 'Audio duration (s)', color: '#deebff', font: { family: 'IBM Plex Sans', size: 11, weight: '600' } },
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
            title: { display: true, text: 'Audio duration (s)', color: '#deebff', font: { family: 'IBM Plex Sans', size: 11, weight: '600' } },
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
          borderColor: '#172742',
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#dce9fb', font: { family: 'IBM Plex Sans', size: 11 } },
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
            ticks: { color: '#c6d5ed', maxRotation: 30, minRotation: 30, font: { family: 'IBM Plex Mono', size: 9 } },
          },
          y: {
            ...chartBase('Mean ms').scales.y,
            stacked: true,
          },
        },
      },
    };

    return { encDec, rtfxRunOrder, rtfxDuration, durPre, trend, bottleneck, compareStages };
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

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>Parakeet Benchmark Lab</h1>
        </div>

        <section className="sidebar-section">
          <h2>Hardware</h2>
          <button className="btn" onClick={refreshHardwareProfile} disabled={isLoadingHardware || isRunning}>
            {isLoadingHardware ? 'Refreshing...' : 'Refresh hardware'}
          </button>
          <p className="status-text">{hardwareStatus}</p>
          <div className="kv-grid">
            <div className="kv-row"><span>CPU</span><strong>{hardwareSummary.cpuLabel}</strong></div>
            <div className="kv-row"><span>System RAM</span><strong>{hardwareSummary.systemMemoryLabel}</strong></div>
            <div className="kv-row"><span>WebGPU</span><strong>{hardwareSummary.webgpuLabel}</strong></div>
            <div className="kv-row"><span>GPU</span><strong>{hardwareSummary.gpuLabel}</strong></div>
            <div className="kv-row"><span>GPU model</span><strong>{hardwareSummary.gpuModelLabel}</strong></div>
            <div className="kv-row"><span>GPU cores</span><strong>{hardwareSummary.gpuCoresLabel}</strong></div>
            <div className="kv-row"><span>VRAM</span><strong>{hardwareSummary.vramLabel}</strong></div>
          </div>
          {hardwareProfile?.webgpu?.features?.length ? <p className="subtle">WebGPU features: {hardwareProfile.webgpu.features.slice(0, 8).join(', ')}{hardwareProfile.webgpu.features.length > 8 ? ' ...' : ''}</p> : null}
        </section>

        <section className="sidebar-section">
          <label>Model<select value={modelKey} onChange={(e) => setModelKey(e.target.value)} disabled={isLoadingModel || isRunning}>{MODEL_OPTIONS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}</select></label>
          <label>Backend<select value={backend} onChange={(e) => setBackend(e.target.value)} disabled={isLoadingModel || isRunning}>{BACKENDS.map((v) => <option key={v} value={v}>{v}</option>)}</select></label>
          <div className="row-2">
            <label>Encoder<select value={encoderQuant} onChange={(e) => setEncoderQuant(e.target.value)} disabled={isLoadingModel || isRunning}>{encoderQuantOptions.map((v) => <option key={v} value={v}>{v}</option>)}</select></label>
            <label>Decoder<select value={decoderQuant} onChange={(e) => setDecoderQuant(e.target.value)} disabled={isLoadingModel || isRunning}>{decoderQuantOptions.map((v) => <option key={v} value={v}>{v}</option>)}</select></label>
          </div>
          <label>Preprocessor<select value={preprocessorBackend} onChange={(e) => setPreprocessorBackend(e.target.value)} disabled={isLoadingModel || isRunning}><option value="js">JS (meljs)</option><option value="onnx">ONNX (nemo128.onnx)</option></select></label>
          <label>CPU threads<input type="number" min="1" max="64" value={cpuThreads} onChange={(e) => setCpuThreads(clamp(e.target.value, cpuThreads, 1, 64))} disabled={isLoadingModel || isRunning} /></label>
          <label className="check"><input type="checkbox" checked={enableProfiling} onChange={(e) => setEnableProfiling(e.target.checked)} disabled={isLoadingModel || isRunning} />Enable profiling</label>
          <button className="btn btn-primary" onClick={loadModel} disabled={isLoadingModel || isRunning || isModelReady}>{isLoadingModel ? 'Loading...' : isModelReady ? 'Model ready' : 'Load model'}</button>
          <p className="status-text">{modelStatus}</p>
          {modelProgress ? <p className="subtle">{modelProgress}</p> : null}
          {resolvedModelInfo ? <p className="subtle">{resolvedModelInfo}</p> : null}
        </section>

        <section className="sidebar-section">
          <h2>Dataset</h2>
          <label>Dataset<input value={datasetId} onChange={(e) => setDatasetId(e.target.value)} disabled={isLoadingDataset || isRunning} /></label>
          <button className="btn" onClick={() => refreshDatasetMeta(datasetId, datasetConfig, datasetSplit, true)} disabled={isLoadingDataset || isRunning}>{isLoadingDataset ? 'Refreshing...' : 'Refresh metadata'}</button>
          <div className="row-2">
            <label>Config<select value={datasetConfig} onChange={(e) => refreshDatasetMeta(datasetId, e.target.value, datasetSplit)} disabled={isLoadingDataset || isRunning}>{configs.map((c) => <option key={c} value={c}>{c}</option>)}</select></label>
            <label>Split<select value={datasetSplit} onChange={(e) => setDatasetSplit(e.target.value)} disabled={isLoadingDataset || isRunning}>{splits.map((s) => <option key={s} value={s}>{s}</option>)}</select></label>
          </div>
          <div className="row-2">
            <label>Offset<input type="number" min="0" value={offset} onChange={(e) => setOffset(clamp(e.target.value, offset, 0, 1_000_000))} disabled={isRunning} /></label>
            <label>Samples<input type="number" min="1" max="100" value={sampleCount} onChange={(e) => setSampleCount(clamp(e.target.value, sampleCount, 1, 100))} disabled={isRunning} /></label>
          </div>
          <button className="btn" onClick={prepareSampleRows} disabled={isRunning || isLoadingDataset}>Preview sample set</button>
          <p className="status-text">{datasetStatus}</p>
          {features.length ? <p className="subtle">Features: {features.join(', ')}</p> : null}
          {splitCounts?.[datasetSplit]?.num_examples ? <p className="subtle">Rows: {splitCounts[datasetSplit].num_examples}</p> : null}
        </section>

        <section className="sidebar-section">
          <h2>Run Plan</h2>
          <div className="row-2">
            <label>Repeats<input type="number" min="1" max="100" value={repeatCount} onChange={(e) => setRepeatCount(clamp(e.target.value, repeatCount, 1, 100))} disabled={isRunning} /></label>
            <label>Warmups<input type="number" min="0" max="10" value={warmups} onChange={(e) => setWarmups(clamp(e.target.value, warmups, 0, 10))} disabled={isRunning} /></label>
          </div>
          <label className="check"><input type="checkbox" checked={randomize} onChange={(e) => setRandomize(e.target.checked)} disabled={isRunning} />Random sample rows</label>
          <label>Random seed<input value={randomSeed} onChange={(e) => setRandomSeed(e.target.value)} placeholder="e.g. 42 or exp-a" disabled={isRunning || !randomize} /></label>
          <button className="btn btn-primary" onClick={runBenchmark} disabled={isRunning || isLoadingModel || !isModelReady}>{isRunning ? 'Running...' : 'Start benchmark'}</button>
          <button className="btn btn-danger" onClick={stopRun} disabled={!isRunning}>Stop</button>
          <div className="row-2">
            <button className="btn" onClick={exportJson} disabled={!runs.length || isRunning}>Export JSON</button>
            <button className="btn" onClick={exportCsv} disabled={!runs.length || isRunning}>Export CSV</button>
          </div>
          <button className="btn" onClick={() => setRuns([])} disabled={!runs.length || isRunning}>Clear runs</button>
          <label>Snapshot name<input value={snapshotName} onChange={(e) => setSnapshotName(e.target.value)} placeholder="optional label" disabled={isRunning} /></label>
          <button className="btn" onClick={saveCurrentSnapshot} disabled={!runs.length || isRunning}>Save snapshot</button>
          <p className="status-text">{benchStatus}</p>
          {progress.total > 0 ? <p className="subtle">{progress.stage} ({progress.current}/{progress.total})</p> : null}
        </section>

        <section className="sidebar-section">
          <h2>Compare Stored</h2>
          <label>A<select value={compareAId} onChange={(e) => setCompareAId(e.target.value)} disabled={!snapshots.length}>{snapshots.length ? snapshots.map((s) => <option key={s.id} value={s.id}>{s.label}</option>) : <option value="">No snapshots</option>}</select></label>
          <label>B<select value={compareBId} onChange={(e) => setCompareBId(e.target.value)} disabled={!snapshots.length}>{snapshots.length ? snapshots.map((s) => <option key={s.id} value={s.id}>{s.label}</option>) : <option value="">No snapshots</option>}</select></label>
          <p className="subtle">{snapshots.length} stored snapshot(s)</p>
        </section>
      </aside>

      <main className="main">
        <div className="topbar">
          <h2>Repeated Transcription Analytics</h2>
          <div className="meta-text">{okRuns.length} successful / {runs.length} total{runErrors ? `, ${runErrors} errors` : ''}</div>
        </div>

        <div className="kpi-row">
          <div className="kpi-card teal"><div className="kpi-label">Total</div><div className="kpi-value">{ms(metrics.total.mean)}</div><div className="kpi-sub">p90 {ms(metrics.total.p90)}</div></div>
          <div className="kpi-card green"><div className="kpi-label">Preprocess</div><div className="kpi-value">{ms(metrics.preprocess.mean)}</div><div className="kpi-sub">share {pct(Number.isFinite(metrics.preprocess.mean) && Number.isFinite(metrics.total.mean) && metrics.total.mean > 0 ? metrics.preprocess.mean / metrics.total.mean : null)}</div></div>
          <div className="kpi-card orange"><div className="kpi-label">Decode</div><div className="kpi-value">{ms(metrics.decode.mean)}</div><div className="kpi-sub">std {ms(metrics.decode.stddev)}</div></div>
          <div className="kpi-card teal"><div className="kpi-label">Encode</div><div className="kpi-value">{ms(metrics.encode.mean)}</div><div className="kpi-sub">Tokenize {ms(metrics.tokenize.mean)}</div></div>
          <div className="kpi-card orange"><div className="kpi-label">Exact Repeatability</div><div className="kpi-value">{pct(repeatability.exactRate)}</div><div className="kpi-sub">sim {pct(repeatability.similarityMean)}</div></div>
          <div className="kpi-card purple"><div className="kpi-label">RTF Median</div><div className="kpi-value">{Number.isFinite(metrics.rtf.median) ? `${metrics.rtf.median.toFixed(2)}x` : '-'}</div><div className="kpi-sub">duration avg {Number.isFinite(mean(okRuns.map((r) => r.audioDurationSec).filter(Number.isFinite))) ? mean(okRuns.map((r) => r.audioDurationSec).filter(Number.isFinite)).toFixed(2) : '-'} s</div></div>
          <div className="kpi-card teal"><div className="kpi-label">Encoder RTFx</div><div className="kpi-value">{rtfx(metrics.encodeRtfx.median)}</div><div className="kpi-sub">std {rtfx(metrics.encodeRtfx.stddev)}</div></div>
          <div className="kpi-card green"><div className="kpi-label">Decoder RTFx</div><div className="kpi-value">{rtfx(metrics.decodeRtfx.median)}</div><div className="kpi-sub">std {rtfx(metrics.decodeRtfx.stddev)}</div></div>
        </div>

        <section className="table-panel">
          <div className="table-header"><h3>Snapshot A/B Comparison</h3></div>
          <div className="table-wrap">
            {compareA && compareB ? (
              <table>
                <thead><tr><th>Metric</th><th>A ({compareA.label})</th><th>B ({compareB.label})</th><th>Delta B vs A</th></tr></thead>
                <tbody>
                  <tr><td>Model</td><td>{compareA.settings?.modelKey || '-'}</td><td>{compareB.settings?.modelKey || '-'}</td><td>-</td></tr>
                  <tr><td>Backend</td><td>{compareA.settings?.backend || '-'}</td><td>{compareB.settings?.backend || '-'}</td><td>-</td></tr>
                  <tr><td>CPU</td><td>{compareA.hardwareSummary?.cpuLabel || '-'}</td><td>{compareB.hardwareSummary?.cpuLabel || '-'}</td><td>-</td></tr>
                  <tr><td>GPU</td><td>{compareA.hardwareSummary?.gpuLabel || '-'}</td><td>{compareB.hardwareSummary?.gpuLabel || '-'}</td><td>-</td></tr>
                  <tr><td>GPU model</td><td>{compareA.hardwareSummary?.gpuModelLabel || '-'}</td><td>{compareB.hardwareSummary?.gpuModelLabel || '-'}</td><td>-</td></tr>
                  <tr><td>GPU cores</td><td>{compareA.hardwareSummary?.gpuCoresLabel || '-'}</td><td>{compareB.hardwareSummary?.gpuCoresLabel || '-'}</td><td>-</td></tr>
                  <tr><td>VRAM</td><td>{compareA.hardwareSummary?.vramLabel || '-'}</td><td>{compareB.hardwareSummary?.vramLabel || '-'}</td><td>-</td></tr>
                  <tr><td>System RAM</td><td>{compareA.hardwareSummary?.systemMemoryLabel || '-'}</td><td>{compareB.hardwareSummary?.systemMemoryLabel || '-'}</td><td>-</td></tr>
                  <tr><td>Preprocessor</td><td>{compareA.settings?.preprocessorBackend || '-'}</td><td>{compareB.settings?.preprocessorBackend || '-'}</td><td>-</td></tr>
                  <tr><td>Random seed</td><td>{compareA.settings?.randomize ? (compareA.settings?.randomSeed ?? 'random') : 'off'}</td><td>{compareB.settings?.randomize ? (compareB.settings?.randomSeed ?? 'random') : 'off'}</td><td>-</td></tr>
                  <tr><td>Total mean</td><td>{ms(compareA.summary?.totalMean)}</td><td>{ms(compareB.summary?.totalMean)}</td><td>{deltaPercent(compareA.summary?.totalMean, compareB.summary?.totalMean, true)}</td></tr>
                  <tr><td>Preprocess mean</td><td>{ms(compareA.summary?.preprocessMean)}</td><td>{ms(compareB.summary?.preprocessMean)}</td><td>{deltaPercent(compareA.summary?.preprocessMean, compareB.summary?.preprocessMean, true)}</td></tr>
                  <tr><td>Encode mean</td><td>{ms(compareA.summary?.encodeMean)}</td><td>{ms(compareB.summary?.encodeMean)}</td><td>{deltaPercent(compareA.summary?.encodeMean, compareB.summary?.encodeMean, true)}</td></tr>
                  <tr><td>Decode mean</td><td>{ms(compareA.summary?.decodeMean)}</td><td>{ms(compareB.summary?.decodeMean)}</td><td>{deltaPercent(compareA.summary?.decodeMean, compareB.summary?.decodeMean, true)}</td></tr>
                  <tr><td>RTF median</td><td>{Number.isFinite(compareA.summary?.rtfMedian) ? `${compareA.summary.rtfMedian.toFixed(2)}x` : '-'}</td><td>{Number.isFinite(compareB.summary?.rtfMedian) ? `${compareB.summary.rtfMedian.toFixed(2)}x` : '-'}</td><td>{deltaPercent(compareA.summary?.rtfMedian, compareB.summary?.rtfMedian, false)}</td></tr>
                  <tr><td>Encoder RTFx median</td><td>{rtfx(compareA.summary?.encodeRtfxMedian)}</td><td>{rtfx(compareB.summary?.encodeRtfxMedian)}</td><td>{deltaPercent(compareA.summary?.encodeRtfxMedian, compareB.summary?.encodeRtfxMedian, false)}</td></tr>
                  <tr><td>Decoder RTFx median</td><td>{rtfx(compareA.summary?.decodeRtfxMedian)}</td><td>{rtfx(compareB.summary?.decodeRtfxMedian)}</td><td>{deltaPercent(compareA.summary?.decodeRtfxMedian, compareB.summary?.decodeRtfxMedian, false)}</td></tr>
                  <tr><td>Exact repeatability</td><td>{pct(compareA.summary?.exactRate)}</td><td>{pct(compareB.summary?.exactRate)}</td><td>{deltaPercent(compareA.summary?.exactRate, compareB.summary?.exactRate, false)}</td></tr>
                  <tr><td>Similarity mean</td><td>{pct(compareA.summary?.similarityMean)}</td><td>{pct(compareB.summary?.similarityMean)}</td><td>{deltaPercent(compareA.summary?.similarityMean, compareB.summary?.similarityMean, false)}</td></tr>
                  <tr><td>Runs</td><td>{compareA.summary?.runCount ?? '-'}</td><td>{compareB.summary?.runCount ?? '-'}</td><td>-</td></tr>
                </tbody>
              </table>
            ) : <div className="empty-row">Save snapshots, then choose A/B in the sidebar.</div>}
          </div>
        </section>

        {chartConfigs ? (
          <div className="chart-grid">
            <ChartCard title="Encoder vs Decoder" badge="scatter" config={chartConfigs.encDec} />
            <ChartCard title="Run Order vs Stage RTFx" badge="scatter" config={chartConfigs.rtfxRunOrder} />
            <ChartCard title="Audio Duration vs Stage RTFx" badge="scatter" config={chartConfigs.rtfxDuration} />
            <ChartCard title="Audio Duration vs Preprocess" badge="scatter" config={chartConfigs.durPre} />
            <ChartCard title="Repeat Trend" badge="mean" config={chartConfigs.trend} />
            <ChartCard title="Stage Bottleneck Share" badge="doughnut" config={chartConfigs.bottleneck} />
            <ChartCard title="Config Stage Compare" badge="stacked" config={chartConfigs.compareStages} />
          </div>
        ) : (
          <div className="empty-state"><p>Run a benchmark batch to generate charts.</p></div>
        )}

        <section className="table-panel">
          <div className="table-header"><h3>Per-Sample Repeatability</h3></div>
          <div className="table-wrap">
            {sampleStats.length ? (
              <table>
                <thead><tr><th>Sample</th><th>Runs</th><th>Unique</th><th>Exact</th><th>Similarity</th><th>Preproc mean</th><th>Encode mean</th><th>Decode mean</th><th>Decode std</th><th>Enc RTFx</th><th>Enc RTFx std</th><th>Dec RTFx</th><th>Dec RTFx std</th><th>Total mean</th></tr></thead>
                <tbody>
                  {sampleStats.map((s) => (
                    <tr key={s.sampleKey}><td>{s.sampleKey}</td><td>{s.runs}</td><td>{s.uniqueOutputs}</td><td>{pct(s.exactRate)}</td><td>{pct(s.similarity)}</td><td>{ms(s.preprocessMean)}</td><td>{ms(s.encodeMean)}</td><td>{ms(s.decodeMean)}</td><td>{ms(s.decodeStd)}</td><td>{rtfx(s.encodeRtfxMean)}</td><td>{rtfx(s.encodeRtfxStd)}</td><td>{rtfx(s.decodeRtfxMean)}</td><td>{rtfx(s.decodeRtfxStd)}</td><td>{ms(s.totalMean)}</td></tr>
                  ))}
                </tbody>
              </table>
            ) : <div className="empty-row">No sample stats yet.</div>}
          </div>
        </section>

        <section className="table-panel">
          <div className="table-header"><h3>Config Bottleneck Comparison</h3></div>
          <div className="table-wrap">
            {configStats.length ? (
              <table>
                <thead><tr><th>Config</th><th>Runs</th><th>Preproc mean</th><th>Encode mean</th><th>Decode mean</th><th>Tokenize mean</th><th>Total mean</th><th>Preproc share</th><th>Decode share</th></tr></thead>
                <tbody>
                  {configStats.map((c) => (
                    <tr key={c.key}><td>{c.key}</td><td>{c.runs}</td><td>{ms(c.preprocessMean)}</td><td>{ms(c.encodeMean)}</td><td>{ms(c.decodeMean)}</td><td>{ms(c.tokenizeMean)}</td><td>{ms(c.totalMean)}</td><td>{pct(c.preprocessShare)}</td><td>{pct(c.decodeShare)}</td></tr>
                  ))}
                </tbody>
              </table>
            ) : <div className="empty-row">Run at least one benchmark batch.</div>}
          </div>
        </section>

        <section className="table-panel">
          <div className="table-header"><h3>Stored Snapshots</h3></div>
          <div className="table-wrap">
            {snapshots.length ? (
              <table>
                <thead><tr><th>Label</th><th>Created</th><th>Model</th><th>Backend</th><th>CPU</th><th>GPU model</th><th>VRAM</th><th>Quant</th><th>Preproc</th><th>Seed</th><th>Total mean</th><th>Runs</th><th>Action</th></tr></thead>
                <tbody>
                  {snapshots.map((s) => (
                    <tr key={s.id}>
                      <td>{s.label}</td>
                      <td>{s.createdAt ? new Date(s.createdAt).toLocaleString() : '-'}</td>
                      <td>{s.settings?.modelKey || '-'}</td>
                      <td>{s.settings?.backend || '-'}</td>
                      <td>{s.hardwareSummary?.cpuLabel || '-'}</td>
                      <td>{s.hardwareSummary?.gpuModelLabel || s.hardwareSummary?.gpuLabel || '-'}</td>
                      <td>{s.hardwareSummary?.vramLabel || '-'}</td>
                      <td>{`e:${s.settings?.encoderQuant || '-'} d:${s.settings?.decoderQuant || '-'}`}</td>
                      <td>{s.settings?.preprocessorBackend || '-'}</td>
                      <td>{s.settings?.randomize ? (s.settings?.randomSeed ?? 'random') : 'off'}</td>
                      <td>{ms(s.summary?.totalMean)}</td>
                      <td>{s.summary?.runCount ?? '-'}</td>
                      <td><button className="btn" onClick={() => deleteSnapshot(s.id)}>Delete</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <div className="empty-row">No snapshots saved yet.</div>}
          </div>
        </section>

        <section className="table-panel">
          <div className="table-header"><h3>Recent Runs ({recentRuns.length})</h3></div>
          <div className="table-wrap">
            {recentRuns.length ? (
              <table>
                <thead><tr><th>Run</th><th>Sample</th><th>Repeat</th><th>Duration</th><th>Preproc</th><th>Encode</th><th>Decode</th><th>Tokenize</th><th>Total</th><th>RTF</th><th>Enc RTFx</th><th>Dec RTFx</th><th>Exact</th><th>Sim</th><th>Error</th></tr></thead>
                <tbody>
                  {recentRuns.map((r) => (
                    <tr key={r.id} className={r.error ? 'row-error' : ''}><td>{r.id}</td><td>{r.sampleKey}</td><td>{r.repeatIndex}</td><td>{Number.isFinite(r.audioDurationSec) ? `${r.audioDurationSec.toFixed(2)} s` : '-'}</td><td>{ms(r.metrics?.preprocess_ms)}</td><td>{ms(r.metrics?.encode_ms)}</td><td>{ms(r.metrics?.decode_ms)}</td><td>{ms(r.metrics?.tokenize_ms)}</td><td>{ms(r.metrics?.total_ms)}</td><td>{Number.isFinite(r.metrics?.rtf) ? `${r.metrics.rtf.toFixed(2)}x` : '-'}</td><td>{rtfx(calcRtfx(r.audioDurationSec, r.metrics?.encode_ms))}</td><td>{rtfx(calcRtfx(r.audioDurationSec, r.metrics?.decode_ms))}</td><td>{typeof r.exactMatchToFirst === 'boolean' ? (r.exactMatchToFirst ? 'yes' : 'no') : '-'}</td><td>{Number.isFinite(r.similarityToFirst) ? `${(r.similarityToFirst * 100).toFixed(1)}%` : '-'}</td><td className="text-cell">{r.error || '-'}</td></tr>
                  ))}
                </tbody>
              </table>
            ) : <div className="empty-row">No runs yet.</div>}
          </div>
        </section>

        <section className="table-panel">
          <div className="table-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
            <h3>Prepared Samples ({preparedSamples.length})</h3>
            <button className="btn" onClick={() => setShowPreparedSamples((v) => !v)}>
              {showPreparedSamples ? 'Fold' : 'Unfold'}
            </button>
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
              ) : <div className="empty-row">No prepared samples yet.</div>}
            </div>
          ) : (
            <div className="empty-row">Folded. Unfold to inspect sampled rows.</div>
          )}
        </section>
      </main>
    </div>
  );
}
