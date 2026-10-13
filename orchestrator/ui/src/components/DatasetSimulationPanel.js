import React, { useCallback, useEffect, useRef, useState } from 'react';
import { shallowEqual, useSelector } from 'react-redux';
import { MdFolderOpen } from 'react-icons/md';
import { selectInferenceTaskInfo } from '../features/tasks/taskSlice';
import { InferencePhase } from '../constants/taskPhases';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import FileBrowserModal from './FileBrowserModal';

export function DatasetSimulationImages({ frame }) {
  const images = Object.entries(frame?.images || {});
  return (
    <div className="flex h-full w-full items-center justify-center gap-2 overflow-hidden">
      {images.length === 0 && <div className="text-sm text-gray-500">Load a dataset simulation to show its camera images.</div>}
      {images.map(([name, src]) => (
        <div key={name} className="relative flex h-full min-w-0 flex-1 items-center justify-center rounded-xl bg-gray-100">
          <img src={src} alt={name} className="max-h-full max-w-full object-contain" />
          <span className="absolute bottom-2 left-2 rounded bg-black/60 px-2 py-1 text-xs text-white">{name} · frame {frame.frame}</span>
        </div>
      ))}
    </div>
  );
}

export default function DatasetSimulationPanel({ enabled, active = true, onEnabledChange, onFrame, onSessionChange }) {
  const info = useSelector(selectInferenceTaskInfo, shallowEqual);
  const robotType = useSelector((state) => state.tasks.robotType);
  const phase = useSelector((state) => state.tasks.inferenceStatus.inferencePhase);
  const { callService } = useRosServiceCaller();
  const [datasetPath, setDatasetPath] = useState('');
  const [showDatasetBrowser, setShowDatasetBrowser] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [initialFrame, setInitialFrame] = useState(0);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [running, setRunning] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const runRef = useRef(false);
  const busyRef = useRef(false);
  const mountedRef = useRef(true);
  const loadedRef = useRef(false);
  const pendingRef = useRef(Promise.resolve());
  const callRef = useRef(callService);
  callRef.current = callService;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      runRef.current = false;
      // Let the in-flight chunk finish before freeing its model.
      if (loadedRef.current) pendingRef.current.finally(() => callRef.current(
        '/lerobot/engine_command', 'interfaces/srv/EngineCommand', { command: 5 }, 0
      ).catch(() => {}));
    };
  }, []);

  useEffect(() => {
    if (!active) {
      runRef.current = false;
      setRunning(false);
    }
  }, [active]);

  const command = useCallback(async (value) => {
    const response = await callService('/lerobot/engine_command', 'interfaces/srv/EngineCommand', {
      command: value,
      model_path: info.policyPath || '',
      robot_type: robotType,
      ik_backend: info.ikBackend || 'cyclo_control',
      ik_model_path: info.ikModelPath || '',
      action_request_mode: info.actionRequestMode || 'async',
      chunkrl_checkpoint: info.chunkrlCheckpoint || '',
      task_instruction: info.taskInstruction?.[0] || '',
      inference_hz: Number(info.inferenceHz) || 30,
      simulation_config: JSON.stringify({ dataset_path: datasetPath, episode: Number(episode), frame: Number(initialFrame), control_hz: Number(info.controlHz) || 100 }),
    }, 0);
    if (!response.success) throw new Error(response.message || 'Dataset simulation failed');
    return JSON.parse(response.simulation_state);
  }, [callService, info.policyPath, info.taskInstruction, info.inferenceHz, info.controlHz, info.ikBackend, info.ikModelPath, info.actionRequestMode, info.chunkrlCheckpoint, robotType, datasetPath, episode, initialFrame]);

  const display = useCallback(async (data) => {
    for (const frame of data.frames || []) {
      if (frame.dt_s != null) await new Promise((resolve) => setTimeout(resolve, frame.dt_s * 1000));
      if (!mountedRef.current) return;
      onFrame({ ...data, ...frame });
      if (frame.dt_s == null && data.frames.length > 1) await new Promise((resolve) => setTimeout(resolve, 1000 / data.fps));
    }
    if (mountedRef.current) setStatus(`Episode ${data.episode} · frame ${data.frame}/${data.length - 1}${data.ik_backend ? ' · ' + data.ik_backend : ''}${data.action_request_mode === 'chunkrl' ? ' · ChunkRL ' + (data.chunkrl_decisions || []).map((d) => `${d.arm}:${d.switched ? 'switch' : 'keep'}@${d.index}`).join(' / ') : ''}${data.finished ? ' · finished' : ''}`);
  }, [onFrame]);

  const execute = useCallback((operation) => {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setError('');
    const pending = operation().catch((err) => {
      runRef.current = false;
      if (mountedRef.current) setError(err.message || String(err));
    }).finally(() => {
      busyRef.current = false;
      if (mountedRef.current) {
        setBusy(false);
        setRunning(false);
      }
    });
    pendingRef.current = pending;
  }, []);

  const load = () => execute(async () => {
    const data = await command(3);
    loadedRef.current = true;
    if (!mountedRef.current) {
      await command(5);
      return;
    }
    setLoaded(true);
    onSessionChange(true);
    await display(data);
  });
  const step = (continuous) => execute(async () => {
    runRef.current = continuous;
    setRunning(continuous);
    do {
      const data = await command(4);
      await display(data);
      if (data.finished) runRef.current = false;
    } while (runRef.current && mountedRef.current);
  });
  const unload = () => execute(async () => {
    await command(5);
    loadedRef.current = false;
    setLoaded(false);
    onSessionChange(false);
    onFrame(null);
    setStatus('Unloaded');
  });

  const toggle = async (nextEnabled) => {
    if (nextEnabled) {
      onEnabledChange(true);
      onFrame(null);
      return;
    }
    runRef.current = false;
    setRunning(false);
    setStopping(true);
    try {
      // A load or chunk may still be using the model. Finish it before unload.
      await pendingRef.current;
      if (loadedRef.current) await command(5);
      loadedRef.current = false;
      setLoaded(false);
      onSessionChange(false);
      onFrame(null);
      onEnabledChange(false);
      setStatus('Unloaded');
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setStopping(false);
    }
  };

  const available = info.serviceType === 'lerobot' && (info.inferenceMode || 'simulation') === 'simulation' && phase === InferencePhase.READY;
  const inputClass = 'w-full rounded border border-gray-300 px-2 py-1 text-sm disabled:bg-gray-100';
  return (
    <div className="my-2 w-full max-w-[350px] rounded-xl border border-gray-200 bg-white p-3 text-sm">
      <label className="flex items-center gap-2 font-medium">
        <input type="checkbox" checked={enabled} disabled={stopping || (!enabled && !available)} onChange={(e) => toggle(e.target.checked)} />
        Dataset feedback simulation
      </label>
      {enabled && (
        <div className="mt-3 space-y-2">
          <div>
            <label htmlFor="simulation-dataset-path" className="block">Dataset path</label>
            <div className="flex items-center gap-2">
              <input id="simulation-dataset-path" aria-label="Simulation dataset path" className={inputClass} value={datasetPath} disabled={loaded || busy || stopping} onChange={(e) => setDatasetPath(e.target.value)} placeholder="/workspace/dataset/Task_0_tape_MCAP_lerobot_v30" />
              <button type="button" aria-label="Browse for simulation dataset folder" disabled={loaded || busy || stopping} onClick={() => setShowDatasetBrowser(true)} className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-gray-200 text-blue-500 hover:text-blue-700 disabled:cursor-not-allowed disabled:opacity-50">
                <MdFolderOpen className="h-5 w-5" />
              </button>
            </div>
          </div>
          <div className="flex gap-2">
            <label>Episode<input aria-label="Simulation episode" className={inputClass} type="number" min="0" value={episode} disabled={loaded || busy} onChange={(e) => setEpisode(e.target.value)} /></label>
            <label>Initial frame<input aria-label="Simulation initial frame" className={inputClass} type="number" min="0" value={initialFrame} disabled={loaded || busy} onChange={(e) => setInitialFrame(e.target.value)} /></label>
          </div>
          <div className="flex flex-wrap gap-2">
            {[
              ['Load / Reset', load, loaded || busy || !available || !datasetPath || !info.policyPath],
              ['Step chunk', () => step(false), !loaded || busy],
              ['Run', () => step(true), !loaded || busy],
              ['Pause after chunk', () => { runRef.current = false; setRunning(false); }, !running],
              ['Unload', unload, !loaded || busy],
            ].map(([label, action, disabled]) => <button key={label} className="rounded bg-indigo-600 px-2 py-1 text-xs text-white disabled:bg-gray-300" disabled={disabled || stopping} onClick={action}>{label}</button>)}
          </div>
          <p className="text-xs text-gray-500">Initial state: dataset. Next state: predicted IK joints. Images advance with dataset frames. Robot commands are disabled.</p>
          {(busy || stopping) && <p role="status">{stopping ? 'Finishing chunk and disabling' : running ? 'Running' : 'Processing'}…</p>}
          {status && <p>{status}</p>}
          {error && <p role="alert" className="text-red-600">{error}</p>}
        </div>
      )}
      {showDatasetBrowser && (
        <FileBrowserModal
          isOpen
          onClose={() => setShowDatasetBrowser(false)}
          onFileSelect={(item) => { if (item?.full_path) setDatasetPath(item.full_path); }}
          title="Select simulation dataset folder"
          selectButtonText="Select"
          allowDirectorySelect
          allowFileSelect={false}
          initialPath={datasetPath || '/workspace/dataset'}
          defaultPath="/workspace/dataset"
          homePath="/workspace/dataset"
        />
      )}
    </div>
  );
}
