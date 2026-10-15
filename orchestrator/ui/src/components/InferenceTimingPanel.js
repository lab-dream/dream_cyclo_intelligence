import React, { useEffect, useState } from 'react';

const STAGES = [
  ['observation', 'Observation'], ['preprocess', 'Preprocess'], ['policy', 'Policy'],
  ['postprocess', 'Postprocess'], ['request', 'Request'], ['buffer', 'Buffer'], ['total', 'Total'],
];

export default function InferenceTimingPanel({ telemetry }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const timer = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(timer); }, []);
  const timings = telemetry?.timings_ms || {};
  const age = telemetry ? Math.max(0, (now - telemetry.receivedAt) / 1000) : null;
  const value = (key) => {
    const time = timings[key] ?? (['total', 'request'].includes(key) ? timings.engine_request : undefined);
    return Number.isFinite(time) && time >= 0 ? `${time.toFixed(1)} ms` : '—';
  };
  return (
    <section aria-label="Inference timing" className="mx-2 my-1 rounded-lg border border-gray-200 bg-white dark:border-slate-700 dark:bg-slate-900 px-3 py-2 text-xs">
      <div className="flex flex-wrap items-center justify-between gap-2 text-gray-600 dark:text-slate-300">
        <strong>Inference timing {telemetry?.action_request_mode ? `· ${telemetry.action_request_mode}` : ''}</strong>
        <span>{age === null ? 'Waiting for inference' : `Last update ${age.toFixed(1)} s ago`}</span>
      </div>
      <dl className="mt-1 grid grid-cols-4 gap-x-3 gap-y-1 xl:grid-cols-7">
        {STAGES.map(([key, title]) => <div key={key}><dt className="text-gray-500 dark:text-slate-400">{title}</dt><dd className="font-mono font-semibold text-blue-700 dark:text-sky-300">{value(key)}</dd></div>)}
      </dl>
      {Number.isFinite(timings.ik) && <p className="mt-1 text-gray-500 dark:text-slate-400">Postprocess includes IK {timings.ik.toFixed(1)} ms{Number.isFinite(timings.chunkrl) ? ` · ChunkRL ${timings.chunkrl.toFixed(1)} ms` : ''}</p>}
      {telemetry?.chunkrl_decisions?.length > 0 && <p className="mt-1 text-gray-500 dark:text-slate-400">{telemetry.chunkrl_decisions.map((d) => `${d.arm}: ${d.switched ? 'switch' : 'keep'} @${d.index}`).join(' / ')}</p>}
      {telemetry?.success === false && <p className="mt-1 text-red-600 dark:text-red-300" role="status">{telemetry.message || 'Inference request failed'}</p>}
      <p className="mt-1 text-[10px] text-gray-400 dark:text-slate-400">{telemetry?.source === 'dataset'
        ? 'Dataset STEP includes feedback frames and image responses; no Main buffer sample.'
        : 'Request includes engine transport; Buffer includes action interpolation. Total = Request + Buffer.'} Last sample, including warm-up.</p>
    </section>
  );
}
