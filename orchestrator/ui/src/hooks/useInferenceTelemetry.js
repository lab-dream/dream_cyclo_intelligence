import { useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import ROSLIB from 'roslib';
import rosConnectionManager from '../utils/rosConnectionManager';

export default function useInferenceTelemetry(backend, robotType, enabled = true) {
  const url = useSelector((state) => state.ros.rosbridgeUrl);
  const [telemetry, setTelemetry] = useState(null);
  useEffect(() => {
    setTelemetry(null);
    if (!enabled || !url || !['lerobot', 'groot'].includes(backend)) return undefined;
    let cancelled = false;
    let topic;
    let latest = 0;
    rosConnectionManager.getConnection(url).then((ros) => {
      if (cancelled || !ros) return;
      topic = new ROSLIB.Topic({ ros, name: `/${backend}/inference_telemetry`, messageType: 'std_msgs/msg/String', throttle_rate: 100 });
      topic.subscribe((message) => {
        try {
          const value = JSON.parse(message.data);
          if (!value || typeof value !== 'object' || !Number.isFinite(value.timestamp_ms) || value.timestamp_ms < latest) return;
          latest = value.timestamp_ms;
          if (value.event === 'reset') { setTelemetry(null); return; }
          setTelemetry({ ...value, receivedAt: Date.now() });
        } catch (error) { /* Ignore malformed diagnostics without interrupting inference. */ }
      });
    }).catch(() => {});
    return () => { cancelled = true; topic?.unsubscribe(); };
  }, [url, backend, robotType, enabled]);
  return telemetry;
}
