import React from 'react';
import { render, screen } from '@testing-library/react';
import InferenceTimingPanel from './InferenceTimingPanel';

test('shows whole-request stages, zero duration, selected ChunkRL arm and failures', () => {
  const { rerender } = render(<InferenceTimingPanel telemetry={{
    receivedAt: Date.now(), success: true, action_request_mode: 'chunkrl',
    timings_ms: { observation: 30, preprocess: 0, policy: 80, postprocess: 12, ik: 10, chunkrl: 2, request: 125, buffer: 3, total: 128 },
    chunkrl_decisions: [{ arm: 'right', switched: true, index: 3 }],
  }} />);
  expect(screen.getByText('0.0 ms')).toBeInTheDocument();
  expect(screen.getByText('128.0 ms')).toBeInTheDocument();
  expect(screen.getByText('right: switch @3')).toBeInTheDocument();
  expect(screen.getByText(/Postprocess includes IK 10.0 ms/)).toBeInTheDocument();
  rerender(<InferenceTimingPanel telemetry={{ receivedAt: Date.now(), success: false, message: 'IK target rejected' }} />);
  expect(screen.getByRole('status')).toHaveTextContent('IK target rejected');
  rerender(<InferenceTimingPanel telemetry={null} />);
  expect(screen.getByText('Waiting for inference')).toBeInTheDocument();
});
