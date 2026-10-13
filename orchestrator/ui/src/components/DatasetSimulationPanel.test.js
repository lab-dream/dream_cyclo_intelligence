import React from 'react';
import { configureStore } from '@reduxjs/toolkit';
import { Provider } from 'react-redux';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import DatasetSimulationPanel, { DatasetSimulationImages } from './DatasetSimulationPanel';
import taskReducer from '../features/tasks/taskSlice';
import { InferencePhase } from '../constants/taskPhases';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

jest.mock('../hooks/useRosServiceCaller', () => ({ useRosServiceCaller: jest.fn() }));

function setup(overrides = {}) {
  const initial = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({ reducer: { tasks: taskReducer }, preloadedState: { tasks: {
    ...initial, robotType: 'ffw_sg2_rev1',
    inferenceTaskInfo: { ...initial.inferenceTaskInfo, serviceType: 'lerobot', policyPath: '/models/482000', inferenceMode: 'simulation', ...overrides },
    inferenceStatus: { ...initial.inferenceStatus, inferencePhase: InferencePhase.READY },
  } } });
  const onFrame = jest.fn();
  const onSessionChange = jest.fn();
  const onEnabledChange = jest.fn();
  const callService = jest.fn().mockImplementation(async (_name, _type, request) => ({
    success: true,
    simulation_state: JSON.stringify(request.command === 5 ? { unloaded: true } : {
      episode: 0, frame: request.command === 3 ? 0 : 2, length: 3, fps: 1000,
      finished: request.command === 4,
      frames: request.command === 3 ? [{ frame: 0, joint_state: { name: ['lift_joint'], position: [-0.2] } }] : [
        { frame: 1, joint_state: { name: ['lift_joint'], position: [-0.1] } },
        { frame: 2, joint_state: { name: ['lift_joint'], position: [0] } },
      ],
    }),
  }));
  useRosServiceCaller.mockReturnValue({ callService });
  render(<Provider store={store}><DatasetSimulationPanel enabled onEnabledChange={onEnabledChange} onFrame={onFrame} onSessionChange={onSessionChange} /></Provider>);
  return { callService, onFrame, onSessionChange, onEnabledChange };
}

test('turning off during a running chunk waits, unloads, and restores the normal mode', async () => {
  const { callService, onFrame, onSessionChange, onEnabledChange } = setup();
  fireEvent.change(screen.getByLabelText('Simulation dataset path'), { target: { value: '/workspace/dataset/task' } });
  fireEvent.click(screen.getByRole('button', { name: 'Load / Reset' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Run' })).toBeEnabled());
  let finishChunk;
  callService.mockImplementationOnce(() => new Promise((resolve) => { finishChunk = resolve; }));
  fireEvent.click(screen.getByRole('button', { name: 'Run' }));
  fireEvent.click(screen.getByRole('checkbox', { name: 'Dataset feedback simulation' }));
  expect(screen.getByText('Finishing chunk and disabling…')).toBeInTheDocument();
  expect(callService.mock.calls.map(([, , request]) => request.command)).toEqual([3, 4]);
  expect(onEnabledChange).not.toHaveBeenCalled();
  finishChunk({ success: true, simulation_state: JSON.stringify({ episode: 0, frame: 1, length: 3, fps: 30, frames: [] }) });
  await waitFor(() => expect(onEnabledChange).toHaveBeenCalledWith(false));
  expect(callService.mock.calls.map(([, , request]) => request.command)).toEqual([3, 4, 5]);
  expect(onSessionChange).toHaveBeenLastCalledWith(false);
  expect(onFrame).toHaveBeenLastCalledWith(null);
});

test('load uses existing policy path; chunk playback displays every IK state and then unloads', async () => {
  const { callService, onFrame, onSessionChange } = setup({ ikBackend: 'nodeik', ikModelPath: '/models/nodeik/pose' });
  fireEvent.change(screen.getByLabelText('Simulation dataset path'), { target: { value: '/workspace/dataset/task' } });
  fireEvent.click(screen.getByRole('button', { name: 'Load / Reset' }));
  await waitFor(() => expect(onSessionChange).toHaveBeenCalledWith(true));
  expect(callService).toHaveBeenCalledWith('/lerobot/engine_command', 'interfaces/srv/EngineCommand', expect.objectContaining({
    command: 3, model_path: '/models/482000', robot_type: 'ffw_sg2_rev1',
    ik_backend: 'nodeik', ik_model_path: '/models/nodeik/pose',
    simulation_config: JSON.stringify({ dataset_path: '/workspace/dataset/task', episode: 0, frame: 0, control_hz: 100 }),
  }), 0);
  fireEvent.click(screen.getByRole('button', { name: 'Step chunk' }));
  expect(await screen.findByText(/frame 2\/2.*finished/)).toBeInTheDocument();
  expect(onFrame.mock.calls.map(([frame]) => frame.frame)).toEqual([0, 1, 2]);
  expect(onFrame.mock.calls[2][0].joint_state.position).toEqual([0]);
  fireEvent.click(screen.getByRole('button', { name: 'Unload' }));
  await waitFor(() => expect(onSessionChange).toHaveBeenCalledWith(false));
  expect(onFrame).toHaveBeenLastCalledWith(null);
  expect(callService.mock.calls.every(([, , request]) => !request.publish_to_robot)).toBe(true);
});

test('robot deployment cannot load a dataset simulation', () => {
  setup({ inferenceMode: 'robot' });
  expect(screen.getByRole('button', { name: 'Load / Reset' })).toBeDisabled();
});

test('NodeIK subframes use their control timing and forward the selected control rate', async () => {
  const { callService, onFrame } = setup({ ikBackend: 'nodeik', controlHz: 80 });
  fireEvent.change(screen.getByLabelText('Simulation dataset path'), { target: { value: '/workspace/dataset/task' } });
  fireEvent.click(screen.getByRole('button', { name: 'Load / Reset' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Step chunk' })).toBeEnabled());
  expect(JSON.parse(callService.mock.calls[0][2].simulation_config).control_hz).toBe(80);
  callService.mockResolvedValueOnce({ success: true, simulation_state: JSON.stringify({
    episode: 0, frame: 1, length: 3, fps: 30, ik_backend: 'nodeik',
    frames: [{ frame: 0, dt_s: 0.001 }, { frame: 1, dt_s: 0.002 }],
  }) });
  jest.useFakeTimers();
  const timer = jest.spyOn(global, 'setTimeout');
  try {
    fireEvent.click(screen.getByRole('button', { name: 'Step chunk' }));
    await screen.findByText(/frame 1\/2.*nodeik/);
    expect(onFrame.mock.calls.map(([frame]) => frame.frame)).toEqual([0, 0, 1]);
    expect(timer).toHaveBeenCalledWith(expect.any(Function), 1);
    expect(timer).toHaveBeenCalledWith(expect.any(Function), 2);
    expect(timer).not.toHaveBeenCalledWith(expect.any(Function), 1000 / 30);
  } finally {
    timer.mockRestore();
    jest.useRealTimers();
  }
});

test('dataset images use the supplied frame instead of live streams', () => {
  render(<DatasetSimulationImages frame={{ frame: 12, images: { cam_left_wrist: 'data:image/jpeg;base64,example' } }} />);
  expect(screen.getByAltText('cam_left_wrist')).toHaveAttribute('src', 'data:image/jpeg;base64,example');
  expect(screen.getByText(/frame 12/)).toBeInTheDocument();
});
