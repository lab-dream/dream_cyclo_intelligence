import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import InferencePanel from './InferencePanel';
import taskReducer from '../features/tasks/taskSlice';
import { InferencePhase } from '../constants/taskPhases';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    error: jest.fn(),
    success: jest.fn(),
  },
}));

jest.mock('../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: jest.fn(),
}));

jest.mock('./InferenceModelSelector', () => () => <div />);
jest.mock('./PolicyBackendControl', () => () => <div />);
jest.mock('./TrtEngineControl', () => () => <div />);
jest.mock('./FileBrowserModal', () => () => null);
jest.mock('./Tooltip', () => ({ children }) => <>{children}</>);

const renderPanel = ({
  inferenceMode,
  inferencePhase = InferencePhase.READY,
  initialPoseSync = true,
  inferenceHz = 15,
  controlHz = 100,
  policyType = 'act',
} = {}) => {
  const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
  useRosServiceCaller.mockReturnValue({ sendRecordCommand });
  const initialTasks = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: { tasks: taskReducer },
    preloadedState: {
      tasks: {
        ...initialTasks,
        inferenceTaskInfo: {
          ...initialTasks.inferenceTaskInfo,
          inferenceMode,
          initialPoseSync,
          initialPoseSyncDurationS: 5.0,
          inferenceHz,
          controlHz,
          policyType,
        },
        taskInfo: {
          ...initialTasks.taskInfo,
          inferenceMode,
          initialPoseSync,
          initialPoseSyncDurationS: 5.0,
          inferenceHz,
          controlHz,
        },
        inferenceStatus: {
          ...initialTasks.inferenceStatus,
          inferencePhase,
        },
      },
    },
  });

  render(
    <Provider store={store}>
      <InferencePanel />
    </Provider>
  );
  return { store, sendRecordCommand };
};

describe('InferencePanel initial pose sync settings', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('selects NodeIK and syncs its model path with the inference task', async () => {
    jest.useFakeTimers();
    const { store, sendRecordCommand } = renderPanel({ inferenceMode: 'simulation', policyType: 'diffusion' });
    expect(screen.getByLabelText('EEF IK backend')).toHaveValue('cyclo_control');
    fireEvent.change(screen.getByLabelText('EEF IK backend'), { target: { value: 'nodeik' } });
    fireEvent.change(screen.getByLabelText('NodeIK model path'), { target: { value: '/models/nodeik/pose' } });
    expect(store.getState().tasks.inferenceTaskInfo).toMatchObject({ ikBackend: 'nodeik', ikModelPath: '/models/nodeik/pose' });
    await act(async () => { jest.runOnlyPendingTimers(); });
    expect(sendRecordCommand).toHaveBeenCalledWith('set_task_info', { autofillEmptyTaskFields: false });
    jest.useRealTimers();
  });

  test('selects DLS for diffusion without requiring an IK checkpoint', async () => {
    jest.useFakeTimers();
    const { store, sendRecordCommand } = renderPanel({ inferenceMode: 'simulation', policyType: 'diffusion' });
    fireEvent.change(screen.getByLabelText('EEF IK backend'), { target: { value: 'dls' } });
    expect(store.getState().tasks.inferenceTaskInfo.ikBackend).toBe('dls');
    expect(screen.queryByLabelText('NodeIK model path')).not.toBeInTheDocument();
    expect(screen.getByText(/Keeps the starting elbow posture/)).toBeInTheDocument();
    await act(async () => { jest.runOnlyPendingTimers(); });
    expect(sendRecordCommand).toHaveBeenCalledWith('set_task_info', { autofillEmptyTaskFields: false });
    jest.useRealTimers();
  });

  test('does not expose DLS settings for ACT', () => {
    renderPanel({ policyType: 'act' });
    expect(screen.queryByLabelText('EEF IK backend')).not.toBeInTheDocument();
  });

  test('locks DLS selection during inference', () => {
    renderPanel({ policyType: 'diffusion', inferencePhase: InferencePhase.INFERENCING });
    expect(screen.getByLabelText('EEF IK backend')).toBeDisabled();
  });

  test('applies the local diffusion checkpoint and syncs its dataset FPS', async () => {
    jest.useFakeTimers();
    const { store, sendRecordCommand } = renderPanel({
      inferenceMode: 'simulation', policyType: 'diffusion',
    });
    fireEvent.click(screen.getByRole('button', { name: 'Use Task 0 · 964000 (30 FPS)' }));
    expect(store.getState().tasks.inferenceTaskInfo).toMatchObject({
      policyPath: '/workspace/model/lerobot/diffusion_task0_964000/pretrained_model',
      inferenceHz: 30, controlHz: 100, actionRequestMode: 'async',
      inferenceMode: 'simulation',
    });
    await act(async () => { jest.advanceTimersByTime(700); });
    expect(sendRecordCommand).toHaveBeenCalledWith('set_task_info', {
      autofillEmptyTaskFields: false,
    });
    jest.useRealTimers();
  });

  test('locks the diffusion preset while a model is active', () => {
    renderPanel({ policyType: 'diffusion', inferencePhase: InferencePhase.INFERENCING });
    expect(screen.getByRole('button', { name: 'Use Task 0 · 964000 (30 FPS)' })).toBeDisabled();
  });

  test('preserves but disables initial pose sync in simulation mode', () => {
    renderPanel({ inferenceMode: 'simulation' });

    expect(screen.getByRole('checkbox', { name: 'Initial Pose Sync' }))
      .toBeChecked();
    expect(screen.getByRole('checkbox', { name: 'Initial Pose Sync' }))
      .toBeDisabled();
    expect(screen.getByRole('spinbutton', { name: 'Initial Pose Sync duration' }))
      .toBeDisabled();
  });

  test('allows initial pose sync editing for an idle real robot session', () => {
    renderPanel({ inferenceMode: 'robot' });

    expect(screen.getByRole('checkbox', { name: 'Initial Pose Sync' }))
      .toBeEnabled();
    expect(screen.getByRole('spinbutton', { name: 'Initial Pose Sync duration' }))
      .toBeEnabled();
  });

  test('shows duration only after initial pose sync is enabled', () => {
    renderPanel({ inferenceMode: 'robot', initialPoseSync: false });

    expect(screen.queryByRole('spinbutton', { name: 'Initial Pose Sync duration' }))
      .not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('checkbox', { name: 'Initial Pose Sync' }));

    expect(screen.getByRole('spinbutton', { name: 'Initial Pose Sync duration' }))
      .toBeEnabled();
  });

  test('makes initial pose sync settings read-only while synchronizing', () => {
    renderPanel({
      inferenceMode: 'robot',
      inferencePhase: InferencePhase.SYNCING,
    });

    expect(screen.getByRole('checkbox', { name: 'Initial Pose Sync' }))
      .toBeDisabled();
    expect(screen.getByRole('spinbutton', { name: 'Initial Pose Sync duration' }))
      .toBeDisabled();
  });

  test('keeps Dataset FPS blank while the user replaces its value', () => {
    jest.useFakeTimers();
    const { sendRecordCommand } = renderPanel({ inferenceMode: 'robot' });
    const datasetFpsInput = screen.getByRole('spinbutton', { name: 'Dataset FPS' });

    fireEvent.change(datasetFpsInput, { target: { value: '' } });
    expect(datasetFpsInput).toHaveValue(null);

    act(() => {
      jest.advanceTimersByTime(1000);
    });
    expect(datasetFpsInput).toHaveValue(null);
    expect(sendRecordCommand).not.toHaveBeenCalled();
    jest.useRealTimers();
  });

  test('shows a non-blocking warning for unusual Dataset FPS', () => {
    renderPanel({ inferenceMode: 'robot', inferenceHz: 1515, controlHz: 200 });

    expect(screen.getByRole('status', { name: 'Timing warnings' }))
      .toHaveTextContent('Dataset FPS is unusually high (1515)');
  });
});


describe('ChunkRL inference mode', () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => { jest.clearAllTimers(); jest.useRealTimers(); });

  test('selects ChunkRL and sends its checkpoint through task synchronization', async () => {
    const { store, sendRecordCommand } = renderPanel({ policyType: 'diffusion' });
    fireEvent.click(screen.getByRole('button', { name: 'Use ChunkRL action requests' }));
    expect(store.getState().tasks.inferenceTaskInfo.actionRequestMode).toBe('chunkrl');
    const input = screen.getByLabelText('ChunkRL checkpoint');
    fireEvent.change(input, { target: { value: '/workspace/model/chunkrl/custom.pt' } });
    await act(async () => { jest.advanceTimersByTime(700); });
    expect(store.getState().tasks.inferenceTaskInfo.chunkrlCheckpoint).toBe('/workspace/model/chunkrl/custom.pt');
    expect(sendRecordCommand).toHaveBeenCalled();
    fireEvent.click(screen.getByText('Use local relative Diffusion model'));
    expect(store.getState().tasks.inferenceTaskInfo.policyPath).toContain('diffusion_task0_relative_482000');
    expect(store.getState().tasks.inferenceTaskInfo.inferenceHz).toBe(30);
  });

  test('keeps ChunkRL unavailable to non-Diffusion policies', () => {
    renderPanel({ policyType: 'act' });
    expect(screen.queryByRole('button', { name: 'Use ChunkRL action requests' })).not.toBeInTheDocument();
  });

  test('locks ChunkRL mode while a model is active', () => {
    renderPanel({ policyType: 'diffusion', inferencePhase: InferencePhase.INFERENCING });
    expect(screen.getByRole('button', { name: 'Use ChunkRL action requests' })).toBeDisabled();
  });
});
