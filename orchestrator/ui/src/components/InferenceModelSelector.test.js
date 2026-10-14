import { configureStore } from '@reduxjs/toolkit';
import { fireEvent, render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import InferenceModelSelector from './InferenceModelSelector';
import taskReducer from '../features/tasks/taskSlice';

test('switching away from Diffusion clears a DLS backend that other policies cannot use', () => {
  const tasks = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: { tasks: taskReducer },
    preloadedState: { tasks: {
      ...tasks,
      inferenceTaskInfo: { ...tasks.inferenceTaskInfo, serviceType: 'lerobot', policyType: 'diffusion', ikBackend: 'dls' },
    } },
  });
  render(<Provider store={store}><InferenceModelSelector /></Provider>);
  fireEvent.change(screen.getByRole('combobox'), { target: { value: 'lerobot:act' } });
  expect(store.getState().tasks.inferenceTaskInfo).toMatchObject({ policyType: 'act', ikBackend: 'cyclo_control' });
});
