import { configureStore } from '@reduxjs/toolkit';
import configureReducer from '../features/configureSlice';

export default configureStore({
  reducer: {
    configuration: configureReducer
  },
})