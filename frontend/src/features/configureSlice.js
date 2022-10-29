import { createSlice } from '@reduxjs/toolkit'

export const configureSlice = createSlice({
    name: 'configuration',
    initialState: {
        processState: 0,
        melodyRes: {},
        lyricRes: {}
    },
    reducers: {
        setProcessState: {
            reducer: (state, action) => {
                const step = action.payload;
                state.processState = step;
            },
            prepare: (step) => {
                return { payload: step };
            }
        },
        setMelodyRes: {
            reducer: (state, action) => {
                const result = action.payload;
                state.melodyRes = result;
            },
            prepare: (result) => {
                return { payload: result };
            }
        },
        setLyricRes: {
            reducer: (state, action) => {
                const result = action.payload;
                state.lyricRes = result;
            },
            prepare: (result) => {
                return { payload: result };
            }
        }
    }
})


export const { setProcessState, setMelodyRes, setLyricRes } = configureSlice.actions

export default configureSlice.reducer