import React, {useState, useEffect} from "react";
import { useSelector, useDispatch } from 'react-redux'
import { setProcessState, setMelodyRes, setLyricRes } from "./features/configureSlice";
import PianoRoll from "./components/PianoRoll";

export function ResultsSection(melody_output, lyrics_output) {
    const configuration = useSelector((state) => state.configuration);
    const dispatch = useDispatch();

  return (
    <div>
        <h2>Results</h2>
        {
            (configuration.processState !== 2) 
                ? <p> Start Analysing some music!</p> 
                : <div>
                    <p> Done Processing! {console.log(melody_output,lyrics_output)}</p>
                    
                    <h2>Lyrics Output: </h2>
                    <h3>{lyrics_output.text}</h3>
                    <PianoRoll
                      bpm={150}
                      width={500}
                      height={350}
                      zoom={6}
                      resolution={2}
                      gridLineColor={0x333333}
                      blackGridBgColor={0x1e1e1e}
                      whiteGridBgColor={0x282828}
                      noteFormat={"MIDI"}
                      noteData={melody_output}
                    />
                  </div>
        }
    </div>
  );
}
