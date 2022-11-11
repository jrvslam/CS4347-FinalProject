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
                      noteData={[
                        ["0:0:0", "F5", ""],
                        ["0:0:0", "C4", "2n"],
                        ["0:0:0", "D4", "2n"],
                        ["0:0:0", "E4", "2n"],
                        ["0:2:0", "B4", "4n"],
                        ["0:3:0", "A#4", "4n"],
                        ["0:0:0", "F2", ""],
                      ]}
                    />
                  </div>
        }
    </div>
  );
}
