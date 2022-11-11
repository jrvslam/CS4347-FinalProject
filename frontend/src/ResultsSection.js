import React, { useEffect, useState } from "react";
import { useSelector } from 'react-redux';
import Plot from 'react-plotly.js';
import PianoRoll from "./components/PianoRoll";

export function ResultsSection() {
  const processState = useSelector((state) => state.configuration.processState);
  const melodyRes = useSelector((state) => state.configuration.melodyRes);
  const lyricRes = useSelector((state) => state.configuration.lyricRes);
  const [x, setX] = useState([]);
  const [y, setY] = useState([]);

  useEffect(() => {
    const newX = [];
    const newY = [];
    melodyRes.forEach((note) => {
      console.log(note);
      newX.push(note[0]);
      newX.push(note[1]);
      newY.push(note[2]);
      newY.push(note[2]);
    })
    setX(newX);
    setY(newY);
  }, [melodyRes]);

  return (
    <div>
        <h2>Results</h2>
        {
            (processState !== 2) 
                ? <p> Start Analysing some music!</p> 
                : <div>
                  <p> Lyrics: {lyricRes}</p>
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
                    <Plot
                    data={[
                      {
                        x,
                        y,
                        type: 'scatter',
                      },
                    ]}
                    layout={{
                      width: 500,
                      height: 350,
                      title: 'Melody',
                      xaxis: {
                        title: {
                          text: 'Time (s)',
                        },
                      },
                      yaxis: {
                        title: {
                          text: 'MIDI Note',
                        }
                      }
                    }}
                  />
                  </div>
        }
    </div>
  );
}
