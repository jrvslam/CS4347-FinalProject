import React, { useEffect, useState } from "react";
import { useSelector } from 'react-redux'
import Plot from 'react-plotly.js';

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
                  <Plot
                    data={[
                      {
                        x,
                        y,
                        type: 'scatter',
                      },
                    ]}
                    layout={{width: 500, height: 350, title: 'Melody'}}
                  />
                  <p> Lyrics: {lyricRes}</p>
                  </div>
        }
    </div>
  );
}
