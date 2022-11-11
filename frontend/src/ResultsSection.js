import React, { useEffect, useState } from "react";
import { useSelector } from 'react-redux';
import Plot from 'react-plotly.js';
import PianoRoll from "./components/PianoRoll";
import { Box, FormControl, FormControlLabel, FormLabel, Radio, RadioGroup } from "@mui/material";

let bpm = 150

function bar_maker_start(sixcount){
  let bcount = Math.floor(sixcount/16).toString()
  sixcount = sixcount%16
  let qcount = Math.floor(sixcount/4).toString()
  let unit = (sixcount % 4 ). toString()
  return bcount.concat(":",qcount,":",unit)
}

function melody_processing(testing) {
  let startsix=0
  let endssix=0
  let note=0
  const processed_piano = [];
  // console.log(testing.length)
  // console.log(typeof testing)
  // console.log(typeof testing[0])
  for (let i = 0; i < testing.length; i++) {
    // console.log(testing[i])
    testing[i][2]=parseFloat(testing[i][2])
    testing[i][0]=parseFloat(testing[i][0])
    testing[i][1]=parseFloat(testing[i][1])
    startsix=parseFloat(testing[i][0]*10).toFixed(0)
    console.log(startsix)
    endssix=parseFloat(testing[i][1]*10).toFixed(0)
    console.log(endssix)

    note=testing[i][2]
    while( (endssix-startsix) >= 16){
      processed_piano.push([bar_maker_start(startsix), note, 1])
      startsix=startsix+16
    }
    // console.log(endssix-startsix)
    switch(endssix-startsix) {
      case 15:
        processed_piano.push([bar_maker_start(startsix), note, "2n"])
        processed_piano.push([bar_maker_start(startsix+8), note, "4n"])
        processed_piano.push([bar_maker_start(startsix+12), note, "8n"])
        processed_piano.push([bar_maker_start(startsix+14), note, "16n"])
        break;
      case 14:
        processed_piano.push([bar_maker_start(startsix), note, "2n"])
        processed_piano.push([bar_maker_start(startsix+8), note, "4n"])
        processed_piano.push([bar_maker_start(startsix+12), note, "8n"])
        break;
      case 13:
        processed_piano.push([bar_maker_start(startsix), note, "2n"])
        processed_piano.push([bar_maker_start(startsix+8), note, "4n"])
        processed_piano.push([bar_maker_start(startsix+12), note, "16n"])
        break;
      case 12:
          processed_piano.push([bar_maker_start(startsix), note, "2n"])
          processed_piano.push([bar_maker_start(startsix+8), note, "4n"])
          break;
      case 11:
          processed_piano.push([bar_maker_start(startsix), note, "2n"])
          processed_piano.push([bar_maker_start(startsix+8), note, "8n"])
          processed_piano.push([bar_maker_start(startsix+10), note, "16n"])
          break;    
      case 10:
        processed_piano.push([bar_maker_start(startsix), note, "2n"])
        processed_piano.push([bar_maker_start(startsix+8), note, "8n"])
        break;        
      case 9:
        processed_piano.push([bar_maker_start(startsix), note, "2n"])
        processed_piano.push([bar_maker_start(startsix+8), note, "16n"])
        break;    
      case 8:
        processed_piano.push([bar_maker_start(startsix), note, "2n"])
        break;    
      case 7:
        processed_piano.push([bar_maker_start(startsix), note, "4n"])
        processed_piano.push([bar_maker_start(startsix+4), note, "8n"])
        processed_piano.push([bar_maker_start(startsix+2), note, "16n"])
        break;
      case 6:
        processed_piano.push([bar_maker_start(startsix), note, "4n"])
        processed_piano.push([bar_maker_start(startsix+4), note, "8n"])
        break;
      case 5:
        processed_piano.push([bar_maker_start(startsix), note, "4n"])
        processed_piano.push([bar_maker_start(startsix+2), note, "16n"])
        break;
      case 4:
        processed_piano.push([bar_maker_start(startsix), note, "4n"])
        break;
      case 3:
        processed_piano.push([bar_maker_start(startsix), note, "8n"])
        processed_piano.push([bar_maker_start(startsix+2), note, "16n"])
        break;
      case 2:
        processed_piano.push([bar_maker_start(startsix), note, "8n"])
        break;
      case 1:
        processed_piano.push([bar_maker_start(startsix), note, "16n"])
        break;
      }
      // console.log(processed_piano[-1])
  }
  return processed_piano
}

export function ResultsSection() {
  const processState = useSelector((state) => state.configuration.processState);
  const melodyRes = useSelector((state) => state.configuration.melodyRes);
  const lyricRes = useSelector((state) => state.configuration.lyricRes);
  const [x, setX] = useState([]);
  const [y, setY] = useState([]);
  const [pianoData, setPianoData] = useState([]);
  const [display, setDisplay] = useState('piano');

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

    setPianoData(melody_processing(melodyRes));
  }, [melodyRes]);

  const handleChange = (event) => setDisplay(event.target.value);

  const pianoRoll = <PianoRoll
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

  const graph = <Plot
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

  return (
    <div>
        <Box display="flex" flexDirection="row" justifyContent="space-between">
          <Box display="flex" width="55%" justifyContent="flex-end">
            <h2>Results</h2>
          </Box>
          <FormControl>
            <FormLabel>Display</FormLabel>
            <RadioGroup
              value={display}
              onChange={handleChange}
            >
              <FormControlLabel value="piano" control={<Radio />} label="Piano Roll" />
              <FormControlLabel value="graph" control={<Radio />} label="Graph" />
            </RadioGroup>
          </FormControl>
        </Box>
        {
            (processState !== 2) 
                ? <p> Start Analysing some music!</p> 
                : <div>
                  <p> Lyrics: {lyricRes}</p>
                  {display === 'piano' ? pianoRoll : graph}
                  </div>
        }
    </div>
  );
}
