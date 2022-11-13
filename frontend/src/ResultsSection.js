import React, { useEffect, useState, useRef } from "react";
import { useSelector } from "react-redux";
import Plot from "react-plotly.js";
import PianoRoll from "./components/PianoRoll";
import {
  Box,
  Button,
  FormControl,
  FormControlLabel,
  FormLabel,
  Radio,
  RadioGroup,
  Stack,
} from "@mui/material";

const smallest = 1 / 16;
var defbpm = 150;
var defzoom = 2;
var a;

function bar_maker_start(sixcount) {
  let bcount = Math.floor(sixcount / 16).toString();
  sixcount = sixcount % 16;
  let qcount = Math.floor(sixcount / 4).toString();
  let unit = (sixcount % 4).toString();
  return bcount.concat(":", qcount, ":", unit);
}

function melodyProcessing(rawMelData) {
  const processedPiano = [];
  let noteDurArr = [];
  let smallestNoteDur = Number.MAX_VALUE;
  let lowestNote = Number.MAX_VALUE;
  let highestNote = 0;
  let prevOffset = 0;
  // normalize notes
  for (let i = 0; i < rawMelData.length; i++) {
    let onset = rawMelData[i][0] * 10;
    let offset = rawMelData[i][1] * 10;
    let note = rawMelData[i][2];
    onset = onset.toFixed(0);
    offset = offset.toFixed(0);

    let noteBetween = onset - prevOffset;
    if (noteBetween > 10) {
      noteDurArr.push([noteBetween, 0]);
    }
    prevOffset = offset;

    let noteDur = offset - onset;
    // noteDur is length of each note in seconds * 10
    if (noteDur < smallestNoteDur) {
      smallestNoteDur = noteDur;
    }

    if (note < lowestNote) {
      lowestNote = note;
    }
    if (note > highestNote) {
      highestNote = note;
    }

    noteDurArr.push([noteDur, note]);
  }

  //assume smallestNoteDur is a sixteenth note
  //convert noteDurArr from arr of noteDur to arr of (total sixteenth notes per note length i.e if note is crochet, value is 4)
  let totalSixteenthNotes = 0;
  for (let i = 0; i < noteDurArr.length; i++) {
    let normNoteDur = noteDurArr[i][0] / smallestNoteDur;
    normNoteDur = normNoteDur.toFixed(0);

    //convert to required note notation
    for (let j = 0; j < normNoteDur; j++) {
      let currIdx = totalSixteenthNotes + j;
      let noteDurStr = "0:0:" + currIdx;

      if (noteDurArr[i][1] !== 0) {
        processedPiano.push([noteDurStr, noteDurArr[i][1], "16n"]);
      }
    }

    totalSixteenthNotes = parseInt(totalSixteenthNotes) + parseInt(normNoteDur);
  }
  processedPiano.push(["0:0:0", parseInt(highestNote) + 2, ""]);
  processedPiano.push(["0:0:0", parseInt(lowestNote) - 2, ""]);
  if (rawMelData.length > 0) {
    defbpm =
      (Math.ceil([processedPiano.length - 3][0] / 4) /
        rawMelData[rawMelData.length - 1][1]) *
      60;
    defzoom = Math.ceil([processedPiano.length - 3][0] / 16);
  }
  console.log("new mel proc: ", processedPiano);
  return processedPiano;
}

function melody_processing(testing) {
  const processed_piano = [];
  for (let i = 0; i < testing.length; i++) {
    let onset = (testing[i][0] / smallest).toFixed(0);
    const offset = (testing[i][1] / smallest).toFixed(0);
    const note = testing[i][2];

    while (offset - onset >= 16) {
      processed_piano.push([bar_maker_start(onset), note, 1]);
      onset = onset + 16;
    }

    switch (offset - onset) {
      case 15:
        processed_piano.push([bar_maker_start(onset), note, "2n"]);
        processed_piano.push([bar_maker_start(onset + 8), note, "4n"]);
        processed_piano.push([bar_maker_start(onset + 12), note, "8n"]);
        processed_piano.push([bar_maker_start(onset + 14), note, "16n"]);
        break;
      case 14:
        processed_piano.push([bar_maker_start(onset), note, "2n"]);
        processed_piano.push([bar_maker_start(onset + 8), note, "4n"]);
        processed_piano.push([bar_maker_start(onset + 12), note, "8n"]);
        break;
      case 13:
        processed_piano.push([bar_maker_start(onset), note, "2n"]);
        processed_piano.push([bar_maker_start(onset + 8), note, "4n"]);
        processed_piano.push([bar_maker_start(onset + 12), note, "16n"]);
        break;
      case 12:
        processed_piano.push([bar_maker_start(onset), note, "2n"]);
        processed_piano.push([bar_maker_start(onset + 8), note, "4n"]);
        break;
      case 11:
        processed_piano.push([bar_maker_start(onset), note, "2n"]);
        processed_piano.push([bar_maker_start(onset + 8), note, "8n"]);
        processed_piano.push([bar_maker_start(onset + 10), note, "16n"]);
        break;
      case 10:
        processed_piano.push([bar_maker_start(onset), note, "2n"]);
        processed_piano.push([bar_maker_start(onset + 8), note, "8n"]);
        break;
      case 9:
        processed_piano.push([bar_maker_start(onset), note, "2n"]);
        processed_piano.push([bar_maker_start(onset + 8), note, "16n"]);
        break;
      case 8:
        processed_piano.push([bar_maker_start(onset), note, "2n"]);
        break;
      case 7:
        processed_piano.push([bar_maker_start(onset), note, "4n"]);
        processed_piano.push([bar_maker_start(onset + 4), note, "8n"]);
        processed_piano.push([bar_maker_start(onset + 2), note, "16n"]);
        break;
      case 6:
        processed_piano.push([bar_maker_start(onset), note, "4n"]);
        processed_piano.push([bar_maker_start(onset + 4), note, "8n"]);
        break;
      case 5:
        processed_piano.push([bar_maker_start(onset), note, "4n"]);
        processed_piano.push([bar_maker_start(onset + 2), note, "16n"]);
        break;
      case 4:
        processed_piano.push([bar_maker_start(onset), note, "4n"]);
        break;
      case 3:
        processed_piano.push([bar_maker_start(onset), note, "8n"]);
        processed_piano.push([bar_maker_start(onset + 2), note, "16n"]);
        break;
      case 2:
        processed_piano.push([bar_maker_start(onset), note, "8n"]);
        break;
      case 1:
        processed_piano.push([bar_maker_start(onset), note, "16n"]);
        break;
      default:
        break;
    }
  }

  console.log("processed: ", processed_piano);
  return processed_piano;
}

export function ResultsSection() {
  const processState = useSelector((state) => state.configuration.processState);
  const melodyRes = useSelector((state) => state.configuration.melodyRes);
  const lyricRes = useSelector((state) => state.configuration.lyricRes);
  const audioURL = useSelector((state) => state.configuration.audioURL);
  const [x, setX] = useState([]);
  const [y, setY] = useState([]);
  const [melodyData, setMelodyData] = useState([]);
  const [display, setDisplay] = useState("piano");
  const [audio, setAudio] = useState();
  const playbackRef = useRef();

  useEffect(() => {
    const newX = [];
    const newY = [];
    melodyRes.forEach((note) => {
      newX.push(note[0]);
      newX.push(note[1]);
      newY.push(note[2]);
      newY.push(note[2]);
      newX.push(null);
      newY.push(null);
    });
    setX(newX);
    setY(newY);
    setMelodyData(melodyProcessing(melodyRes));
    defbpm = parseInt(defbpm);
    console.log("bpm: " + defbpm + ", zoom: " + defzoom);
  }, [melodyRes]);

  useEffect(() => {
    setAudio(audioURL);
  }, [audioURL]);

  useEffect(() => {
    if (a) {
      a.pause();
      a = null;
    }
    if (audio) {
      a = new Audio(audio);
      a.onended = () => {
        playbackRef.current.seek("0:0:0");
        playbackRef.current.pause();
      };
    }
  }, [audio]);

  const handlePlay = () => {
    a.play();
    playbackRef.current.play();
  };

  const handlePause = () => {
    a.pause();
    playbackRef.current.pause();
  };

  const handleReset = () => {
    a.pause();
    a.currentTime = 0;
    playbackRef.current.seek("0:0:0");
    playbackRef.current.pause();
  };

  const handleChange = (event) => setDisplay(event.target.value);

  const pianoRoll = (
    <div>
      <PianoRoll
        bpm={defbpm}
        width={600}
        height={350}
        zoom={defzoom}
        // resolution={2}
        gridLineColor={0x333333}
        blackGridBgColor={0x1e1e1e}
        whiteGridBgColor={0x282828}
        noteFormat="MIDI"
        noteData={melodyData}
        ref={playbackRef}
      />
      <Stack
        direction="row"
        spacing={2}
        alignItems="center"
        justifyContent="center"
      >
        <Button
          variant="contained"
          style={{
            backgroundColor: "#ec66ca",
          }}
          onClick={() => {
            handlePlay();
          }}
        >
          Play
        </Button>
        <Button
          variant="contained"
          style={{
            backgroundColor: "#ec66ca",
          }}
          onClick={() => {
            handlePause();
          }}
        >
          Pause
        </Button>
        <Button
          variant="contained"
          style={{
            backgroundColor: "#ec66ca",
          }}
          onClick={() => {
            handleReset();
          }}
        >
          Reset
        </Button>
      </Stack>
    </div>
  );

  const graph = (
    <Plot
      data={[
        {
          x,
          y,
          type: "scatter",
        },
      ]}
      layout={{
        width: 600,
        height: 350,
        title: "Melody",
        xaxis: {
          title: {
            text: "Time (s)",
          },
        },
        yaxis: {
          title: {
            text: "MIDI Note",
          },
        },
      }}
    />
  );

  return (
    <div>
      <Box display="flex" flexDirection="row" justifyContent="space-between">
        <Box display="flex" width="55%" justifyContent="flex-end">
          <h2>Results</h2>
        </Box>
      </Box>
      {processState === 0 ? (
        <p>Start Analysing some music!</p>
      ) : processState === 1 ? (
        <p>Processing...</p>
      ) : (
        <div>
          <Stack
            direction="column"
            spacing={2}
            alignItems="center"
            justifyContent="center"
          >
            <p> Lyrics: {lyricRes}</p>
            <p> Melody: </p>
            {display === "piano" ? pianoRoll : graph}
            <FormControl justifyContent="center" alignItems="flex-start">
              <FormLabel>Display</FormLabel>
              <RadioGroup
                value={display}
                onChange={handleChange}
                row
                width="auto"
                justifyContent="center"
                alignItems="flex-start"
              >
                <FormControlLabel
                  value="piano"
                  control={<Radio />}
                  label="Piano Roll"
                />
                <FormControlLabel
                  value="graph"
                  control={<Radio />}
                  label="Graph"
                />
              </RadioGroup>
            </FormControl>
          </Stack>
        </div>
      )}
    </div>
  );
}
