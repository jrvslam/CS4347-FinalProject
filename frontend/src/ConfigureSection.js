import React, {useState, useEffect} from "react";
import { useSelector, useDispatch } from 'react-redux'
import { setProcessState, setMelodyRes, setLyricRes } from "./features/configureSlice";
import { Button } from "@mui/material";
import LoadingButton from '@mui/lab/LoadingButton';
import FileUploadIcon from "@mui/icons-material/FileUpload";
let bpm = 150

var a;
var processed_piano=[];
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

export function ConfigureSection(getMelodyOutput, getLyricsOutput) {
    const configuration = useSelector((state) => state.configuration);
    const dispatch = useDispatch();

    const [audio, setAudio] = useState();
    const [selectedFile, setSelectedFile] = useState();
    const [isSelected, setIsSelected] = useState(false);
    const [buttonName, setButtonName] = useState("Play");

    const changeHandler = (event) => {
        if (event.target.files[0]) {
            setAudio(URL.createObjectURL(event.target.files[0]));
            setSelectedFile(event.target.files[0]);
            setIsSelected(true);
        }
    }

    useEffect(() => {
        if (a) {
          a.pause();
          a = null;
          setButtonName("Play");
        }
        if (audio) {
          a = new Audio(audio);
          a.onended = () => {
            setButtonName("Play");
          };
        }
      }, [audio]);

    const handlePlay = () => {
        if (buttonName === "Play") {
            a.play();
            setButtonName("Pause");
        } else {
            a.pause();
            setButtonName("Play");
        }
    }

    const handleProcess = () => {
        dispatch(setProcessState(1))
        var melodyRes;
        var lyricRes;
        var doneCheck = false;
        var data = new FormData();
        data.append('file', selectedFile);
        //get Melody Extraction
        fetch(
            "http://localhost:5000/melody",
            {
                method: "POST",
                mode: 'cors',
                headers: {
                    'Access-Control-Allow-Origin':'*'
                },
                body: data
            }
        )
        .then(res => {
            console.log(res);
            if (!doneCheck) {
                doneCheck = !doneCheck;
            } else {
                dispatch(setProcessState(2));
            }
            
            return res.json();
        })
        .then(result => {
            melodyRes = melody_processing(result.result);
            dispatch(setMelodyRes(melodyRes));
            
        })

        //get Lyric Extraction
        fetch(
            "http://localhost:5000/lyrics",
            {
                method: "POST",
                mode: 'cors',
                headers: {
                    'Access-Control-Allow-Origin':'*'
                },
                body: data
            }
        )
        .then(res => {
            if (!doneCheck) {
                doneCheck = !doneCheck;
            } else {
                dispatch(setProcessState(2));
            }

            return res.json();
        })
        .then(result => {
            lyricRes = result.text;
            dispatch(setLyricRes(lyricRes));
            
        })
    };


  return (
    <div>
      <h3>Upload your music file</h3>
      {/* <input type="file" name="file" onChange={changeHandler} /> */}
      <Button
        variant="contained"
        component="label"
        style={{
          backgroundColor: "#ec66ca",
        }}
        startIcon={<FileUploadIcon />}
      >
        Upload
        <input
            type="file"
            hidden
            onChange={changeHandler}
            name="file"
        />
      </Button>
      {isSelected ? (
        <div>
          <p style={{ wordBreak: "break-word" }}>
            Filename: {selectedFile.name}
          </p>
        </div>
      ) : (
            <p>Select a file to show details</p>
      )}
      <Button
        variant="contained"
        style={{
          backgroundColor: "#ec66ca",
        }}
        onClick={handlePlay}
      >
        {buttonName}
      </Button>

      <h3>Analyse your music file</h3>
      <LoadingButton
        variant="contained"
        style={{
          backgroundColor: "#ec66ca",
        }}
        onClick={handleProcess}
        loading={configuration.processState === 1}
      >
        Process
      </LoadingButton>
    </div>
  );
}
