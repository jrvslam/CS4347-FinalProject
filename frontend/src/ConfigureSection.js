import React, {useState, useEffect} from "react";
import { useSelector, useDispatch } from 'react-redux'
import { setProcessState, setMelodyRes, setLyricRes } from "./features/configureSlice";

import { Box } from "@mui/system";
import { Button, Paper } from "@mui/material";
import LoadingButton from '@mui/lab/LoadingButton';
import FileUploadIcon from "@mui/icons-material/FileUpload";

var a;

export function ConfigureSection() {
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
            console.log("Results: ", result.result);
            melodyRes = result.result;
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
            console.log(res);
            if (!doneCheck) {
                doneCheck = !doneCheck;
            } else {
                dispatch(setProcessState(2));
            }

            return res.json();
        })
        .then(result => {
            console.log("Results: ", result.result);
            lyricRes = result.result;
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
            <p>Filename: {selectedFile.name}</p>
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
