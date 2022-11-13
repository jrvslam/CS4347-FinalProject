import React, { useState } from "react";
import { useSelector, useDispatch } from "react-redux";
import {
  setProcessState,
  setMelodyRes,
  setLyricRes,
  setAudioURL,
} from "./features/configureSlice";
import { Button } from "@mui/material";
import LoadingButton from "@mui/lab/LoadingButton";
import FileUploadIcon from "@mui/icons-material/FileUpload";

export function ConfigureSection() {
  const configuration = useSelector((state) => state.configuration);
  const dispatch = useDispatch();

  const [selectedFile, setSelectedFile] = useState();
  const [isSelected, setIsSelected] = useState(false);

  const changeHandler = (event) => {
    if (event.target.files[0]) {
      dispatch(setAudioURL(URL.createObjectURL(event.target.files[0])));
      setSelectedFile(event.target.files[0]);
      setIsSelected(true);
    }
  };

  const handleProcess = () => {
    dispatch(setProcessState(1));
    var melodyRes;
    var lyricRes;
    var doneCheck = false;
    var data = new FormData();
    data.append("file", selectedFile);
    //get Melody Extraction
    fetch("http://localhost:5000/melody", {
      method: "POST",
      mode: "cors",
      headers: {
        "Access-Control-Allow-Origin": "*",
      },
      body: data,
    })
      .then((res) => {
        if (!doneCheck) {
          doneCheck = !doneCheck;
        } else {
          dispatch(setProcessState(2));
        }

        return res.json();
      })
      .then((result) => {
        melodyRes = result.result;
        dispatch(setMelodyRes(melodyRes));
      });

    //get Lyric Extraction
    fetch("http://localhost:5000/lyrics", {
      method: "POST",
      mode: "cors",
      headers: {
        "Access-Control-Allow-Origin": "*",
      },
      body: data,
    })
      .then((res) => {
        if (!doneCheck) {
          doneCheck = !doneCheck;
        } else {
          dispatch(setProcessState(2));
        }

        return res.json();
      })
      .then((result) => {
        lyricRes = result.text;
        dispatch(setLyricRes(lyricRes));
      });
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
        <input type="file" hidden onChange={changeHandler} name="file" />
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

      <h3>Analyse your music file</h3>
      <LoadingButton
        variant="contained"
        style={{
          backgroundColor: "#ec66ca",
        }}
        disabled={!isSelected}
        onClick={handleProcess}
        loading={configuration.processState === 1}
      >
        Process
      </LoadingButton>
    </div>
  );
}
