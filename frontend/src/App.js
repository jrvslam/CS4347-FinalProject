import "./App.css";
import Grid from "@mui/material/Grid";
import { Box } from "@mui/system";
import { Button, Paper } from "@mui/material";
import FileUploadIcon from "@mui/icons-material/FileUpload";

import { ConfigureSection } from "./ConfigureSection.js";
import { ResultsSection } from "./ResultsSection.js";
import { useState,useCallback } from "react";
function App() {
  const [melodyOutput, setMelofyOutput] = useState([])
  const [lyricsOutput, setLyricsOutput] = useState([])
  
  const getMelodyOutput = useCallback(value => {
      setMelofyOutput(value)
    },[]
  );

  const getLyricsOutput = useCallback(value => {
    setLyricsOutput(value)
  },[]
);


  return (
    <div>
      <div className="App">
        <header className="App-header">
          {/* <img src={logo} className="App-logo" alt="logo" /> */}
          <h2>Musify</h2>
        </header>
      </div>
      <div className="body">
        <Grid container spacing={2} alignItems="center" justifyContent="center">
          <Grid xs={3} align="center">
            <Box m={2}>
              <Paper elevation={3}>
                <Box p={2} sx={{ height: 500 }}>
                  {
                    ConfigureSection(getMelodyOutput, getLyricsOutput)
                  }
                </Box>
              </Paper>
            </Box>
          </Grid>
          <Grid xs={9} align="center">
            <Box m={2}>
              <Paper elevation={3}>
                <Box p={2} sx={{ height: 500 }}>
                  {
                    ResultsSection(melodyOutput,lyricsOutput)
                  }
                </Box>
              </Paper>
            </Box>
          </Grid>
        </Grid>
      </div>
    </div>
  );
}

export default App;
