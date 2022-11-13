import "./App.css";
import Grid from "@mui/material/Grid";
import { Box } from "@mui/system";
import { Paper } from "@mui/material";

import { ConfigureSection } from "./ConfigureSection.js";
import { ResultsSection } from "./ResultsSection.js";

function App() {
  return (
    <div>
      <div className="App">
        <header className="App-header">
          <h2>Musify</h2>
        </header>
      </div>
      <div className="body">
        <Grid container spacing={2} alignItems="center" justifyContent="center">
          <Grid xs={3} align="center">
            <Box m={2}>
              <Paper elevation={3}>
                <Box p={2} sx={{ height: 700 }}>
                  {
                    ConfigureSection()
                  }
                </Box>
              </Paper>
            </Box>
          </Grid>
          <Grid xs={9} align="center">
            <Box m={2}>
              <Paper elevation={3}>
                <Box p={2} sx={{ height: 700 }}>
                  {
                    ResultsSection()
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
