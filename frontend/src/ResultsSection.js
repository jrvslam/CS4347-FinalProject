import React from "react";
import { useSelector } from 'react-redux'

export function ResultsSection() {
  const configuration = useSelector((state) => state.configuration);

  return (
    <div>
        <h2>Results</h2>
        {
            (configuration.processState !== 2) 
                ? <p> Start Analysing some music!</p> 
                : <p> Lyrics: {configuration.lyricRes}</p>
        }
    </div>
  );
}
