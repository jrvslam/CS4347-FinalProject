import React, {useState, useEffect} from "react";
import { useSelector, useDispatch } from 'react-redux'
import { setProcessState, setMelodyRes, setLyricRes } from "./features/configureSlice";

export function ResultsSection() {
    const configuration = useSelector((state) => state.configuration);
    const dispatch = useDispatch();


  return (
    <div>
        <h2>Results</h2>
        {
            (configuration.processState !== 2) 
                ? <p> Start Analysing some music!</p> 
                : <p> Done Processing! </p>
        }
    </div>
  );
}
