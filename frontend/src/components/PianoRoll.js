// Workaround to duplicating piano roll
// Taken from:
// https://stackoverflow.com/questions/71139905/how-to-stop-react-piano-roll-component-from-duplicating-itself-on-state-update
// https://codesandbox.io/s/boring-mopsa-cppq2v?file=/src/components/PianoRoll/PianoRoll.js

import React, {
  useEffect,
  useRef,
  useImperativeHandle,
  forwardRef
} from "react";
import pixiPianoRoll from "./pixiPianoRoll.js";

var PianoRoll = function PianoRoll(props, playbackRef) {
  var container = useRef();
  var pianoRoll = pixiPianoRoll(props);

  useImperativeHandle(playbackRef, function () {
    return pianoRoll.playback;
  });
  useEffect(function () {
    container.current.replaceChildren(pianoRoll.view);
  });
  return React.createElement("div", {
    ref: container
  });
};

export default forwardRef(PianoRoll);