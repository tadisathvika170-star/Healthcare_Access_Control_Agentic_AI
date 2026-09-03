/*
script.js

Captures keystroke timings for authentication.
*/

const typingBox = document.getElementById("typingBox");
const clearButton = document.getElementById("clearButton");
const submitButton = document.getElementById("submitButton");
const status = document.getElementById("status");

let keyDownTimes = {};
let keystrokes = [];

// --------------------------------------------
// Key Down
// --------------------------------------------
typingBox.addEventListener("keydown", function(event) {

    const key = event.key;

    keyDownTimes[key] = performance.now();

});

// --------------------------------------------
// Key Up
// --------------------------------------------
typingBox.addEventListener("keyup", function(event) {

    const key = event.key;

    const keyUpTime = performance.now();

    if (keyDownTimes[key] !== undefined) {

        keystrokes.push({

            key: key,

            key_down: keyDownTimes[key],

            key_up: keyUpTime

        });

    }

});

// --------------------------------------------
// Clear
// --------------------------------------------
clearButton.addEventListener("click", function() {

    typingBox.value = "";

    keystrokes = [];

    keyDownTimes = {};

    status.innerHTML = "Typing cleared.";

});

// --------------------------------------------
// Submit
// --------------------------------------------
submitButton.addEventListener("click", function() {

    if (keystrokes.length === 0) {

        status.innerHTML = "No typing captured.";

        return;

    }

    console.log("Captured Keystrokes");

    console.log(keystrokes);

    status.innerHTML =
        "Captured " +
        keystrokes.length +
        " keystrokes successfully.";

    /*
    Later this data will be sent to Streamlit:

    window.parent.postMessage(
        {
            type: "keystrokes",
            data: keystrokes
        },
        "*"
    );
    */

});
