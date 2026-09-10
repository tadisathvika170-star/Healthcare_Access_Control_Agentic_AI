(function () {
    "use strict";

    const REQUIRED_REPETITIONS = 1;
    const REQUIRED_PHRASE = ".tie5Roanl";

    const input = document.getElementById("typingInput");
    const clearButton = document.getElementById("clearButton");
    const submitButton = document.getElementById("submitButton");
    const counterElement = document.getElementById("counter");
    const statusElement = document.getElementById("status");

    let events = [];
    let keyDownTimes = {};
    let repetitions = 0;

    function sendMessage(type, data) {
        window.parent.postMessage({
            isStreamlitMessage: true,
            type: type,
            ...data
        }, "*");
    }

    function componentReady() {
        sendMessage("streamlit:componentReady", { apiVersion: 1 });
    }

    function setFrameHeight() {
        sendMessage("streamlit:setFrameHeight", {
            height: document.body.scrollHeight
        });
    }

    function sendComponentValue(value) {
        sendMessage("streamlit:setComponentValue", { value: value });
    }

    function updateRepetitions() {
        const lines = input.value.split("\n").filter(function (line) {
            return line.trim() !== "";
        });

        repetitions = Math.min(lines.length, REQUIRED_REPETITIONS);
        counterElement.textContent = repetitions + " / " + REQUIRED_REPETITIONS;
        submitButton.disabled = repetitions !== REQUIRED_REPETITIONS;

        if (repetitions === 0) {
            statusElement.textContent = "Start typing the phrase.";
        } else if (repetitions < REQUIRED_REPETITIONS) {
            statusElement.textContent = "Repetition " + repetitions + " recorded. Continue typing.";
        } else {
            statusElement.textContent = "The repetition is recorded. Click Submit.";
        }

        setFrameHeight();
    }

    function normalizeKey(event) {
        if (event.key === " ") return "Space";
        if (event.key === "Enter") return "Enter";
        if (event.key === "Shift") return "Shift";
        if (/^[a-zA-Z]$/.test(event.key)) return event.key.toLowerCase();
        return event.key;
    }

    input.addEventListener("keydown", function (event) {
        const now = performance.now();
        const key = normalizeKey(event);

        keyDownTimes[key] = now;
        events.push({
            key: key,
            key_down: now,
            key_up: null,
            hold_time: null
        });

        if (event.key === "Enter") {
            setTimeout(updateRepetitions, 0);
        }
    });

    input.addEventListener("keyup", function (event) {
        const now = performance.now();
        const key = normalizeKey(event);
        const downTime = keyDownTimes[key];

        for (let i = events.length - 1; i >= 0; i--) {
            if (events[i].key === key && events[i].key_up === null) {
                events[i].key_up = now;
                if (downTime !== undefined) {
                    events[i].hold_time = now - downTime;
                }
                break;
            }
        }

        setFrameHeight();
    });

    input.addEventListener("input", updateRepetitions);

    clearButton.addEventListener("click", function () {
        input.value = "";
        events = [];
        keyDownTimes = {};
        repetitions = 0;
        counterElement.textContent = "0 / 1";
        submitButton.disabled = true;
        statusElement.textContent = "Start typing the phrase.";
        sendComponentValue(null);
        input.focus();
        setFrameHeight();
    });

    submitButton.addEventListener("click", function () {
        const lines = input.value.split("\n").filter(function (line) {
            return line.trim() !== "";
        });

        if (lines.length !== REQUIRED_REPETITIONS) {
            statusElement.textContent = "Please complete the repetition first.";
            return;
        }

        if (lines[0].trim() !== REQUIRED_PHRASE) {
            statusElement.textContent = "The phrase is incorrect. Please type " + REQUIRED_PHRASE;
            return;
        }

        const submission = {
            submitted: true,
            submission_id: Date.now(),
            phrase: REQUIRED_PHRASE,
            repetitions: 1,
            events: events
        };

        statusElement.textContent = "Keystroke data submitted for verification.";
        sendComponentValue(submission);
    });

    window.addEventListener("message", function (event) {
        if (!event.data) return;
        if (event.data.type === "streamlit:render") setFrameHeight();
    });

    counterElement.textContent = "0 / 1";
    submitButton.disabled = true;

    setTimeout(function () {
        componentReady();
        setFrameHeight();
    }, 100);
})();
