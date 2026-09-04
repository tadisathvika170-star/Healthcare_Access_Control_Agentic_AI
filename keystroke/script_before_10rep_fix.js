(function () {
    "use strict";

    const REQUIRED_REPETITIONS = 3;
    const REQUIRED_PHRASE = ".tie5Roanl";

    const input = document.getElementById("typingInput");
    const clearButton = document.getElementById("clearButton");
    const submitButton = document.getElementById("submitButton");

    const counterElement = document.getElementById("counter");
    const statusElement = document.getElementById("status");

    let events = [];
    let keyDownTimes = {};
    let repetitions = 0;


    // =========================================================
    // STREAMLIT V1 COMMUNICATION
    // =========================================================

    function sendMessage(type, data) {

        window.parent.postMessage(
            {
                isStreamlitMessage: true,
                type: type,
                ...data
            },
            "*"
        );
    }


    function componentReady() {

        sendMessage(
            "streamlit:componentReady",
            {
                apiVersion: 1
            }
        );
    }


    function setFrameHeight() {

        sendMessage(
            "streamlit:setFrameHeight",
            {
                height: document.body.scrollHeight
            }
        );
    }


    function sendComponentValue(value) {

        sendMessage(
            "streamlit:setComponentValue",
            {
                value: value
            }
        );
    }


    // =========================================================
    // COUNTER
    // =========================================================

    function updateRepetitions() {

        const lines = input.value
            .split("\n")
            .filter(function (line) {
                return line.trim() !== "";
            });

        repetitions = lines.length;

        if (repetitions > REQUIRED_REPETITIONS) {
            repetitions = REQUIRED_REPETITIONS;
        }

        counterElement.textContent =
            repetitions + " / " + REQUIRED_REPETITIONS;

        submitButton.disabled =
            repetitions !== REQUIRED_REPETITIONS;

        if (repetitions === 0) {

            statusElement.textContent =
                "Start typing the phrase.";

        } else if (repetitions < REQUIRED_REPETITIONS) {

            statusElement.textContent =
                "Repetition " +
                repetitions +
                " recorded. Continue typing.";

        } else {

            statusElement.textContent =
                "All 3 repetitions recorded. Click Submit.";
        }

        setFrameHeight();
    }


    // =========================================================
    // KEY NORMALIZATION
    // =========================================================

    function normalizeKey(event) {

        if (event.key === " ") {
            return "Space";
        }

        if (event.key === "Enter") {
            return "Enter";
        }

        if (event.key === "Shift") {
            return "Shift";
        }

        return event.key;
    }


    // =========================================================
    // KEY DOWN
    // =========================================================

    input.addEventListener(
        "keydown",
        function (event) {

            const now = performance.now();

            const key = normalizeKey(event);

            keyDownTimes[key] = now;

            events.push({
                key: key,
                key_down: now,
                key_up: null,
                hold_time: null
            });

            // DO NOT prevent Enter.
            // This allows a new line to be created.

            if (event.key === "Enter") {

                setTimeout(
                    updateRepetitions,
                    0
                );
            }
        }
    );


    // =========================================================
    // KEY UP
    // =========================================================

    input.addEventListener(
        "keyup",
        function (event) {

            const now = performance.now();

            const key = normalizeKey(event);

            const downTime =
                keyDownTimes[key];

            for (
                let i = events.length - 1;
                i >= 0;
                i--
            ) {

                if (
                    events[i].key === key &&
                    events[i].key_up === null
                ) {

                    events[i].key_up = now;

                    if (
                        downTime !== undefined
                    ) {

                        events[i].hold_time =
                            now - downTime;
                    }

                    break;
                }
            }

            setFrameHeight();
        }
    );


    // =========================================================
    // INPUT
    // =========================================================

    input.addEventListener(
        "input",
        updateRepetitions
    );


    // =========================================================
    // CLEAR
    // =========================================================

    clearButton.addEventListener(
        "click",
        function () {

            input.value = "";

            events = [];
            keyDownTimes = {};
            repetitions = 0;

            counterElement.textContent = "0 / 3";

            submitButton.disabled = true;

            statusElement.textContent =
                "Start typing the phrase.";

            sendComponentValue(null);

            input.focus();

            setFrameHeight();
        }
    );


    // =========================================================
    // SUBMIT
    // =========================================================

    submitButton.addEventListener(
        "click",
        function () {

            const lines = input.value
                .split("\n")
                .filter(function (line) {
                    return line.trim() !== "";
                });


            if (lines.length !== 3) {

                statusElement.textContent =
                    "Please complete all 3 repetitions first.";

                return;
            }


            // Validate every repetition.

            for (
                let i = 0;
                i < lines.length;
                i++
            ) {

                if (
                    lines[i].trim() !==
                    REQUIRED_PHRASE
                ) {

                    statusElement.textContent =
                        "Repetition " +
                        (i + 1) +
                        " is incorrect. Please type " +
                        REQUIRED_PHRASE;

                    return;
                }
            }


            // =================================================
            // SUBMISSION
            // =================================================

            const submission = {

                submitted: true,

                submission_id:
                    Date.now(),

                phrase:
                    REQUIRED_PHRASE,

                repetitions:
                    3,

                events:
                    events
            };


            statusElement.textContent =
                "Keystroke data submitted for verification.";


            sendComponentValue(
                submission
            );
        }
    );


    // =========================================================
    // STREAMLIT RENDER
    // =========================================================

    window.addEventListener(
        "message",
        function (event) {

            if (!event.data) {
                return;
            }

            if (
                event.data.type ===
                "streamlit:render"
            ) {

                setFrameHeight();
            }
        }
    );


    // =========================================================
    // INITIALIZE
    // =========================================================

    counterElement.textContent = "0 / 3";

    submitButton.disabled = true;

    setTimeout(
        function () {

            componentReady();

            setFrameHeight();

        },
        100
    );

})();