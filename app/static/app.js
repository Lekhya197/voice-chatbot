const micBtn = document.getElementById("micBtn");
const stateText = document.getElementById("stateText");
const recognizedText = document.getElementById("recognizedText");
const chatLog = document.getElementById("chatLog");
const textForm = document.getElementById("textForm");
const textInput = document.getElementById("textInput");
const notice = document.getElementById("notice");
const asrToggle = document.getElementById("asrToggle");
const uploadBtn = document.getElementById("uploadBtn");
const muteToggle = document.getElementById("muteToggle");
const permissionBtn = document.getElementById("permissionBtn");

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const isLocalhost = ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
const hasSecureMicContext = window.isSecureContext || isLocalhost;

let recognition = null;
let mediaRecorder = null;
let chunks = [];
let recordingUpload = false;

function setState(state) {
  stateText.textContent = state;
  micBtn.className = "mic " + state;
  micBtn.setAttribute("aria-label", state === "idle" ? "Start voice input" : "Voice input " + state);
}

function setNotice(message, tone = "info") {
  notice.textContent = message || "";
  notice.dataset.tone = tone;
}

function setTranscript(text) {
  recognizedText.textContent = text || "Your transcript will appear here.";
}

function chromeMicHelp() {
  return "Chrome is blocking the microphone. Click the lock/tune icon in the address bar, set Microphone to Allow for this site, then reload the page.";
}

function insecureContextHelp() {
  return "Chrome only allows microphone access on HTTPS or localhost. Open http://localhost:7860 locally, or use the deployed HTTPS URL.";
}

function handleMicError(error) {
  setState("idle");
  const name = error && error.name ? error.name : "";
  if (!hasSecureMicContext) {
    setNotice(insecureContextHelp(), "error");
  } else if (name === "NotAllowedError" || name === "SecurityError" || name === "PermissionDeniedError") {
    setNotice(chromeMicHelp(), "error");
  } else if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    setNotice("Chrome cannot find a microphone. Check your system input device, then try again.", "error");
  } else if (name === "NotReadableError" || name === "TrackStartError") {
    setNotice("Your microphone is busy in another app. Close other recording apps and try again.", "error");
  } else {
    setNotice("Microphone access failed. You can still type your question below.", "error");
  }
}

async function requestMicStream() {
  if (!hasSecureMicContext) {
    throw new DOMException(insecureContextHelp(), "SecurityError");
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new DOMException("Audio recording is not supported in this browser.", "NotSupportedError");
  }
  return navigator.mediaDevices.getUserMedia({ audio: true });
}

async function checkMicPermission() {
  try {
    const stream = await requestMicStream();
    stream.getTracks().forEach((track) => track.stop());
    setNotice("Microphone is allowed. Press the main mic button and ask your campus question.", "success");
  } catch (error) {
    handleMicError(error);
  }
}

async function refreshPermissionHint() {
  if (!hasSecureMicContext) {
    setNotice(insecureContextHelp(), "error");
    return;
  }
  if (!navigator.permissions || !navigator.permissions.query) {
    return;
  }
  try {
    const status = await navigator.permissions.query({ name: "microphone" });
    if (status.state === "denied") {
      setNotice(chromeMicHelp(), "error");
    } else if (status.state === "prompt") {
      setNotice("Chrome will ask for microphone permission when you press the mic button.", "info");
    }
    status.onchange = refreshPermissionHint;
  } catch {
    // Some browsers do not expose microphone permission status.
  }
}

function addBubble(kind, text, meta) {
  const bubble = document.createElement("div");
  bubble.className = "bubble " + kind;
  bubble.textContent = text;
  if (meta) {
    const details = document.createElement("div");
    details.className = "meta";

    const line = document.createElement("div");
    line.textContent = `Intent: ${meta.intent}${meta.fallback ? " (fallback)" : ""} - Confidence: ${Math.round(meta.confidence * 100)}%`;
    details.appendChild(line);

    if (meta.top_tokens && meta.top_tokens.length) {
      const chips = document.createElement("div");
      chips.className = "chips";
      meta.top_tokens.forEach((token) => {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = "why: " + token;
        chips.appendChild(chip);
      });
      details.appendChild(chips);
    }

    const bars = document.createElement("div");
    bars.className = "bars";
    (meta.top3 || []).forEach((item) => {
      const row = document.createElement("div");
      row.className = "barRow";
      const label = document.createElement("span");
      label.textContent = item.intent;
      const track = document.createElement("div");
      track.className = "barTrack";
      const fill = document.createElement("div");
      fill.className = "barFill";
      fill.style.width = Math.round(item.prob * 100) + "%";
      track.appendChild(fill);
      const value = document.createElement("span");
      value.textContent = Math.round(item.prob * 100) + "%";
      row.append(label, track, value);
      bars.appendChild(row);
    });
    details.appendChild(bars);
    bubble.appendChild(details);
  }
  chatLog.appendChild(bubble);
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function ask(text) {
  const clean = (text || "").trim();
  if (!clean) {
    setNotice("Please say or type a question first.", "error");
    return;
  }
  setTranscript(clean);
  addBubble("user", clean);
  setState("thinking");
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: clean }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Chat request failed");
    }
    addBubble("bot", data.response, data);
    if (!muteToggle.checked && "speechSynthesis" in window) {
      const utterance = new SpeechSynthesisUtterance(data.response);
      utterance.lang = "en-IN";
      speechSynthesis.speak(utterance);
    }
  } catch (error) {
    addBubble("bot", "Sorry, I could not process that request. " + error.message);
  } finally {
    setState("idle");
  }
}

async function browserListen() {
  if (!SpeechRecognition) {
    asrToggle.checked = true;
    setNotice("Browser speech recognition is unavailable. Try server ASR or typed input.", "error");
    return;
  }

  try {
    const stream = await requestMicStream();
    stream.getTracks().forEach((track) => track.stop());
  } catch (error) {
    handleMicError(error);
    return;
  }

  recognition = new SpeechRecognition();
  recognition.lang = "en-IN";
  recognition.continuous = false;
  recognition.interimResults = true;
  let finalText = "";

  recognition.onstart = () => {
    setState("listening");
    setNotice("Listening...", "info");
  };
  recognition.onerror = (event) => {
    if (event.error === "not-allowed" || event.error === "service-not-allowed") {
      handleMicError(new DOMException(chromeMicHelp(), "NotAllowedError"));
    } else {
      setState("idle");
      setNotice("Speech recognition error: " + event.error + ". You can type your question below.", "error");
    }
  };
  recognition.onresult = (event) => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const text = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalText += text;
      } else {
        interim += text;
      }
    }
    setTranscript((finalText + " " + interim).trim());
  };
  recognition.onend = () => {
    setState("idle");
    if (finalText.trim()) {
      setNotice("", "info");
      ask(finalText.trim());
    }
  };
  recognition.start();
}

async function recordUpload() {
  if (!window.MediaRecorder) {
    setNotice("Audio recording is not supported in this browser. Use browser ASR or typed input.", "error");
    return;
  }
  if (recordingUpload && mediaRecorder) {
    mediaRecorder.stop();
    return;
  }

  try {
    const stream = await requestMicStream();
    chunks = [];
    mediaRecorder = new MediaRecorder(stream);
    recordingUpload = true;
    uploadBtn.textContent = "Stop & transcribe";
    setState("listening");
    setNotice("Recording for Whisper upload. Press Stop when finished.", "info");

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size) {
        chunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      recordingUpload = false;
      uploadBtn.textContent = "Record & upload";
      stream.getTracks().forEach((track) => track.stop());
      setState("transcribing");

      const blob = new Blob(chunks, { type: "audio/webm" });
      const form = new FormData();
      form.append("file", blob, "speech.webm");

      try {
        const response = await fetch("/api/transcribe", { method: "POST", body: form });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "Transcription failed");
        }
        setTranscript(data.text);
        await ask(data.text);
      } catch (error) {
        setNotice(error.message, "error");
        setState("idle");
      }
    };
    mediaRecorder.start();
  } catch (error) {
    handleMicError(error);
  }
}

micBtn.addEventListener("click", () => {
  if (asrToggle.checked) {
    recordUpload();
  } else {
    browserListen();
  }
});
uploadBtn.addEventListener("click", recordUpload);
permissionBtn.addEventListener("click", checkMicPermission);
textForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = textInput.value;
  textInput.value = "";
  ask(text);
});

if (!SpeechRecognition) {
  asrToggle.checked = true;
  setNotice("Browser ASR unavailable here; server ASR is selected.", "info");
}
refreshPermissionHint();
setState("idle");
