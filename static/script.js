let voiceEnabled = false;
let currentUtterance = null;

async function sendMessage() {
  const input = document.getElementById("user-input");
  const chat = document.getElementById("chat");
  const userMessage = input.value.trim();
  if (userMessage === "") return;

  // Append user message bubble
  const userBubble = document.createElement("div");
  userBubble.className = "message user";
  userBubble.textContent = userMessage;
  chat.appendChild(userBubble);
  chat.scrollTop = chat.scrollHeight;
  input.value = "";

  // Append bot message bubble with placeholder
  const botBubble = document.createElement("div");
  botBubble.className = "message bot";
  const avatar = document.createElement("img");
  avatar.src = "/static/Kani.jpg";
  avatar.alt = "Bot Avatar";
  avatar.className = "bot-avatar";
  botBubble.appendChild(avatar);
  const botText = document.createElement("span");
  botText.className = "bot-text";
  botText.innerHTML = '<div class="typing-bubbles"><span></span><span></span><span></span></div>';

  botBubble.appendChild(botText);
  chat.appendChild(botBubble);
  chat.scrollTop = chat.scrollHeight;

  try {
    const response = await fetch("/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userMessage })
    });
    const data = await response.json();
    botText.innerHTML = data.message;

    // Check the state of the voice toggle and speak text if enabled
    if (voiceEnabled) {
      speakText(data.message.replace(/<[^>]*>/g, ''));
    }
  } catch (error) {
    console.error("Error:", error);
    botText.textContent = "Sorry, something went wrong.";
  }
  chat.scrollTop = chat.scrollHeight;
}

function startVoiceRecognition() {
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    alert("Speech recognition not supported.");
    return;
  }
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = new SpeechRecognition();
  recognition.lang = 'en-US';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  recognition.onresult = function(event) {
    const transcript = event.results[0][0].transcript;
    document.getElementById("user-input").value = transcript;
    sendMessage();
  };

  recognition.onerror = function(event) {
    alert("Error during voice recognition: " + event.error);
  };

  recognition.start();
}

document.getElementById("user-input").addEventListener("keydown", function(e) {
  if (e.key === "Enter") sendMessage();
});

function speakText(text) {
  if (!window.speechSynthesis) return;

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'en-US';
  // Use voiceEnabled value to determine volume
  utterance.volume = voiceEnabled ? 1 : 0;

  currentUtterance = utterance;
  speechSynthesis.speak(utterance);
}

document.getElementById("voice-toggle").addEventListener("change", function(e) {
  voiceEnabled = e.target.checked;
});
