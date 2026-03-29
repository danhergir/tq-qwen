const state = {
  busy: false,
};

const elements = {
  chatLog: document.getElementById("chat-log"),
  messageInput: document.getElementById("message-input"),
  sendButton: document.getElementById("send-button"),
  stopButton: document.getElementById("stop-button"),
  resetButton: document.getElementById("reset-button"),
  statusPill: document.getElementById("status-pill"),
  modelName: document.getElementById("model-name"),
  modelSource: document.getElementById("model-source"),
  modeLine: document.getElementById("mode-line"),
  template: document.getElementById("message-template"),
};

function setStatus(label, tone = "ready") {
  elements.statusPill.textContent = label;
  elements.statusPill.classList.remove("busy", "error");
  if (tone === "busy") elements.statusPill.classList.add("busy");
  if (tone === "error") elements.statusPill.classList.add("error");
}

function syncControls() {
  elements.sendButton.disabled = state.busy;
  elements.stopButton.disabled = !state.busy;
  elements.messageInput.disabled = false;
}

function scrollToBottom() {
  elements.chatLog.scrollTop = elements.chatLog.scrollHeight;
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderInline(text) {
  let safe = escapeHtml(text);
  safe = safe.replace(/`([^`]+)`/g, "<code>$1</code>");
  safe = safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  safe = safe.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return safe;
}

function renderRichText(text) {
  const normalized = text.replace(/\r\n/g, "\n").trim();
  if (!normalized) {
    return "";
  }

  const blocks = normalized.split(/\n{2,}/);
  const html = [];

  for (const block of blocks) {
    const trimmed = block.trim();
    if (!trimmed) continue;

    if (trimmed.startsWith("```") && trimmed.endsWith("```")) {
      const code = trimmed.replace(/^```[a-zA-Z0-9_-]*\n?/, "").replace(/\n?```$/, "");
      html.push(`<pre><code>${escapeHtml(code)}</code></pre>`);
      continue;
    }

    const lines = trimmed.split("\n");
    if (lines.every((line) => /^\s*[-*]\s+/.test(line))) {
      const items = lines
        .map((line) => line.replace(/^\s*[-*]\s+/, ""))
        .map((line) => `<li>${renderInline(line)}</li>`)
        .join("");
      html.push(`<ul>${items}</ul>`);
      continue;
    }

    if (lines.every((line) => /^\s*\d+\.\s+/.test(line))) {
      const items = lines
        .map((line) => line.replace(/^\s*\d+\.\s+/, ""))
        .map((line) => `<li>${renderInline(line)}</li>`)
        .join("");
      html.push(`<ol>${items}</ol>`);
      continue;
    }

    const paragraph = lines.map((line) => renderInline(line)).join("<br>");
    html.push(`<p>${paragraph}</p>`);
  }

  return html.join("");
}

function createMessage(role, text = "", options = {}) {
  const fragment = elements.template.content.cloneNode(true);
  const root = fragment.querySelector(".message");
  const roleNode = fragment.querySelector(".message-role");
  const metaNode = fragment.querySelector(".message-meta");
  const bodyNode = fragment.querySelector(".message-body");
  root.classList.add(role);
  roleNode.textContent = role;
  metaNode.textContent = options.meta || "";
  if (!options.meta) {
    metaNode.hidden = true;
  }
  if (options.html) {
    bodyNode.innerHTML = text;
  } else {
    bodyNode.textContent = text;
  }
  elements.chatLog.appendChild(fragment);
  scrollToBottom();
  const message = elements.chatLog.lastElementChild;
  return {
    root: message,
    role: message.querySelector(".message-role"),
    meta: message.querySelector(".message-meta"),
    body: message.querySelector(".message-body"),
  };
}

function setMessageText(messageRef, text, rich = false) {
  if (rich) {
    messageRef.body.innerHTML = renderRichText(text);
  } else {
    messageRef.body.textContent = text;
  }
}

function setMessageMeta(messageRef, meta) {
  messageRef.meta.hidden = !meta;
  messageRef.meta.textContent = meta || "";
}

function formatProgressMeta(tokenCount = 0, tokensPerSecond = 0) {
  if (tokenCount <= 0) {
    return "Waiting for first answer tokens";
  }

  const stats = [`${tokenCount} tokens`];
  if (tokensPerSecond > 0) {
    stats.push(`${tokensPerSecond.toFixed(2)} tok/s`);
  }
  return stats.join(" · ");
}

function setThinkingState(messageRef, tokenCount = 0, tokensPerSecond = 0) {
  setMessageMeta(messageRef, formatProgressMeta(tokenCount, tokensPerSecond));
  messageRef.body.innerHTML = `
    <div class="thinking-text">
      <span>Thinking</span>
      <span class="thinking-dots"><span></span><span></span><span></span></span>
    </div>
  `;
}

function renderMessages(messages) {
  elements.chatLog.innerHTML = "";
  if (!messages.length) {
    createMessage(
      "system",
      renderRichText("Model is ready. Ask a question to start."),
      { html: true }
    );
    return;
  }

  for (const message of messages) {
    createMessage(message.role, renderRichText(message.content), { html: true });
  }
}

async function refreshState() {
  const response = await fetch("/api/state");
  if (!response.ok) {
    throw new Error("Unable to load app state.");
  }

  const payload = await response.json();
  state.busy = payload.busy;
  elements.modelName.textContent = payload.model;
  elements.modelSource.textContent = payload.model_source;
  elements.modeLine.textContent = payload.offline ? "Offline cache only" : "Hub fallback allowed";
  setStatus(payload.busy ? "Generating" : "Ready", payload.busy ? "busy" : "ready");
  syncControls();
  renderMessages(payload.messages);
}

async function sendMessage() {
  const message = elements.messageInput.value.trim();
  if (!message || state.busy) {
    return;
  }

  createMessage("user", renderRichText(message), { html: true });
  const assistantNode = createMessage("assistant", "", {
    meta: "Waiting for first answer tokens",
  });
  setThinkingState(assistantNode, 0, 0);
  elements.messageInput.value = "";

  state.busy = true;
  setStatus("Generating", "busy");
  syncControls();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    if (!response.ok || !response.body) {
      const payload = await response.json().catch(() => ({ error: "Request failed." }));
      throw new Error(payload.error || "Request failed.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalText = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "token") {
          setThinkingState(
            assistantNode,
            event.token_count || 0,
            event.tokens_per_second || 0
          );
          scrollToBottom();
        } else if (event.type === "error") {
          throw new Error(event.error);
        } else if (event.type === "done") {
          finalText = event.text || "";
          if (event.stopped && !finalText) {
            finalText = "Generation stopped.";
          } else if (event.stopped) {
            finalText += "\n\nStopped before completion.";
          }
          setMessageMeta(
            assistantNode,
            formatProgressMeta(
              event.token_count || 0,
              event.tokens_per_second || 0
            )
          );
          setMessageText(assistantNode, finalText, true);
        }
      }
    }

    setStatus("Ready");
  } catch (error) {
    setMessageMeta(assistantNode, "");
    setMessageText(assistantNode, `[Error] ${error.message}`);
    setStatus("Error", "error");
  } finally {
    state.busy = false;
    syncControls();
    scrollToBottom();
  }
}

async function stopGeneration() {
  if (!state.busy) return;
  await fetch("/api/stop", { method: "POST" });
}

async function resetConversation() {
  const response = await fetch("/api/reset", { method: "POST" });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ error: "Unable to reset." }));
    setStatus(payload.error || "Unable to reset", "error");
    return;
  }

  await refreshState();
}

elements.sendButton.addEventListener("click", sendMessage);
elements.stopButton.addEventListener("click", stopGeneration);
elements.resetButton.addEventListener("click", resetConversation);
elements.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

refreshState().catch((error) => {
  setStatus("Offline", "error");
  createMessage("system", renderRichText(`[Startup error] ${error.message}`), {
    html: true,
  });
});
