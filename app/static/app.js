const state = {
  sessionId: localStorage.getItem("docpilot-session") || createId(),
  busy: false,
};

localStorage.setItem("docpilot-session", state.sessionId);

const $ = (selector) => document.querySelector(selector);
const chat = $("#chat");
const welcome = $("#welcome");
const questionInput = $("#question");
const sendButton = $("#send");
const fileInput = $("#file-input");
const documentList = $("#document-list");
const toast = $("#toast");

function createId() {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 2800);
}

function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight;
}

function addMessage(role, text) {
  welcome?.remove();
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "你" : "D";
  const body = document.createElement("div");
  body.className = "message-body";
  const paragraph = document.createElement("p");
  paragraph.className = "message-text";
  paragraph.textContent = text;
  body.appendChild(paragraph);
  article.append(avatar, body);
  chat.appendChild(article);
  scrollToBottom();
  return { article, body, paragraph };
}

async function loadDocuments() {
  try {
    const response = await fetch("/api/documents");
    if (!response.ok) throw new Error("无法读取文档列表");
    const payload = await response.json();
    renderDocuments(payload.documents || []);
  } catch (error) {
    documentList.innerHTML = `<p class="empty-docs">${error.message}</p>`;
  }
}

function renderDocuments(documents) {
  $("#doc-count").textContent = `${documents.length} 篇`;
  documentList.replaceChildren();
  if (!documents.length) {
    const empty = document.createElement("p");
    empty.className = "empty-docs";
    empty.textContent = "还没有文档，上传一份开始吧。";
    documentList.appendChild(empty);
    return;
  }

  for (const doc of documents) {
    const item = document.createElement("div");
    item.className = "document-item";

    const icon = document.createElement("span");
    icon.className = "document-icon";
    icon.textContent = doc.filename.split(".").pop().slice(0, 3).toUpperCase();

    const meta = document.createElement("div");
    meta.className = "document-meta";
    const name = document.createElement("span");
    name.className = "document-name";
    name.textContent = doc.filename;
    name.title = doc.filename;
    const detail = document.createElement("span");
    detail.className = "document-sub";
    detail.textContent = `${doc.chunks} 个片段 · ${formatBytes(doc.size)}`;
    meta.append(name, detail);

    const remove = document.createElement("button");
    remove.className = "delete-doc";
    remove.type = "button";
    remove.title = "删除文档";
    remove.textContent = "×";
    remove.addEventListener("click", () => deleteDocument(doc));

    item.append(icon, meta, remove);
    documentList.appendChild(item);
  }
}

async function uploadFile(file) {
  if (!file) return;
  if (file.size > 20 * 1024 * 1024) {
    showToast("文件超过 20 MB");
    return;
  }
  const form = new FormData();
  form.append("file", file);
  showToast(`正在处理 ${file.name}…`);
  try {
    const response = await fetch("/api/documents", { method: "POST", body: form });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || "上传失败");
    showToast(`已索引 ${payload.chunks} 个片段`);
    await loadDocuments();
  } catch (error) {
    showToast(error.message);
  } finally {
    fileInput.value = "";
  }
}

async function deleteDocument(doc) {
  if (!confirm(`删除《${doc.filename}》？`)) return;
  try {
    const response = await fetch(`/api/documents/${doc.id}`, { method: "DELETE" });
    if (!response.ok) throw new Error("删除失败");
    await loadDocuments();
    showToast("文档已删除");
  } catch (error) {
    showToast(error.message);
  }
}

function parseEventBlock(block) {
  let event = "message";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return null;
  }
}

function renderSources(container, sources) {
  if (!sources || !sources.length) return;
  const wrap = document.createElement("div");
  wrap.className = "sources";
  for (const source of sources) {
    const chip = document.createElement("span");
    chip.className = "source-chip";
    chip.title = source.excerpt || "";
    const label = document.createElement("strong");
    label.textContent = "[来源]";
    chip.append(label, document.createTextNode(`${source.filename}${source.page ? ` · 第 ${source.page} 页` : ""}`));
    wrap.appendChild(chip);
  }
  container.appendChild(wrap);
}

async function sendQuestion(question) {
  if (state.busy || !question.trim()) return;
  state.busy = true;
  sendButton.disabled = true;
  addMessage("user", question.trim());
  questionInput.value = "";
  resizeInput();

  const { body, paragraph } = addMessage("assistant", "");
  const status = document.createElement("div");
  status.className = "message-status";
  status.textContent = "正在分析问题";
  body.insertBefore(status, paragraph);

  let answer = "";
  let sources = [];

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question.trim(), session_id: state.sessionId }),
    });
    if (!response.ok || !response.body) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || "请求失败");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() || "";
      for (const block of blocks) {
        const parsed = parseEventBlock(block);
        if (!parsed) continue;
        const { event, data } = parsed;
        if (event === "status") {
          status.textContent = data.message;
        } else if (event === "token") {
          status.remove();
          answer += data.content;
          paragraph.textContent = answer;
          scrollToBottom();
        } else if (event === "sources") {
          sources = data || [];
        } else if (event === "error") {
          throw new Error(data.message || "生成失败");
        } else if (event === "done") {
          sources = data.sources || sources;
        }
      }
    }

    if (!answer.trim()) paragraph.textContent = "模型未返回有效内容。";
    renderSources(body, sources);
  } catch (error) {
    status.remove();
    paragraph.remove();
    const box = document.createElement("div");
    box.className = "error-box";
    box.textContent = error.message;
    body.appendChild(box);
  } finally {
    state.busy = false;
    sendButton.disabled = false;
    scrollToBottom();
    questionInput.focus();
  }
}

function resizeInput() {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 160)}px`;
}

$("#upload-trigger").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => uploadFile(fileInput.files[0]));

$("#new-chat").addEventListener("click", () => {
  state.sessionId = createId();
  localStorage.setItem("docpilot-session", state.sessionId);
  chat.replaceChildren();
  const fresh = document.createElement("div");
  fresh.className = "welcome";
  fresh.id = "welcome";
  fresh.innerHTML = `
    <div class="welcome-orbit"><span>?</span></div>
    <h3>新对话已开始</h3>
    <p>继续基于知识库提问。对话上下文只保留在当前会话中。</p>`;
  chat.appendChild(fresh);
  questionInput.focus();
});

$("#chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuestion(questionInput.value);
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    sendQuestion(questionInput.value);
  }
});
questionInput.addEventListener("input", resizeInput);

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => sendQuestion(button.textContent));
});

loadDocuments();
questionInput.focus();
