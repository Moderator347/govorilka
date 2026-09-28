/* ЭмоЧиталка — frontend SPA logic */
"use strict";

const EMO_LABELS = {
  neutral: "Нейтрально", happy: "Радость", sad: "Грусть", angry: "Гнев",
  fear: "Страх", excited: "Восторг", tender: "Нежность", mysterious: "Таинственно",
  heroic: "Героически", ironic: "Ирония",
};
const EMO_ICONS = {
  neutral: "😐", happy: "😊", sad: "😢", angry: "😠", fear: "😱",
  excited: "🤩", tender: "🥰", mysterious: "🌫️", heroic: "⚔️", ironic: "😏",
};
const EMO_COLORS = {
  neutral: "#8a93a6", happy: "#ffce54", sad: "#5aa9ff", angry: "#ff6b5e",
  fear: "#b06bff", excited: "#ff9d3d", tender: "#ff8fbf", mysterious: "#4fd6be",
  heroic: "#ffd166", ironic: "#c0f56b",
};

let books = [];            // list summaries
let current = null;        // full book json
let curIdx = 0;            // current passage index
let playing = false;
let pollTimer = null;
const audio = new Audio();
audio.preload = "auto";

const $ = (s) => document.querySelector(s);

/* ---------------- upload ---------------- */
const drop = $("#drop"), fileInput = $("#file");
["dragover", "dragenter"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add("hover"); }));
["dragleave", "drop"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove("hover"); }));
drop.addEventListener("drop", ev => { if (ev.dataTransfer.files.length) upload(ev.dataTransfer.files[0]); });
fileInput.addEventListener("change", () => { if (fileInput.files.length) upload(fileInput.files[0]); });

function upload(file) {
  $("#uperr").textContent = "";
  const bar = $("#upbar"); bar.hidden = false; bar.value = 0;
  const fd = new FormData(); fd.append("file", file);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/books");
  xhr.upload.onprogress = e => { if (e.lengthComputable) bar.value = Math.round(e.loaded / e.total * 70); };
  xhr.onload = () => {
    bar.value = 100;
    setTimeout(() => bar.hidden = true, 600);
    if (xhr.status !== 200) {
      let msg = "Ошибка загрузки";
      try { msg = JSON.parse(xhr.responseText).detail || msg; } catch {}
      $("#uperr").textContent = msg; return;
    }
    const book = JSON.parse(xhr.responseText);
    refreshLibrary().then(() => openBook(book.id));
  };
  xhr.onerror = () => { $("#uperr").textContent = "Сетевая ошибка"; bar.hidden = true; };
  xhr.send(fd);
}

/* ---------------- library ---------------- */
async function refreshLibrary() {
  const r = await fetch("/api/books");
  books = await r.json();
  const lib = $("#library");
  if (!books.length) { lib.innerHTML = '<div class="empty">Пока пусто — загрузите книгу</div>'; return; }
  lib.innerHTML = "";
  for (const b of books) {
    const div = document.createElement("div");
    div.className = "book-item" + (current && current.id === b.id ? " active" : "");
    const statusTxt = b.status === "processing" ? `озвучка ${Math.round(b.progress * 100)}%`
                    : b.status === "error" ? "ошибка" : "готово";
    div.innerHTML = `<div style="min-width:0">
        <div class="t" title="${esc(b.title)}">${esc(b.title)}</div>
        <div class="a">${esc(b.author || "")} · ${b.num_passages} фрагм.</div>
      </div>
      <span class="badge-status">${statusTxt}</span><span class="del" title="Удалить">✕</span>`;
    div.onclick = (ev) => {
      if (ev.target.classList.contains("del")) return;
      openBook(b.id);
    };
    div.querySelector(".del").onclick = async (ev) => {
      ev.stopPropagation();
      if (!confirm(`Удалить «${b.title}»?`)) return;
      await fetch(`/api/books/${b.id}`, { method: "DELETE" });
      if (current && current.id === b.id) stopAll();
      refreshLibrary();
    };
    lib.appendChild(div);
  }
}

/* ---------------- reader ---------------- */
async function openBook(id) {
  stopPolling();
  const r = await fetch(`/api/books/${id}`);
  if (!r.ok) return;
  current = await r.json();
  curIdx = firstUnplayed();
  renderReader();
  refreshLibrary();
  if (current.status === "processing") startPolling();
}

function firstUnplayed() {
  if (!current) return 0;
  const i = current.passages.findIndex(p => !p.ready);
  return i === -1 ? 0 : Math.max(0, i - 1);
}

function startPolling() {
  stopPolling();
  pollTimer = setInterval(async () => {
    if (!current) return;
    const r = await fetch(`/api/books/${current.id}`);
    if (!r.ok) return;
    const updated = await r.json();
    const wasReadyCount = readyCount(current), nowReady = readyCount(updated);
    current = updated;
    updateProgressUI();
    if (nowReady > wasReadyCount) rerenderPassageList();
    if (current.status !== "processing") { stopPolling(); refreshLibrary(); }
  }, 2500);
}
function stopPolling() { if (pollTimer) clearInterval(pollTimer); pollTimer = null; }
function readyCount(b) { return b.passages.filter(p => p.ready).length; }

function renderReader() {
  const el = $("#reader");
  if (!current) { el.innerHTML = '<div class="empty">Выберите книгу 🎧</div>'; return; }
  el.innerHTML = `
   <div class="panel now-playing" style="background:var(--panel2)">
     <div class="np-emotion-bg" id="npbg"></div>
     <div class="np-head">
       <div><b>${esc(current.title)}</b> <span class="np-chapter" id="npchapter"></span></div>
       <span class="chip" id="npchip">—</span>
     </div>
     <div class="np-text" id="nptext">…</div>
     <div class="np-reason" id="npreason"></div>
     <div class="controls" style="margin-top:10px">
       <button class="ctrl" id="prev">⏮</button>
       <button class="ctrl primary" id="playpause">▶ Играть</button>
       <button class="ctrl" id="next">⏭</button>
       <span id="pos" style="font-size:13px;color:var(--muted)"></span>
       <div class="speed" style="margin-left:auto">скорость
         <input type="range" id="rate" min="0.6" max="1.6" step="0.05" value="1">
         <span id="ratev">1.0×</span></div>
     </div>
     <progress id="genbar" max="100" value="0" style="margin-top:10px"></progress>
   </div>
   <div id="passages"></div>`;
  $("#playpause").onclick = togglePlay;
  $("#prev").onclick = () => jump(-1);
  $("#next").onclick = () => jump(+1);
  $("#rate").oninput = e => { audio.playbackRate = +e.target.value; $("#ratev").textContent = (+e.target.value).toFixed(1) + "×"; };
  rerenderPassageList();
  showPassage(curIdx);
  updateProgressUI();
}

function updateProgressUI() {
  const bar = $("#genbar"); if (!bar || !current) return;
  const done = readyCount(current), total = current.passages.length;
  bar.value = Math.round(done / total * 100);
  bar.title = `Озвучено ${done} из ${total}`;
  if (done === total) bar.style.display = "none";
}

function rerenderPassageList() {
  const list = $("#passages"); if (!list || !current) return;
  list.innerHTML = "";
  current.passages.forEach((p, i) => {
    const d = document.createElement("div");
    d.className = "passage" + (i === curIdx ? " current" : "") + (p.ready ? "" : " pending");
    d.dataset.i = i;
    const emo = p.emotion || "neutral";
    d.innerHTML = `<span class="pdot" style="background:${p.ready ? EMO_COLORS[emo] : '#3a2f58'}"></span>
      <span class="ptxt">${esc(p.text.slice(0, 180))}${p.text.length > 180 ? "…" : ""}</span>
      <span class="pemo">${p.ready ? EMO_ICONS[emo] + " " + EMO_LABELS[emo] : "⏳ озвучка…"}</span>`;
    d.onclick = () => { stopAudio(); select(i); if (playing) playCurrent(); };
    list.appendChild(d);
  });
}

function select(i) {
  curIdx = Math.max(0, Math.min(i, current.passages.length - 1));
  showPassage(curIdx);
  document.querySelectorAll(".passage").forEach(el =>
    el.classList.toggle("current", +el.dataset.i === curIdx));
  const cur = document.querySelector(".passage.current");
  if (cur) cur.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function showPassage(i) {
  const p = current.passages[i]; if (!p) return;
  const emo = p.emotion || "neutral";
  $("#nptext").textContent = p.text;
  $("#npchapter").textContent = "· " + (p.chapter || "");
  const chip = $("#npchip");
  chip.textContent = `${EMO_ICONS[emo]} ${EMO_LABELS[emo]}${p.confidence ? " " + Math.round(p.confidence * 100) + "%" : ""}`;
  chip.style.background = EMO_COLORS[emo];
  $("#npbg").style.background = `radial-gradient(600px 200px at 20% 0%, ${EMO_COLORS[emo]}, transparent)`;
  $("#npreason").textContent = p.reason ? "💡 " + p.reason : "";
  $("#pos").textContent = `${i + 1} / ${current.passages.length}`;
  $("#playpause").textContent = playing ? "⏸ Пауза" : (p.ready ? "▶ Играть" : "⏳ ждём аудио");
}

/* ---------------- playback ---------------- */
function togglePlay() {
  if (playing) { stopAudio(); return; }
  playing = true;
  playCurrent();
}

function playCurrent() {
  const p = current.passages[curIdx];
  if (!p) { playing = false; return; }
  if (!p.ready) {
    $("#playpause").textContent = "⏳ аудио генерируется…";
    startPolling();
    setTimeout(() => { if (playing && current.passages[curIdx].ready) playCurrent(); }, 2600);
    return;
  }
  audio.src = `/api/books/${current.id}/audio/${p.index}`;
  audio.playbackRate = +($("#rate")?.value || 1);
  audio.onended = () => { if (!playing) return; select(curIdx + 1); playCurrent(); };
  audio.onerror = () => { if (!playing) return; select(curIdx + 1); playCurrent(); };
  audio.play().catch(() => {});
  showPassage(curIdx);
  $("#playpause").textContent = "⏸ Пауза";
}

function stopAudio() {
  playing = false;
  audio.pause(); audio.removeAttribute("src");
  if ($("#playpause")) $("#playpause").textContent = "▶ Играть";
}
function stopAll() { stopAudio(); stopPolling(); current = null; renderReader(); refreshLibrary(); }
function jump(d) { stopAudio(); select(curIdx + d); }

function esc(s) { return (s || "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

document.addEventListener("keydown", e => {
  if (!current) return;
  if (e.code === "Space") { e.preventDefault(); togglePlay(); }
  if (e.code === "ArrowRight") jump(1);
  if (e.code === "ArrowLeft") jump(-1);
});

refreshLibrary();
