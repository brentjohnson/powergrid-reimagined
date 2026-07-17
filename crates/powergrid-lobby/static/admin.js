"use strict";

const API = "/admin/api";
const TOKEN_KEY = "pg_admin_token";

// ---- Token handling ----
function getToken() { return sessionStorage.getItem(TOKEN_KEY) || ""; }
function setToken(t) { sessionStorage.setItem(TOKEN_KEY, t); }
function clearToken() { sessionStorage.removeItem(TOKEN_KEY); }

async function api(path, opts = {}) {
  const res = await fetch(API + path, {
    ...opts,
    headers: { ...(opts.headers || {}), Authorization: "Bearer " + getToken() },
  });
  if (res.status === 401) {
    clearToken();
    showGate(true);
    throw new Error("unauthorized");
  }
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).error || msg; } catch (_) {}
    throw new Error(msg);
  }
  if (res.status === 204) return null;
  return res.json();
}

// ---- Elements ----
const gate = document.getElementById("gate");
const app = document.getElementById("app");
const view = document.getElementById("view");

function showGate(showError) {
  document.getElementById("gate-error").classList.toggle("hidden", !showError);
  gate.classList.remove("hidden");
  app.classList.add("hidden");
  document.getElementById("gate-token").focus();
}
function showApp() {
  gate.classList.add("hidden");
  app.classList.remove("hidden");
  route();
}

document.getElementById("gate-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const t = document.getElementById("gate-token").value.trim();
  if (!t) return;
  setToken(t);
  try {
    await api("/metrics"); // probe
    showApp();
  } catch (_) {
    clearToken();
    showGate(true);
  }
});

document.getElementById("lock").addEventListener("click", () => {
  clearToken();
  showGate(false);
});

// ---- Helpers ----
function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) e.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined) continue;
    e.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return e;
}
function fmtDate(s) {
  if (!s) return "—";
  const d = new Date(s);
  return d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
function fmtDay(s) {
  return new Date(s).toLocaleDateString([], { month: "short", day: "numeric" });
}
function num(v, digits = 0) {
  return v === null || v === undefined ? "—" : Number(v).toFixed(digits);
}
function setLoading() { view.replaceChildren(el("div", { class: "loading" }, "Loading…")); }
function showError(msg) { view.replaceChildren(el("div", { class: "empty error" }, "Error: " + msg)); }

// ---- Router ----
function route() {
  if (!getToken()) { showGate(false); return; }
  const hash = location.hash.slice(1) || "players";
  const [name, arg] = hash.split("/");
  document.querySelectorAll(".nav a").forEach((a) =>
    a.classList.toggle("active", a.dataset.route === name)
  );
  if (name === "players") renderPlayers();
  else if (name === "player" && arg) renderPlayerDetail(arg);
  else if (name === "metrics") renderMetrics();
  else if (name === "games") renderGames();
  else renderPlayers();
}
window.addEventListener("hashchange", route);

// ---- Players list ----
let playerSort = { key: "created_at", dir: 1 };

async function renderPlayers() {
  setLoading();
  let data;
  try { data = await api("/players"); } catch (e) { return showError(e.message); }
  const players = data.players;

  const cols = [
    { key: "username", label: "Username", num: false },
    { key: "email", label: "Email", num: false },
    { key: "created_at", label: "Created", num: false },
    { key: "last_login", label: "Last Login", num: false },
    { key: "games_played", label: "Games", num: true },
    { key: "wins", label: "Wins", num: true },
    { key: "avg_finish", label: "Avg Finish", num: true },
  ];

  function sorted() {
    const { key, dir } = playerSort;
    return [...players].sort((a, b) => {
      let x = a[key], y = b[key];
      if (x === null) return 1;
      if (y === null) return -1;
      if (typeof x === "string") return x.localeCompare(y) * dir;
      return (x - y) * dir;
    });
  }

  const container = el("div");
  container.append(el("h2", {}, "Players"));
  container.append(el("p", { class: "section-sub" }, players.length + " registered"));

  function draw() {
    const thead = el("tr", {},
      cols.map((c) => {
        const arrow = playerSort.key === c.key ? (playerSort.dir === 1 ? " ▲" : " ▼") : "";
        return el("th", {
          class: "sortable" + (c.num ? " num" : ""),
          onclick: () => {
            if (playerSort.key === c.key) playerSort.dir *= -1;
            else playerSort = { key: c.key, dir: 1 };
            draw();
          },
        }, c.label, el("span", { class: "arrow" }, arrow));
      }),
      el("th", {}, "")
    );

    const rows = sorted().map((p) => {
      const resetBtn = el("button", {
        class: "ghost",
        onclick: (ev) => { ev.stopPropagation(); confirmReset(p); },
      }, "Reset");
      return el("tr", { class: "clickable", onclick: () => { location.hash = "player/" + p.id; } },
        el("td", {}, p.username),
        el("td", { class: "muted" }, p.email),
        el("td", {}, fmtDate(p.created_at)),
        el("td", {}, fmtDate(p.last_login)),
        el("td", { class: "num" }, p.games_played),
        el("td", { class: "num" }, p.wins),
        el("td", { class: "num" }, num(p.avg_finish, 2)),
        el("td", { class: "num" }, resetBtn),
      );
    });

    const table = el("div", { class: "card table-wrap" },
      el("table", {}, el("thead", {}, thead), el("tbody", {}, rows.length ? rows : el("tr", {}, el("td", { colspan: 8, class: "empty" }, "No players"))))
    );
    const existing = container.querySelector(".card");
    if (existing) existing.replaceWith(table); else container.append(table);
  }

  draw();
  view.replaceChildren(container);
}

// ---- Player detail ----
async function renderPlayerDetail(id) {
  setLoading();
  let data;
  try { data = await api("/players/" + id); } catch (e) { return showError(e.message); }
  const p = data.player;
  const games = data.games;
  const counts = data.position_counts;

  const winRate = p.games_played > 0 ? (100 * p.wins / p.games_played).toFixed(1) + "%" : "—";

  const container = el("div");
  container.append(el("a", { class: "back-link", href: "#players" }, "← Players"));
  container.append(el("div", { class: "profile-head" }, el("h2", {}, p.username), el("span", { class: "muted" }, p.email)));

  container.append(el("div", { class: "tiles" },
    tile(p.games_played, "Games played"),
    tile(p.wins, "Wins"),
    tile(winRate, "Win rate"),
    tile(num(p.avg_finish, 2), "Avg finish"),
    tile(fmtDate(p.last_login), "Last login"),
  ));

  // Finish-position distribution
  const maxCount = Math.max(1, ...counts.map((c) => c.count));
  const distRows = counts.length ? counts.map((c) =>
    el("div", { class: "bar-row" },
      el("span", { class: "rowlabel" }, ordinal(c.finish_position)),
      el("div", { class: "bar-track" }, el("div", { class: "bar-fill" + (c.finish_position === 1 ? " win" : ""), style: `width:${(100 * c.count / maxCount).toFixed(1)}%` })),
      el("span", { class: "rowval" }, c.count),
    )
  ) : [el("div", { class: "empty" }, "No finished games")];

  container.append(el("div", { class: "grid cols-2" },
    el("div", { class: "card" }, el("h3", {}, "Finish position distribution"), el("div", { class: "bars-h" }, distRows)),
    el("div", { class: "card" }, el("h3", {}, "Recent games"), gamesTable(games)),
  ));

  view.replaceChildren(container);
}

function gamesTable(games) {
  if (!games.length) return el("div", { class: "empty" }, "No games yet");
  const rows = games.map((g) =>
    el("tr", {},
      el("td", {}, fmtDate(g.finished_at)),
      el("td", {}, g.room_name),
      el("td", {}, el("span", { class: "pill " + (g.finish_position === 1 ? "win" : "human") }, ordinal(g.finish_position))),
      el("td", { class: "num" }, g.cities),
      el("td", { class: "num" }, g.money),
      el("td", { class: "num" }, g.powered),
      el("td", { class: "num" }, g.plants),
    )
  );
  return el("div", { class: "table-wrap" }, el("table", {},
    el("thead", {}, el("tr", {},
      el("th", {}, "Date"), el("th", {}, "Room"), el("th", {}, "Place"),
      el("th", { class: "num" }, "Cities"), el("th", { class: "num" }, "Money"),
      el("th", { class: "num" }, "Powered"), el("th", { class: "num" }, "Plants"))),
    el("tbody", {}, rows)));
}

// ---- Metrics ----
async function renderMetrics() {
  setLoading();
  let m;
  try { m = await api("/metrics"); } catch (e) { return showError(e.message); }

  const container = el("div");
  container.append(el("h2", {}, "Metrics"));
  container.append(el("p", { class: "section-sub" }, "Server-wide game statistics"));

  container.append(el("div", { class: "tiles" },
    tile(m.total_users, "Users"),
    tile(m.total_games, "Games played"),
    tile(m.games_last_7d, "Games (7d)"),
    tile(num(m.avg_rounds, 1), "Avg rounds"),
    tile(num(m.avg_players, 1), "Avg players"),
  ));

  // Games per day (vertical bars)
  const gpd = m.games_per_day;
  const maxG = Math.max(1, ...gpd.map((d) => d.count));
  const vbars = gpd.length ? gpd.map((d) =>
    el("div", { class: "vbar", title: fmtDay(d.day) + ": " + d.count },
      el("div", { class: "vfill", style: `height:${(100 * d.count / maxG).toFixed(1)}%` }),
      el("div", { class: "vtip" }, d.count))
  ) : [el("div", { class: "empty" }, "No games in the last 30 days")];

  // Human vs bot wins
  const totalWins = m.human_wins + m.bot_wins;
  const humanPct = totalWins ? (100 * m.human_wins / totalWins) : 0;
  const botPct = totalWins ? (100 * m.bot_wins / totalWins) : 0;
  const splitCard = el("div", { class: "card" },
    el("h3", {}, "Wins: human vs bot"),
    totalWins ? el("div", {},
      el("div", { class: "split-bar" },
        el("div", { class: "seg", style: `flex:${humanPct};background:var(--series-1)` }, m.human_wins > 0 ? m.human_wins : ""),
        el("div", { class: "seg", style: `flex:${botPct};background:var(--series-2)` }, m.bot_wins > 0 ? m.bot_wins : ""),
      ),
      el("div", { class: "split-legend" },
        legendItem("var(--series-1)", `Human (${humanPct.toFixed(0)}%)`),
        legendItem("var(--series-2)", `Bot (${botPct.toFixed(0)}%)`),
      )
    ) : el("div", { class: "empty" }, "No wins recorded")
  );

  container.append(el("div", { class: "grid cols-2" },
    el("div", { class: "card" }, el("h3", {}, "Games per day (30d)"), el("div", { class: "bars-v" }, vbars)),
    splitCard,
  ));

  // Winner averages
  container.append(el("div", { class: "card", style: "margin-top:16px" },
    el("h3", {}, "Winner averages"),
    el("div", { class: "tiles", style: "margin:0" },
      tile(num(m.winner_avg_cities, 1), "Cities"),
      tile(num(m.winner_avg_money, 0), "Money"),
      tile(num(m.winner_avg_plants, 1), "Plants"),
      tile(num(m.winner_avg_powered, 1), "Powered"),
    )
  ));

  // Leaderboard
  const lbRows = m.leaderboard.length ? m.leaderboard.map((r, i) =>
    el("tr", {},
      el("td", { class: "num muted" }, i + 1),
      el("td", {}, r.username),
      el("td", { class: "num" }, r.games_played),
      el("td", { class: "num" }, r.wins),
      el("td", { class: "num" }, num(r.avg_finish, 2)),
    )
  ) : [el("tr", {}, el("td", { colspan: 5, class: "empty" }, "No games yet"))];
  container.append(el("div", { class: "card", style: "margin-top:16px" },
    el("h3", {}, "Leaderboard (most wins)"),
    el("div", { class: "table-wrap" }, el("table", {},
      el("thead", {}, el("tr", {},
        el("th", { class: "num" }, "#"), el("th", {}, "Player"),
        el("th", { class: "num" }, "Games"), el("th", { class: "num" }, "Wins"),
        el("th", { class: "num" }, "Avg Finish"))),
      el("tbody", {}, lbRows)))
  ));

  view.replaceChildren(container);
}

// ---- Recent games ----
async function renderGames() {
  setLoading();
  let data;
  try { data = await api("/games?limit=100"); } catch (e) { return showError(e.message); }
  const games = data.games;

  const container = el("div");
  container.append(el("h2", {}, "Recent games"));
  container.append(el("p", { class: "section-sub" }, games.length + " games"));

  const rows = games.map((g) =>
    el("tr", {},
      el("td", {}, fmtDate(g.finished_at)),
      el("td", {}, g.room_name),
      el("td", { class: "muted" }, g.map_name),
      el("td", { class: "num" }, g.num_players),
      el("td", { class: "num" }, g.rounds),
      el("td", {}, g.winner_name
        ? el("span", {}, g.winner_name, " ", el("span", { class: "pill " + (g.winner_is_bot ? "bot" : "human") }, g.winner_is_bot ? "bot" : "human"))
        : el("span", { class: "muted" }, "—")),
    )
  );
  container.append(el("div", { class: "card table-wrap" }, el("table", {},
    el("thead", {}, el("tr", {},
      el("th", {}, "Finished"), el("th", {}, "Room"), el("th", {}, "Map"),
      el("th", { class: "num" }, "Players"), el("th", { class: "num" }, "Rounds"), el("th", {}, "Winner"))),
    el("tbody", {}, rows.length ? rows : el("tr", {}, el("td", { colspan: 6, class: "empty" }, "No games recorded"))))));

  view.replaceChildren(container);
}

// ---- Small UI helpers ----
function tile(value, label) {
  return el("div", { class: "tile" }, el("div", { class: "value" }, value), el("div", { class: "label" }, label));
}
function legendItem(color, label) {
  return el("span", { class: "item" }, el("span", { class: "swatch", style: `background:${color}` }), label);
}
function ordinal(n) {
  const s = ["th", "st", "nd", "rd"], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

// ---- Modal / reset flow ----
const modal = document.getElementById("modal");
const modalBody = document.getElementById("modal-body");
const modalConfirm = document.getElementById("modal-confirm");
const modalCancel = document.getElementById("modal-cancel");

function closeModal() { modal.classList.add("hidden"); }
modalCancel.addEventListener("click", closeModal);
modal.addEventListener("click", (e) => { if (e.target === modal) closeModal(); });

function confirmReset(p) {
  modalBody.replaceChildren(
    el("h3", {}, "Reset password"),
    el("p", {}, "Generate a new temporary password for ", el("strong", {}, p.username), "? This immediately logs them out of all sessions.")
  );
  modalConfirm.textContent = "Reset password";
  modalConfirm.className = "danger";
  modalCancel.classList.remove("hidden");
  modalConfirm.onclick = async () => {
    modalConfirm.disabled = true;
    try {
      const res = await api("/players/" + p.id + "/reset-password", { method: "POST" });
      showTempPassword(p.username, res.temp_password);
    } catch (e) {
      modalBody.replaceChildren(el("h3", {}, "Error"), el("p", { class: "error" }, e.message));
    } finally {
      modalConfirm.disabled = false;
    }
  };
  modal.classList.remove("hidden");
}

function showTempPassword(username, pw) {
  const pwBox = el("div", { class: "temp-pw" }, el("span", {}, pw),
    el("button", { class: "ghost", onclick: () => navigator.clipboard && navigator.clipboard.writeText(pw) }, "Copy"));
  modalBody.replaceChildren(
    el("h3", {}, "Password reset"),
    el("p", {}, "New temporary password for ", el("strong", {}, username), ":"),
    pwBox,
    el("p", { class: "warn-text" }, "This is shown only once. Copy it now and share it securely.")
  );
  modalConfirm.textContent = "Done";
  modalConfirm.className = "";
  modalCancel.classList.add("hidden");
  modalConfirm.onclick = closeModal;
}

// ---- Boot ----
if (getToken()) {
  // Validate silently; fall back to gate on failure.
  api("/metrics").then(showApp).catch(() => showGate(false));
} else {
  showGate(false);
}
