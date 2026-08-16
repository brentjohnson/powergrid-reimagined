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
  const navName = name === "game" ? "games" : name === "player" ? "players" : name;
  document.querySelectorAll(".nav a").forEach((a) =>
    a.classList.toggle("active", a.dataset.route === navName)
  );
  if (name === "players") renderPlayers();
  else if (name === "player" && arg) renderPlayerDetail(arg);
  else if (name === "metrics") renderMetrics();
  else if (name === "games") renderGames();
  else if (name === "game" && arg) renderGameDetail(arg);
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
  const stats = data.stats || {};
  const favPlants = data.favorite_plants || [];

  const winRate = p.games_played > 0 ? (100 * p.wins / p.games_played).toFixed(1) + "%" : "—";

  const container = el("div");
  container.append(el("a", { class: "back-link", href: "#players" }, "← Players"));
  container.append(el("div", { class: "profile-head" }, el("h2", {}, p.username), el("span", { class: "muted" }, p.email)));

  container.append(el("div", { class: "tiles" },
    tile(p.games_played, "Games played"),
    tile(p.wins, "Wins"),
    tile(winRate, "Win rate"),
    tile(num(p.avg_finish, 2), "Avg finish"),
    tile(stats.best_finish != null ? ordinal(stats.best_finish) : "—", "Best finish"),
    tile(fmtDate(p.last_login), "Last login"),
  ));

  container.append(el("div", { class: "card", style: "margin-bottom:16px" },
    el("h3", {}, "Career averages (per game)"),
    el("div", { class: "tiles", style: "margin:0 0 12px" },
      tile(num(stats.avg_cities, 1), "Cities (end)"),
      tile(num(stats.avg_money, 0), "Money (end)"),
      tile(num(stats.avg_powered, 1), "Powered"),
      tile(num(stats.avg_plants, 1), "Plants (end)"),
    ),
    el("div", { class: "tiles", style: "margin:0" },
      tile(num(stats.avg_plants_bought, 1), "Plants bought"),
      tile(num(stats.avg_spent_on_plants, 0), "Spent on plants"),
      tile(num(stats.avg_resources_bought, 1), "Resources bought"),
      tile(num(stats.avg_spent_on_resources, 0), "Spent on resources"),
      tile(num(stats.avg_cities_bought, 1), "Cities built"),
      tile(num(stats.avg_spent_on_cities, 0), "Spent on cities"),
    )
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
    el("div", { class: "card" }, el("h3", {}, "Favorite plants"), favoritePlantsTable(favPlants)),
  ));

  container.append(el("div", { class: "card", style: "margin-top:16px" },
    el("h3", {}, "Recent games"), gamesTable(games)));

  view.replaceChildren(container);
}

function favoritePlantsTable(plants) {
  if (!plants.length) return el("div", { class: "empty" }, "No plants recorded");
  const rows = plants.map((p) =>
    el("tr", {},
      el("td", {}, plantBadge(p.plant_number, p.kind)),
      el("td", { class: "muted" }, kindLabel(p.kind)),
      el("td", { class: "num" }, p.capacity),
      el("td", { class: "num" }, p.times_held),
      el("td", { class: "num" }, num(p.avg_finish, 2)),
    )
  );
  return el("div", { class: "table-wrap" }, el("table", {},
    el("thead", {}, el("tr", {},
      el("th", {}, "Plant"), el("th", {}, "Fuel"), el("th", { class: "num" }, "Powers"),
      el("th", { class: "num" }, "Times held"), el("th", { class: "num" }, "Avg finish"))),
    el("tbody", {}, rows)));
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
    tile(m.total_seats, "Seats played"),
    tile(m.games_last_7d, "Games (7d)"),
    tile(num(m.avg_rounds, 1), "Avg rounds"),
    tile(num(m.avg_players, 1), "Avg players"),
    tile(m.avg_game_minutes != null ? num(m.avg_game_minutes, 1) + "m" : "—", "Avg length"),
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

  // Averages by finish position (snapshot + full economy)
  container.append(el("div", { class: "card", style: "margin-top:16px" },
    el("h3", {}, "Averages by finish position"),
    el("p", { class: "section-sub", style: "margin-top:-4px" },
      "Mean end-of-game and total-spend figures for every seat that finished in each place, across all recorded games."),
    finishAvgTable(m.finish_position_averages)));

  // Performance by AI strength & by color/turn order
  container.append(el("div", { class: "grid cols-2", style: "margin-top:16px" },
    el("div", { class: "card" }, el("h3", {}, "Performance by opponent type"),
      perfTable(m.difficulty_stats, "difficulty", (r) => difficultyBadge(r.difficulty))),
    el("div", { class: "card" }, el("h3", {}, "Performance by color"),
      perfTable(m.color_stats, "color", (r) => colorBadge(r.color))),
  ));

  // Turn order + player count + rounds histogram
  container.append(el("div", { class: "grid cols-2", style: "margin-top:16px" },
    el("div", { class: "card" }, el("h3", {}, "Win rate by seat / turn order"),
      perfTable(m.turn_order_stats, "turn_order", (r) => "Seat " + r.turn_order)),
    el("div", { class: "card" }, el("h3", {}, "Table size"),
      tableSizeTable(m.player_count_dist)),
  ));

  // Rounds histogram
  const rh = m.rounds_histogram;
  const maxR = Math.max(1, ...rh.map((d) => d.count));
  const rbars = rh.length ? rh.map((d) =>
    el("div", { class: "vbar", title: d.rounds + " rounds: " + d.count + " games" },
      el("div", { class: "vfill", style: `height:${(100 * d.count / maxR).toFixed(1)}%` }),
      el("div", { class: "vtip" }, d.count),
      el("div", { class: "vlabel" }, d.rounds))
  ) : [el("div", { class: "empty" }, "No games recorded")];
  container.append(el("div", { class: "card", style: "margin-top:16px" },
    el("h3", {}, "Game length (rounds)"), el("div", { class: "bars-v tall" }, rbars)));

  // Fuel-kind effectiveness
  container.append(el("div", { class: "card", style: "margin-top:16px" },
    el("h3", {}, "Fuel kinds (end-of-game plants)"),
    plantKindTable(m.plant_kind_stats)));

  // Per-plant effectiveness
  container.append(el("div", { class: "card", style: "margin-top:16px" },
    el("h3", {}, "Power plant effectiveness"),
    el("p", { class: "section-sub", style: "margin-top:-4px" },
      "Plants held at game end — how often each shows up and the average finish while holding it."),
    plantStatsTable(m.plant_stats)));

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
    el("tr", { class: "clickable", onclick: () => { location.hash = "game/" + g.id; } },
      el("td", {}, fmtDate(g.finished_at)),
      el("td", {}, g.room_name),
      el("td", { class: "muted" }, g.map_name),
      el("td", { class: "num" }, g.num_players),
      el("td", { class: "num" }, g.rounds),
      el("td", { class: "num muted" }, fmtDuration(g.started_at, g.finished_at)),
      el("td", {}, g.winner_name
        ? el("span", {}, g.winner_name, " ", el("span", { class: "pill " + (g.winner_is_bot ? "bot" : "human") }, g.winner_is_bot ? "bot" : "human"))
        : el("span", { class: "muted" }, "—")),
    )
  );
  container.append(el("div", { class: "card table-wrap" }, el("table", {},
    el("thead", {}, el("tr", {},
      el("th", {}, "Finished"), el("th", {}, "Room"), el("th", {}, "Map"),
      el("th", { class: "num" }, "Players"), el("th", { class: "num" }, "Rounds"),
      el("th", { class: "num" }, "Length"), el("th", {}, "Winner"))),
    el("tbody", {}, rows.length ? rows : el("tr", {}, el("td", { colspan: 7, class: "empty" }, "No games recorded"))))));

  view.replaceChildren(container);
}

// ---- Game detail ----
async function renderGameDetail(id) {
  setLoading();
  let data;
  try { data = await api("/games/" + id); } catch (e) { return showError(e.message); }
  const g = data.game;
  const seats = data.seats;
  const plants = data.plants;

  // Group plants by finish_position.
  const plantsBySeat = {};
  for (const p of plants) (plantsBySeat[p.finish_position] ||= []).push(p);

  const container = el("div");
  container.append(el("a", { class: "back-link", href: "#games" }, "← Games"));
  container.append(el("div", { class: "profile-head" },
    el("h2", {}, g.room_name), el("span", { class: "muted" }, g.map_name)));

  container.append(el("div", { class: "tiles" },
    tile(g.num_players, "Players"),
    tile(g.rounds, "Rounds"),
    tile(fmtDuration(g.started_at, g.finished_at), "Length"),
    tile(fmtDate(g.finished_at), "Finished"),
  ));

  const seatCards = seats.map((s) => {
    const held = plantsBySeat[s.finish_position] || [];
    const kindTag = s.is_bot
      ? difficultyBadge(s.bot_difficulty || "bot")
      : el("span", { class: "pill human" }, "human");
    return el("div", { class: "card seat-card" + (s.finish_position === 1 ? " winner" : "") },
      el("div", { class: "seat-head" },
        el("span", { class: "place" }, ordinal(s.finish_position)),
        colorBadge(s.color),
        el("span", { class: "seat-name" }, s.player_name),
        kindTag,
      ),
      el("div", { class: "seat-stats" },
        statChip("Cities", s.cities), statChip("Powered", s.powered),
        statChip("Money", s.money), statChip("Plants", s.plants),
        statChip("Seat", s.turn_order != null ? s.turn_order : "—"),
      ),
      s.plants_bought != null ? el("div", { class: "seat-econ" },
        econItem("Plants bought", s.plants_bought, s.spent_on_plants),
        econItem("Resources bought", s.resources_bought, s.spent_on_resources),
        econItem("Cities built", s.cities_bought, s.spent_on_cities),
      ) : null,
      held.length
        ? el("div", { class: "plant-row" }, held.map((p) => plantBadge(p.plant_number, p.kind, p.capacity)))
        : el("div", { class: "muted small" }, "No plants held at game end"),
    );
  });
  container.append(el("div", { class: "seat-list" }, seatCards));

  view.replaceChildren(container);
}

// ---- Small UI helpers ----
function tile(value, label) {
  return el("div", { class: "tile" }, el("div", { class: "value" }, value), el("div", { class: "label" }, label));
}
function statChip(label, value) {
  return el("span", { class: "chip" }, el("span", { class: "chip-k" }, label), el("span", { class: "chip-v" }, value));
}
// "<label>: <count> (spent <money>)" — a count paired with the elektro spent on it.
function econItem(label, count, spent) {
  return el("span", { class: "econ" },
    el("span", { class: "chip-k" }, label), " ",
    el("span", { class: "chip-v" }, count),
    spent != null ? el("span", { class: "muted" }, ` · spent ${spent}`) : null);
}
function legendItem(color, label) {
  return el("span", { class: "item" }, el("span", { class: "swatch", style: `background:${color}` }), label);
}
function ordinal(n) {
  const s = ["th", "st", "nd", "rd"], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}
function fmtDuration(start, end) {
  if (!start || !end) return "—";
  const secs = (new Date(end) - new Date(start)) / 1000;
  if (!(secs > 0)) return "—";
  const m = Math.floor(secs / 60), s = Math.round(secs % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

// ---- Fuel-kind palette + labels ----
const KIND_COLORS = {
  coal: "#8a5a2b", oil: "#6b6b6b", gasoroil: "#5f7d9e", gas: "#3987e5",
  uranium: "#d03b3b", wind: "#199e70",
};
const KIND_LABELS = {
  coal: "Coal", oil: "Oil", gasoroil: "Gas/Oil", gas: "Gas",
  uranium: "Uranium", wind: "Wind",
};
function kindColor(k) { return KIND_COLORS[k] || "var(--muted)"; }
function kindLabel(k) { return KIND_LABELS[k] || k; }

function plantBadge(number, kind, capacity) {
  return el("span", { class: "plant-badge", title: kindLabel(kind) + (capacity != null ? ` · powers ${capacity}` : ""), style: `border-color:${kindColor(kind)}` },
    el("span", { class: "pdot", style: `background:${kindColor(kind)}` }),
    el("span", {}, "#" + number),
    capacity != null ? el("span", { class: "muted" }, "→" + capacity) : null);
}

const COLOR_SWATCH = {
  red: "#d03b3b", blue: "#3987e5", green: "#199e70", yellow: "#e0b93a",
  purple: "#9b5fd0", white: "#dddddd",
};
function colorBadge(color) {
  const c = COLOR_SWATCH[color] || "var(--muted)";
  return el("span", { class: "color-badge" },
    el("span", { class: "swatch", style: `background:${c}` }), color);
}
function difficultyBadge(diff) {
  return el("span", { class: "pill diff-" + diff }, diff);
}

// A win-rate / avg-finish table shared by the difficulty, color and turn-order
// breakdowns. `labelFn(row)` renders the first cell.
function perfTable(rows, key, labelFn) {
  if (!rows || !rows.length) return el("div", { class: "empty" }, "No data yet");
  const body = rows.map((r) => {
    const wr = r.seats > 0 ? (100 * r.wins / r.seats) : 0;
    return el("tr", {},
      el("td", {}, labelFn(r)),
      el("td", { class: "num" }, r.seats),
      el("td", { class: "num" }, r.wins),
      el("td", {},
        el("div", { class: "mini-bar" },
          el("div", { class: "mini-fill", style: `width:${wr.toFixed(1)}%` })),
        el("span", { class: "mini-val" }, wr.toFixed(0) + "%")),
      el("td", { class: "num" }, num(r.avg_finish, 2)),
    );
  });
  return el("div", { class: "table-wrap" }, el("table", {},
    el("thead", {}, el("tr", {},
      el("th", {}, ""), el("th", { class: "num" }, "Seats"), el("th", { class: "num" }, "Wins"),
      el("th", {}, "Win rate"), el("th", { class: "num" }, "Avg finish"))),
    el("tbody", {}, body)));
}

// Averages-by-finish-position: one column per place, rows are the metrics.
function finishAvgTable(rows) {
  if (!rows || !rows.length) return el("div", { class: "empty" }, "No games yet");
  const metrics = [
    ["Seats (n)", (r) => r.seats, 0],
    ["Cities (end)", (r) => r.avg_cities, 1],
    ["Capacity (end)", (r) => r.avg_capacity, 1],
    ["Powered", (r) => r.avg_powered, 1],
    ["Money (end)", (r) => r.avg_money, 0],
    ["Plants (end)", (r) => r.avg_plants, 1],
    ["Plants bought", (r) => r.avg_plants_bought, 1],
    ["Spent on plants", (r) => r.avg_spent_on_plants, 0],
    ["Resources bought", (r) => r.avg_resources_bought, 1],
    ["Spent on resources", (r) => r.avg_spent_on_resources, 0],
    ["Spent on cities", (r) => r.avg_spent_on_cities, 0],
  ];
  const head = el("tr", {}, el("th", {}, "Metric"),
    rows.map((r) => el("th", { class: "num" }, ordinal(r.finish_position))));
  const body = metrics.map(([label, fn, digits], i) =>
    el("tr", { class: i === 0 ? "muted" : "" },
      el("td", {}, label),
      rows.map((r) => el("td", { class: "num" }, num(fn(r), digits)))));
  return el("div", { class: "table-wrap" }, el("table", {},
    el("thead", {}, head), el("tbody", {}, body)));
}

function tableSizeTable(rows) {
  if (!rows || !rows.length) return el("div", { class: "empty" }, "No games yet");
  const total = rows.reduce((a, r) => a + Number(r.count), 0) || 1;
  const body = rows.map((r) => {
    const pct = 100 * r.count / total;
    return el("tr", {},
      el("td", {}, r.num_players + "p"),
      el("td", { class: "num" }, r.count),
      el("td", {},
        el("div", { class: "mini-bar" }, el("div", { class: "mini-fill", style: `width:${pct.toFixed(1)}%` })),
        el("span", { class: "mini-val" }, pct.toFixed(0) + "%")),
      el("td", { class: "num" }, num(r.avg_rounds, 1)),
    );
  });
  return el("div", { class: "table-wrap" }, el("table", {},
    el("thead", {}, el("tr", {},
      el("th", {}, "Size"), el("th", { class: "num" }, "Games"),
      el("th", {}, "Share"), el("th", { class: "num" }, "Avg rounds"))),
    el("tbody", {}, body)));
}

function plantKindTable(rows) {
  if (!rows || !rows.length) return el("div", { class: "empty" }, "No plant data yet");
  const maxHeld = Math.max(1, ...rows.map((r) => Number(r.times_held)));
  const body = rows.map((r) => {
    const wr = r.times_held > 0 ? (100 * r.wins / r.times_held) : 0;
    return el("tr", {},
      el("td", {}, el("span", { class: "pdot", style: `background:${kindColor(r.kind)};margin-right:8px` }), kindLabel(r.kind)),
      el("td", {},
        el("div", { class: "mini-bar" }, el("div", { class: "mini-fill", style: `width:${(100 * r.times_held / maxHeld).toFixed(1)}%;background:${kindColor(r.kind)}` })),
        el("span", { class: "mini-val" }, r.times_held)),
      el("td", { class: "num" }, wr.toFixed(0) + "%"),
      el("td", { class: "num" }, num(r.avg_finish, 2)),
    );
  });
  return el("div", { class: "table-wrap" }, el("table", {},
    el("thead", {}, el("tr", {},
      el("th", {}, "Fuel"), el("th", {}, "Times held"),
      el("th", { class: "num" }, "Win rate"), el("th", { class: "num" }, "Avg finish"))),
    el("tbody", {}, body)));
}

let plantSort = { key: "plant_number", dir: 1 };
function plantStatsTable(rows) {
  if (!rows || !rows.length) return el("div", { class: "empty" }, "No plant data yet");
  const cols = [
    { key: "plant_number", label: "Plant", num: false },
    { key: "kind", label: "Fuel", num: false },
    { key: "capacity", label: "Powers", num: true },
    { key: "times_held", label: "Times held", num: true },
    { key: "win_rate", label: "Win rate", num: true },
    { key: "avg_finish", label: "Avg finish", num: true },
  ];
  const withWr = rows.map((r) => ({ ...r, win_rate: r.times_held > 0 ? r.wins / r.times_held : 0 }));
  const wrap = el("div", { class: "table-wrap" });
  function draw() {
    const { key, dir } = plantSort;
    const sorted = [...withWr].sort((a, b) => {
      let x = a[key], y = b[key];
      if (typeof x === "string") return x.localeCompare(y) * dir;
      return ((x ?? 0) - (y ?? 0)) * dir;
    });
    const head = el("tr", {}, cols.map((c) => {
      const arrow = plantSort.key === c.key ? (plantSort.dir === 1 ? " ▲" : " ▼") : "";
      return el("th", {
        class: "sortable" + (c.num ? " num" : ""),
        onclick: () => {
          if (plantSort.key === c.key) plantSort.dir *= -1;
          else plantSort = { key: c.key, dir: c.num ? -1 : 1 };
          draw();
        },
      }, c.label, el("span", { class: "arrow" }, arrow));
    }));
    const body = sorted.map((r) =>
      el("tr", {},
        el("td", {}, plantBadge(r.plant_number, r.kind)),
        el("td", { class: "muted" }, kindLabel(r.kind)),
        el("td", { class: "num" }, r.capacity),
        el("td", { class: "num" }, r.times_held),
        el("td", { class: "num" }, (100 * r.win_rate).toFixed(0) + "%"),
        el("td", { class: "num" }, num(r.avg_finish, 2)),
      ));
    wrap.replaceChildren(el("table", {}, el("thead", {}, head), el("tbody", {}, body)));
  }
  draw();
  return wrap;
}

// ---- Modal / reset flow ----
const modal = document.getElementById("modal");
const modalBody = document.getElementById("modal-body");
const modalConfirm = document.getElementById("modal-confirm");
const modalCancel = document.getElementById("modal-cancel");

function closeModal() { modal.classList.add("hidden"); }
modalCancel.addEventListener("click", closeModal);
modal.addEventListener("click", (e) => { if (e.target === modal) closeModal(); });

const MIN_PW = 8;
const MAX_PW = 128;

// Generate a readable random password (ambiguous chars omitted).
function generatePassword() {
  const chars = "abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  const arr = new Uint32Array(16);
  crypto.getRandomValues(arr);
  let s = "";
  for (let i = 0; i < arr.length; i++) s += chars[arr[i] % chars.length];
  return s;
}

function confirmReset(p) {
  const errLine = el("p", { class: "error hidden" });
  const input = el("input", {
    type: "text",
    class: "pw-input",
    placeholder: "New password",
    autocomplete: "new-password",
    spellcheck: "false",
  });
  const genBtn = el("button", {
    class: "ghost", type: "button",
    onclick: () => { input.value = generatePassword(); errLine.classList.add("hidden"); input.focus(); },
  }, "Generate");

  modalBody.replaceChildren(
    el("h3", {}, "Reset password"),
    el("p", {}, "Set a new password for ", el("strong", {}, p.username), ". This immediately logs them out of all sessions."),
    el("div", { class: "pw-row" }, input, genBtn),
    errLine
  );
  modalConfirm.textContent = "Save";
  modalConfirm.className = "";
  modalCancel.classList.remove("hidden");
  modalConfirm.onclick = async () => {
    const pw = input.value;
    if (pw.length < MIN_PW || pw.length > MAX_PW) {
      errLine.textContent = "Password must be " + MIN_PW + "–" + MAX_PW + " characters.";
      errLine.classList.remove("hidden");
      input.focus();
      return;
    }
    modalConfirm.disabled = true;
    try {
      await api("/players/" + p.id + "/reset-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password: pw }),
      });
      showResetDone(p.username);
    } catch (e) {
      errLine.textContent = e.message;
      errLine.classList.remove("hidden");
    } finally {
      modalConfirm.disabled = false;
    }
  };
  modal.classList.remove("hidden");
  input.focus();
}

function showResetDone(username) {
  modalBody.replaceChildren(
    el("h3", {}, "Password reset"),
    el("p", {}, "The password for ", el("strong", {}, username), " has been updated and their sessions were revoked.")
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
