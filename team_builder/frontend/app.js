function getApiCandidates() {
  const fromQuery = new URLSearchParams(window.location.search).get("apiBase");
  const host = window.location.hostname;
  const list = [
    fromQuery,
    `${window.location.protocol}//${host}:8000`,
    "http://127.0.0.1:8000",
    "http://localhost:8000",
  ].filter(Boolean);
  return Array.from(new Set(list));
}

const TEAM_OPTIONS = [
  { id: 1611, name: "맨유 (Manchester United)" },
  { id: 1625, name: "맨시티 (Manchester City)" },
  { id: 1644, name: "리버풀 (Liverpool)" },
  { id: 1628, name: "첼시 (Chelsea)" },
  { id: 1651, name: "아스널 (Arsenal)" },
  { id: 1633, name: "토트넘 (Tottenham)" },
  { id: 1662, name: "바르셀로나 (Barcelona)" },
  { id: 1673, name: "레알 마드리드 (Real Madrid)" },
  { id: 1680, name: "AT 마드리드 (Atletico Madrid)" },
  { id: 1697, name: "바이에른 뮌헨 (Bayern Munich)" },
  { id: 1718, name: "PSG" },
  { id: 1762, name: "유벤투스 (Juventus)" },
];

function getApiCandidates() {
  const fromQuery = new URLSearchParams(window.location.search).get("apiBase");
  const host = window.location.hostname;
  const list = [
    fromQuery,
    `${window.location.protocol}//${host}:8000`,
    "http://127.0.0.1:8000",
    "http://localhost:8000",
  ].filter(Boolean);
  return Array.from(new Set(list));
}

const state = {
  apiBase: null,
  teamId: 1611,
  opponentId: 1625,
  activeAlternative: null,
  activeTacticTeam: "our",
  tacticData: {
    homeAttack: [],
    homeDefense: [],
    awayAttack: [],
    awayDefense: [],
  },
  tacticWeights: {
    homeAttack: {},
    homeDefense: {},
    awayAttack: {},
    awayDefense: {},
  },
  mode: "offensive",
  selectedPlayers: [],
  candidatePlayers: [],
  opponentPlayers: [],
  allPlayers: [],
  alternatives: [],
  constraints: {
    forwards: { min: 1, max: 3 },
    midfielders: { min: 3, max: 5 },
    defenders: { min: 3, max: 5 },
    goalkeeper: { min: 1, max: 1 },
  },
  synergy: {
    offensive: { yPlayers: [], xGroups: [], matrix: [] },
    defensive: { yPlayers: [], xGroups: [], matrix: [] },
  },
};

async function resolveApiBase() {
  if (state.apiBase) return state.apiBase;
  const candidates = getApiCandidates();
  for (const base of candidates) {
    try {
      const res = await fetch(`${base}/health`, { method: "GET" });
      if (res.ok) {
        state.apiBase = base;
        return base;
      }
    } catch {
      // try next candidate
    }
  }
  return null;
}

const dom = {
  teamId: document.getElementById("teamId"),
  opponentId: document.getElementById("opponentId"),
  teamNameSelect: document.getElementById("teamNameSelect"),
  opponentNameSelect: document.getElementById("opponentNameSelect"),
  runOptimizeBtn: document.getElementById("runOptimizeBtn"),
  tacticTeamToggle: document.getElementById("tacticTeamToggle"),
  tacticColTitleLeft: document.getElementById("tacticColTitleLeft"),
  tacticColTitleRight: document.getElementById("tacticColTitleRight"),
  homeTacticList: document.getElementById("homeTacticList"),
  awayTacticList: document.getElementById("awayTacticList"),
  mainPitch: document.getElementById("mainPitch"),
  candidateList: document.getElementById("candidateList"),
  lineupAlternatives: document.getElementById("lineupAlternatives"),
  matrixYLabels: document.getElementById("matrixYLabels"),
  matrixMain: document.getElementById("matrixMain"),
  matrixXGroups: document.getElementById("matrixXGroups"),
  synergyMatrix: document.getElementById("synergyMatrix"),
  tacticLegend: document.getElementById("tacticLegend"),
  constraintsPanel: document.getElementById("constraintsPanel"),
  lineupModal: document.getElementById("lineupModal"),
  modalTitle: document.getElementById("modalTitle"),
  modalSpinner: document.getElementById("modalSpinner"),
  modalBody: document.getElementById("modalBody"),
  closeModalBtn: document.getElementById("closeModalBtn"),
  toast: document.getElementById("toast"),
};

function showToast(message) {
  dom.toast.textContent = message;
  dom.toast.classList.add("show");
  setTimeout(() => dom.toast.classList.remove("show"), 2200);
}

function populateTeamSelectors() {
  const selectors = [dom.teamNameSelect, dom.opponentNameSelect];
  selectors.forEach((sel) => {
    if (!sel) return;
    sel.innerHTML = "";
    TEAM_OPTIONS.forEach((team) => {
      const opt = document.createElement("option");
      opt.value = String(team.id);
      opt.textContent = team.name;
      sel.appendChild(opt);
    });
  });
}

async function loadTeamsFromApi() {
  const base = await resolveApiBase();
  if (!base) return;
  try {
    const res = await fetch(`${base}/api/teams`);
    if (!res.ok) throw new Error(`teams api ${res.status}`);
    const data = await res.json();
    const teams = Array.isArray(data?.teams) ? data.teams : [];
    if (teams.length > 0) {
      TEAM_OPTIONS.length = 0;
      teams.forEach((team) => {
        const id = Number(team.id ?? team.team_id);
        const name = String(team.name ?? team.team_name ?? `Team ${id}`);
        if (Number.isFinite(id)) TEAM_OPTIONS.push({ id, name });
      });
    }
  } catch {
    // fallback to built-in TEAM_OPTIONS
  }
}

function syncNameSelectorsFromIds() {
  if (dom.teamNameSelect) dom.teamNameSelect.value = String(state.teamId);
  if (dom.opponentNameSelect) dom.opponentNameSelect.value = String(state.opponentId);
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function normalizeByMax(values, fallback = 1) {
  const max = Math.max(...values, fallback);
  return values.map((v) => (max > 0 ? v / max : 0));
}

function defaultTacticData() {
  return {
    homeAttack: [
      { id: "ha_1", name: "Build-up Left", startAction: "Pass", frequency: 0.82, successRate: 0.64, weight: 1.0, path: [[10, 70], [30, 55], [52, 42], [76, 35]] },
      { id: "ha_2", name: "Direct Counter", startAction: "Free kick", frequency: 0.61, successRate: 0.58, weight: 1.0, path: [[14, 52], [40, 40], [72, 24]] },
      { id: "ha_3", name: "Half-space Cut", startAction: "Corner kick", frequency: 0.47, successRate: 0.53, weight: 1.0, path: [[18, 65], [42, 52], [58, 43], [74, 31]] },
      { id: "ha_4", name: "Switch Play", startAction: "Pass", frequency: 0.56, successRate: 0.61, weight: 1.0, path: [[16, 58], [44, 54], [74, 42]] },
      { id: "ha_5", name: "Set Piece A", startAction: "Free kick", frequency: 0.39, successRate: 0.66, weight: 1.0, path: [[12, 46], [34, 38], [62, 26]] },
      { id: "ha_6", name: "Corner Near Post", startAction: "Corner kick", frequency: 0.33, successRate: 0.57, weight: 1.0, path: [[10, 82], [20, 62], [34, 40]] },
    ],
    homeDefense: [
      { id: "hd_1", name: "Press 1st Line", startAction: "Intercept", frequency: 0.69, successRate: 0.56, weight: 1.0, path: [[82, 28], [64, 36], [40, 46], [22, 56]] },
      { id: "hd_2", name: "Recover Mid", startAction: "Foul", frequency: 0.62, successRate: 0.61, weight: 1.0, path: [[78, 36], [58, 42], [38, 48], [20, 58]] },
      { id: "hd_3", name: "Box Compact", startAction: "Tackle", frequency: 0.51, successRate: 0.66, weight: 1.0, path: [[84, 30], [66, 38], [44, 48]] },
      { id: "hd_4", name: "Counter Press", startAction: "Intercept", frequency: 0.57, successRate: 0.63, weight: 1.0, path: [[70, 32], [54, 44]] },
      { id: "hd_5", name: "Smart Foul", startAction: "Foul", frequency: 0.44, successRate: 0.69, weight: 1.0, path: [[66, 46], [52, 52]] },
      { id: "hd_6", name: "Last-man Tackle", startAction: "Tackle", frequency: 0.36, successRate: 0.59, weight: 1.0, path: [[78, 24], [60, 36]] },
    ],
    awayAttack: [
      { id: "aa_1", name: "Right Channel Run", startAction: "Pass", frequency: 0.71, successRate: 0.59, weight: 1.0, path: [[12, 38], [34, 44], [58, 36], [80, 28]] },
      { id: "aa_2", name: "Central Through", startAction: "Free kick", frequency: 0.55, successRate: 0.52, weight: 1.0, path: [[16, 50], [38, 46], [60, 38], [78, 30]] },
      { id: "aa_3", name: "Wide Switch", startAction: "Corner kick", frequency: 0.44, successRate: 0.49, weight: 1.0, path: [[12, 62], [34, 54], [58, 44], [82, 36]] },
      { id: "aa_4", name: "Diagonal Pass", startAction: "Pass", frequency: 0.63, successRate: 0.61, weight: 1.0, path: [[16, 60], [42, 50], [76, 34]] },
      { id: "aa_5", name: "FK Routine", startAction: "Free kick", frequency: 0.37, successRate: 0.65, weight: 1.0, path: [[18, 48], [34, 34], [62, 24]] },
      { id: "aa_6", name: "Corner Far Post", startAction: "Corner kick", frequency: 0.34, successRate: 0.56, weight: 1.0, path: [[12, 18], [24, 30], [40, 42]] },
    ],
    awayDefense: [
      { id: "ad_1", name: "Mid Block", startAction: "Intercept", frequency: 0.76, successRate: 0.62, weight: 1.0, path: [[76, 34], [58, 40], [38, 50], [16, 58]] },
      { id: "ad_2", name: "Press Trap", startAction: "Foul", frequency: 0.55, successRate: 0.51, weight: 1.0, path: [[70, 28], [56, 36], [42, 48], [26, 60]] },
      { id: "ad_3", name: "Box Protect", startAction: "Tackle", frequency: 0.40, successRate: 0.68, weight: 1.0, path: [[82, 30], [68, 36], [52, 44], [36, 50]] },
      { id: "ad_4", name: "Lane Cut", startAction: "Intercept", frequency: 0.58, successRate: 0.64, weight: 1.0, path: [[72, 42], [54, 48]] },
      { id: "ad_5", name: "Tactical Foul", startAction: "Foul", frequency: 0.43, successRate: 0.67, weight: 1.0, path: [[66, 54], [48, 60]] },
      { id: "ad_6", name: "Edge Tackle", startAction: "Tackle", frequency: 0.32, successRate: 0.6, weight: 1.0, path: [[80, 26], [64, 34]] },
    ],
  };
}

function defaultPlayers() {
  const selectedPlayers = [
    { id: 1, number: 1, name: "D. de Gea", position: "GK", x: 50, y: 92, vi: 0.1, io: 0.03, idv: 0.12 },
    { id: 2, number: 25, name: "A. Valencia", position: "DF", x: 20, y: 72, vi: 0.2, io: 0.08, idv: 0.2 },
    { id: 3, number: 12, name: "C. Smalling", position: "DF", x: 38, y: 70, vi: 0.19, io: 0.07, idv: 0.18 },
    { id: 4, number: 4, name: "P. Jones", position: "DF", x: 62, y: 70, vi: 0.21, io: 0.08, idv: 0.17 },
    { id: 5, number: 18, name: "A. Young", position: "DF", x: 80, y: 72, vi: 0.18, io: 0.09, idv: 0.16 },
    { id: 6, number: 31, name: "N. Matic", position: "MF", x: 28, y: 52, vi: 0.28, io: 0.15, idv: 0.11 },
    { id: 7, number: 6, name: "P. Pogba", position: "MF", x: 50, y: 48, vi: 0.31, io: 0.19, idv: 0.10 },
    { id: 8, number: 14, name: "J. Lingard", position: "MF", x: 72, y: 52, vi: 0.27, io: 0.14, idv: 0.12 },
    { id: 9, number: 11, name: "A. Martial", position: "FW", x: 24, y: 28, vi: 0.35, io: 0.21, idv: 0.06 },
    { id: 10, number: 9, name: "R. Lukaku", position: "FW", x: 50, y: 22, vi: 0.39, io: 0.25, idv: 0.04 },
    { id: 11, number: 19, name: "M. Rashford", position: "FW", x: 76, y: 28, vi: 0.34, io: 0.22, idv: 0.05 },
  ];

  const mapPlayersToFormation = (points, playerPool = selectedPlayers) => playerPool.map((p, i) => ({ ...p, x: points[i][0], y: points[i][1] }));

  const alt2Players = [
    { id: 1, number: 1, name: "D. de Gea", position: "GK", vi: 0.11, io: 0.03, idv: 0.12 },
    { id: 2, number: 25, name: "A. Valencia", position: "DF", vi: 0.19, io: 0.08, idv: 0.2 },
    { id: 3, number: 3, name: "Bailly", position: "DF", vi: 0.2, io: 0.07, idv: 0.19 },
    { id: 4, number: 5, name: "Rojo", position: "DF", vi: 0.2, io: 0.08, idv: 0.18 },
    { id: 5, number: 18, name: "A. Young", position: "DF", vi: 0.17, io: 0.09, idv: 0.16 },
    { id: 6, number: 31, name: "N. Matic", position: "MF", vi: 0.27, io: 0.14, idv: 0.11 },
    { id: 7, number: 21, name: "Herrera", position: "MF", vi: 0.25, io: 0.16, idv: 0.1 },
    { id: 8, number: 14, name: "J. Lingard", position: "MF", vi: 0.29, io: 0.15, idv: 0.1 },
    { id: 9, number: 7, name: "Alexis", position: "FW", vi: 0.34, io: 0.24, idv: 0.06 },
    { id: 10, number: 9, name: "R. Lukaku", position: "FW", vi: 0.38, io: 0.24, idv: 0.05 },
    { id: 11, number: 11, name: "A. Martial", position: "FW", vi: 0.33, io: 0.21, idv: 0.06 },
  ];

  const alt3Players = [
    { id: 1, number: 1, name: "D. de Gea", position: "GK", vi: 0.1, io: 0.03, idv: 0.13 },
    { id: 2, number: 25, name: "A. Valencia", position: "DF", vi: 0.19, io: 0.07, idv: 0.2 },
    { id: 3, number: 12, name: "C. Smalling", position: "DF", vi: 0.2, io: 0.08, idv: 0.18 },
    { id: 4, number: 4, name: "P. Jones", position: "DF", vi: 0.21, io: 0.08, idv: 0.17 },
    { id: 5, number: 5, name: "Rojo", position: "DF", vi: 0.2, io: 0.09, idv: 0.16 },
    { id: 6, number: 27, name: "Fellaini", position: "MF", vi: 0.26, io: 0.12, idv: 0.12 },
    { id: 7, number: 6, name: "P. Pogba", position: "MF", vi: 0.32, io: 0.2, idv: 0.1 },
    { id: 8, number: 8, name: "Mata", position: "MF", vi: 0.28, io: 0.17, idv: 0.1 },
    { id: 9, number: 19, name: "M. Rashford", position: "FW", vi: 0.35, io: 0.23, idv: 0.05 },
    { id: 10, number: 9, name: "R. Lukaku", position: "FW", vi: 0.37, io: 0.24, idv: 0.05 },
    { id: 11, number: 7, name: "Alexis", position: "FW", vi: 0.34, io: 0.22, idv: 0.05 },
  ];

  const opponentPlayers = [
    { id: 201, number: 31, name: "Ederson" },
    { id: 202, number: 2, name: "Walker" },
    { id: 203, number: 5, name: "Stones" },
    { id: 204, number: 14, name: "Laporte" },
    { id: 205, number: 22, name: "Mendy" },
    { id: 206, number: 25, name: "Fernandinho" },
    { id: 207, number: 17, name: "K. De Bruyne" },
    { id: 208, number: 21, name: "D. Silva" },
    { id: 209, number: 7, name: "Sterling" },
    { id: 210, number: 10, name: "Aguero" },
    { id: 211, number: 19, name: "Sane" },
    { id: 212, number: 8, name: "Gundogan" },
    { id: 213, number: 20, name: "B. Silva" },
    { id: 214, number: 30, name: "Otamendi" },
  ];

  return {
    selectedPlayers,
    candidatePlayers: [
      { id: 12, number: 8, name: "Mata", vi: 0.24, io: 0.12, idv: 0.09 },
      { id: 13, number: 16, name: "Carrick", vi: 0.18, io: 0.16, idv: 0.08 },
      { id: 14, number: 27, name: "Fellaini", vi: 0.27, io: 0.08, idv: 0.06 },
      { id: 15, number: 3, name: "Bailly", vi: 0.14, io: 0.12, idv: 0.13 },
      { id: 16, number: 5, name: "Rojo", vi: 0.22, io: 0.1, idv: 0.11 },
      { id: 17, number: 21, name: "Herrera", vi: 0.2, io: 0.17, idv: 0.09 },
      { id: 18, number: 7, name: "Alexis", vi: 0.3, io: 0.2, idv: 0.05 },
    ],
    alternatives: [
      { name: "대안 1", winningRate: 0.54, appearCount: 14, expectedStats: { shots: 12.8, goals: 1.72, intercepts: 9.6, tackles: 15.1 }, formationPoints: [[50,92],[20,72],[38,70],[62,70],[80,72],[28,52],[50,48],[72,52],[24,28],[50,22],[76,28]], players: mapPlayersToFormation([[50,92],[20,72],[38,70],[62,70],[80,72],[28,52],[50,48],[72,52],[24,28],[50,22],[76,28]]) },
      { name: "대안 2", winningRate: 0.58, appearCount: 9, expectedStats: { shots: 13.6, goals: 1.91, intercepts: 8.9, tackles: 14.2 }, formationPoints: [[50,92],[14,74],[34,70],[50,67],[68,70],[86,74],[28,53],[50,49],[72,53],[40,29],[60,29]], players: mapPlayersToFormation([[50,92],[14,74],[34,70],[50,67],[68,70],[86,74],[28,53],[50,49],[72,53],[40,29],[60,29]], alt2Players) },
      { name: "대안 3", winningRate: 0.61, appearCount: 6, expectedStats: { shots: 14.1, goals: 2.04, intercepts: 8.1, tackles: 13.8 }, formationPoints: [[50,92],[18,73],[38,69],[62,69],[82,73],[22,50],[40,47],[60,47],[78,50],[38,26],[62,24]], players: mapPlayersToFormation([[50,92],[18,73],[38,69],[62,69],[82,73],[22,50],[40,47],[60,47],[78,50],[38,26],[62,24]], alt3Players) },
    ],
    opponentPlayers,
  };
}

function defaultSynergy() {
  const ourYPlayers = [
    { number: 25, name: "A. Valencia" },
    { number: 12, name: "C. Smalling" },
    { number: 4, name: "P. Jones" },
    { number: 31, name: "N. Matic" },
    { number: 6, name: "P. Pogba" },
    { number: 14, name: "J. Lingard" },
    { number: 11, name: "A. Martial" },
    { number: 9, name: "R. Lukaku" },
    { number: 19, name: "M. Rashford" },
    { number: 1, name: "D. de Gea" },
    { number: 18, name: "A. Young" },
  ];
  const oppAttackers = [
    { number: 10, name: "S. Agüero" },
    { number: 7, name: "R. Sterling" },
    { number: 19, name: "L. Sané" },
    { number: 17, name: "K. De Bruyne" },
    { number: 21, name: "D. Silva" },
    { number: 25, name: "Fernandinho" },
  ];
  const xGroups = [
    { group: "F", players: [{ number: 11, name: "A. Martial" }, { number: 9, name: "R. Lukaku" }, { number: 19, name: "M. Rashford" }] },
    { group: "M", players: [{ number: 31, name: "N. Matic" }, { number: 6, name: "P. Pogba" }, { number: 14, name: "J. Lingard" }] },
    { group: "D", players: [{ number: 25, name: "A. Valencia" }, { number: 12, name: "C. Smalling" }, { number: 4, name: "P. Jones" }, { number: 18, name: "A. Young" }] },
    { group: "G", players: [{ number: 1, name: "D. de Gea" }] },
  ];

  function randomMatrix() {
    const totalCols = xGroups.flatMap((g) => g.players).length;
    return Array.from({ length: ourYPlayers.length }, () =>
      Array.from({ length: totalCols }, () => Number((Math.random() * 0.9 + 0.1).toFixed(3)))
    );
  }

  function randomMatrixWithRows(rowCount) {
    const totalCols = xGroups.flatMap((g) => g.players).length;
    return Array.from({ length: rowCount }, () =>
      Array.from({ length: totalCols }, () => Number((Math.random() * 0.9 + 0.1).toFixed(3)))
    );
  }

  return {
    offensive: { yPlayers: ourYPlayers, xGroups, matrix: randomMatrix() },
    defensive: {
      yPlayers: ourYPlayers,
      xPlayers: oppAttackers,
      matrix: randomMatrixWithRows(ourYPlayers.length).map((r) =>
        r.slice(0, oppAttackers.length).map((v) => (Math.random() < 0.18 ? 0 : v))
      ),
    },
  };
}

function toPlayerLabel(player) {
  if (typeof player === "string") return player;
  if (!player) return "-";
  const num = player.number ?? player.jersey_number ?? player.jerseyNumber;
  const name = player.name ?? player.player_name ?? player.playerName ?? "Unknown";
  return num !== undefined && num !== null ? `${num} ${name}` : name;
}

function snapToCellCenter(value) {
  const centers = [100 / 6, 50, 500 / 6];
  let nearest = centers[0];
  let minDistance = Math.abs(value - centers[0]);
  for (let idx = 1; idx < centers.length; idx += 1) {
    const dist = Math.abs(value - centers[idx]);
    if (dist < minDistance) {
      minDistance = dist;
      nearest = centers[idx];
    }
  }
  return nearest;
}

function snapPathToGridCenters(path) {
  return (path || []).map(([x, y]) => [snapToCellCenter(Number(x)), snapToCellCenter(Number(y))]);
}

function createActionIconSvg(actionType, size = 20) {
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));

  const mk = (tag, attrs) => {
    const el = document.createElementNS(svgNS, tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, String(v)));
    return el;
  };

  const stroke = "#ffffff";
  const lower = String(actionType || "pass").toLowerCase();

  if (lower.includes("corner")) {
    svg.appendChild(mk("path", { d: "M3 3 v7 h7", fill: "none", stroke: "#f8fafc", "stroke-width": 1.6 }));
    svg.appendChild(mk("path", { d: "M7 8 C11 9, 14 11, 18 15", fill: "none", stroke, "stroke-width": 1.5 }));
    svg.appendChild(mk("path", { d: "M18 15 l-1.9 -0.4 l0.8 -1.4 Z", fill: stroke }));
  } else if (lower.includes("free")) {
    svg.appendChild(mk("path", { d: "M4 16 A6 6 0 0 1 12 10", fill: "none", stroke: "#f8fafc", "stroke-width": 1.4 }));
    svg.appendChild(mk("rect", { x: 13, y: 10, width: 2.2, height: 7, fill: "#94a3b8", rx: 0.7 }));
    svg.appendChild(mk("rect", { x: 16, y: 10, width: 2.2, height: 7, fill: "#94a3b8", rx: 0.7 }));
    svg.appendChild(mk("rect", { x: 19, y: 10, width: 2.2, height: 7, fill: "#94a3b8", rx: 0.7 }));
  } else if (lower.includes("pass")) {
    svg.appendChild(mk("path", { d: "M4 18 L19 7", stroke, "stroke-width": 1.7, fill: "none" }));
    svg.appendChild(mk("path", { d: "M19 7 l-2.3 0.35 l1.15 1.55 Z", fill: stroke }));
  } else if (lower.includes("tackle")) {
    svg.appendChild(mk("path", { d: "M5 6 L19 18", stroke: "#f8fafc", "stroke-width": 1.8, fill: "none" }));
    svg.appendChild(mk("path", { d: "M19 6 L5 18", stroke: "#f8fafc", "stroke-width": 1.8, fill: "none" }));
  } else if (lower.includes("intercept")) {
    svg.appendChild(mk("path", { d: "M4 16 L20 8", stroke: "#f8fafc", "stroke-width": 1.6, fill: "none" }));
    svg.appendChild(mk("path", { d: "M11 11 L14 14", stroke: "#f59e0b", "stroke-width": 2.2, fill: "none" }));
  } else if (lower.includes("foul")) {
    svg.appendChild(mk("circle", { cx: 10, cy: 11, r: 4.5, fill: "none", stroke: "#f8fafc", "stroke-width": 1.4 }));
    svg.appendChild(mk("rect", { x: 13.5, y: 9.3, width: 6, height: 2.1, fill: "#f8fafc", rx: 0.7 }));
    svg.appendChild(mk("circle", { cx: 8.1, cy: 17.2, r: 1.2, fill: "#f8fafc" }));
    svg.appendChild(mk("circle", { cx: 11.2, cy: 18.3, r: 1.1, fill: "#f8fafc" }));
  } else {
    svg.appendChild(mk("path", { d: "M4 18 L19 7", stroke, "stroke-width": 1.7, fill: "none" }));
    svg.appendChild(mk("path", { d: "M19 7 l-2.3 0.35 l1.15 1.55 Z", fill: stroke }));
  }
  return svg;
}

function renderTacticLegend() {
  if (!dom.tacticLegend) return;
  dom.tacticLegend.innerHTML = "";
  const groups = [
    { cls: "off", items: ["Corner kick", "Free kick", "Pass"] },
    { cls: "def", items: ["Tackle", "Intercept", "Foul"] },
  ];

  groups.forEach((group) => {
    const panel = document.createElement("div");
    panel.className = `legend-group ${group.cls}`;

    group.items.forEach((item) => {
    const wrap = document.createElement("div");
    wrap.className = "legend-item";
      const icon = document.createElement("span");
      icon.className = `legend-icon legend-badge ${group.cls}`;
      icon.appendChild(createActionIconSvg(item, 16));
    const text = document.createElement("span");
    text.className = "legend-text";
      text.textContent = item.toLowerCase().replace(" kick", "kick");
    wrap.append(icon, text);
      panel.appendChild(wrap);
    });

    dom.tacticLegend.appendChild(panel);
  });
}

function drawSymbolInsideSvg(svg, startPoint, actionType, theme = "off") {
  if (!svg || !startPoint) return;
  const svgNS = "http://www.w3.org/2000/svg";
  const [cx, cy] = startPoint;
  const baseColor = theme === "off" ? "#6ea3d7" : "#f39a34";
  const white = "#ffffff";
  const k = 2.1;

  const sx = (dx) => cx + dx * k;
  const sy = (dy) => cy + dy * k;
  const sw = (v) => Math.max(0.7, v * k);

  const mk = (tag, attrs) => {
    const el = document.createElementNS(svgNS, tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, String(v)));
    return el;
  };

  const base = mk("circle", { cx, cy, r: 9.2, fill: baseColor });
  svg.appendChild(base);

  const lower = String(actionType || "pass").toLowerCase();
  if (lower.includes("corner")) {
    svg.appendChild(mk("path", { d: `M${sx(-2.2)} ${sy(2.2)} V${sy(-1.8)} H${sx(2.0)}`, fill: "none", stroke: white, "stroke-width": sw(0.9) }));
  } else if (lower.includes("free")) {
    svg.appendChild(mk("path", { d: `M${sx(-2.4)} ${sy(1.6)} A${2.8 * k} ${2.8 * k} 0 0 1 ${sx(-0.3)} ${sy(-1.9)}`, fill: "none", stroke: white, "stroke-width": sw(0.85) }));
    svg.appendChild(mk("line", { x1: sx(0.7), y1: sy(-1.5), x2: sx(0.7), y2: sy(1.8), stroke: white, "stroke-width": sw(0.75) }));
    svg.appendChild(mk("line", { x1: sx(1.7), y1: sy(-1.5), x2: sx(1.7), y2: sy(1.8), stroke: white, "stroke-width": sw(0.75) }));
  } else if (lower.includes("tackle")) {
    svg.appendChild(mk("line", { x1: sx(-1.9), y1: sy(-1.9), x2: sx(1.9), y2: sy(1.9), stroke: white, "stroke-width": sw(0.95) }));
    svg.appendChild(mk("line", { x1: sx(1.9), y1: sy(-1.9), x2: sx(-1.9), y2: sy(1.9), stroke: white, "stroke-width": sw(0.95) }));
  } else if (lower.includes("intercept")) {
    svg.appendChild(mk("line", { x1: sx(-2.2), y1: sy(0), x2: sx(2.2), y2: sy(0), stroke: white, "stroke-width": sw(0.95) }));
    svg.appendChild(mk("line", { x1: sx(-0.4), y1: sy(-1.3), x2: sx(0.4), y2: sy(1.3), stroke: white, "stroke-width": sw(0.95) }));
  } else if (lower.includes("foul")) {
    svg.appendChild(mk("line", { x1: sx(-0.2), y1: sy(-1.9), x2: sx(-0.2), y2: sy(0.8), stroke: white, "stroke-width": sw(1.0) }));
    svg.appendChild(mk("circle", { cx: sx(-0.2), cy: sy(1.7), r: 0.36 * k, fill: white }));
  } else {
    svg.appendChild(mk("line", { x1: sx(-2.1), y1: sy(0), x2: sx(1.7), y2: sy(0), stroke: white, "stroke-width": sw(0.95) }));
    svg.appendChild(mk("path", { d: `M${sx(1.7)} ${sy(0)} l${-0.95 * k} ${-0.65 * k} l0 ${1.3 * k} Z`, fill: white }));
  }
}

function renderMiniTrajectory(container, path, theme = "off", actionType = "Pass", isDefense = false) {
  container.innerHTML = "";
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", "0 0 100 100");
  const points = snapPathToGridCenters(path);
  const start = points[0] || [50, 50];

  const defs = document.createElementNS(svgNS, "defs");
  const marker = document.createElementNS(svgNS, "marker");
  marker.setAttribute("id", `arrow-${Math.random().toString(36).slice(2)}`);
  marker.setAttribute("markerWidth", "8");
  marker.setAttribute("markerHeight", "8");
  marker.setAttribute("refX", "7");
  marker.setAttribute("refY", "4");
  marker.setAttribute("orient", "auto");
  const arrowPath = document.createElementNS(svgNS, "path");
  arrowPath.setAttribute("d", "M0,0 L8,4 L0,8 Z");
  arrowPath.setAttribute("fill", "#111111");
  marker.appendChild(arrowPath);
  defs.appendChild(marker);
  svg.appendChild(defs);

  if (!isDefense && points.length >= 2) {
    const d = points
      .map(([x, y], idx) => `${idx === 0 ? "M" : "L"}${x} ${y}`)
      .join(" ");
    const traj = document.createElementNS(svgNS, "path");
    traj.setAttribute("d", d);
    traj.setAttribute("fill", "none");
    traj.setAttribute("stroke", "#111111");
    traj.setAttribute("stroke-width", "1.4");
    traj.setAttribute("stroke-opacity", "0.95");
    traj.setAttribute("marker-end", `url(#${marker.getAttribute("id")})`);
    svg.appendChild(traj);

    points.slice(1).forEach(([x, y]) => {
      const node = document.createElementNS(svgNS, "circle");
      node.setAttribute("cx", x);
      node.setAttribute("cy", y);
      node.setAttribute("r", "1.6");
      node.setAttribute("fill", theme === "off" ? "#6ea3d7" : "#f39a34");
      svg.appendChild(node);
    });
  }

  const last = points[points.length - 1];
  if (last) {
    const endNode = document.createElementNS(svgNS, "circle");
    endNode.setAttribute("cx", last[0]);
    endNode.setAttribute("cy", last[1]);
    endNode.setAttribute("r", "2.2");
    endNode.setAttribute("fill", theme === "off" ? "#6ea3d7" : "#f39a34");
    svg.appendChild(endNode);
  }

  drawSymbolInsideSvg(svg, start, actionType, theme);

  container.appendChild(svg);
}

function isDefensiveAction(actionType = "") {
  const lower = String(actionType).toLowerCase();
  return lower.includes("intercept") || lower.includes("foul") || lower.includes("tackle");
}

function createTacticCard(item, sideKey, options = {}) {
  const { showSlider = true } = options;
  const card = document.createElement("article");
  card.className = "tactic-card";
  const isDefense = sideKey.toLowerCase().includes("defense") || isDefensiveAction(item.startAction);
  const theme = isDefense ? "def" : "off";
  card.classList.add(`tactic-${theme}`);

  const pitchWrap = document.createElement("div");
  pitchWrap.className = "mini-pitch-wrap";

  const miniPitch = document.createElement("div");
  miniPitch.className = "mini-pitch";

  const snappedPath = snapPathToGridCenters(item.path || []);
  const startPoint = snappedPath[0] || [50, 50];
  if (isDefense) {
    renderMiniTrajectory(miniPitch, [startPoint], theme, item.startAction, true);
  } else {
    const connectedPath = [...snappedPath];
    if (connectedPath.length === 0) connectedPath.push(startPoint);
    connectedPath[0] = startPoint;
    renderMiniTrajectory(miniPitch, connectedPath, theme, item.startAction, false);
  }

  const sideLabels = document.createElement("div");
  sideLabels.className = "pitch-side-labels";
  sideLabels.innerHTML = "<span>sideway</span><span>central</span><span>sideway</span>";

  const bottomLabels = document.createElement("div");
  bottomLabels.className = "pitch-bottom-labels";
  bottomLabels.innerHTML = isDefense
    ? "<span>attacking third</span><span>middle third</span><span>defensive third</span>"
    : "<span>defensive third</span><span>middle third</span><span>attacking third</span>";

  pitchWrap.append(miniPitch, sideLabels, bottomLabels);

  const freqRow = document.createElement("div");
  freqRow.className = "freq-row";
  const freqLabel = document.createElement("span");
  freqLabel.className = "freq-label";
  freqLabel.textContent = "Freq";
  const freqBar = document.createElement("div");
  freqBar.className = "freq-bar";
  const freqFill = document.createElement("div");
  freqFill.style.width = `${clamp01(item.frequency) * 100}%`;
  freqBar.appendChild(freqFill);
  freqRow.append(freqLabel, freqBar);

  const successRow = document.createElement("div");
  successRow.className = "success-row";
  const successLabel = document.createElement("span");
  successLabel.className = "success-label";
  successLabel.textContent = "Success Rate";
  const successBar = document.createElement("div");
  successBar.className = "success-bar";
  const successFill = document.createElement("div");
  successFill.style.width = `${clamp01(item.successRate ?? 0) * 100}%`;
  const successVal = document.createElement("span");
  successVal.className = "success-val";
  successVal.textContent = `${(clamp01(item.successRate ?? 0) * 100).toFixed(0)}%`;
  successBar.appendChild(successFill);
  successRow.append(successLabel, successBar, successVal);

  card.append(pitchWrap, freqRow, successRow);

  if (showSlider) {
    const weightRow = document.createElement("div");
    weightRow.className = "weight-row";
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0.5";
    slider.max = "2.0";
    slider.step = "0.1";
    slider.value = String(item.weight ?? 1.0);

    const weightVal = document.createElement("span");
    weightVal.className = "weight-val";
    weightVal.textContent = Number(slider.value).toFixed(1);

    slider.addEventListener("input", (e) => {
      const value = Number(e.target.value);
      weightVal.textContent = value.toFixed(1);
      state.tacticWeights[sideKey][item.id] = value;
    });

    weightRow.append(slider, weightVal);
    card.appendChild(weightRow);
  }
  return card;
}

function pickTopThreeBy(list, key) {
  return [...(list || [])]
    .sort((a, b) => Number(b[key] ?? 0) - Number(a[key] ?? 0))
    .slice(0, 3);
}

function pickTopSixFreqAndSuccess(list) {
  const byFreq = pickTopThreeBy(list, "frequency");
  const bySuccess = pickTopThreeBy(list, "successRate");
  const merged = [...byFreq, ...bySuccess];
  const unique = [];
  const seen = new Set();
  merged.forEach((item) => {
    if (!item?.id || seen.has(item.id)) return;
    seen.add(item.id);
    unique.push(item);
  });

  if (unique.length < 6) {
    const fallback = [...(list || [])].sort(
      (a, b) => Number((b.frequency ?? 0) + (b.successRate ?? 0)) - Number((a.frequency ?? 0) + (a.successRate ?? 0))
    );
    fallback.forEach((item) => {
      if (unique.length >= 6) return;
      if (!item?.id || seen.has(item.id)) return;
      seen.add(item.id);
      unique.push(item);
    });
  }

  return unique.slice(0, 6);
}

function renderTacticLists(data) {
  state.tacticData = {
    homeAttack: data.homeAttack || [],
    homeDefense: data.homeDefense || [],
    awayAttack: data.awayAttack || [],
    awayDefense: data.awayDefense || [],
  };

  dom.homeTacticList.innerHTML = "";
  dom.awayTacticList.innerHTML = "";

  const isOur = state.activeTacticTeam === "our";
  dom.tacticColTitleLeft.textContent = `${isOur ? "Our" : "Opponent"} Offensive Tactic (Top 6)`;
  dom.tacticColTitleRight.textContent = `${isOur ? "Our" : "Opponent"} Defensive Tactic (Top 6)`;

  const leftRaw = isOur ? state.tacticData.homeAttack : state.tacticData.awayAttack;
  const rightRaw = isOur ? state.tacticData.homeDefense : state.tacticData.awayDefense;
  const leftList = pickTopSixFreqAndSuccess(leftRaw);
  const rightList = pickTopSixFreqAndSuccess(rightRaw);
  const leftWeightKey = isOur ? "homeAttack" : "awayAttack";
  const rightWeightKey = isOur ? "homeDefense" : "awayDefense";
  const showSlider = isOur;

  (leftList || []).forEach((item) => {
    state.tacticWeights[leftWeightKey][item.id] = item.weight ?? state.tacticWeights[leftWeightKey][item.id] ?? 1.0;
    dom.homeTacticList.appendChild(createTacticCard(item, leftWeightKey, { showSlider }));
  });

  (rightList || []).forEach((item) => {
    state.tacticWeights[rightWeightKey][item.id] = item.weight ?? state.tacticWeights[rightWeightKey][item.id] ?? 1.0;
    dom.awayTacticList.appendChild(createTacticCard(item, rightWeightKey, { showSlider }));
  });
}

function renderMainPitch(players) {
  dom.mainPitch.innerHTML = "";
  players.forEach((player, idx) => {
    const node = document.createElement("div");
    node.className = "player-node";
    node.style.left = `${player.x}%`;
    node.style.top = `${player.y}%`;
    node.textContent = String(player.number ?? idx + 1);

    const label = document.createElement("div");
    label.className = "player-name-tag";
    label.style.left = `${player.x}%`;
    label.style.top = `${player.y}%`;
    label.textContent = `${player.number ?? idx + 1} ${player.name}`;

    dom.mainPitch.append(node, label);
  });
}

function renderCandidateList(players) {
  dom.candidateList.innerHTML = "";
  const totals = players.map((p) => (p.vi + p.io + p.idv) || 1);

  players.forEach((player, idx) => {
    const item = document.createElement("div");
    item.className = "candidate-item";

    const title = document.createElement("div");
    title.className = "candidate-title";
    const numBadge = document.createElement("span");
    numBadge.className = "candidate-num";
    numBadge.textContent = String(player.number ?? "-");
    const nameText = document.createElement("span");
    nameText.className = "candidate-name";
    nameText.textContent = player.name;
    const meta = document.createElement("span");
    meta.className = "candidate-meta";
    meta.textContent = `#${idx + 1}`;
    title.append(numBadge, nameText, meta);

    const bar = document.createElement("div");
    bar.className = "stacked-bar";

    const total = totals[idx] || 1;
    const viW = `${(player.vi / total) * 100}%`;
    const ioW = `${(player.io / total) * 100}%`;
    const idW = `${(player.idv / total) * 100}%`;

    const segVi = document.createElement("div");
    segVi.className = "seg-vi";
    segVi.style.width = viW;

    const segIo = document.createElement("div");
    segIo.className = "seg-io";
    segIo.style.width = ioW;

    const segId = document.createElement("div");
    segId.className = "seg-id";
    segId.style.width = idW;

    const values = document.createElement("div");
    values.className = "stacked-values";
    values.innerHTML = `<span>VI ${player.vi.toFixed(2)}</span><span>IO ${player.io.toFixed(2)}</span><span>ID ${player.idv.toFixed(2)}</span>`;

    bar.append(segVi, segIo, segId);
    item.append(title, bar, values);
    dom.candidateList.appendChild(item);
  });
}

function renderAlternatives(rows) {
  dom.lineupAlternatives.innerHTML = "";
  const maxAppear = Math.max(1, ...rows.map((row) => Number(row.appearCount ?? row.appear_count ?? 0)));

  rows.forEach((row, i) => {
    const item = document.createElement("div");
    item.className = "alt-row";
    if (state.activeAlternative === (row.name || `대안 ${i + 1}`)) item.classList.add("active");

    const name = document.createElement("div");
    name.textContent = row.name || `대안 ${i + 1}`;

    const gaugeWrap = document.createElement("div");
    gaugeWrap.className = "metrics-grid";

    const winMetric = document.createElement("div");
    const winTitle = document.createElement("div");
    winTitle.className = "metric-title";
    winTitle.textContent = "Winning Rate";
    const gauge = document.createElement("div");
    gauge.className = "gauge";
    const fill = document.createElement("div");
    fill.style.width = `${clamp01(row.winningRate) * 100}%`;
    gauge.appendChild(fill);
    const winVal = document.createElement("div");
    winVal.className = "metric-val";
    winVal.textContent = `${(clamp01(row.winningRate) * 100).toFixed(1)}%`;
    winMetric.append(winTitle, gauge, winVal);

    const appearMetric = document.createElement("div");
    const appearTitle = document.createElement("div");
    appearTitle.className = "metric-title";
    appearTitle.textContent = "Appear";
    const appearGauge = document.createElement("div");
    appearGauge.className = "gauge appear";
    const appearFill = document.createElement("div");
    const appearCount = Number(row.appearCount ?? row.appear_count ?? 0);
    appearFill.style.width = `${clamp01(appearCount / maxAppear) * 100}%`;
    appearGauge.appendChild(appearFill);
    const appearVal = document.createElement("div");
    appearVal.className = "metric-val";
    appearVal.textContent = `${appearCount}회`;
    appearMetric.append(appearTitle, appearGauge, appearVal);

    gaugeWrap.append(winMetric, appearMetric);

    const thumb = document.createElement("div");
    thumb.className = "mini-thumb";
    (row.formationPoints || []).forEach((point) => {
      const [x, y] = point;
      const dot = document.createElement("div");
      dot.className = "mini-dot";
      dot.style.left = `${x}%`;
      dot.style.top = `${y}%`;
      thumb.appendChild(dot);
    });

    const expectedWrap = document.createElement("div");
    expectedWrap.className = "expected-wrap";

    const expectedToggle = document.createElement("button");
    expectedToggle.type = "button";
    expectedToggle.className = "expected-toggle";
    expectedToggle.textContent = "예상 지표 보기";

    const expectedPanel = document.createElement("div");
    expectedPanel.className = "expected-panel";
    const expected = row.expectedStats || row.expected_stats || { shots: 0, goals: 0, intercepts: 0, tackles: 0 };
    expectedPanel.innerHTML = `
      <div class="expected-grid">
        <div class="expected-stat"><span class="k">Shots</span><span class="v">${Number(expected.shots ?? 0).toFixed(1)}</span></div>
        <div class="expected-stat"><span class="k">Goals</span><span class="v">${Number(expected.goals ?? 0).toFixed(2)}</span></div>
        <div class="expected-stat"><span class="k">Intercepts</span><span class="v">${Number(expected.intercepts ?? 0).toFixed(1)}</span></div>
        <div class="expected-stat"><span class="k">Tackles</span><span class="v">${Number(expected.tackles ?? 0).toFixed(1)}</span></div>
      </div>
    `;

    expectedToggle.addEventListener("click", (event) => {
      event.stopPropagation();
      expectedPanel.classList.toggle("show");
      expectedToggle.textContent = expectedPanel.classList.contains("show") ? "예상 지표 숨기기" : "예상 지표 보기";
    });

    expectedWrap.append(expectedToggle, expectedPanel);

    item.append(name, gaugeWrap, thumb, expectedWrap);
    item.addEventListener("click", () => {
      state.activeAlternative = row.name || `대안 ${i + 1}`;
      if (Array.isArray(row.players) && row.players.length > 0) {
        state.selectedPlayers = row.players.map((player, idx) => ({ ...player, number: player.number ?? idx + 1 }));
      } else if (Array.isArray(row.formationPoints) && row.formationPoints.length === state.selectedPlayers.length) {
        state.selectedPlayers = state.selectedPlayers.map((p, idx) => ({
          ...p,
          x: row.formationPoints[idx][0],
          y: row.formationPoints[idx][1],
        }));
      }
      renderMainPitch(state.selectedPlayers);
      renderAlternatives(state.alternatives);
      openLineupDetailModal(row);
    });
    dom.lineupAlternatives.appendChild(item);
  });
}

function openLineupDetailModal(row) {
  if (!dom.lineupModal) return;
  dom.modalTitle.textContent = `${row.name || "대안 라인업"} 상세`;
  dom.modalBody.innerHTML = "";
  dom.modalSpinner.classList.add("show");
  dom.lineupModal.classList.add("show");
  dom.lineupModal.setAttribute("aria-hidden", "false");

  setTimeout(() => {
    const formation = document.createElement("div");
    formation.className = "modal-formation";
    (row.formationPoints || []).forEach((point, idx) => {
      const [x, y] = point;
      const dot = document.createElement("div");
      dot.className = "mini-dot modal-node";
      dot.style.left = `${x}%`;
      dot.style.top = `${y}%`;
      formation.appendChild(dot);

      const p = (row.players || state.selectedPlayers || [])[idx] || {};
      const label = document.createElement("div");
      label.className = "modal-node-label";
      label.style.left = `${x}%`;
      label.style.top = `${y}%`;
      label.textContent = `${p.number ?? idx + 1} ${p.name ?? `Player ${idx + 1}`}`;
      formation.appendChild(label);
    });

    const squad = document.createElement("div");
    squad.className = "modal-squad";
    squad.innerHTML = `<h4>전체 스쿼드</h4><div class="modal-squad-list"></div>`;
    const squadList = squad.querySelector(".modal-squad-list");
    const players = row.players || state.selectedPlayers;
    players.forEach((player, idx) => {
      const li = document.createElement("div");
      li.className = "modal-squad-item";
      li.textContent = `${player.number ?? idx + 1} ${player.name ?? `Player ${idx + 1}`}`;
      squadList.appendChild(li);
    });

    dom.modalBody.append(formation, squad);
    dom.modalSpinner.classList.remove("show");
  }, 240);
}

function closeLineupDetailModal() {
  dom.lineupModal?.classList.remove("show");
  dom.lineupModal?.setAttribute("aria-hidden", "true");
}

function renderSynergyMatrix(mode) {
  const data = state.synergy[mode];
  if (!data) return;

  dom.matrixMain?.classList.toggle("defensive-scroll", mode === "defensive");

  dom.matrixYLabels.innerHTML = "";
  dom.matrixXGroups.innerHTML = "";
  dom.synergyMatrix.innerHTML = "";

  const matrixYPlayers =
    mode === "defensive"
      ? state.selectedPlayers.map((p) => ({ number: p.number, name: p.name }))
      : data.yPlayers || [];

  matrixYPlayers.forEach((player) => {
    const y = document.createElement("div");
    y.className = "y-item";
    y.textContent = toPlayerLabel(player);
    dom.matrixYLabels.appendChild(y);
  });

  const groupRow = document.createElement("div");
  groupRow.className = "x-group-row";
  const playerRow = document.createElement("div");
  playerRow.className = "x-players-row";

  const allX = [];
  if (mode === "defensive") {
    const opp = ((data.xPlayers && data.xPlayers.length ? data.xPlayers : state.opponentPlayers) || []).map((p) => ({
      number: p.number ?? p.jersey_number ?? p.jerseyNumber,
      name: p.name ?? p.player_name ?? p.playerName,
    }));
    const g = document.createElement("div");
    g.className = "x-group";
    g.textContent = "Opponent Players";
    g.style.minWidth = `${opp.length * 74}px`;
    groupRow.appendChild(g);

    opp.forEach((player) => {
      allX.push(player);
      const p = document.createElement("div");
      p.className = "x-player";
      p.textContent = toPlayerLabel(player);
      p.classList.add("player-label-strong");
      playerRow.appendChild(p);
    });
  } else {
    data.xGroups.forEach((group) => {
      const g = document.createElement("div");
      g.className = "x-group";
      g.textContent = `${group.group}`;
      g.style.minWidth = `${group.players.length * 74}px`;
      groupRow.appendChild(g);

      group.players.forEach((player) => {
        allX.push(player);
        const p = document.createElement("div");
        p.className = "x-player";
        p.textContent = toPlayerLabel(player);
        p.classList.add("player-label-strong");
        playerRow.appendChild(p);
      });
    });
  }

  dom.matrixXGroups.append(groupRow, playerRow);

  requestAnimationFrame(() => {
    const headerHeight = dom.matrixXGroups.offsetHeight || 54;
    dom.matrixYLabels.style.setProperty("--matrix-head-h", `${headerHeight}px`);
  });

  const matrixSource = data.matrix || [];
  const flatValues = matrixSource.flatMap((row) => row).filter((v) => Number(v) > 0);
  const normalized = normalizeByMax(flatValues, 1);
  let idx = 0;

  const rowCount = matrixYPlayers.length;
  const colCount = allX.length;
  for (let r = 0; r < rowCount; r += 1) {
    const row = matrixSource[r] || [];
    const rowEl = document.createElement("div");
    rowEl.className = "matrix-row";

    for (let c = 0; c < colCount; c += 1) {
      const cell = document.createElement("div");
      cell.className = "matrix-cell";

      const raw = Number(row[c] ?? 0);
      if (raw > 0) {
        const valueNorm = normalized[idx++] ?? 0;
        const box = document.createElement("div");
        box.className = "synergy-box";
        const side = Math.max(6, Math.floor(26 * valueNorm));
        box.style.width = `${side}px`;
        box.style.height = `${side}px`;
        cell.appendChild(box);
      }
      rowEl.appendChild(cell);
    }

    dom.synergyMatrix.appendChild(rowEl);
  }
}

function renderAll() {
  renderMainPitch(state.selectedPlayers);
  renderCandidateList(state.allPlayers);
  renderAlternatives(state.alternatives);
  renderSynergyMatrix(state.mode);
}

function bindTabEvents() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    });
  });

  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
      renderSynergyMatrix(state.mode);
    });
  });
}

function applyApiResponse(payload) {
  // expected schema (flexible):
  // {
  //   tactics: { home_attack: [...], away_defense: [...] },
  //   lineup: { selected_players: [...], candidate_players: [...] },
  //   alternatives: [{ name, winning_rate, formation_points }],
  //   synergy: {
  //     offensive: { y_players, x_groups, matrix },
  //     defensive: { y_players, x_groups, matrix }
  //   }
  // }
  const tactics = payload.tactics || {};
  renderTacticLists({
    homeAttack: tactics.home_attack || tactics.homeAttack || defaultTacticData().homeAttack,
    homeDefense: tactics.home_defense || tactics.homeDefense || defaultTacticData().homeDefense,
    awayAttack: tactics.away_attack || tactics.awayAttack || defaultTacticData().awayAttack,
    awayDefense: tactics.away_defense || tactics.awayDefense || defaultTacticData().awayDefense,
  });

  const lineup = payload.lineup || {};
  state.selectedPlayers = (lineup.selected_players || lineup.selectedPlayers || state.selectedPlayers).map((p, i) => ({
    id: p.id ?? p.player_id ?? i,
    number: p.number ?? p.jersey_number ?? p.jerseyNumber ?? i + 1,
    name: p.name ?? p.player_name ?? `P${i + 1}`,
    position: p.position ?? "MF",
    x: p.x ?? p.pitch_x ?? 50,
    y: p.y ?? p.pitch_y ?? 50,
    vi: Number(p.vi ?? 0),
    io: Number(p.io ?? 0),
    idv: Number(p.idv ?? p.i_d ?? p.defensive ?? 0),
  }));

  state.candidatePlayers = (lineup.candidate_players || lineup.candidatePlayers || state.candidatePlayers).map((p, i) => ({
    id: p.id ?? p.player_id ?? i,
    number: p.number ?? p.jersey_number ?? p.jerseyNumber ?? i + 1,
    name: p.name ?? p.player_name ?? `Bench ${i + 1}`,
    vi: Number(p.vi ?? 0),
    io: Number(p.io ?? 0),
    idv: Number(p.idv ?? p.i_d ?? p.defensive ?? 0),
  }));

  state.opponentPlayers = (lineup.opponent_players || lineup.opponentPlayers || state.opponentPlayers).map((p, i) => ({
    id: p.id ?? p.player_id ?? i,
    number: p.number ?? p.jersey_number ?? p.jerseyNumber ?? i + 1,
    name: p.name ?? p.player_name ?? `Opp ${i + 1}`,
  }));

  const apiAllPlayers = lineup.all_players || lineup.allPlayers || [];
  state.allPlayers = (apiAllPlayers.length ? apiAllPlayers : [...state.selectedPlayers, ...state.candidatePlayers]).map((p, i) => ({
    id: p.id ?? p.player_id ?? i,
    number: p.number ?? p.jersey_number ?? p.jerseyNumber ?? i + 1,
    name: p.name ?? p.player_name ?? `Player ${i + 1}`,
    vi: Number(p.vi ?? 0),
    io: Number(p.io ?? 0),
    idv: Number(p.idv ?? p.i_d ?? p.defensive ?? 0),
  }));

  state.alternatives = (payload.alternatives || []).map((a, i) => ({
    name: a.name || `대안 ${i + 1}`,
    winningRate: Number(a.winning_rate ?? a.winningRate ?? 0),
    appearCount: Number(a.appear_count ?? a.appearCount ?? 0),
    expectedStats: a.expected_stats || a.expectedStats || { shots: 0, goals: 0, intercepts: 0, tackles: 0 },
    formationPoints: a.formation_points || a.formationPoints || [],
    players: (a.players || a.lineup_players || []).map((p, idx) => ({
      id: p.id ?? p.player_id ?? idx,
      number: p.number ?? p.jersey_number ?? p.jerseyNumber ?? idx + 1,
      name: p.name ?? p.player_name ?? `Player ${idx + 1}`,
      position: p.position ?? "MF",
      x: p.x ?? p.pitch_x ?? 50,
      y: p.y ?? p.pitch_y ?? 50,
      vi: Number(p.vi ?? 0),
      io: Number(p.io ?? 0),
      idv: Number(p.idv ?? p.i_d ?? p.defensive ?? 0),
    })),
  }));

  const sy = payload.synergy || {};
  if (sy.offensive) {
    state.synergy.offensive = {
      yPlayers: sy.offensive.y_players || sy.offensive.yPlayers || [],
      xGroups: sy.offensive.x_groups || sy.offensive.xGroups || [],
      matrix: sy.offensive.matrix || [],
    };
  }
  if (sy.defensive) {
    const defensiveY =
      sy.defensive.opponent_y_players ||
      sy.defensive.opponentYPlayers ||
      sy.defensive.y_players_opponent ||
      sy.defensive.yPlayersOpponent ||
      sy.defensive.y_players ||
      sy.defensive.yPlayers ||
      [];

    state.synergy.defensive = {
      yPlayers: defensiveY,
      xPlayers:
        sy.defensive.opponent_x_players ||
        sy.defensive.opponentXPlayers ||
        sy.defensive.x_players_opponent ||
        sy.defensive.xPlayersOpponent ||
        sy.defensive.x_players ||
        sy.defensive.xPlayers ||
        state.opponentPlayers ||
        [],
      xGroups: sy.defensive.x_groups || sy.defensive.xGroups || [],
      matrix: sy.defensive.matrix || [],
    };
  }

  renderAll();
}

async function optimizeLineup() {
  state.teamId = Number(dom.teamNameSelect?.value || dom.teamId.value);
  state.opponentId = Number(dom.opponentNameSelect?.value || dom.opponentId.value);
  dom.teamId.value = String(state.teamId);
  dom.opponentId.value = String(state.opponentId);

  const payload = {
    team_id: state.teamId,
    opponent_id: state.opponentId,
    opponent_team_id: state.opponentId,
    tactic_weights: state.tacticWeights,
    tactic_view_context: {
      active_team: "our",
    },
    constraints: {
      formation: "4-3-3",
      locked_players: [],
      excluded_players: [],
      formation_constraints: {
        forwards: state.constraints.forwards,
        midfielders: state.constraints.midfielders,
        defenders: state.constraints.defenders,
        goalkeeper: { min: 1, max: 1 },
      },
    },
    view_mode: state.mode,
  };

  dom.runOptimizeBtn.disabled = true;
  dom.runOptimizeBtn.textContent = "최적화 실행 중...";

  try {
    const base = await resolveApiBase();
    if (!base) throw new Error("Backend not reachable on port 8000");

    const res = await fetch(`${base}/api/optimize_lineup`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`API ${res.status}: ${text}`);
    }

    const data = await res.json();
    applyApiResponse(data);
    showToast("최적화 결과가 업데이트되었습니다.");
  } catch (error) {
    console.error(error);
    showToast("백엔드 연결 실패(8000포트): 서버 실행 후 다시 시도하세요.");
  } finally {
    dom.runOptimizeBtn.disabled = false;
    dom.runOptimizeBtn.textContent = "최적화 실행";
  }
}

function initStateFromInputs() {
  state.teamId = Number(dom.teamNameSelect?.value || dom.teamId.value);
  state.opponentId = Number(dom.opponentNameSelect?.value || dom.opponentId.value);
  dom.teamId.value = String(state.teamId);
  dom.opponentId.value = String(state.opponentId);
  syncNameSelectorsFromIds();
}

function bindInputs() {
  dom.teamId.addEventListener("change", () => {
    state.teamId = Number(dom.teamId.value);
    syncNameSelectorsFromIds();
  });
  dom.opponentId.addEventListener("change", () => {
    state.opponentId = Number(dom.opponentId.value);
    syncNameSelectorsFromIds();
  });

  dom.teamNameSelect?.addEventListener("change", () => {
    state.teamId = Number(dom.teamNameSelect.value);
    dom.teamId.value = String(state.teamId);
  });
  dom.teamNameSelect?.addEventListener("input", () => {
    state.teamId = Number(dom.teamNameSelect.value);
    dom.teamId.value = String(state.teamId);
  });

  dom.opponentNameSelect?.addEventListener("change", () => {
    state.opponentId = Number(dom.opponentNameSelect.value);
    dom.opponentId.value = String(state.opponentId);
  });
  dom.opponentNameSelect?.addEventListener("input", () => {
    state.opponentId = Number(dom.opponentNameSelect.value);
    dom.opponentId.value = String(state.opponentId);
  });

  dom.runOptimizeBtn.addEventListener("click", optimizeLineup);

  dom.tacticTeamToggle?.querySelectorAll(".team-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      dom.tacticTeamToggle.querySelectorAll(".team-tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.activeTacticTeam = btn.dataset.team;
      renderTacticLists(state.tacticData);
    });
  });

  dom.constraintsPanel?.querySelectorAll(".constraint-row").forEach((row) => {
    const role = row.dataset.role;
    const slider = row.querySelector('input[data-key="count"]');
    const valueSpan = row.querySelector('.constraint-val[data-key="value"]');

    const syncConstraint = () => {
      const current = Number(slider.value);
      const temp = {
        forwards: role === "forwards" ? current : Number(dom.constraintsPanel.querySelector('.constraint-row[data-role="forwards"] input[data-key="count"]').value),
        midfielders: role === "midfielders" ? current : Number(dom.constraintsPanel.querySelector('.constraint-row[data-role="midfielders"] input[data-key="count"]').value),
        defenders: role === "defenders" ? current : Number(dom.constraintsPanel.querySelector('.constraint-row[data-role="defenders"] input[data-key="count"]').value),
      };

      const sum = temp.forwards + temp.midfielders + temp.defenders;
      if (sum > 10) {
        const overflow = sum - 10;
        slider.value = String(Math.max(1, current - overflow));
      }

      const finalVal = Number(slider.value);
      valueSpan.textContent = String(finalVal);
      state.constraints[role] = { min: finalVal, max: finalVal };

      const totalNow =
        Number(dom.constraintsPanel.querySelector('.constraint-row[data-role="forwards"] input[data-key="count"]').value) +
        Number(dom.constraintsPanel.querySelector('.constraint-row[data-role="midfielders"] input[data-key="count"]').value) +
        Number(dom.constraintsPanel.querySelector('.constraint-row[data-role="defenders"] input[data-key="count"]').value);

      const note = document.getElementById("constraintNote");
      if (note) {
        note.textContent = `F+M+D = ${totalNow} (최대 10, GK 1명 고정)`;
        note.style.color = totalNow > 10 ? "#fca5a5" : "#94a3b8";
      }
    };

    slider?.addEventListener("input", syncConstraint);
    syncConstraint();
  });

  dom.matrixMain?.addEventListener("scroll", () => {
    dom.matrixYLabels.scrollTop = dom.matrixMain.scrollTop;
  });

  dom.closeModalBtn?.addEventListener("click", closeLineupDetailModal);
  dom.lineupModal?.addEventListener("click", (e) => {
    if (e.target === dom.lineupModal) closeLineupDetailModal();
  });
}

async function initialize() {
  await resolveApiBase();
  await loadTeamsFromApi();
  populateTeamSelectors();
  initStateFromInputs();

  const t = defaultTacticData();
  const p = defaultPlayers();
  const s = defaultSynergy();

  state.selectedPlayers = p.selectedPlayers;
  state.candidatePlayers = p.candidatePlayers;
  state.opponentPlayers = p.opponentPlayers;
  state.allPlayers = [...p.selectedPlayers, ...p.candidatePlayers];
  state.alternatives = p.alternatives;
  state.synergy = s;

  renderTacticLists(t);
  renderTacticLegend();
  bindTabEvents();
  bindInputs();
  renderAll();
}

initialize();
