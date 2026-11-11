// Echo desktop (version 4): a compact, always-on-top window around the agent_v4 panel.
// It's an ordinary window: it shows up in any screen share or recording that includes it.
const { app, BrowserWindow, Menu, globalShortcut, ipcMain, screen, shell } = require("electron");
const { spawn } = require("node:child_process");
const path = require("node:path");

const ROOT = path.resolve(__dirname, "..");
const PORT = process.env.PORT || "8766";
const BASE = `http://127.0.0.1:${PORT}`;
const SHORTCUT_ANSWER = "CommandOrControl+Shift+A";
const SHORTCUT_TOGGLE = "CommandOrControl+Shift+Space";

let win;
let backend; // only set when this app started the agent, so we only stop what we started

async function agentUp() {
  try {
    return (await fetch(`${BASE}/api/docs`)).ok;
  } catch {
    return false;
  }
}

// Reuses an agent that's already running (./run.sh v4), otherwise starts one.
async function ensureAgent() {
  if (await agentUp()) return;
  backend = spawn(path.join(ROOT, "run.sh"), ["v4"], { cwd: ROOT, stdio: "inherit", env: { ...process.env, PORT } });
  backend.on("exit", () => (backend = null));
  const deadline = Date.now() + 180_000; // first run loads the Parakeet model
  while (Date.now() < deadline) {
    if (await agentUp()) return;
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`Echo didn't start on ${BASE} (see the terminal for errors)`);
}

// Top-right corner of the display's usable area.
function placeOn(display) {
  const { x, y, width, height } = display.workArea;
  const [w, h] = win.getSize();
  const hh = Math.min(h, height - 32);
  win.setBounds({ x: x + width - w - 16, y: y + 16, width: w, height: hh });
}

function toggle() {
  if (win.isVisible()) win.hide();
  else win.showInactive(); // don't steal focus from the call
}

const answerNow = () => win.webContents.send("answer-now");

function buildMenu() {
  const current = screen.getDisplayMatching(win.getBounds());
  const primary = screen.getPrimaryDisplay();
  const displays = screen.getAllDisplays().map((d, i) => ({
    label: `${d.label || `Display ${i + 1}`} (${d.size.width}×${d.size.height})${d.id === primary.id ? " · primary" : ""}`,
    type: "radio",
    checked: d.id === current.id,
    click: () => {
      placeOn(d);
      buildMenu();
    },
  }));
  Menu.setApplicationMenu(
    Menu.buildFromTemplate([
      { role: "appMenu" },
      { role: "editMenu" },
      {
        label: "Assistant",
        submenu: [
          // Global shortcuts below handle these keys; registerAccelerator: false avoids firing twice.
          { label: "Answer now", accelerator: SHORTCUT_ANSWER, registerAccelerator: false, click: answerNow },
          { label: "Show / hide", accelerator: SHORTCUT_TOGGLE, registerAccelerator: false, click: toggle },
          { type: "separator" },
          {
            label: "Always on top",
            type: "checkbox",
            checked: win.isAlwaysOnTop(),
            click: (item) => win.setAlwaysOnTop(item.checked, "floating"),
          },
          { label: "Move to display", submenu: displays },
          { type: "separator" },
          { label: "Open full panel in browser", click: () => shell.openExternal(`http://localhost:${PORT}`) },
        ],
      },
      { role: "windowMenu" },
    ]),
  );
}

app.whenReady().then(async () => {
  win = new BrowserWindow({
    width: 420,
    height: 640,
    minWidth: 320,
    minHeight: 360,
    title: "Echo",
    backgroundColor: "#2d2b2b",
    alwaysOnTop: true,
    webPreferences: { preload: path.join(__dirname, "preload.js") },
  });
  win.setAlwaysOnTop(true, "floating");
  win.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
  // Answers are text only: the window stays muted unless the panel's Sound setting turns the chime on.
  win.webContents.setAudioMuted(true);
  ipcMain.on("sound", (_, on) => win.webContents.setAudioMuted(!on));

  placeOn(screen.getPrimaryDisplay());
  buildMenu();
  screen.on("display-added", buildMenu);
  screen.on("display-removed", buildMenu);
  win.on("moved", buildMenu);

  globalShortcut.register(SHORTCUT_ANSWER, answerNow);
  globalShortcut.register(SHORTCUT_TOGGLE, toggle);

  win.loadURL(`data:text/html,<body style="background:%232d2b2b;color:%23bab6b6;font:15px system-ui;padding:24px">Starting Echo…</body>`);
  try {
    await ensureAgent();
    await win.loadURL(`${BASE}/?compact=1`);
  } catch (e) {
    win.loadURL(`data:text/html,<body style="background:%232d2b2b;color:%23fb923c;font:15px system-ui;padding:24px">${encodeURIComponent(e.message)}</body>`);
  }
});

app.on("will-quit", () => {
  globalShortcut.unregisterAll();
  backend?.kill();
});
app.on("window-all-closed", () => app.quit());
