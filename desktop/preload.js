// Bridge between the panel page and the desktop app (the page works without it in a browser).
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("echoDesktop", {
  onAnswerNow: (fn) => ipcRenderer.on("answer-now", () => fn()),
  setSound: (on) => ipcRenderer.send("sound", Boolean(on)),
});
