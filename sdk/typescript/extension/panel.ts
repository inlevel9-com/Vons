import { importBundle, verifyBundle, makeRequest, type LocalBundle } from "./bundle.ts";
import { modelStorage } from "./storage.ts";
import { validateResponse, type DecisionResponse } from "../src/index.ts";

function element<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing panel element: ${id}`);
  return node as T;
}
const files = element<HTMLInputElement>("bundle-files");
const remember = element<HTMLInputElement>("remember");
const run = element<HTMLButtonElement>("run");
const cancel = element<HTMLButtonElement>("cancel");
const forget = element<HTMLButtonElement>("forget");
const provider = element<HTMLSelectElement>("provider");
const form = element<HTMLFormElement>("decision-form");
let bundle: LocalBundle | undefined;
let worker: Worker | undefined;
let workerKey = "";
let busy = false;
let importing = false;
let lastResult = "";
let timeout: ReturnType<typeof setTimeout> | undefined;

function status(text: string, error = false): void {
  const node = element("status"); node.textContent = text; node.dataset.error = String(error);
}
function controls(): void {
  run.disabled = busy || importing || !bundle;
  forget.disabled = busy || importing || !bundle;
  files.disabled = busy || importing;
  remember.disabled = busy || importing;
  provider.disabled = busy || importing;
  for (const id of ["state", "question", "options", "example"]) element<HTMLInputElement | HTMLButtonElement>(id).disabled = busy;
  cancel.hidden = !busy;
  form.setAttribute("aria-busy", String(busy));
}
function stop(): void {
  worker?.terminate(); worker = undefined; workerKey = "";
  clearTimeout(timeout); busy = false; controls();
}
function modelLabel(): void {
  element("model-badge").textContent = bundle ? bundle.backend : "Not loaded";
  element("model-description").textContent = bundle
    ? `${bundle.modelId} · ${(bundle.assets.reduce((sum, a) => sum + a.bytes.byteLength, 0) / 1048576).toFixed(1)} MiB · SHA-256 checked`
    : "Import a Vons ONNX bundle folder. Model weights are supplied separately.";
  controls();
}
function errorMessage(error: unknown): string { return error instanceof Error ? error.message : "The operation failed."; }
function showResult(data: { response: DecisionResponse; manifestHash: string; runtime: unknown; loadMs: number; inferenceMs: number }): void {
  validateResponse(data.response);
  const answer = data.response.answers[0];
  if (!answer) throw new Error("The model returned no answer.");
  element("decision-label").textContent = answer.status === "abstain" ? "NO RECOMMENDATION" : "MODEL PROPOSAL";
  element("choice").textContent = answer.choice ?? "The model abstained.";
  element("reason").textContent = answer.abstain_reason?.replaceAll("_", " ") ?? "Review this choice yourself before taking any action.";
  const ranking = element("ranking"); ranking.replaceChildren();
  for (const entry of [...answer.probabilities].sort((a, b) => b.probability - a.probability)) {
    const item = document.createElement("li"); const row = document.createElement("div"); row.className = "rank-label";
    const label = document.createElement("span"); label.textContent = entry.option;
    const value = document.createElement("span"); value.textContent = `${(entry.probability * 100).toFixed(1)}%`;
    const bar = document.createElement("progress"); bar.max = 1; bar.value = entry.probability; bar.setAttribute("aria-label", `${entry.option} model score`);
    row.append(label, value); item.append(row, bar); ranking.append(item);
  }
  element("timing").textContent = `Local run: ${data.inferenceMs.toFixed(0)} ms · load/check: ${data.loadMs.toFixed(0)} ms. This is one run, not a benchmark.`;
  lastResult = JSON.stringify(data, null, 2); element("raw").textContent = lastResult;
  element("result").hidden = false;
}
files.addEventListener("change", async () => {
  if (!files.files?.length) return;
  importing = true; stop(); controls(); element("result").hidden = true;
  status("Reading and verifying the selected model files…");
  try {
    const next = await importBundle(Array.from(files.files));
    // Do not make a new model active until its requested storage operation succeeds.
    if (remember.checked) await modelStorage("write", next);
    else await modelStorage("clear");
    bundle = next; status("Model ready. Add context and candidates, then compare.");
  } catch (error) { status(errorMessage(error), true); }
  finally { importing = false; files.value = ""; modelLabel(); }
});
remember.addEventListener("change", async () => {
  importing = true; controls();
  try {
    if (remember.checked && bundle) await modelStorage("write", bundle);
    else await modelStorage("clear");
    status(remember.checked ? "This model will be available next time. Text is never saved." : "Stored model removed. The current model remains in this session only.");
  } catch (error) { remember.checked = !remember.checked; status(errorMessage(error), true); }
  finally { importing = false; controls(); }
});
forget.addEventListener("click", async () => {
  importing = true; stop(); controls();
  try {
    await modelStorage("clear"); bundle = undefined; remember.checked = false;
    element("result").hidden = true; lastResult = ""; element("raw").textContent = "";
    status("Model removed from this device.");
  } catch (error) { status(errorMessage(error), true); }
  finally { importing = false; modelLabel(); }
});
element("example").addEventListener("click", () => {
  element<HTMLInputElement>("state").value = "A document has unsaved edits. The author wants to keep the changes before closing it.";
  element<HTMLInputElement>("question").value = "What should happen next?";
  element<HTMLInputElement>("options").value = "Save the document\nDiscard the changes\nAsk the author for more context";
  element("result").hidden = true;
  status("Example loaded. This is an illustrative input, not a quality test.");
});
form.addEventListener("submit", (event) => {
  event.preventDefault(); if (busy || importing || !bundle) return;
  try {
    const request = makeRequest(element<HTMLInputElement>("state").value, element<HTMLInputElement>("question").value, element<HTMLInputElement>("options").value);
    const selected = provider.value === "webgpu" ? "webgpu" : "wasm";
    const key = `${bundle.manifestHash}:${selected}`;
    if (workerKey !== key) stop();
    if (!worker) {
      worker = new Worker(new URL("worker.js", import.meta.url), { type: "module" });
      worker.onmessage = (message) => {
        const data = message.data;
        if (data.type === "progress") { status(data.text); return; }
        clearTimeout(timeout); busy = false; controls();
        if (data.type === "error") { stop(); status(data.text, true); return; }
        try { showResult(data); status("Comparison complete. No page action was taken."); }
        catch (error) { stop(); status(errorMessage(error), true); }
      };
      worker.onerror = () => { stop(); status("The local runtime stopped unexpectedly. You can try again with WASM.", true); };
    }
    busy = true; controls(); element("result").hidden = true;
    status("Starting the local comparison…");
    worker.postMessage({ request, provider: selected, ...(workerKey === key ? {} : { bundle }) }); workerKey = key;
    timeout = setTimeout(() => { stop(); status("The comparison exceeded two minutes and was stopped. Try fewer or shorter candidates.", true); }, 120000);
  } catch (error) { stop(); status(errorMessage(error), true); }
});
cancel.addEventListener("click", () => { stop(); status("Canceled. No decision was returned. The model can be used again."); });
element("copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(lastResult); status("Response JSON copied. It contains your candidate text."); }
  catch { status("Clipboard access was unavailable. Select the JSON text to copy it manually.", true); }
});
window.addEventListener("pagehide", stop);
async function restore(): Promise<void> {
  importing = true; controls();
  try {
    const saved = await modelStorage("read");
    if (saved) { await verifyBundle(saved); bundle = saved; remember.checked = true; status("Your saved model is ready. Context and decisions were not retained."); }
  } catch (error) { status(`${errorMessage(error)} You can import a model folder again.`, true); }
  finally { importing = false; modelLabel(); }
}
void restore();
