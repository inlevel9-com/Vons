import type { LocalBundle } from "./bundle.ts";
const DATABASE = "vons-local-model";
async function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE, 1);
    request.onupgradeneeded = () => request.result.createObjectStore("model");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(new Error("Local model storage is unavailable."));
    request.onblocked = () => reject(new Error("Close other Vons panels before updating storage."));
  });
}
export async function modelStorage(action: "read" | "write" | "clear", model?: LocalBundle): Promise<LocalBundle | undefined> {
  const db = await open();
  try {
    return await new Promise((resolve, reject) => {
      const transaction = db.transaction("model", action === "read" ? "readonly" : "readwrite");
      const store = transaction.objectStore("model");
      const request = action === "read" ? store.get("active") : action === "write" ? store.put(model, "active") : store.clear();
      transaction.oncomplete = () => resolve(action === "read" ? request.result as LocalBundle | undefined : undefined);
      transaction.onabort = transaction.onerror = () => reject(new Error("Could not update local model storage. Check available space."));
    });
  } finally { db.close(); }
}
