let documents = [];
let selectedDocumentId = null;
let selectedDocument = null;
let selectedPageId = null;
let pages = [];
let selectedBulkDocumentIds = new Set();

const modelStatus = document.querySelector("#modelStatus");
const createGroupButton = document.querySelector("#createGroupButton");
const pageFileInput = document.querySelector("#pageFileInput");
const folderInput = document.querySelector("#folderInput");
const refreshButton = document.querySelector("#refreshButton");
const selectAllDocuments = document.querySelector("#selectAllDocuments");
const selectedDocumentsCount = document.querySelector("#selectedDocumentsCount");
const extractSelectedDocumentsButton = document.querySelector("#extractSelectedDocumentsButton");
const searchInput = document.querySelector("#searchInput");
const searchButton = document.querySelector("#searchButton");
const documentList = document.querySelector("#documentList");
const emptyState = document.querySelector("#emptyState");
const documentView = document.querySelector("#documentView");
const pageTabs = document.querySelector("#pageTabs");
const previewImage = document.querySelector("#previewImage");
const documentTitle = document.querySelector("#documentTitle");
const documentMeta = document.querySelector("#documentMeta");
const currentDocumentTitle = document.querySelector("#currentDocumentTitle");
const renameDocumentButton = document.querySelector("#renameDocumentButton");
const deleteDocumentButton = document.querySelector("#deleteDocumentButton");
const renumberPageButton = document.querySelector("#renumberPageButton");
const deletePageButton = document.querySelector("#deletePageButton");
const extractPageButton = document.querySelector("#extractPageButton");
const extractDocumentButton = document.querySelector("#extractDocumentButton");
const keyInput = document.querySelector("#keyInput");
const kvTable = document.querySelector("#kvTable");
const rawOutput = document.querySelector("#rawOutput");
let activeExtractionId = null;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

async function loadStatus() {
  const status = await api("/api/model/status");
  modelStatus.textContent = status.message;
  modelStatus.classList.toggle("bad", !status.ready);
}

async function loadDocuments() {
  const q = searchInput.value.trim();
  documents = await api(`/api/documents${q ? `?q=${encodeURIComponent(q)}` : ""}`);
  const visibleIds = new Set(documents.map((doc) => doc.id));
  selectedBulkDocumentIds = new Set([...selectedBulkDocumentIds].filter((id) => visibleIds.has(id)));
  renderList();
}

function renderList() {
  documentList.innerHTML = "";
  updateBulkSelectionUi();
  if (!documents.length) {
    documentList.innerHTML = '<p class="status">No groups found.</p>';
    return;
  }
  for (const doc of documents) {
    const row = document.createElement("div");
    row.className = "docRow";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "docCheck";
    checkbox.checked = selectedBulkDocumentIds.has(doc.id);
    checkbox.title = "Use this group for bulk extraction";
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        selectedBulkDocumentIds.add(doc.id);
      } else {
        selectedBulkDocumentIds.delete(doc.id);
      }
      updateBulkSelectionUi();
    });

    const item = document.createElement("button");
    item.type = "button";
    item.className = `docItem ${doc.id === selectedDocumentId ? "active" : ""}`;
    item.innerHTML = `
      <strong>${escapeHtml(doc.filename)}</strong>
      <span class="status">${escapeHtml(doc.status)} - ${doc.processed_pages || 0}/${doc.page_count || 0} documents - ${formatBytes(doc.size_bytes)}</span>
    `;
    item.addEventListener("click", () => selectDocument(doc.id));

    row.appendChild(checkbox);
    row.appendChild(item);
    documentList.appendChild(row);
  }
  updateBulkSelectionUi();
}

async function selectDocument(id) {
  selectedDocumentId = id;
  selectedDocument = await api(`/api/documents/${id}`);
  pages = await api(`/api/documents/${id}/pages`);
  selectedPageId = pages[0]?.id || null;
  renderList();
  renderDocument();
}

function renderDocument() {
  emptyState.classList.add("hidden");
  documentView.classList.remove("hidden");
  documentTitle.textContent = selectedDocument.filename;
  renderTabs();
  renderPage(currentPage());
}

function renderTabs() {
  pageTabs.innerHTML = "";
  for (const page of pages) {
    const tab = document.createElement("button");
    tab.type = "button";
    tab.className = `pageTab ${page.id === selectedPageId ? "active" : ""}`;
    tab.textContent = pageLabel(page);
    tab.title = `#${page.page_number} - ${page.status}`;
    tab.addEventListener("click", () => {
      selectedPageId = page.id;
      renderTabs();
      renderPage(page);
    });
    pageTabs.appendChild(tab);
  }
}

function currentPage() {
  return pages.find((page) => page.id === selectedPageId) || pages[0] || null;
}

function renderPage(page) {
  if (!page) {
    documentMeta.textContent = "";
    currentDocumentTitle.textContent = "";
    currentDocumentTitle.title = "";
    kvTable.innerHTML = '<tr><td colspan="5">No documents in this group.</td></tr>';
    previewImage.removeAttribute("src");
    rawOutput.textContent = "";
    return;
  }
  documentMeta.textContent = "";
  currentDocumentTitle.textContent = pageLabel(page);
  currentDocumentTitle.title = pageLabel(page);
  previewImage.src = `/api/pages/${page.id}/image`;
  clearBboxOverlay();
  const rows = page.extractions || [];
  if (activeExtractionId && !rows.some((item) => item.id === activeExtractionId)) {
    activeExtractionId = null;
  }
  kvTable.innerHTML = "";
  for (const extraction of rows) {
    const tr = document.createElement("tr");
    tr.className = extraction.id === activeExtractionId ? "selectedRow" : "";
    tr.innerHTML = `
      <td>${escapeHtml(extraction.query_key)}</td>
      <td>${escapeHtml(extraction.value_text || "")}</td>
      <td>${escapeHtml(JSON.stringify(extraction.bbox || ""))}</td>
      <td>${escapeHtml(shorten(extraction.raw_output || ""))}</td>
      <td><button type="button" class="smallButton" data-id="${extraction.id}">Delete</button></td>
    `;
    tr.addEventListener("click", () => selectExtraction(extraction));
    tr.querySelector("button").addEventListener("click", (event) => deleteExtraction(extraction.id, event));
    kvTable.appendChild(tr);
  }
  if (!rows.length) {
    kvTable.innerHTML = '<tr><td colspan="5">No key queries extracted yet.</td></tr>';
  }
  const selectedExtraction = rows.find((item) => item.id === activeExtractionId);
  rawOutput.textContent = page.error ? `Error: ${page.error}` : selectedExtraction?.raw_output || "";
}

async function uploadDocumentFiles(input) {
  if (!selectedDocumentId) return alert("Select a group first.");
  const selectedFiles = Array.from(input.files || []);
  const files = selectedFiles.filter((file) => allowedFile(file.name));
  if (!files.length) {
    input.value = "";
    return alert("Select image or PDF files.");
  }
  const form = new FormData();
  for (const file of files) {
    form.append("files", file, file.webkitRelativePath || file.name);
  }
  const doc = await api(`/api/documents/${selectedDocumentId}/pages`, { method: "POST", body: form });
  selectedDocument = doc;
  pages = doc.pages || (await api(`/api/documents/${selectedDocumentId}/pages`));
  selectedPageId = pages[pages.length - 1]?.id || selectedPageId;
  input.value = "";
  await loadDocuments();
  renderDocument();
}

async function createDocumentGroup() {
  const groupName = prompt("Group name");
  if (!groupName || !groupName.trim()) return;
  const form = new FormData();
  form.append("group_name", groupName.trim());
  const doc = await api("/api/documents", { method: "POST", body: form });
  selectedDocumentId = doc.id;
  await loadDocuments();
  await selectDocument(doc.id);
}

async function extractSelectedPage() {
  if (!selectedPageId) return;
  const key = keyInput.value.trim();
  if (!key) return alert("Enter a key first.");
  setBusy(extractPageButton, "Processing...");
  const form = new FormData();
  form.append("key", key);
  try {
    const page = await api(`/api/pages/${selectedPageId}/extract`, { method: "POST", body: form });
    pages = pages.map((item) => (item.id === page.id ? page : item));
    renderTabs();
    renderPage(page);
    await loadDocuments();
  } finally {
    setReady(extractPageButton, "Extract document");
  }
}

async function extractSelectedDocument() {
  if (!selectedDocumentId) return;
  const key = keyInput.value.trim();
  if (!key) return alert("Enter a key first.");
  setBusy(extractDocumentButton, "Processing...");
  const form = new FormData();
  form.append("key", key);
  try {
    await api(`/api/documents/${selectedDocumentId}/extract`, { method: "POST", body: form });
    pages = await api(`/api/documents/${selectedDocumentId}/pages`);
    renderTabs();
    renderPage(currentPage());
    await loadDocuments();
  } finally {
    setReady(extractDocumentButton, "Extract group");
  }
}

async function extractBulkDocuments() {
  const key = keyInput.value.trim();
  if (!key) return alert("Enter a key first.");
  const selectedIds = [...selectedBulkDocumentIds];
  if (!selectedIds.length) return alert("Select at least one group.");

  const button = extractSelectedDocumentsButton;
  const idleText = "Extract selected";
  setBusy(button, "Processing...");
  const form = new FormData();
  form.append("key", key);
  form.append("all_documents", "false");
  for (const id of selectedIds) {
    form.append("document_ids", String(id));
  }

  try {
    const result = await api("/api/extract", { method: "POST", body: form });
    await loadDocuments();
    if (selectedDocumentId) {
      selectedDocument = await api(`/api/documents/${selectedDocumentId}`);
      pages = await api(`/api/documents/${selectedDocumentId}/pages`);
      renderDocument();
    }
    const message = `Done: ${result.processed_document_count}/${result.document_count} groups, ${result.processed_page_count} documents.`;
    rawOutput.textContent = result.failed_document_count ? `${message}\nFailed: ${JSON.stringify(result.failed_documents, null, 2)}` : message;
  } finally {
    setReady(button, idleText);
  }
}

async function renameDocument() {
  if (!selectedDocumentId) return;
  const filename = prompt("New group name", selectedDocument.filename);
  if (!filename) return;
  const form = new FormData();
  form.append("filename", filename);
  selectedDocument = await api(`/api/documents/${selectedDocumentId}`, { method: "PATCH", body: form });
  await loadDocuments();
  renderDocument();
}

async function deleteDocument() {
  if (!selectedDocumentId || !confirm("Delete this group and all documents inside it?")) return;
  await api(`/api/documents/${selectedDocumentId}`, { method: "DELETE" });
  selectedDocumentId = null;
  selectedDocument = null;
  selectedPageId = null;
  pages = [];
  documentView.classList.add("hidden");
  emptyState.classList.remove("hidden");
  await loadDocuments();
}

async function renumberPage() {
  const page = currentPage();
  if (!page) return;
  const title = prompt("Document title", page.page_title || pageLabel(page));
  if (title === null) return;
  const value = prompt("Document order number", page.page_number);
  if (value === null) return;
  const form = new FormData();
  form.append("page_number", value);
  form.append("page_title", title);
  const updated = await api(`/api/pages/${page.id}`, { method: "PATCH", body: form });
  pages = pages.map((item) => (item.id === updated.id ? updated : item)).sort((a, b) => a.page_number - b.page_number);
  renderTabs();
  renderPage(updated);
}

async function deletePage() {
  const page = currentPage();
  if (!page || !confirm("Delete this document from the group?")) return;
  await api(`/api/pages/${page.id}`, { method: "DELETE" });
  pages = pages.filter((item) => item.id !== page.id);
  selectedPageId = pages[0]?.id || null;
  renderTabs();
  renderPage(currentPage());
  await loadDocuments();
}

async function deleteExtraction(id, event) {
  event?.stopPropagation?.();
  if (!confirm("Delete this extraction result?")) return;
  await api(`/api/extractions/${id}`, { method: "DELETE" });
  if (activeExtractionId === id) {
    activeExtractionId = null;
    rawOutput.textContent = "";
    clearBboxOverlay();
  }
  pages = await api(`/api/documents/${selectedDocumentId}/pages`);
  renderPage(currentPage());
}

function selectExtraction(extraction) {
  activeExtractionId = extraction.id;
  renderPage(currentPage());
  rawOutput.textContent = extraction.raw_output || "";
  drawBbox(extraction.bbox);
}

function clearBboxOverlay() {
  document.querySelector(".bboxOverlay")?.remove();
}

function drawBbox(bbox) {
  clearBboxOverlay();
  const box = Array.isArray(bbox) && Array.isArray(bbox[0]) ? bbox[0] : bbox;
  if (!Array.isArray(box) || box.length !== 4) return;
  if (!previewImage.complete || !previewImage.naturalWidth) {
    previewImage.addEventListener("load", () => drawBbox(bbox), { once: true });
    return;
  }

  const imageRect = previewImage.getBoundingClientRect();
  const parentRect = previewImage.parentElement.getBoundingClientRect();
  const overlay = document.createElement("div");
  overlay.className = "bboxOverlay";
  const [x1, y1, x2, y2] = box.map(Number);
  const left = imageRect.left - parentRect.left + (x1 / 999) * imageRect.width;
  const top = imageRect.top - parentRect.top + (y1 / 999) * imageRect.height;
  const width = ((x2 - x1) / 999) * imageRect.width;
  const height = ((y2 - y1) / 999) * imageRect.height;
  overlay.style.left = `${left}px`;
  overlay.style.top = `${top}px`;
  overlay.style.width = `${width}px`;
  overlay.style.height = `${height}px`;
  previewImage.parentElement.appendChild(overlay);
}

function setBusy(button, text) {
  button.disabled = true;
  button.textContent = text;
}

function setReady(button, text) {
  button.disabled = false;
  button.textContent = text;
}

function updateBulkSelectionUi() {
  const count = selectedBulkDocumentIds.size;
  if (selectedDocumentsCount) {
    selectedDocumentsCount.textContent = `${count} selected`;
  }
  if (selectAllDocuments) {
    selectAllDocuments.checked = Boolean(documents.length) && documents.every((doc) => selectedBulkDocumentIds.has(doc.id));
    selectAllDocuments.indeterminate = count > 0 && !selectAllDocuments.checked;
  }
}

function pageLabel(page) {
  const title = String(page?.page_title || "").trim();
  return title || `Document ${page.page_number}`;
}

function allowedFile(name) {
  return /\.(png|jpe?g|webp|bmp|tiff?|pdf)$/i.test(name || "");
}

function formatBytes(value) {
  if (!value) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

function shorten(value) {
  return value.length > 120 ? `${value.slice(0, 120)}...` : value;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[char]);
}

on(createGroupButton, "click", createDocumentGroup);
on(pageFileInput, "change", () => uploadDocumentFiles(pageFileInput));
on(folderInput, "change", () => uploadDocumentFiles(folderInput));
on(refreshButton, "click", loadDocuments);
on(selectAllDocuments, "change", () => {
  if (selectAllDocuments.checked) {
    documents.forEach((doc) => selectedBulkDocumentIds.add(doc.id));
  } else {
    documents.forEach((doc) => selectedBulkDocumentIds.delete(doc.id));
  }
  renderList();
});
on(extractSelectedDocumentsButton, "click", extractBulkDocuments);
on(searchButton, "click", loadDocuments);
on(searchInput, "keydown", (event) => {
  if (event.key === "Enter") loadDocuments();
});
on(extractPageButton, "click", extractSelectedPage);
on(extractDocumentButton, "click", extractSelectedDocument);
on(renameDocumentButton, "click", renameDocument);
on(deleteDocumentButton, "click", deleteDocument);
on(renumberPageButton, "click", renumberPage);
on(deletePageButton, "click", deletePage);

function on(element, eventName, handler) {
  if (element) element.addEventListener(eventName, handler);
}

loadStatus().catch((error) => {
  modelStatus.textContent = error.message;
  modelStatus.classList.add("bad");
});
loadDocuments().catch(console.error);
