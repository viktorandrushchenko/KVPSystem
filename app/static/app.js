let documents = [];
let selectedDocumentId = null;
let selectedDocument = null;
let selectedPageId = null;
let pages = [];
let selectedBulkDocumentIds = new Set();
let checkpoints = [];
let annotationSets = [];
let selectedAnnotationSetId = null;
let annotationPages = [];
let annotationBbox = null;
let annotationDrag = null;

const modelStatus = document.querySelector("#modelStatus");
const createGroupButton = document.querySelector("#createGroupButton");
const extractTabButton = document.querySelector("#extractTabButton");
const annotateTabButton = document.querySelector("#annotateTabButton");
const extractTab = document.querySelector("#extractTab");
const annotateTab = document.querySelector("#annotateTab");
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
const checkpointSelect = document.querySelector("#checkpointSelect");
const renameCheckpointButton = document.querySelector("#renameCheckpointButton");
const kvTable = document.querySelector("#kvTable");
const rawOutput = document.querySelector("#rawOutput");
const newAnnotationSetButton = document.querySelector("#newAnnotationSetButton");
const annotationSetSelect = document.querySelector("#annotationSetSelect");
const annotationSetList = document.querySelector("#annotationSetList");
const annotationGroupSelect = document.querySelector("#annotationGroupSelect");
const annotationPageSelect = document.querySelector("#annotationPageSelect");
const annotationKeyInput = document.querySelector("#annotationKeyInput");
const annotationValueInput = document.querySelector("#annotationValueInput");
const saveAnnotationButton = document.querySelector("#saveAnnotationButton");
const trainAnnotationSetButton = document.querySelector("#trainAnnotationSetButton");
const trainingStepsInput = document.querySelector("#trainingStepsInput");
const bboxCoordinates = document.querySelector("#bboxCoordinates");
const annotationStatus = document.querySelector("#annotationStatus");
const annotationPreview = document.querySelector("#annotationPreview");
const annotationImage = document.querySelector("#annotationImage");
const annotationList = document.querySelector("#annotationList");
const refreshAnnotationButton = document.querySelector("#refreshAnnotationButton");
const trainingLog = document.querySelector("#trainingLog");
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
  renderAnnotationSources();
}

async function loadCheckpoints() {
  checkpoints = await api("/api/checkpoints");
  if (!checkpointSelect) return;
  checkpointSelect.innerHTML = "";
  for (const checkpoint of checkpoints) {
    const option = document.createElement("option");
    option.value = checkpoint.path;
    option.textContent = checkpoint.folder_name && checkpoint.folder_name !== checkpoint.name
      ? `${checkpoint.name} (${checkpoint.folder_name})`
      : checkpoint.name;
    checkpointSelect.appendChild(option);
  }
}

async function loadAnnotationSets() {
  annotationSets = await api("/api/annotation-sets");
  if (!selectedAnnotationSetId && annotationSets.length) {
    selectedAnnotationSetId = annotationSets[0].id;
  }
  renderAnnotationSets();
  await renderAnnotations();
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
  appendCheckpoint(form);
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
  appendCheckpoint(form);
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
  appendCheckpoint(form);
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

function appendCheckpoint(form) {
  const value = checkpointSelect?.value;
  if (value) form.append("checkpoint_path", value);
}

function switchTab(name) {
  const annotationMode = name === "annotate";
  extractTab.classList.toggle("active", !annotationMode);
  annotateTab.classList.toggle("active", annotationMode);
  extractTabButton.classList.toggle("active", !annotationMode);
  annotateTabButton.classList.toggle("active", annotationMode);
  if (annotationMode) {
    renderAnnotationSources();
    loadAnnotationSets().catch(showAnnotationError);
  }
}

function renderAnnotationSets() {
  if (!annotationSetList) return;
  annotationSetList.innerHTML = "";
  if (annotationSetSelect) {
    annotationSetSelect.innerHTML = '<option value="">Create/select dataset</option>';
    for (const set of annotationSets) {
      const option = document.createElement("option");
      option.value = set.id;
      option.textContent = `${set.name} (${set.annotation_count || 0})`;
      annotationSetSelect.appendChild(option);
    }
    annotationSetSelect.value = selectedAnnotationSetId ? String(selectedAnnotationSetId) : "";
  }
  if (!annotationSets.length) {
    annotationSetList.innerHTML = '<p class="status">No annotation datasets yet. It will be created automatically on first save.</p>';
    return;
  }
  for (const set of annotationSets) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `docItem ${set.id === selectedAnnotationSetId ? "active" : ""}`;
    button.innerHTML = `
      <strong>${escapeHtml(set.name)}</strong>
      <span class="status">${escapeHtml(set.status)} - ${set.annotation_count || 0} annotations</span>
    `;
    button.addEventListener("click", async () => {
      selectedAnnotationSetId = set.id;
      renderAnnotationSets();
      await renderAnnotations();
    });
    annotationSetList.appendChild(button);
  }
}

function renderAnnotationSources() {
  if (!annotationGroupSelect) return;
  const previousGroup = annotationGroupSelect.value || String(selectedDocumentId || "");
  annotationGroupSelect.innerHTML = '<option value="">Select group</option>';
  for (const doc of documents) {
    const option = document.createElement("option");
    option.value = doc.id;
    option.textContent = doc.filename;
    annotationGroupSelect.appendChild(option);
  }
  if ([...annotationGroupSelect.options].some((option) => option.value === previousGroup)) {
    annotationGroupSelect.value = previousGroup;
  }
  loadAnnotationPages().catch(showAnnotationError);
}

async function loadAnnotationPages() {
  const groupId = annotationGroupSelect?.value;
  annotationPages = [];
  annotationPageSelect.innerHTML = '<option value="">Select document</option>';
  clearAnnotationSelection();
  if (!groupId) return;
  annotationPages = await api(`/api/documents/${groupId}/pages`);
  for (const page of annotationPages) {
    const option = document.createElement("option");
    option.value = page.id;
    option.textContent = pageLabel(page);
    annotationPageSelect.appendChild(option);
  }
  if (annotationPages.length) {
    annotationPageSelect.value = annotationPages[0].id;
    renderAnnotationImage();
  }
}

function renderAnnotationImage() {
  clearAnnotationSelection();
  const pageId = annotationPageSelect?.value;
  if (!pageId) {
    annotationImage.removeAttribute("src");
    return;
  }
  annotationImage.src = `/api/pages/${pageId}/image`;
}

async function createAnnotationSet() {
  const name = prompt("Dataset name");
  if (!name || !name.trim()) return;
  const form = new FormData();
  form.append("name", name.trim());
  const set = await api("/api/annotation-sets", { method: "POST", body: form });
  selectedAnnotationSetId = set.id;
  await loadAnnotationSets();
}

async function ensureAnnotationSet() {
  if (selectedAnnotationSetId) return selectedAnnotationSetId;
  const groupOption = annotationGroupSelect.options[annotationGroupSelect.selectedIndex];
  const name = groupOption?.value ? `${groupOption.textContent} annotations` : "Manual annotations";
  const form = new FormData();
  form.append("name", name.trim());
  const set = await api("/api/annotation-sets", { method: "POST", body: form });
  selectedAnnotationSetId = set.id;
  await loadAnnotationSets();
  return selectedAnnotationSetId;
}

async function renderAnnotations() {
  if (!annotationList) return;
  annotationList.innerHTML = "";
  trainingLog.textContent = "";
  if (!selectedAnnotationSetId) {
    annotationList.innerHTML = '<p class="status">Select or create a dataset.</p>';
    return;
  }
  const set = await api(`/api/annotation-sets/${selectedAnnotationSetId}`);
  trainingLog.textContent = set.training_log || "";
  const rows = set.annotations || [];
  if (!rows.length) {
    annotationList.innerHTML = '<p class="status">No saved annotations.</p>';
    return;
  }
  for (const item of rows) {
    const row = document.createElement("div");
    row.className = "annotationItem";
    row.innerHTML = `
      <strong>${escapeHtml(item.query_key)} = ${escapeHtml(item.value_text)}</strong>
      <span class="status">${escapeHtml(item.group_name)} / ${escapeHtml(item.page_title || `Document ${item.page_number}`)}</span>
      <span class="status">${escapeHtml(JSON.stringify(item.bbox))}</span>
      <button type="button" class="smallButton dangerButton">Delete</button>
    `;
    row.addEventListener("click", () => showAnnotation(item));
    row.querySelector("button").addEventListener("click", (event) => deleteAnnotation(item.id, event));
    annotationList.appendChild(row);
  }
}

async function saveAnnotation() {
  const pageId = annotationPageSelect.value;
  const key = annotationKeyInput.value.trim();
  const value = annotationValueInput.value.trim();
  if (!pageId) return alert("Select a document.");
  if (!key || !value) return alert("Enter key and value.");
  if (!annotationBbox) return alert("Draw a bbox on the document.");
  const setId = await ensureAnnotationSet();
  const form = new FormData();
  form.append("annotation_set_id", String(setId));
  form.append("page_id", pageId);
  form.append("query_key", key);
  form.append("value_text", value);
  form.append("bbox_json", JSON.stringify([annotationBbox]));
  await api("/api/annotations", { method: "POST", body: form });
  annotationStatus.textContent = "Annotation saved.";
  annotationValueInput.value = "";
  clearAnnotationSelection();
  await loadAnnotationSets();
}

async function deleteAnnotation(id, event) {
  event.stopPropagation();
  if (!confirm("Delete this annotation?")) return;
  await api(`/api/annotations/${id}`, { method: "DELETE" });
  await loadAnnotationSets();
}

async function trainAnnotationSet() {
  if (!selectedAnnotationSetId) return alert("Select an annotation dataset first.");
  setBusy(trainAnnotationSetButton, "Training...");
  try {
    const form = new FormData();
    form.append("steps", String(Math.max(1, Number(trainingStepsInput?.value || 100))));
    const set = await api(`/api/annotation-sets/${selectedAnnotationSetId}/train`, { method: "POST", body: form });
    trainingLog.textContent = set.training_log || "Training queued.";
    await loadAnnotationSets();
    await loadCheckpoints();
  } finally {
    setReady(trainAnnotationSetButton, "Train model");
  }
}

async function renameSelectedCheckpoint() {
  const path = checkpointSelect?.value;
  if (!path) return alert("Select a checkpoint first.");
  const checkpoint = checkpoints.find((item) => item.path === path);
  const currentName = checkpoint?.name || checkpoint?.folder_name || "";
  const name = prompt("Checkpoint name", currentName);
  if (!name || !name.trim()) return;
  const form = new FormData();
  form.append("path", path);
  form.append("name", name.trim());
  await api("/api/checkpoints/name", { method: "PATCH", body: form });
  await loadCheckpoints();
  checkpointSelect.value = path;
}

function showAnnotation(item) {
  annotationKeyInput.value = item.query_key || "";
  annotationValueInput.value = item.value_text || "";
  const group = documents.find((doc) => doc.id === item.document_id);
  if (group) {
    annotationGroupSelect.value = group.id;
    loadAnnotationPages().then(() => {
      annotationPageSelect.value = item.page_id;
      renderAnnotationImage();
      annotationImage.addEventListener("load", () => drawAnnotationBox(item.bbox?.[0] || item.bbox), { once: true });
    });
  }
}

function startAnnotationDraw(event) {
  if (event.button !== 0) return;
  event.preventDefault();
  if (!annotationImage.src || !annotationImage.complete) return;
  const point = imagePoint(event);
  if (!point) return;
  annotationDrag = { start: point, end: point };
  drawAnnotationRect(point, point);
}

function moveAnnotationDraw(event) {
  if (!annotationDrag) return;
  event.preventDefault();
  const point = imagePoint(event);
  if (!point) return;
  annotationDrag.end = point;
  drawAnnotationRect(annotationDrag.start, annotationDrag.end);
}

function finishAnnotationDraw() {
  if (!annotationDrag) return;
  const box = pointsToNormalizedBox(annotationDrag.start, annotationDrag.end);
  annotationDrag = null;
  if (!box) {
    clearAnnotationSelection();
    return;
  }
  annotationBbox = box;
  drawAnnotationBox(box);
  updateBboxText(box);
}

function imagePoint(event) {
  const rect = annotationImage.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  if (x < 0 || y < 0 || x > rect.width || y > rect.height) return null;
  return { x, y, width: rect.width, height: rect.height };
}

function pointsToNormalizedBox(a, b) {
  const left = Math.max(0, Math.min(a.x, b.x));
  const top = Math.max(0, Math.min(a.y, b.y));
  const right = Math.min(a.width, Math.max(a.x, b.x));
  const bottom = Math.min(a.height, Math.max(a.y, b.y));
  if (right - left < 4 || bottom - top < 4) return null;
  return [
    Math.round((left / a.width) * 999),
    Math.round((top / a.height) * 999),
    Math.round((right / a.width) * 999),
    Math.round((bottom / a.height) * 999),
  ];
}

function drawAnnotationRect(a, b) {
  const box = pointsToNormalizedBox(a, b);
  if (box) drawAnnotationBox(box, "annotationDraftBox");
}

function drawAnnotationBox(box, className = "annotationBbox") {
  document.querySelector(".annotationBbox")?.remove();
  document.querySelector(".annotationDraftBox")?.remove();
  if (!Array.isArray(box) || box.length !== 4) return;
  const imageRect = annotationImage.getBoundingClientRect();
  const parentRect = annotationPreview.getBoundingClientRect();
  const overlay = document.createElement("div");
  overlay.className = className;
  const [x1, y1, x2, y2] = box.map(Number);
  overlay.style.left = `${imageRect.left - parentRect.left + (x1 / 999) * imageRect.width}px`;
  overlay.style.top = `${imageRect.top - parentRect.top + (y1 / 999) * imageRect.height}px`;
  overlay.style.width = `${((x2 - x1) / 999) * imageRect.width}px`;
  overlay.style.height = `${((y2 - y1) / 999) * imageRect.height}px`;
  annotationPreview.appendChild(overlay);
  if (className === "annotationBbox") {
    annotationBbox = box;
    updateBboxText(box);
  }
}

function clearAnnotationSelection() {
  annotationBbox = null;
  annotationDrag = null;
  document.querySelector(".annotationBbox")?.remove();
  document.querySelector(".annotationDraftBox")?.remove();
  updateBboxText(null);
}

function updateBboxText(box) {
  if (!bboxCoordinates) return;
  bboxCoordinates.textContent = box ? `BBox: [${box.join(", ")}]` : "BBox: not selected";
}

function showAnnotationError(error) {
  if (annotationStatus) annotationStatus.textContent = error.message;
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
on(extractTabButton, "click", () => switchTab("extract"));
on(annotateTabButton, "click", () => switchTab("annotate"));
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
on(renameCheckpointButton, "click", () => renameSelectedCheckpoint().catch(console.error));
on(renameDocumentButton, "click", renameDocument);
on(deleteDocumentButton, "click", deleteDocument);
on(renumberPageButton, "click", renumberPage);
on(deletePageButton, "click", deletePage);
on(newAnnotationSetButton, "click", createAnnotationSet);
on(annotationSetSelect, "change", async () => {
  selectedAnnotationSetId = annotationSetSelect.value ? Number(annotationSetSelect.value) : null;
  renderAnnotationSets();
  await renderAnnotations();
});
on(annotationGroupSelect, "change", () => loadAnnotationPages().catch(showAnnotationError));
on(annotationPageSelect, "change", renderAnnotationImage);
on(saveAnnotationButton, "click", () => saveAnnotation().catch(showAnnotationError));
on(trainAnnotationSetButton, "click", () => trainAnnotationSet().catch(showAnnotationError));
on(refreshAnnotationButton, "click", () => loadAnnotationSets().catch(showAnnotationError));
on(annotationPreview, "mousedown", startAnnotationDraw);
on(annotationPreview, "contextmenu", (event) => event.preventDefault());
on(annotationPreview, "mousemove", moveAnnotationDraw);
on(document, "mouseup", finishAnnotationDraw);

function on(element, eventName, handler) {
  if (element) element.addEventListener(eventName, handler);
}

loadStatus().catch((error) => {
  modelStatus.textContent = error.message;
  modelStatus.classList.add("bad");
});
loadDocuments().catch(console.error);
loadCheckpoints().catch(console.error);
loadAnnotationSets().catch(console.error);
