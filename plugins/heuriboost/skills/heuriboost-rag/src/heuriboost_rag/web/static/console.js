(() => {
  const form = document.querySelector("#upload-form");
  const result = document.querySelector("#preflight");
  const baseDataset = document.querySelector("#base-dataset");
  const productionCasesDataset = document.querySelector("#production-cases-dataset");
  const startRun = document.querySelector("#start-run");
  const runStartResult = document.querySelector("#run-start-result");
  const datasetsTableBody = document.querySelector("#datasets-table-body");
  if (!form || !result) return;

  const csrfToken = () => document.querySelector('meta[name="csrf-token"]')?.content || "";
  const csrfHeaders = (extra = {}) => {
    const token = csrfToken();
    return token ? { ...extra, "X-CSRF-Token": token } : extra;
  };

  const nextIdempotencyKey = (prefix) => {
    if (globalThis.crypto?.randomUUID) return `${prefix}-${globalThis.crypto.randomUUID()}`;
    return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  };

  const requestJson = async (url, options = {}) => {
    const response = await fetch(url, options);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.message || "请求失败");
    }
    return payload;
  };

  const mappingForRole = (role, columns) => {
    const targets = new Set([
      "domain",
      "query_id",
      "query",
      "doc_id",
      "text",
      "relevance",
      "split",
      "rank",
      "score",
      "case_id",
      "shown_doc_id",
      "shown_doc_text",
      "user_verdict",
    ]);
    const aliases = role === "base"
      ? {
          document: "text",
          doc_text: "text",
          query_text: "query",
          label: "relevance",
          raw_final_score: "score",
          verdict: "relevance",
        }
      : {
          text: "shown_doc_text",
          document: "shown_doc_text",
          doc_text: "shown_doc_text",
          query_text: "query",
          relevance: "user_verdict",
          label: "user_verdict",
          raw_final_score: "score",
          verdict: "user_verdict",
        };
    const mapping = {};
    const usedTargets = new Set();
    for (const column of columns) {
      if (targets.has(column)) {
        mapping[column] = column;
        usedTargets.add(column);
      }
    }
    for (const column of columns) {
      const target = aliases[column];
      if (target && !usedTargets.has(target)) {
        mapping[column] = target;
        usedTargets.add(target);
      }
    }
    return mapping;
  };

  const setOptions = (select, datasets, role, placeholder, preferredId, preferredRole) => {
    if (!select) return;
    const current = preferredRole === role ? preferredId : select.value;
    select.replaceChildren(new Option(placeholder, ""));
    for (const dataset of datasets.filter((item) => item.role === role)) {
      select.append(new Option(dataset.id, dataset.id));
    }
    if (current && Array.from(select.options).some((option) => option.value === current)) {
      select.value = current;
    }
  };

  const updateStartButton = () => {
    if (!startRun || !baseDataset || !productionCasesDataset) return;
    startRun.disabled = !(baseDataset.value && productionCasesDataset.value);
  };

  const renderDatasetTable = (datasets) => {
    if (!datasetsTableBody) return;
    datasetsTableBody.replaceChildren();
    if (!datasets.length) {
      const row = datasetsTableBody.insertRow();
      const cell = row.insertCell();
      cell.colSpan = 3;
      cell.textContent = "暂无数据集";
      return;
    }
    for (const dataset of datasets) {
      const row = datasetsTableBody.insertRow();
      row.insertCell().textContent = dataset.id;
      row.insertCell().textContent = dataset.role;
      row.insertCell().textContent = dataset.status;
    }
  };

  const refreshDatasets = async (preferredId = "", preferredRole = "") => {
    const datasets = await requestJson("/api/datasets");
    setOptions(baseDataset, datasets, "base", "选择基础数据集", preferredId, preferredRole);
    setOptions(productionCasesDataset, datasets, "production_cases", "选择生产 case 数据集", preferredId, preferredRole);
    renderDatasetTable(datasets);
    updateStartButton();
  };

  baseDataset?.addEventListener("change", updateStartButton);
  productionCasesDataset?.addEventListener("change", updateStartButton);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    result.className = "result";
    result.textContent = "";
    try {
      const role = form.querySelector("#dataset-role")?.value || "";
      const upload = await requestJson("/api/imports", {
        method: "POST",
        headers: csrfHeaders(),
        body: new FormData(form),
      });
      const dataset = await requestJson(`/api/imports/${upload.id}/normalize`, {
        method: "POST",
        headers: csrfHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({
          role,
          mapping: mappingForRole(role, upload.inspection.columns || []),
        }),
      });
      await refreshDatasets(dataset.id, role);
      result.textContent = `已入库：${dataset.id}（${dataset.metadata.rows} 行）。`;
    } catch (error) {
      result.classList.add("error");
      result.textContent = error.message || "上传失败";
    }
  });

  startRun?.addEventListener("click", async () => {
    if (!baseDataset || !productionCasesDataset || !runStartResult) return;
    runStartResult.className = "result";
    runStartResult.textContent = "";
    try {
      const run = await requestJson("/api/runs", {
        method: "POST",
        headers: csrfHeaders({
          "Content-Type": "application/json",
          "Idempotency-Key": nextIdempotencyKey("run"),
        }),
        body: JSON.stringify({
          base_dataset_id: baseDataset.value,
          production_cases_id: productionCasesDataset.value,
        }),
      });
      runStartResult.textContent = `已创建运行：${run.id}（${run.state} / ${run.job_status}）。`;
    } catch (error) {
      runStartResult.classList.add("error");
      runStartResult.textContent = error.message || "启动失败";
    }
  });

  updateStartButton();
})();
