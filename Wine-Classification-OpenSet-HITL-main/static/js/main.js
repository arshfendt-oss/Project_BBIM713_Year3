function readJsonData(element, attribute, fallback) {
    if (!element) return fallback;
    try {
        return JSON.parse(element.dataset[attribute] || "null") || fallback;
    } catch {
        return fallback;
    }
}

function createBarChart(id, label, color) {
    const canvas = document.getElementById(id);
    if (!canvas || typeof Chart === "undefined") return;
    const labels = readJsonData(canvas, "labels", []);
    const values = readJsonData(canvas, "values", []);
    if (!labels.length || !values.length) return;
    new Chart(canvas, {
        type: "bar",
        data: {
            labels,
            datasets: [{ label, data: values, backgroundColor: color, borderColor: color, borderWidth: 1 }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { beginAtZero: true, max: values.every(v => v <= 1) ? 1 : undefined } },
            plugins: { legend: { display: false } }
        }
    });
}

function initCharts() {
    createBarChart("classificationChart", "Score", "#2454a6");
    createBarChart("osrChart", "Score", "#0f766e");
    createBarChart("retrainingMiniChart", "Score", "#7c3aed");
    createBarChart("retrainingChart", "Score", "#2454a6");

    const clusterCanvas = document.getElementById("clusterChart");
    if (clusterCanvas && typeof Chart !== "undefined") {
        const clusters = readJsonData(clusterCanvas, "clusters", []);
        if (clusters.length) {
            new Chart(clusterCanvas, {
                type: "bar",
                data: {
                    labels: clusters.map(item => `Cluster ${item.cluster}`),
                    datasets: [{ label: "Records", data: clusters.map(item => item.count), backgroundColor: "#0f766e" }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: { y: { beginAtZero: true } },
                    plugins: { legend: { display: false } }
                }
            });
        }
    }
}

function initSidebar() {
    const button = document.getElementById("sidebarToggle");
    const sidebar = document.getElementById("sidebar");
    if (!button || !sidebar) return;
    button.addEventListener("click", () => sidebar.classList.toggle("open"));
}

function initTableSearch() {
    document.querySelectorAll(".table-search").forEach(input => {
        input.addEventListener("input", () => {
            const table = document.getElementById(input.dataset.table);
            if (!table) return;
            const query = input.value.toLowerCase();
            table.querySelectorAll("tbody tr").forEach(row => {
                row.style.display = row.innerText.toLowerCase().includes(query) ? "" : "none";
            });
        });
    });
}

function initClusterFilter() {
    const select = document.getElementById("clusterFilter");
    const table = document.getElementById("clusterTable");
    if (!select || !table) return;
    select.addEventListener("change", () => {
        const value = select.value;
        table.querySelectorAll("tbody tr").forEach(row => {
            row.style.display = value === "all" || row.dataset.cluster === value ? "" : "none";
        });
    });
}

function initOsrFilter() {
    const buttons = document.querySelectorAll(".osr-filter");
    const table = document.getElementById("osrTable");
    if (!buttons.length || !table) return;
    buttons.forEach(button => {
        button.addEventListener("click", () => {
            buttons.forEach(item => item.classList.remove("active"));
            button.classList.add("active");
            const filter = button.dataset.filter;
            table.querySelectorAll("tbody tr").forEach(row => {
                row.style.display = filter === "all" || row.dataset.osr === filter ? "" : "none";
            });
        });
    });
}

function initPredictionForm() {
    const form = document.getElementById("predictionForm");
    const result = document.getElementById("predictionResult");
    if (!form || !result) return;
    form.addEventListener("submit", async event => {
        event.preventDefault();
        const text = form.querySelector("[name='wine_text']").value.trim();
        if (!text) {
            result.className = "result-card mt-3";
            result.innerHTML = "<div class='alert alert-danger mb-0'>Please enter a wine name or description.</div>";
            return;
        }
        const body = new FormData(form);
        const response = await fetch("/classifier/predict", { method: "POST", body });
        const data = await response.json();
        result.className = "result-card mt-3";
        if (!response.ok) {
            result.innerHTML = `<div class="alert alert-danger mb-0">${data.error || "Prediction failed."}</div>`;
            return;
        }
        const badge = data.osr_status === "KNOWN" ? "text-bg-success" : "text-bg-warning";
        result.innerHTML = `
            <div class="result-grid">
                <div><span>Predicted Wine Key</span><strong>${data.predicted_class}</strong></div>
                <div><span>Confidence Score</span><strong>${Number(data.confidence).toFixed(4)}</strong></div>
                <div><span>Classification Status</span><strong>${data.classification_status}</strong></div>
                <div><span>OSR Status</span><strong><span class="badge ${badge}">${data.osr_status}</span></strong></div>
            </div>
            ${data.closest_known_class ? `<div class="mt-3"><span>Closest Known Class</span><strong>${data.closest_known_class}</strong></div>` : ""}
            <div class="mt-2 text-muted">Threshold: ${Number(data.threshold).toFixed(4)}</div>`;
    });
}

function initConfirmations() {
    document.querySelectorAll(".confirm-action").forEach(button => {
        button.addEventListener("click", event => {
            const action = button.value;
            if (!window.confirm(`Confirm ${action} for this review record?`)) {
                event.preventDefault();
            }
        });
    });
}

function initImageModal() {
    const modalElement = document.getElementById("imageModal");
    const modalImg = document.getElementById("imageModalImg");
    const modalTitle = document.getElementById("imageModalTitle");
    if (!modalElement || !modalImg || !modalTitle || typeof bootstrap === "undefined") return;
    const modal = new bootstrap.Modal(modalElement);
    document.querySelectorAll(".image-modal-trigger").forEach(image => {
        image.addEventListener("click", () => {
            modalImg.src = image.src;
            modalTitle.textContent = image.dataset.title || image.alt || "Experiment Figure";
            modal.show();
        });
    });
}

document.addEventListener("DOMContentLoaded", () => {
    initSidebar();
    initCharts();
    initTableSearch();
    initClusterFilter();
    initOsrFilter();
    initPredictionForm();
    initConfirmations();
    initImageModal();
});
