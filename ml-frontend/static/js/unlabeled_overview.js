async function loadTableData() {
    const response = await fetch("/api/unlabeled_datasets");
    const data = await response.json();

    const tbody = document.querySelector("tbody");
    tbody.innerHTML = "";

    data.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

    for (const row of data) {
    const tr = document.createElement("tr");

    const userTd = `<td data-label="User">${row.user}</td>`;
    const totalTd = `<td data-label="Total Images">${row.total_images}</td>`;
    const progressTd = `
        <td data-label="Labeled %">
        <div class="progress">
            <div class="progress-bar" style="width: ${row.labeled_percentage}%;">${row.labeled_percentage}%</div>
        </div>
        </td>`;
    const timestampTd = `<td data-label="Received">${row.received}</td>`;

    tr.innerHTML = userTd + totalTd + progressTd + timestampTd;

    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => {
        window.location.href = `/label?dataset=${encodeURIComponent(row.dataset_id)}`;
    });

    tbody.appendChild(tr);
    }

}

window.onload = loadTableData;