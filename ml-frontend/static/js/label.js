const datasetName = new URLSearchParams(window.location.search).get("dataset");
const carouselList = document.getElementById("carouselList");
const labeledData = [];

async function loadImages() {
  console.log("Loading images for dataset:", datasetName);

  const res = await fetch(`/proxy_image/${datasetName}/selection/selection.json`);
  console.log("Fetched selection.json, status:", res.status);

  if (!res.ok) {
    console.error("Fout bij ophalen selection.json:", await res.text());
    return;
  }

  const json = await res.json();
  const filenames = json.selected_filenames;
  console.log("Selected filenames:", filenames);

  filenames.forEach(filename => {
    console.log("Adding image to carousel:", filename);
    const li = document.createElement("li");
    li.className = "splide__slide";

    // Gebruik proxy endpoint
    const proxiedPath = `/proxy_image/${datasetName}/${filename}`;
    li.innerHTML = `<img src="${proxiedPath}" data-filename="${filename}" alt="image" />`;

    carouselList.appendChild(li);
  });

  console.log("Initializing Splide carousel");
  new Splide("#imageCarousel", {
    perPage: 1,
    pagination: false,
    arrows: true,
  }).mount();
  console.log("Splide carousel mounted");
}

function setupButtons() {
  const buttons = document.querySelectorAll(".emotionButton");
  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      const current = document.querySelector(".splide__slide.is-active img");
      if (!current) return;

      const filename = current.getAttribute("data-filename");
      const emotion = btn.getAttribute("data-emotion");

      fetch(`/proxy_image/${datasetName}/${filename}`)
        .then(res => res.blob())
        .then(blob => {
          const form = new FormData();
          form.append("dataset_name", datasetName);
          form.append("emotion", emotion);
          form.append("file", new File([blob], filename.split("/").pop(), { type: "image/png" }));

          return fetch("/proxy_label", {
            method: "POST",
            body: form,
          });
        })
        .then(res => {
          if (!res.ok) throw new Error("Labeling mislukt");
          labeledData.push({ filename, emotion });
          console.log(`Labeled & uploaded: ${filename} as ${emotion}`);

          const splide = document.querySelector("#imageCarousel").splide;
          splide.go(">");
        })
        .catch(err => {
          console.error("Error bij labelen:", err);
          alert("Er ging iets mis bij het labelen.");
        });
    });
  });
}


window.onload = async () => {
  await loadImages();
  setupButtons();
};
