const API="http://127.0.0.1:8001";

const $=id=>document.getElementById(id);

async function api(path){
  const r=await fetch(API+path);
  if(!r.ok){
    let d=`HTTP ${r.status}`;
    try{
      d=(await r.json()).detail||d;
    }catch{}
    throw Error(d);
  }
  return r.json();
}

function status(text,ok=false){
  $("status").textContent=text;
  $("status").className="status"+(ok?" ok":"");
}

function esc(v){
  return String(v??"")
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

function dist(id,items){
  const el=$(id);
  el.innerHTML="";
  const max=Math.max(...items.map(x=>x.count),1);

  items.forEach(x=>{
    el.insertAdjacentHTML(
      "beforeend",
      `<div class="dist">
        <span>${esc(x.label)}</span>
        <div class="track">
          <div class="bar" style="width:${x.count/max*100}%"></div>
        </div>
        <strong>${x.count}</strong>
      </div>`
    );
  });
}

function thumbnail(faceId){
  return `${API}/api/dashboard/faces/${encodeURIComponent(faceId)}/thumbnail`;
}

function faceCard(f){
  const similarity = f.similarity == null
    ? "—"
    : Number(f.similarity).toFixed(4);

  const detection = f.detection_confidence == null
    ? "—"
    : Number(f.detection_confidence).toFixed(4);

  return `
    <article class="face">
      <img
        class="face-thumb"
        src="${thumbnail(f.face_id)}"
        alt="Face ${esc(f.face_id)} thumbnail"
        loading="lazy"
        onerror="this.classList.add('fallback');this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%2292%22 height=%2292%22%3E%3Crect width=%2292%22 height=%2292%22 fill=%22%23eef2f7%22/%3E%3Ctext x=%2246%22 y=%2248%22 text-anchor=%22middle%22 font-size=%2211%22 fill=%22%23647569%22%3ENo image%3C/text%3E%3C/svg%3E';"
      >

      <div class="face-info">
        <div class="face-title">Face ${esc(f.face_id)}</div>

        <div class="face-row">
          Image ${esc(f.image_id)}
          <div class="muted">Index ${esc(f.face_index)}</div>
        </div>

        <div class="face-row">
          ${esc(f.gender||"Unknown")} · ${esc(f.age_group||"Unknown")}
        </div>

        ${f.age_years != null
          ? `<div class="face-row">Age ${esc(f.age_years)}</div>`
          : ""}

        <div class="face-metric">
          Similarity ${similarity}
        </div>

        <div class="muted">
          Detection ${detection}
        </div>
      </div>
    </article>
  `;
}

async function load(){
  status("Loading dashboard...");

  try{
    const [s,c]=await Promise.all([
      api("/api/dashboard/summary"),
      api("/api/dashboard/clusters")
    ]);

    $("totalFaces").textContent=s.total_faces;
    $("totalClusters").textContent=s.total_clusters;
    $("totalImages").textContent=s.total_images;
    $("clusterCount").textContent=`${c.total_clusters} clusters`;

    dist("genderDistribution",s.gender_distribution);
    dist("ageDistribution",s.age_group_distribution);

    $("clusters").innerHTML=c.clusters.map(x=>`
      <button class="cluster" data-id="${x.cluster_id}">
        <b>Cluster ${x.cluster_id}</b>
        <div class="muted">
          ${x.face_count} face${x.face_count===1?"":"s"}
        </div>
      </button>
    `).join("");

    document.querySelectorAll(".cluster").forEach(b=>{
      b.onclick=()=>cluster(b.dataset.id);
    });

    status("Connected to Workflow 3 Retrieval API",true);

  }catch(e){
    status("Dashboard load failed: "+e.message);
  }
}

async function cluster(id){
  try{
    const [d,imgs]=await Promise.all([
      api(`/api/dashboard/clusters/${id}`),
      api(`/api/dashboard/clusters/${id}/images`)
    ]);

    $("clusterTitle").textContent=`Cluster ${d.cluster_id}`;
    $("clusterMeta").textContent=
      `${d.face_count} face${d.face_count===1?"":"s"}`;

    $("clusterFaces").innerHTML=d.faces.map(faceCard).join("");

    $("clusterImages").innerHTML=imgs.images.map(i=>`
      <div class="image" data-id="${i.image_id}">
        <b>Image ${i.image_id}</b>
        <div class="muted">Upload ${i.upload_id}</div>
        <div class="muted">
          ${i.width||"?"} × ${i.height||"?"} ·
          ${esc(i.content_type||"unknown")}
        </div>
      </div>
    `).join("");

    document.querySelectorAll("#clusterImages .image").forEach(x=>{
      x.onclick=()=>imageFaces(x.dataset.id);
    });

    $("clusterPanel").classList.remove("hidden");
    $("imagePanel").classList.add("hidden");
    $("clusterPanel").scrollIntoView({behavior:"smooth"});

  }catch(e){
    status("Cluster load failed: "+e.message);
  }
}

async function imageFaces(id){
  try{
    const d=await api(`/api/dashboard/images/${id}/faces`);

    $("imageTitle").textContent=`Image ${d.image_id} Faces`;
    $("imageMeta").textContent=
      `${d.total_faces} face${d.total_faces===1?"":"s"} detected`;

    $("imageFaces").innerHTML=d.faces.map(f=>`
      <article class="face">
        <img
          class="face-thumb"
          src="${thumbnail(f.face_id)}"
          alt="Face ${esc(f.face_id)} thumbnail"
          loading="lazy"
          onerror="this.classList.add('fallback');this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%2292%22 height=%2292%22%3E%3Crect width=%2292%22 height=%2292%22 fill=%22%23eef2f7%22/%3E%3Ctext x=%2246%22 y=%2248%22 text-anchor=%22middle%22 font-size=%2211%22 fill=%22%23647569%22%3ENo image%3C/text%3E%3C/svg%3E';"
        >

        <div class="face-info">
          <div class="face-title">Face ${esc(f.face_id)}</div>

          <div class="face-row">
            Image ${esc(f.image_id)}
            <div class="muted">Index ${esc(f.face_index)}</div>
          </div>

          <div class="face-row">
            ${esc(f.gender||"Unknown")} · ${esc(f.age_group||"Unknown")}
          </div>

          <div class="face-row">
            Age ${f.age_years??"—"}
          </div>

          <div class="face-metric">
            Bounding box:
            ${[
              f.bbox_x1,
              f.bbox_y1,
              f.bbox_x2,
              f.bbox_y2
            ].map(x=>Number(x).toFixed(1)).join(", ")}
          </div>

          <div class="muted">
            Detection ${
              f.detection_confidence == null
                ? "—"
                : Number(f.detection_confidence).toFixed(4)
            }
          </div>
        </div>
      </article>
    `).join("");

    $("imagePanel").classList.remove("hidden");
    $("imagePanel").scrollIntoView({behavior:"smooth"});

  }catch(e){
    status("Image load failed: "+e.message);
  }
}

$("refreshBtn").onclick=load;

$("closeCluster").onclick=()=>{
  $("clusterPanel").classList.add("hidden");
  $("imagePanel").classList.add("hidden");
};

$("closeImage").onclick=()=>{
  $("imagePanel").classList.add("hidden");
};

load();
