# Panduan Analisis & Implementasi Lengkap: Deteksi Mind Wandering Menggunakan WebGazer.js

Dokumen ini mendokumentasikan hasil analisis riset dari jurnal:
> **"Webcam-based eye tracking to detect mind wandering"** (*Behavior Research Methods, 2024*)  
> Serta panduan arsitektur dan langkah-langkah implementasi teknis dari nol (*from scratch*) menggunakan **WebGazer.js**.

---

## DAFTAR ISI
1. [Konsep Dasar: WebGazer.js, Machine Learning, atau Deep Learning?](#1-konsep-dasar-webgazerjs-machine-learning-atau-deep-learning)
2. [Ringkasan & Analisis Jurnal Acuan](#2-ringkasan--analisis-jurnal-acuan)
3. [Arsitektur Sistem Eksperimen](#3-arsitektur-sistem-eksperimen)
4. [Desain Stimulus & Definisi Area of Interest (AOI)](#4-desain-stimulus--definisi-area-of-interest-aoi)
5. [Ekstraksi Fitur (Feature Engineering)](#5-ekstraksi-fitur-feature-engineering)
6. [Pipeline Machine Learning & Validasi](#6-pipeline-machine-learning--validasi)
7. [Panduan Setup Projek dari Nol (Step-by-Step)](#7-panduan-setup-projek-dari-nol-step-by-step)
8. [Petunjuk Eksekusi & Best Practices](#8-petunjuk-eksekusi--best-practices)

---

## 1. Konsep Dasar: WebGazer.js, Machine Learning, atau Deep Learning?

### Apakah Eye Tracking lebih baik pakai Deep Learning daripada Machine Learning?
**Jawaban singkat:** Tergantung lapisannya (*layer*).
* **Untuk Ekstraksi Fitur Mata (Computer Vision)**: **Deep Learning (DL) jauh lebih unggul** daripada Computer Vision tradisional karena tahan terhadap variasi pencahayaan, bentuk mata, kacamata, dan orientasi wajah.
* **Untuk Pemetaan Gaze ke Titik Layar (*Calibration Mapping*)**: **Machine Learning Tradisional (seperti Ridge Regression)** sering kali jauh lebih praktis dan cepat, karena mampu beradaptasi (*on-the-fly*) secara instan dengan kalibrasi beberapa titik klik tanpa memerlukan pelatihan ulang jaringan saraf tiruan (*neural network*) yang berat.
* **Untuk Deteksi Status Kognitif (Mind Wandering / Comprehension)**: **Classical Machine Learning (XGBoost, Random Forest, SVM)** lebih disarankan bila ukuran dataset terbatas (ratusan hingga ribuan sampel). Deep learning pada dataset kognitif berukuran kecil sangat rentan mengalami *severe overfitting*.

### WebGazer.js itu apa? Deep Learning atau Machine Learning?
**WebGazer.js adalah sistem Hibrida (Hybrid DL + Classical ML)** yang dibagi menjadi 3 lapisan:

```
[ Input Webcam ] 
       │
       ▼
[ Layer 1: MediaPipe FaceMesh / TFLite ]  ──► DEEP LEARNING (CNN / MobileNet)
  (Mendeteksi 468 titik landmark wajah,       Mengekstrak pupil & kontur mata
   iris, dan kontur mata di browser)           secara real-time via WebGL/Wasm.
       │
       ▼
[ Layer 2: Ridge Regression / Weighted ] ──► CLASSICAL MACHINE LEARNING
  (src/ridgeReg.js)                           Memetakan fitur mata ke koordinat (x, y)
                                               layar berdasarkan titik klik kalibrasi user.
       │
       ▼
[ Layer 3: Kalman Filter ]               ──► STATISTICAL SIGNAL PROCESSING
  (src/kalman.js)                             Meredam jitter / noise koordinat gaze.
       │
       ▼
[ Output: Prediksi Koordinat (x, y) Layar ]
```

Dengan arsitektur hibrida ini:
1. **Tidak butuh GPU server**: Berjalan 100% di browser klien.
2. **Kalibrasi Instan**: *Ridge Regression* menghitung invers matriks secara instan saat pengguna mengklik titik layar, tanpa *backpropagation* lambat.
3. **Privasi Terjamin**: Feed video tidak dikirim ke internet.

---

## 2. Ringkasan & Analisis Jurnal Acuan

* **Judul**: *Webcam-based eye tracking to detect mind wandering* (Behavior Research Methods, 2024).
* **Tujuan**: Memvalidasi apakah pelacakan mata webcam murah (menggunakan WebGazer) cukup andal untuk mendeteksi:
  1. **Task-Unrelated Thought (TUT / Mind Wandering)**: Kondisi pikiran melamun / teralihkan dari bacaan.
  2. **Reading Comprehension**: Kemampuan pembaca menarik kesimpulan yang benar (*correct inference*).
* **Temuan Utama**:
  * WebGazer memiliki akurasi ~4.17° (lebih rendah dibanding Tobii lab <1°), serta mengalami *drift* (geseran kalibrasi).
  * Kendati demikian, deteksi TUT dan pemahaman bacaan **tetap berhasil melampaui performa baseline (chance & reading time)**.
  * **Prediksi TUT**: Fitur *Global* (terutama jumlah titik gaze / `num_samples`) menjadi prediktor terkuat. Partisipan yang melamun menunjukkan penurunan aktivitas fiksasi aktif. Model terbaik: **XGBoost (Kappa = 0.15, F1-TUT = 0.25)**.
  * **Prediksi Comprehension (Correctness)**: Fitur *Local* (AOI opsi) jauh lebih dominan (**Kappa = 0.57**). Pembaca yang menjawab benar cenderung membagi pandangannya secara merata ke seluruh opsi jawaban untuk mempertimbangkan plausibilitasnya.

---

## 3. Arsitektur Sistem Eksperimen

```
┌────────────────────────────────────────────────────────┐
│               FRONTEND (Browser Klien)                 │
│  - WebGazer.js (FaceMesh DL + Ridge ML + Kalman)       │
│  - Stimulus: 65 Cerita Naratif (3 Kalimat per cerita)  │
│  - Opsi: 3 Lingkaran (20% H) & AOI Margin (30% H)      │
│  - 30 TUT Probes Acak (Kalimat 1, 2, atau 3)           │
│  - Choice Screen (Jawaban Benar/Salah)                 │
└──────────────────────────┬─────────────────────────────┘
                           │ POST /api/save-trial (JSON)
                           ▼
┌────────────────────────────────────────────────────────┐
│             BACKEND (FastAPI / Node.js)                │
│  - Menerima raw log per partisipan                     │
│  - Menyimpan ke: data/raw_logs/{participant_id}.json   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│        DATA PROCESSING & ML PIPELINE (Python)          │
│  1. 01_extract_features.py                             │
│     -> Ekstraksi Global, Local, & Baseline Features    │
│  2. 02_train_models.py                                 │
│     -> Participant-level GroupKFold (10-Fold)          │
│     -> StandardScaler (hanya fit pada Train)           │
│     -> SMOTE Imbalance Handling (hanya pada Train)     │
│     -> Classifier: XGBoost, GradientBoost, LogReg      │
│     -> Metrik: Cohen's Kappa, F1-TUT, Weighted-F1      │
│     -> Analisis Kontribusi Fitur: SHAP Values          │
└────────────────────────────────────────────────────────┘
```

---

## 4. Desain Stimulus & Definisi Area of Interest (AOI)

### 4.1 Paradigma Narrative Anticipation Task
* Terdiri dari **65 cerita pendek** (adaptasi Cranford & Moss, 2018).
* Setiap cerita tersusun dari **3 kalimat berurutan**.
* Terdapat **3 opsi kesimpulan akhir** yang ditampilkan sejak awal percobaan.
  * Awalnya ketiga opsi tampak mungkin (*plausible*).
  * Pada kalimat ke-2 dan ke-3, hanya tersisa **1 opsi yang logis dan benar**.
* **TUT Probe**: Disisipkan pada **30 cerita** secara acak:
  * 10 probe setelah kalimat 1.
  * 10 probe setelah kalimat 2.
  * 10 probe setelah kalimat 3.
  * Menanyakan: *"Apakah pikiran Anda teralihkan dari tugas saat ini?"* (Opsi: Ya [Off-Task] / Tidak [On-Task]).

### 4.2 Layout & Toleransi AOI
Untuk mengatasi *webcam drift* dan pergeseran posisi kepala:
* **Ukuran Stimulus Visual**: Tiga opsi diletakkan dalam lingkaran berdiameter **20% tinggi layar (`0.20 * window.innerHeight`)**.
* **Ukuran Deteksi AOI**: Diberi margin toleransi dengan diameter **30% tinggi layar (`0.30 * window.innerHeight`)**.
* **AOI Tengah (Sentence / Center)**: Bounding box area teks bacaan di tengah layar.

---

## 5. Ekstraksi Fitur (Feature Engineering)

Data gaze diagregasi pada jendela bacaan **1 layar persis sebelum TUT probe muncul** (untuk model TUT) atau akumulasi 3 kalimat (untuk model *Comprehension*).

### 5.1 Fitur Global (Karakteristik Umum Gerakan Mata)
1. **`num_samples`**: Total titik koordinat valid yang tercatat selama membaca layar tersebut.
2. **`num_unique_samples`**: Jumlah titik koordinat unik $(x, y)$ setelah dibulatkan ke grid pixel terdekat (mengukur eksplorasi area layar).
3. **`dispersion` (RMS Dispersion)**:
   $$RMS = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left((x_i - \bar{x})^2 + (y_i - \bar{y})^2\right)}$$
   Mengukur seberapa luas pandangan mata menyebar dari titik pusat rata-rata.

### 5.2 Fitur Local (Konteks Terhadap Konten / AOI)
1. **`opt1_gaze`**: Total sampel pandangan yang jatuh pada AOI Opsi 1.
2. **`opt2_gaze`**: Total sampel pandangan yang jatuh pada AOI Opsi 2.
3. **`opt3_gaze`**: Total sampel pandangan yang jatuh pada AOI Opsi 3.
4. **`all_options_gaze`**: Penjumlahan total waktu memandang ketiga opsi (`opt1 + opt2 + opt3`).
5. **`sentence_center_gaze`**: Total sampel yang berada di area kalimat/tengah layar.
6. **`correct_answer_gaze`**: Total sampel pada opsi yang berstatus jawaban benar.
7. **`incorrect_answer_gaze`**: Total sampel pada opsi-opsi yang salah.

### 5.3 Baseline Feature
* **`reading_time`**: Durasi waktu membaca halaman tersebut dalam detik. Digunakan untuk melatih model baseline Logistic Regression.

---

## 6. Pipeline Machine Learning & Validasi

Sesuai metodologi ketat pada jurnal:

1. **Participant-Level 10-Fold Cross-Validation (`GroupKFold`)**:
   Data dari seorang partisipan **tidak boleh** tersebar di data latih (*train*) dan data uji (*test*) pada *fold* yang sama, untuk mencegah kebocoran identitas personal.
2. **Normalisasi Z-Score (`StandardScaler`)**:
   Fitur distandarkan ($z = \frac{x - \mu}{\sigma}$). Parameter mean & std **hanya dihitung dari data train**, lalu diaplikasikan ke data test.
3. **Penanganan Ketidakseimbangan Kelas (`SMOTE`)**:
   Karena kasus TUT hanya ~20–30% (minoritas), algoritma SMOTE digunakan untuk menyintesis sampel minoritas hingga seimbang. SMOTE **hanya boleh diaplikasikan pada data train**! Data uji harus tetap mempertahankan rasio kelas asli.
4. **Metrik Evaluasi**:
   * **Cohen's Kappa ($\kappa$)**: Koreksi terhadap faktor tebakan acak (*chance-corrected*). $\kappa > 0$ menandakan model lebih baik dari tebakan acak.
   * **F1-Score per Kelas**: Fokus pada $F_1$ kelas TUT (minoritas).
   * **Weighted F1, Precision, dan Recall**.
5. **Analisis Kepentingan Fitur (SHAP Values)**:
   Menggunakan `shap.TreeExplainer` untuk menghitung rata-rata nilai absolut SHAP per fitur.

---

## 7. Panduan Setup Projek dari Nol (Step-by-Step)

### Langkah 1: Persiapan Direktori & Lingkungan Virtual
Buka terminal PowerShell:

```powershell
# 1. Buat struktur folder projek
mkdir d:\PROJEK\mind-wandering-gaze
cd d:\PROJEK\mind-wandering-gaze
mkdir frontend frontend\js frontend\css backend analysis data data\raw_logs data\processed

# 2. Buat & aktifkan Virtual Environment Python
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Instal pustaka yang dibutuhkan
pip install fastapi uvicorn pydantic pandas numpy scikit-learn imbalanced-learn xgboost shap
```

### Langkah 2: Salin File WebGazer.js
Ambil file kompilasi WebGazer dari repositori lokal Anda:

```powershell
Copy-Item "d:\PROJEK\WebGazer\dist\webgazer.js" "frontend\js\webgazer.js"
```

### Langkah 3: Buat Server Backend Pengumpul Data (`backend/app.py`)
Simpan kode berikut di `backend/app.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.abspath("../data/raw_logs")
os.makedirs(DATA_DIR, exist_ok=True)

class SessionPayload(BaseModel):
    participant_id: str
    trials: list

@app.post("/api/save-trial")
def save_trial(payload: SessionPayload):
    filename = f"{payload.participant_id}_{int(time.time())}.json"
    target_path = os.path.join(DATA_DIR, filename)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(payload.dict(), f, indent=2)
    return {"status": "success", "file": filename}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

### Langkah 4: Buat Frontend Tugas Eksperimen

#### 1. File Cerita (`frontend/js/stories.js`)
```javascript
const STORIES_DATA = [
    {
        id: 1,
        sentences: [
            "Larry always wanted to know what it was like to live in a foreign country.",
            "He went to read at his favorite store on main street.",
            "The steam rose from the cup as Larry brought it to his lips and slowly..."
        ],
        options: [
            { id: "opt1", text: "sipped coffee", is_correct: true },
            { id: "opt2", text: "bought muffin", is_correct: false },
            { id: "opt3", text: "rolled marble", is_correct: false }
        ],
        probe_sentence: 2 // TUT Probe muncul setelah kalimat ke-2 (1, 2, atau 3)
    }
];
```

#### 2. File Styling (`frontend/css/style.css`)
```css
* { box-sizing: border-box; margin: 0; padding: 0; font-family: sans-serif; }
body { background: #f8fafc; height: 100vh; overflow: hidden; display: flex; flex-direction: column; }

#sentence-box {
    position: absolute; top: 35%; left: 50%; transform: translate(-50%, -50%);
    width: 65%; font-size: 1.4rem; line-height: 1.8; text-align: center;
    padding: 24px; background: #ffffff; border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}

.options-container {
    position: absolute; bottom: 10%; width: 100%;
    display: flex; justify-content: space-around; padding: 0 5%;
}

/* Lingkaran Opsi berdiameter 20% tinggi layar (20vh) */
.option-circle {
    width: 20vh; height: 20vh; border-radius: 50%;
    background: #ffffff; border: 3px solid #64748b;
    display: flex; align-items: center; justify-content: center;
    text-align: center; padding: 12px; font-weight: 600; cursor: pointer;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

#probe-modal {
    display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(15, 23, 42, 0.7); justify-content: center; align-items: center; z-index: 999;
}
.probe-card { background: white; padding: 32px; border-radius: 16px; text-align: center; max-width: 480px; }
.btn { margin: 10px; padding: 12px 24px; border: none; border-radius: 8px; font-size: 1rem; cursor: pointer; }
.btn-yes { background: #ef4444; color: white; }
.btn-no { background: #10b981; color: white; }
```

#### 3. Logika Eksperimen & AOI (`frontend/js/app.js`)
```javascript
let participantId = "P_" + Math.floor(Math.random() * 100000);
let currentStoryIdx = 0;
let currentSentenceIdx = 0;
let trialLogs = [];
let currentTrialData = null;
let currentGazeBuffer = [];
let isCalibrated = false;

window.onload = async function() {
    await webgazer.setRegression('ridge')
        .setGazeListener((data, elapsedTime) => {
            if (!data || !isCalibrated) return;
            recordGaze(data.x, data.y, elapsedTime);
        })
        .saveDataAcrossSessions(false)
        .begin();

    webgazer.showVideoPreview(true)
            .showPredictionPoints(false)
            .applyKalmanFilter(true);
};

function recordGaze(x, y, timestamp) {
    const H = window.innerHeight;
    const aoiRadius = (0.30 * H) / 2; // Radius deteksi AOI 30% H

    let aoiHit = "outside";
    ['opt1', 'opt2', 'opt3'].forEach(optId => {
        const el = document.getElementById(optId);
        if (el) {
            const rect = el.getBoundingClientRect();
            const cx = rect.left + rect.width / 2;
            const cy = rect.top + rect.height / 2;
            if (Math.hypot(x - cx, y - cy) <= aoiRadius) aoiHit = optId;
        }
    });

    const textBox = document.getElementById("sentence-box");
    if (textBox && aoiHit === "outside") {
        const rect = textBox.getBoundingClientRect();
        if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
            aoiHit = "sentence_center";
        }
    }

    currentGazeBuffer.push({ x: Math.round(x), y: Math.round(y), aoi: aoiHit, t: timestamp });
}

function startExperiment() {
    isCalibrated = true;
    document.getElementById("calibration-guide").style.display = "none";
    document.getElementById("experiment-ui").style.display = "block";
    loadStory(0);
}

function loadStory(idx) {
    currentStoryIdx = idx;
    currentSentenceIdx = 0;
    currentGazeBuffer = [];

    const story = STORIES_DATA[currentStoryIdx];
    currentTrialData = {
        participant_id: participantId,
        story_id: story.id,
        sentences: [],
        probe_response: null,
        final_choice: null,
        is_correct: null
    };
    renderScreen();
}

function renderScreen() {
    const story = STORIES_DATA[currentStoryIdx];
    document.getElementById("sentence-box").innerText = story.sentences[currentSentenceIdx];
    document.getElementById("opt1").innerText = story.options[0].text;
    document.getElementById("opt2").innerText = story.options[1].text;
    document.getElementById("opt3").innerText = story.options[2].text;
    currentTrialData.start_time = Date.now();
}

function nextStep() {
    const readingTime = (Date.now() - currentTrialData.start_time) / 1000;
    currentTrialData.sentences.push({
        sentence_index: currentSentenceIdx,
        reading_time_sec: readingTime,
        gaze_points: [...currentGazeBuffer]
    });
    currentGazeBuffer = [];

    const story = STORIES_DATA[currentStoryIdx];
    if (story.probe_sentence === currentSentenceIdx + 1) {
        document.getElementById("probe-modal").style.display = "flex";
        return;
    }
    advanceSentence();
}

function handleProbe(isTUT) {
    currentTrialData.probe_response = isTUT ? 1 : 0;
    document.getElementById("probe-modal").style.display = "none";
    advanceSentence();
}

function advanceSentence() {
    currentSentenceIdx++;
    if (currentSentenceIdx < 3) {
        renderScreen();
    } else {
        alert("Pilih kesimpulan akhir dari 3 opsi lingkaran di bawah!");
    }
}

function selectAnswer(optId) {
    if (currentSentenceIdx < 3) return;
    const story = STORIES_DATA[currentStoryIdx];
    const chosen = story.options.find(o => o.id === optId);
    currentTrialData.final_choice = optId;
    currentTrialData.is_correct = chosen.is_correct ? 1 : 0;

    trialLogs.push(currentTrialData);
    currentStoryIdx++;

    if (currentStoryIdx < STORIES_DATA.length) {
        loadStory(currentStoryIdx);
    } else {
        finishSession();
    }
}

async function finishSession() {
    alert("Selesai! Menyimpan data...");
    await fetch("http://127.0.0.1:8000/api/save-trial", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ participant_id: participantId, trials: trialLogs })
    });
    alert("Data berhasil tersimpan di server!");
}
```

#### 4. File HTML Utama (`frontend/index.html`)
```html
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>Eksperimen Mind Wandering Gaze</title>
    <link rel="stylesheet" href="css/style.css">
    <script src="js/webgazer.js"></script>
    <script src="js/stories.js"></script>
    <script src="js/app.js"></script>
</head>
<body>
    <div id="calibration-guide" style="text-align: center; margin-top: 20vh;">
        <h2>Kalibrasi Kamera</h2>
        <p>Lihat kamera dan klik beberapa area di layar untuk melatih WebGazer.</p>
        <button class="btn btn-no" onclick="startExperiment()">Mulai Eksperimen</button>
    </div>

    <div id="experiment-ui" style="display: none;">
        <div id="sentence-box">Teks narasi akan muncul di sini...</div>
        <button class="btn" style="position: absolute; top: 55%; left: 50%; transform: translateX(-50%);" onclick="nextStep()">Lanjut Kalimat</button>

        <div class="options-container">
            <div id="opt1" class="option-circle" onclick="selectAnswer('opt1')">Opsi 1</div>
            <div id="opt2" class="option-circle" onclick="selectAnswer('opt2')">Opsi 2</div>
            <div id="opt3" class="option-circle" onclick="selectAnswer('opt3')">Opsi 3</div>
        </div>
    </div>

    <div id="probe-modal">
        <div class="probe-card">
            <h3>Pertanyaan Evaluasi</h3>
            <p>Apakah saat membaca barusan pikiran Anda teralihkan (Mind Wandering / Off-Task)?</p>
            <button class="btn btn-yes" onclick="handleProbe(true)">Ya (Off-Task)</button>
            <button class="btn btn-no" onclick="handleProbe(false)">Tidak (On-Task)</button>
        </div>
    </div>
</body>
</html>
```

### Langkah 5: Skrip Ekstraksi Fitur (`analysis/01_extract_features.py`)
```python
import os, json, glob
import numpy as np
import pandas as pd

RAW_DIR = "../data/raw_logs"
OUTPUT_FILE = "../data/processed/features.csv"
os.makedirs("../data/processed", exist_ok=True)

rows = []
for filepath in glob.glob(os.path.join(RAW_DIR, "*.json")):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    p_id = data["participant_id"]
    for trial in data["trials"]:
        if trial.get("probe_response") is None:
            continue

        target_sentence = trial["sentences"][-1]
        gaze_pts = target_sentence["gaze_points"]
        reading_time = target_sentence["reading_time_sec"]

        if len(gaze_pts) < 5:
            continue

        xs = np.array([p["x"] for p in gaze_pts])
        ys = np.array([p["y"] for p in gaze_pts])
        aois = [p["aoi"] for p in gaze_pts]

        num_samples = len(gaze_pts)
        num_unique = len(set(zip(np.round(xs / 20), np.round(ys / 20))))
        mx, my = np.mean(xs), np.mean(ys)
        dispersion = float(np.sqrt(np.mean((xs - mx)**2 + (ys - my)**2)))

        opt1 = sum(1 for a in aois if a == "opt1")
        opt2 = sum(1 for a in aois if a == "opt2")
        opt3 = sum(1 for a in aois if a == "opt3")
        all_opts = opt1 + opt2 + opt3
        sentence = sum(1 for a in aois if a == "sentence_center")

        rows.append({
            "participant_id": p_id,
            "story_id": trial["story_id"],
            "reading_time": reading_time,
            "num_samples": num_samples,
            "num_unique": num_unique,
            "dispersion": dispersion,
            "opt1": opt1, "opt2": opt2, "opt3": opt3,
            "all_opts": all_opts, "sentence": sentence,
            "label_tut": trial["probe_response"],
            "label_correct": trial.get("is_correct", 1)
        })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Sukses! {len(df)} data berhasil diekstrak ke {OUTPUT_FILE}")
```

### Langkah 6: Skrip Pelatihan Model ML (`analysis/02_train_models.py`)
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

df = pd.read_csv("../data/processed/features.csv")

features = ['num_samples', 'num_unique', 'dispersion', 'opt1', 'opt2', 'opt3', 'all_opts', 'sentence']
X = df[features]
y = df['label_tut']
groups = df['participant_id']

n_splits = min(10, df['participant_id'].nunique())
gkf = GroupKFold(n_splits=n_splits)

y_true_all, y_pred_all = [], []

for train_idx, test_idx in gkf.split(X, y, groups=groups):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Standardize (Hanya Train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE (Hanya Train)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # Model XGBoost persis paper
    clf = XGBClassifier(n_estimators=100, max_depth=3, eval_metric='logloss', random_state=42)
    clf.fit(X_train_res, y_train_res)

    preds = clf.predict(X_test_scaled)
    y_true_all.extend(y_test)
    y_pred_all.extend(preds)

print("=== HASIL EVALUASI PREDIKSI MIND WANDERING (TUT) ===")
print(f"Cohen's Kappa : {cohen_kappa_score(y_true_all, y_pred_all):.3f}")
print(f"Weighted F1   : {f1_score(y_true_all, y_pred_all, average='weighted'):.3f}")
print(f"F1 (TUT Class): {f1_score(y_true_all, y_pred_all, pos_label=1):.3f}")
print(f"Precision TUT : {precision_score(y_true_all, y_pred_all, pos_label=1):.3f}")
print(f"Recall TUT    : {recall_score(y_true_all, y_pred_all, pos_label=1):.3f}")
```

---

## 8. Petunjuk Eksekusi & Best Practices

### Cara Menjalankan Sistem
1. **Terminal 1 (Backend)**:
   ```powershell
   cd d:\PROJEK\mind-wandering-gaze\backend
   python app.py
   ```
2. **Terminal 2 (Frontend)**:
   ```powershell
   cd d:\PROJEK\mind-wandering-gaze\frontend
   python -m http.server 3000
   ```
3. Buka browser di `http://localhost:3000`, izinkan webcam, lakukan kalibrasi, dan selesaikan tugas eksperimen.
4. **Terminal 3 (Analisis Data)**:
   ```powershell
   cd d:\PROJEK\mind-wandering-gaze\analysis
   python 01_extract_features.py
   python 02_train_models.py
   ```

### 3 Aturan Emas Riset Jurnal:
1. **Jangan Ada Data Leakage**: SMOTE dan `StandardScaler.fit()` **wajib** berada di dalam loop *fold*. Jangan pernah jalankan SMOTE pada seluruh dataset sebelum split!
2. **Gunakan Margin AOI 30% H**: Jitter dan drift pada webcam pasti terjadi. Margin toleransi 30% adalah kunci keberhasilan ekstraksi fitur lokal.
3. **Fokus pada Pola Relatif, Bukan Fiksasi Kata**: Karena sampling rate webcam dinamis (bergantung spek CPU laptop klien), jangan mencoba menghitung mikro-sakad per kata, melainkan gunakan dispersi RMS dan total waktu di AOI.
