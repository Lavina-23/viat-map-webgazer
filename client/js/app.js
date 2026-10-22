let participantId = "P_" + Math.floor(Math.random() * 100000);
let currentStoryIdx = 0;
let currentSentenceIdx = 0;
let trialLogs = [];
let currentTrialData = null;
let currentGazeBuffer = [];
let isCalibrated = false;

// Status Kalibrasi 9 Titik
const CAL_TARGET = 5;
const calProgress = { Pt1: 0, Pt2: 0, Pt3: 0, Pt4: 0, Pt5: 0, Pt6: 0, Pt7: 0, Pt8: 0, Pt9: 0 };
let calibratedPointsCount = 0;

window.onload = async function () {
    const statusEl = document.getElementById("camera-status");
    try {
        if (!window.isSecureContext && window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
            throw new Error(`Akses via IP (${window.location.hostname}) membutuhkan izin kamera browser. Tambahkan <code>http://${window.location.host}</code> ke <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code> atau gunakan HTTPS.`);
        }
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("Browser memblokir API Kamera pada HTTP non-localhost. Gunakan flag Chrome/Edge atau HTTPS.");
        }

        await webgazer.setRegression('ridge')
            .setGazeListener((data, elapsedTime) => {
                if (!data || !isCalibrated) return;
                recordGaze(data.x, data.y, elapsedTime);
            })
            .saveDataAcrossSessions(false)
            .begin();

        // Tampilkan preview video kamera dan titik merah saat fase kalibrasi
        webgazer.showVideoPreview(true)
            .showPredictionPoints(true)
            .applyKalmanFilter(true);

        statusEl.innerHTML = "🟢 <strong>Webcam Aktif & Terhubung!</strong> Silakan klik titik kalibrasi.";
        statusEl.style.background = "#dcfce7";
        statusEl.style.color = "#166534";
    } catch (err) {
        console.error("Gagal memulai WebGazer:", err);
        statusEl.innerHTML = `⚠️ <strong>Akses Kamera Ditolak / Dibatasi:</strong> ${err.message}<br><small>Jika menggunakan perangkat lain / IP, pastikan kamera diizinkan atau tambahkan URL ke insecure origin flags.</small>`;
        statusEl.style.background = "#fee2e2";
        statusEl.style.color = "#991b1b";
    }
};

function onCalPointClick(id) {
    if (calProgress[id] >= CAL_TARGET) return;
    calProgress[id]++;
    const el = document.getElementById(id);
    const remaining = CAL_TARGET - calProgress[id];

    if (remaining > 0) {
        el.innerText = remaining;
    } else {
        el.innerText = "✓";
        el.classList.add("calibrated");
        calibratedPointsCount++;
        document.getElementById("calibrated-count").innerText = calibratedPointsCount;

        // Buka tombol Mulai Eksperimen jika sudah minimal 5 titik terkalibrasi
        if (calibratedPointsCount >= 5) {
            const btn = document.getElementById("btn-start");
            btn.removeAttribute("disabled");
            btn.style.opacity = "1";
            btn.style.cursor = "pointer";
            btn.innerText = `Mulai Eksperimen (${calibratedPointsCount}/9 Siap)`;
        }
    }
}

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
    document.getElementById("calibration-points").style.display = "none";

    // Sembunyikan titik merah prediksi saat eksperimen membaca berlangsung agar tidak mengganggu fokus
    if (typeof webgazer !== 'undefined') {
        webgazer.showPredictionPoints(false);
    }

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
    const experimentUi = document.getElementById("experiment-ui");
    experimentUi.innerHTML = `
        <div style="text-align: center; margin-top: 25vh;">
            <h2 style="font-size: 2rem; color: #1e293b; margin-bottom: 12px;">Menyimpan Data Eksperimen...</h2>
            <p style="color: #64748b;">Mohon tunggu sebentar, data gaze sedang dikirim ke backend server.</p>
        </div>
    `;

    try {
        const backendHost = window.location.hostname || "127.0.0.1";
        const res = await fetch(`http://${backendHost}:8000/api/save-trial`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ participant_id: participantId, trials: trialLogs })
        });

        if (!res.ok) throw new Error("HTTP error " + res.status);
        const result = await res.json();

        experimentUi.innerHTML = `
            <div style="text-align: center; margin-top: 25vh;">
                <h2 style="font-size: 2.2rem; color: #166534; margin-bottom: 12px;">✅ Eksperimen Selesai!</h2>
                <p style="color: #334155; font-size: 1.1rem; margin-bottom: 8px;">ID Partisipan: <strong>${participantId}</strong></p>
                <p style="color: #64748b; margin-bottom: 24px;">Data tersimpan: <code>${result.file}</code></p>
                <p style="color: #475569;">Data log siap diolah pada tahap ekstraksi fitur di <code>analysis/</code>.</p>
            </div>
        `;
    } catch (err) {
        console.error("Gagal mengirim data:", err);
        experimentUi.innerHTML = `
            <div style="text-align: center; margin-top: 25vh;">
                <h2 style="font-size: 2rem; color: #991b1b; margin-bottom: 12px;">⚠️ Gagal Menyimpan ke Server</h2>
                <p style="color: #64748b; margin-bottom: 16px;">Pastikan backend server FastAPI (port 8000) aktif.</p>
                <p style="color: #ef4444; margin-bottom: 24px;">Detail: ${err.message}</p>
                <button class="btn btn-no" onclick="location.reload()">Coba Lagi</button>
            </div>
        `;
    }
}