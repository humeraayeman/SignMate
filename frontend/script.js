// SignMate frontend

// ------------------------------------------------------------
// State
// ------------------------------------------------------------
let currentUser = null;
let currentMode = 'home';

// Camera & Sign Recognition State
let cameraStream = null;
let cameraInterval = null;
let isCameraActive = false;
let isProcessingFrame = false;
let cameraMirrored = true;
let currentDetectedLetter = '';
let currentDetectedWord = '';
let currentConfidence = 0;
let currentHandCount = 0;
let autoAddEnabled = false;
let stableLetter = '';
let stableCount = 0;
const STABLE_THRESHOLD = 3; // ~1.2s at 400ms interval

// Speech Recognition State
let speechRecognizer = null;
let isListening = false;
let speechTranscript = '';
let speechSlideItems = [];
let speechSlideIndex = 0;
let speechSlideTimer = null;
let speechIsPlaying = false;
let speechSpeedMs = 1000;
let speechViewMode = 'slideshow';
let speechUseNaturalWords = true;

// Text to Sign State
let textSlideItems = [];
let textSlideIndex = 0;
let textSlideTimer = null;
let textIsPlaying = false;
let textSpeedMs = 1000;
let textViewMode = 'slideshow';
let textUseNaturalWords = true;

// Dictionary State
let islAlphabet = [];
let islWords = [];
let currentDictMainTab = 'alphabet'; // 'alphabet' | 'words'
let currentDictAlphaFilter = 'all';  // 'all' | 'One-Hand' | 'Two-Hand'
let currentDictWordFilter = 'all';   // 'all' | 'Greetings' | 'Polite' | etc.
let currentDictQuery = '';
let activeModalItem = null;

// Word lookup (filled if we load word signs)
const ISL_WORDS_MAP = {};

// History State
let fullHistoryList = [];
let currentHistoryFilter = 'all';

// Audio Context for feedback
let audioCtx = null;

// DOM Helper
function $(id) {
    return document.getElementById(id);
}

// ------------------------------------------------------------
// boot
// ------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    checkApiStatus();
    initSpeechRecognition();
    initAudioContext();
    initUserSession();
    fetchDictionaryData();

    // Attach text counter listeners
    if ($('composedText')) {
        $('composedText').addEventListener('input', updateCharCount);
    }
    if ($('textToSignInput')) {
        $('textToSignInput').addEventListener('input', updateWordCount);
    }
});

function initAudioContext() {
    try {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (AudioContext) {
            audioCtx = new AudioContext();
        }
    } catch (e) {
        console.warn('AudioContext not available:', e);
    }
}

function playBeep(freq = 600, duration = 0.08) {
    if (!audioCtx) return;
    try {
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
        gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
        // Ignore audio playback errors
    }
}

// ------------------------------------------------------------
// toasts
// ------------------------------------------------------------
function showToast(message, type = 'info', duration = 3200) {
    const container = $('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    let icon = `<svg class="svg-icon" viewBox="0 0 24 24" style="width: 15px; height: 15px;"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>`;
    if (type === 'success') {
        icon = `<svg class="svg-icon" viewBox="0 0 24 24" style="width: 15px; height: 15px; color: #34d399;"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
    } else if (type === 'error') {
        icon = `<svg class="svg-icon" viewBox="0 0 24 24" style="width: 15px; height: 15px; color: #f87171;"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>`;
    }
    
    toast.innerHTML = `<span>${icon}</span><span>${escapeHtml(message)}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(-10px)';
        toast.style.transition = 'all 0.25s ease';
        setTimeout(() => toast.remove(), 260);
    }, duration);
}

// ------------------------------------------------------------
// /api/status
// ------------------------------------------------------------
async function checkApiStatus() {
    const label = $('apiStatusLabel');
    const pill = $('apiStatusPill');
    try {
        const res = await fetch('/api/status');
        if (res.ok) {
            const data = await res.json();
            if (label) label.textContent = `${data.application} Online`;
            if (pill) pill.classList.remove('offline');
        } else {
            if (label) label.textContent = 'API Offline';
            if (pill) pill.classList.add('offline');
        }
    } catch (err) {
        if (label) label.textContent = 'Disconnected';
        if (pill) pill.classList.add('offline');
    }
}

// ------------------------------------------------------------
// auth
// ------------------------------------------------------------
function initUserSession() {
    const saved = localStorage.getItem('signmate_user');
    if (saved) {
        try {
            currentUser = JSON.parse(saved);
            renderUserProfile();
            showAppScreen();
            return;
        } catch (e) {
            localStorage.removeItem('signmate_user');
        }
    }
    showAuthScreen();
}

function showLoginTab() {
    $('loginTabBtn')?.classList.add('active');
    $('registerTabBtn')?.classList.remove('active');
    $('loginForm')?.classList.remove('hidden');
    $('registerForm')?.classList.add('hidden');
}

function showRegisterTab() {
    $('loginTabBtn')?.classList.remove('active');
    $('registerTabBtn')?.classList.add('active');
    $('loginForm')?.classList.add('hidden');
    $('registerForm')?.classList.remove('hidden');
}

function continueAsGuest() {
    currentUser = {
        id: 0,
        name: 'Guest Explorer',
        email: 'guest@signmate.app',
        isGuest: true
    };
    localStorage.setItem('signmate_user', JSON.stringify(currentUser));
    renderUserProfile();
    showAppScreen();
    showToast('Guest mode — history stays in this browser only.', 'info');
}

async function login() {
    const email = $('loginEmail')?.value.trim();
    const password = $('loginPassword')?.value;
    const msg = $('loginMessage');
    const btn = $('loginSubmitBtn');

    if (!email || !password) {
        if (msg) msg.textContent = 'Please provide both email and password.';
        return;
    }

    try {
        if (btn) btn.disabled = true;
        if (msg) msg.textContent = 'Signing in...';

        const res = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });
        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.detail || 'Login failed.');
        }

        currentUser = {
            id: data.user.id,
            name: data.user.name,
            email: data.user.email,
            isGuest: false
        };
        localStorage.setItem('signmate_user', JSON.stringify(currentUser));

        if (msg) msg.textContent = '';
        renderUserProfile();
        showAppScreen();
        showToast(`Hi ${currentUser.name}.`, 'success');
    } catch (err) {
        if (msg) msg.textContent = err.message || 'Invalid email or password.';
    } finally {
        if (btn) btn.disabled = false;
    }
}

async function register() {
    const name = $('registerName')?.value.trim();
    const email = $('registerEmail')?.value.trim();
    const password = $('registerPassword')?.value;
    const msg = $('registerMessage');
    const btn = $('registerSubmitBtn');

    if (!name || !email || !password) {
        if (msg) msg.textContent = 'Please fill out all fields.';
        return;
    }
    if (password.length < 6) {
        if (msg) msg.textContent = 'Password must be at least 6 characters.';
        return;
    }

    try {
        if (btn) btn.disabled = true;
        if (msg) msg.textContent = 'Creating account...';

        const res = await fetch('/api/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, email, password })
        });
        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.detail || 'Registration failed.');
        }

        if (msg) {
            msg.className = 'auth-feedback success';
            msg.textContent = 'Account created successfully! You can now sign in.';
        }
        showToast('Account created. Sign in with the same email.', 'success');

        setTimeout(() => {
            showLoginTab();
            if ($('loginEmail')) $('loginEmail').value = email;
            if ($('loginPassword')) $('loginPassword').value = '';
            if (msg) {
                msg.className = 'auth-feedback';
                msg.textContent = '';
            }
        }, 1200);
    } catch (err) {
        if (msg) {
            msg.className = 'auth-feedback';
            msg.textContent = err.message || 'Registration failed.';
        }
    } finally {
        if (btn) btn.disabled = false;
    }
}

function logout() {
    stopCamera();
    stopSpeech();
    stopSpeechSlidePlayback();
    stopTextSlidePlayback();

    currentUser = null;
    localStorage.removeItem('signmate_user');
    showAuthScreen();
    showToast('Signed out.', 'info');
}

function showAuthScreen() {
    $('authScreen')?.classList.remove('hidden');
    $('appScreen')?.classList.add('hidden');
}

function showAppScreen() {
    $('authScreen')?.classList.add('hidden');
    $('appScreen')?.classList.remove('hidden');
    switchMode('home');
}

function renderUserProfile() {
    if (!currentUser) return;
    const nameEl = $('userName');
    const avatarEl = $('userAvatar');
    const tagEl = $('userStatusTag');

    if (nameEl) nameEl.textContent = currentUser.name || 'User';
    if (avatarEl) {
        const initial = (currentUser.name || 'U').charAt(0).toUpperCase();
        avatarEl.textContent = initial;
    }
    if (tagEl) {
        tagEl.textContent = currentUser.isGuest ? 'Guest Mode' : 'Verified';
        tagEl.style.color = currentUser.isGuest ? 'var(--warning)' : 'var(--success)';
    }
}

// ------------------------------------------------------------
// tabs
// ------------------------------------------------------------
function switchMode(mode) {
    currentMode = mode;

    const allModes = ['home', 'sign', 'speech', 'text', 'dict', 'history'];
    allModes.forEach(m => {
        const view = $(`${m}Mode`);
        const tabBtn = $(`${m}ModeBtn`);
        if (view) {
            if (m === mode) {
                view.classList.remove('hidden');
            } else {
                view.classList.add('hidden');
            }
        }
        if (tabBtn) {
            if (m === mode) {
                tabBtn.classList.add('active');
            } else {
                tabBtn.classList.remove('active');
            }
        }
    });

    // Lifecycle triggers
    if (mode === 'dict') {
        renderDictionary();
    } else if (mode === 'history') {
        loadHistory();
    } else if (mode !== 'sign' && isCameraActive) {
        stopCamera();
    } else if (mode !== 'speech' && isListening) {
        stopSpeech();
    }
}

// ------------------------------------------------------------
// sign → text
// ------------------------------------------------------------
async function startCamera() {
    if (isCameraActive) return;

    const video = $('camera');
    const overlay = $('cameraOverlay');
    const reticle = $('cameraTargetReticle');
    const startBtn = $('startCameraBtn');
    const stopBtn = $('stopCameraBtn');
    const statusPill = $('cameraStatusPill');

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showToast('Webcam access not supported in this browser.', 'error');
        return;
    }

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });

        if (video) {
            video.srcObject = cameraStream;
            await video.play();
        }

        isCameraActive = true;
        if (overlay) overlay.classList.add('hidden');
        if (reticle) reticle.classList.remove('hidden');
        if (startBtn) startBtn.disabled = true;
        if (stopBtn) stopBtn.disabled = false;
        if (statusPill) {
            statusPill.textContent = 'Camera Active';
            statusPill.className = 'badge badge-success';
        }

        showToast('Camera on. Put your hands in the box.', 'info');

        // Processing loop: check frame every 400ms
        cameraInterval = setInterval(captureAndPredict, 400);

    } catch (err) {
        console.error('Webcam activation error:', err);
        showToast('Unable to open camera. Check permissions in browser settings.', 'error');
        if (statusPill) {
            statusPill.textContent = 'Camera Error';
            statusPill.className = 'badge badge-outline';
        }
    }
}

function stopCamera() {
    if (cameraInterval) {
        clearInterval(cameraInterval);
        cameraInterval = null;
    }

    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }

    const video = $('camera');
    const overlay = $('cameraOverlay');
    const reticle = $('cameraTargetReticle');
    const startBtn = $('startCameraBtn');
    const stopBtn = $('stopCameraBtn');
    const statusPill = $('cameraStatusPill');

    if (video) video.srcObject = null;
    isCameraActive = false;
    isProcessingFrame = false;

    if (overlay) overlay.classList.remove('hidden');
    if (reticle) reticle.classList.add('hidden');
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    if (statusPill) {
        statusPill.textContent = 'Camera Idle';
        statusPill.className = 'badge badge-neutral';
    }

    resetPredictionUI();
}

function toggleCameraMirror() {
    const video = $('camera');
    const toggle = $('mirrorToggle');
    if (!video || !toggle) return;
    cameraMirrored = toggle.checked;
    if (cameraMirrored) {
        video.classList.add('mirrored');
    } else {
        video.classList.remove('mirrored');
    }
}

function toggleAutoAdd() {
    const toggle = $('autoAddToggle');
    const bar = $('holdProgressBar');
    autoAddEnabled = toggle ? toggle.checked : false;
    if (bar) {
        if (autoAddEnabled) {
            bar.classList.remove('hidden');
        } else {
            bar.classList.add('hidden');
        }
    }
    stableLetter = '';
    stableCount = 0;
    updateHoldProgressBar(0);
}

function updateHoldProgressBar(percent) {
    const fill = $('holdProgressFill');
    if (fill) {
        fill.style.width = `${Math.min(100, Math.max(0, percent))}%`;
    }
}

async function captureAndPredict() {
    const video = $('camera');
    if (!video || !isCameraActive || isProcessingFrame || video.readyState < 2) {
        return;
    }

    isProcessingFrame = true;

    try {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(async blob => {
            if (!blob) {
                isProcessingFrame = false;
                return;
            }

            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');

            try {
                const res = await fetch('/api/translate/sign', {
                    method: 'POST',
                    body: formData
                });

                if (res.ok) {
                    const data = await res.json();
                    handlePredictionResult(data);
                }
            } catch (netErr) {
                // Ignore transient frame upload errors
            } finally {
                isProcessingFrame = false;
            }
        }, 'image/jpeg', 0.85);

    } catch (e) {
        isProcessingFrame = false;
    }
}

function handlePredictionResult(data) {
    const letter = data.label ? String(data.label).toUpperCase() : '';
    const confidence = Number(data.confidence || 0);
    const hands = Number(data.hands || 0);
    const success = Boolean(data.success && confidence >= 0.35);

    currentDetectedLetter = success ? letter : '';
    currentConfidence = confidence;
    currentHandCount = hands;

    const letterEl = $('liveLetter');
    const confLabel = $('confidenceLabel');
    const confBar = $('confidenceBar');
    const handBadge = $('handCountBadge');
    const statusBadge = $('translationStateBadge');
    const hint = $('modelFeedback');

    if (letterEl) {
        if (success && letter) {
            if (letterEl.textContent !== letter) {
                letterEl.textContent = letter;
                letterEl.style.transform = 'scale(1.15)';
                setTimeout(() => { letterEl.style.transform = 'scale(1)'; }, 150);
            }
        } else {
            letterEl.textContent = '—';
        }
    }

    const pct = Math.round(confidence * 100);
    if (confLabel) confLabel.textContent = `${pct}%`;
    if (confBar) {
        confBar.style.width = `${pct}%`;
        confBar.className = 'meter-bar-fill';
        if (pct >= 70) confBar.classList.add('high');
        else if (pct >= 45) confBar.classList.add('medium');
        else confBar.classList.add('low');
    }

    if (handBadge) {
        handBadge.textContent = hands === 1 ? '1 hand' : `${hands} hands`;
    }

    if (statusBadge) {
        if (success && pct >= 70) {
            statusBadge.textContent = 'High confidence';
            statusBadge.className = 'badge badge-success';
        } else if (success) {
            statusBadge.textContent = 'Recognized';
            statusBadge.className = 'badge badge-primary';
        } else if (hands > 0) {
            statusBadge.textContent = 'Analyzing posture...';
            statusBadge.className = 'badge badge-neutral';
        } else {
            statusBadge.textContent = 'No hands visible';
            statusBadge.className = 'badge badge-outline';
        }
    }

    if (hint) {
        if (hands === 0) {
            hint.textContent = 'Bring your hands into the frame.';
        } else if (!success) {
            hint.textContent = 'Adjust angle or form the sign more clearly.';
        } else {
            hint.textContent = data.message || `Signing letter '${letter}'`;
        }
    }

    currentDetectedWord = '';

    // Auto-Add on hold logic
    if (autoAddEnabled && success && letter) {
        if (letter === stableLetter) {
            stableCount++;
            const progress = (stableCount / STABLE_THRESHOLD) * 100;
            updateHoldProgressBar(progress);

            if (stableCount >= STABLE_THRESHOLD) {
                addCurrentLetter();
                stableCount = 0;
                updateHoldProgressBar(0);
                showToast(`Auto-added '${letter}'`, 'success', 1500);
            }
        } else {
            stableLetter = letter;
            stableCount = 1;
            updateHoldProgressBar((1 / STABLE_THRESHOLD) * 100);
        }
    } else if (autoAddEnabled) {
        stableCount = 0;
        updateHoldProgressBar(0);
    }
}

function resetPredictionUI() {
    currentDetectedLetter = '';
    currentDetectedWord = '';
    currentConfidence = 0;
    currentHandCount = 0;

    if ($('liveLetter')) $('liveLetter').textContent = '—';
    if ($('confidenceLabel')) $('confidenceLabel').textContent = '0%';
    if ($('confidenceBar')) $('confidenceBar').style.width = '0%';
    if ($('handCountBadge')) $('handCountBadge').textContent = '0 hands';
    if ($('translationStateBadge')) {
        $('translationStateBadge').textContent = 'Waiting for hands';
        $('translationStateBadge').className = 'badge badge-neutral';
    }
    if ($('modelFeedback')) $('modelFeedback').textContent = 'Show clear hand gestures to the camera.';
    updateHoldProgressBar(0);
}

// ------------------------------------------------------------
// Text builder
// ------------------------------------------------------------
function addCurrentLetter() {
    if (!currentDetectedLetter) {
        showToast('No sign currently detected to add.', 'info');
        return;
    }
    const box = $('composedText');
    if (!box) return;

    box.value += currentDetectedLetter;

    updateCharCount();
    playBeep(750, 0.05);
}


function addSpace() {
    const box = $('composedText');
    if (!box) return;
    if (box.value && !box.value.endsWith(' ')) {
        box.value += ' ';
        updateCharCount();
        playBeep(440, 0.04);
    }
}

function removeLastLetter() {
    const box = $('composedText');
    if (!box || !box.value) return;
    box.value = box.value.slice(0, -1);
    updateCharCount();
    playBeep(350, 0.04);
}

function clearComposedText() {
    const box = $('composedText');
    if (!box) return;
    box.value = '';
    updateCharCount();
}

function updateCharCount() {
    const box = $('composedText');
    const label = $('charCountLabel');
    if (box && label) {
        const count = box.value.length;
        label.textContent = `${count} character${count === 1 ? '' : 's'}`;
    }
}

function speakComposedText() {
    const box = $('composedText');
    const text = box ? box.value.trim() : '';
    if (!text) {
        showToast('Nothing to speak yet.', 'info');
        return;
    }
    speakText(text);
}

function speakText(text) {
    if (!window.speechSynthesis) {
        showToast('Speech synthesis not supported in this browser.', 'error');
        return;
    }
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 0.95;
    utter.pitch = 1.0;
    
    // Pick pleasant English voice if available
    const voices = window.speechSynthesis.getVoices();
    const englishVoice = voices.find(v => v.lang.includes('en') || v.lang.includes('IN'));
    if (englishVoice) utter.voice = englishVoice;

    window.speechSynthesis.speak(utter);
}

function copyComposedText() {
    const box = $('composedText');
    const text = box ? box.value.trim() : '';
    if (!text) {
        showToast('No composed text to copy.', 'info');
        return;
    }
    navigator.clipboard.writeText(text).then(() => {
        showToast('Copied.', 'success');
    }).catch(() => {
        showToast('Unable to copy text.', 'error');
    });
}

async function saveSignTranslation() {
    const box = $('composedText');
    const text = box ? box.value.trim() : '';
    if (!text) {
        showToast('Cannot save empty translation.', 'info');
        return;
    }
    await recordHistoryEntry('Sign → Text', 'Camera ISL Gestures', text);
}

// ------------------------------------------------------------
// speech → ISL
// ------------------------------------------------------------
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.warn('Web Speech API is not supported in this browser.');
        return;
    }

    speechRecognizer = new SpeechRecognition();
    speechRecognizer.continuous = true;
    speechRecognizer.interimResults = true;
    speechRecognizer.lang = 'en-IN';

    speechRecognizer.onstart = () => {
        isListening = true;
        $('speechStatusBadge').textContent = 'Listening...';
        $('speechStatusBadge').className = 'badge badge-success';
        $('micPulseArea')?.classList.add('listening');
        $('startSpeechBtn').disabled = true;
        $('stopSpeechBtn').disabled = false;
    };

    speechRecognizer.onresult = event => {
        let finalStr = '';
        let interimStr = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalStr += transcript + ' ';
            } else {
                interimStr += transcript;
            }
        }

        if (finalStr.trim()) {
            speechTranscript += finalStr;
        }

        const combined = (speechTranscript + interimStr).trim();
        const display = $('speechLiveTranscript');
        if (display) {
            display.textContent = combined || 'Listening...';
        }

        if (combined) {
            generateSpeechISL(combined);
        }
    };

    speechRecognizer.onerror = event => {
        console.error('Speech recognition error:', event.error);
        if (event.error === 'not-allowed') {
            showToast('Microphone access denied. Please grant permission in browser.', 'error');
        } else if (event.error !== 'no-speech') {
            showToast(`Speech recognition: ${event.error}`, 'error');
        }
    };

    speechRecognizer.onend = () => {
        isListening = false;
        $('speechStatusBadge').textContent = speechTranscript.trim() ? 'Speech captured' : 'Mic Inactive';
        $('speechStatusBadge').className = speechTranscript.trim() ? 'badge badge-primary' : 'badge badge-neutral';
        $('micPulseArea')?.classList.remove('listening');
        $('startSpeechBtn').disabled = false;
        $('stopSpeechBtn').disabled = true;
    };
}

function startSpeech() {
    if (!speechRecognizer) {
        initSpeechRecognition();
        if (!speechRecognizer) {
            showToast('Speech Recognition not supported in this browser. Please use Chrome.', 'error');
            return;
        }
    }

    try {
        speechTranscript = '';
        const display = $('speechLiveTranscript');
        if (display) display.textContent = 'Listening... speak now...';
        speechRecognizer.start();
            showToast('Mic on. Go ahead.', 'info');
    } catch (e) {
        console.warn('Speech start error:', e);
    }
}

function stopSpeech() {
    if (speechRecognizer && isListening) {
        try {
            speechRecognizer.stop();
        } catch (e) {}
    }
    isListening = false;
}

function clearSpeech() {
    stopSpeech();
    speechTranscript = '';
    const display = $('speechLiveTranscript');
    if (display) display.textContent = 'Click "Start Speaking" and speak clearly into your microphone...';
    speechSlideItems = [];
    renderSpeechViewer();
}

function toggleSpeechEdit() {
    const display = $('speechLiveTranscript');
    const field = $('speechEditField');
    const btnRow = $('speechEditBtnRow');
    const editBtn = $('toggleSpeechEditBtn');

    if (!field || !display) return;
    const isEditing = !field.classList.contains('hidden');

    if (isEditing) {
        cancelSpeechEdit();
    } else {
        stopSpeech();
        field.value = speechTranscript.trim();
        display.classList.add('hidden');
        field.classList.remove('hidden');
        btnRow?.classList.remove('hidden');
        if (editBtn) editBtn.textContent = 'Cancel';
        field.focus();
    }
}

function applySpeechEdit() {
    const display = $('speechLiveTranscript');
    const field = $('speechEditField');
    const btnRow = $('speechEditBtnRow');
    const editBtn = $('toggleSpeechEditBtn');

    if (!field || !display) return;
    speechTranscript = field.value.trim();
    display.textContent = speechTranscript || 'Click "Start Speaking" to record...';

    field.classList.add('hidden');
    btnRow?.classList.add('hidden');
    display.classList.remove('hidden');
    if (editBtn) editBtn.textContent = 'Edit Text';

    if (speechTranscript) {
        generateSpeechISL(speechTranscript);
        showToast('Updated speech translation.', 'success');
    }
}

function cancelSpeechEdit() {
    const display = $('speechLiveTranscript');
    const field = $('speechEditField');
    const btnRow = $('speechEditBtnRow');
    const editBtn = $('toggleSpeechEditBtn');

    if (!field || !display) return;
    field.classList.add('hidden');
    btnRow?.classList.add('hidden');
    display.classList.remove('hidden');
    if (editBtn) editBtn.textContent = 'Edit Text';
}


function generateSpeechISL(text) {
    speechSlideItems = parseTextToSignItems(text);
    speechSlideIndex = 0;
    renderSpeechViewer();
}

function switchSpeechSignView(mode) {
    speechViewMode = mode;
    $('speechViewSlideshowBtn')?.classList.toggle('active', mode === 'slideshow');
    $('speechViewGridBtn')?.classList.toggle('active', mode === 'grid');

    if (mode === 'slideshow') {
        $('speechSlideshowView')?.classList.remove('hidden');
        $('speechGridView')?.classList.add('hidden');
    } else {
        $('speechSlideshowView')?.classList.add('hidden');
        $('speechGridView')?.classList.remove('hidden');
        renderSpeechGrid();
    }
}

function renderSpeechViewer() {
    const stage = $('speechSignViewer');
    const controls = $('speechSlideControls');
    const counter = $('speechSlideCounter');
    const slider = $('speechScrubSlider');

    if (!stage) return;

    if (!speechSlideItems.length) {
        stage.innerHTML = `
            <div class="sign-empty-state">
                <span class="empty-icon">
                    <svg class="svg-icon" viewBox="0 0 24 24" style="width: 36px; height: 36px;"><path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0"></path><path d="M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v2"></path><path d="M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v8"></path><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"></path></svg>
                </span>
                <p>Spoken words will appear as ISL gesture sequences here</p>
            </div>
        `;
        controls?.classList.add('hidden');
        $('speechProgressDots')?.replaceChildren();
        stopSpeechSlidePlayback();
        return;
    }

    controls?.classList.remove('hidden');
    if (slider) {
        slider.max = speechSlideItems.length - 1;
        slider.value = speechSlideIndex;
    }
    if (counter) {
        counter.textContent = `${speechSlideIndex + 1} / ${speechSlideItems.length}`;
    }

    const item = speechSlideItems[speechSlideIndex];
    if (item.isSpace) {
        stage.innerHTML = `
            <div class="active-slide-card">
                <div class="slide-img-wrapper" style="background: var(--slate-100); display: flex; flex-direction: column; align-items: center; justify-content: center; color: var(--text-muted);">
                    <svg class="svg-icon" viewBox="0 0 24 24" style="width: 32px; height: 32px; color: var(--slate-400);"><rect x="4" y="16" width="16" height="4" rx="1"></rect></svg>
                    <strong style="margin-top: 8px; font-size: 0.88rem; color: var(--slate-700);">Word Break</strong>
                </div>
                <div class="slide-details-row">
                    <span class="slide-letter-name" style="font-size: 0.95rem; color: var(--text-muted);">[ Space ]</span>
                </div>
            </div>
        `;
    } else {
        stage.innerHTML = `
            <div class="active-slide-card">
                <div class="slide-img-wrapper">
                    <img src="${item.imageJpg || `/static/isl/${item.letter}.jpg`}" alt="Letter ${item.letter}"
                         onerror="this.onerror=null; this.src='${item.imagePng || `/static/isl/${item.letter}.png`}';">
                </div>
                <div class="slide-details-row">
                    <span class="slide-letter-name">Letter ${item.letter}</span>
                </div>
            </div>
        `;
    }

    renderSpeechProgressDots();
    updateSpeechPlayButtonState();

    if (speechViewMode === 'grid') {
        renderSpeechGrid();
    }
}

function renderSpeechProgressDots() {
    const container = $('speechProgressDots');
    if (!container) return;
    if (!speechSlideItems.length) {
        container.innerHTML = '';
        return;
    }

    container.innerHTML = speechSlideItems.map((item, idx) => {
        const isCurrent = idx === speechSlideIndex;
        const isPassed = idx < speechSlideIndex;
        const dotState = isCurrent ? 'active' : (isPassed ? 'completed' : '');
        const wordClass = item.isWord ? 'dot-word' : (item.isSpace ? 'dot-space' : '');
        const label = item.isSpace ? '␣' : (item.isWord ? (item.icon ? `${item.icon} ${item.word}` : item.word) : item.letter);
        const title = item.isSpace ? 'Space' : (item.isWord ? `Word Sign: ${item.word}` : `Letter: ${item.letter}`);

        return `
            <button type="button" class="progress-dot ${dotState} ${wordClass}" onclick="jumpToSpeechSlide(${idx})" title="${title}">
                <span>${label}</span>
            </button>
        `;
    }).join('');
}

function renderSpeechGrid() {
    const grid = $('speechSignGrid');
    if (!grid) return;

    if (!speechSlideItems.length) {
        grid.innerHTML = '<p class="text-muted" style="grid-column: 1/-1; padding: 24px; text-align: center;">No active signs to display.</p>';
        return;
    }

    grid.innerHTML = speechSlideItems.map((item, idx) => {
        const isCurrent = idx === speechSlideIndex ? 'active' : '';
        if (item.isSpace) {
            return `
                <div class="strip-card ${isCurrent}" onclick="jumpToSpeechSlide(${idx})">
                    <div style="width: 70px; height: 70px; background: var(--slate-100); border-radius: var(--radius-sm); display: flex; align-items: center; justify-content: center; font-size: 0.72rem; color: var(--text-muted); margin-bottom: 6px;">[ Space ]</div>
                    <strong>Break</strong>
                </div>
            `;
        }
        return `
            <div class="strip-card ${isCurrent}" onclick="jumpToSpeechSlide(${idx})">
                <img src="${item.imageJpg || `/static/isl/${item.letter}.jpg`}" alt="Letter ${item.letter}" onerror="this.onerror=null; this.src='${item.imagePng || `/static/isl/${item.letter}.png`}';">
                <strong>Letter ${item.letter}</strong>
            </div>
        `;
    }).join('');
}

function jumpToSpeechSlide(idx) {
    if (idx >= 0 && idx < speechSlideItems.length) {
        speechSlideIndex = idx;
        renderSpeechViewer();
    }
}

function nextSpeechSlide() {
    if (!speechSlideItems.length) return;
    if (speechSlideIndex < speechSlideItems.length - 1) {
        speechSlideIndex++;
        renderSpeechViewer();
    } else {
        showToast('Reached the last sign.', 'info', 1200);
    }
}

function prevSpeechSlide() {
    if (!speechSlideItems.length) return;
    if (speechSlideIndex > 0) {
        speechSlideIndex--;
        renderSpeechViewer();
    }
}

function scrubSpeechSlide(val) {
    speechSlideIndex = parseInt(val, 10) || 0;
    renderSpeechViewer();
}

function toggleSpeechPlay() {
    if (speechIsPlaying) {
        stopSpeechSlidePlayback();
    } else {
        if (speechSlideIndex >= speechSlideItems.length - 1) {
            speechSlideIndex = 0;
            renderSpeechViewer();
        }
        startSpeechSlidePlayback();
    }
}

function startSpeechSlidePlayback() {
    if (!speechSlideItems.length) return;

    if (speechSlideIndex >= speechSlideItems.length - 1) {
        speechSlideIndex = 0;
        renderSpeechViewer();
    }

    speechIsPlaying = true;
    updateSpeechPlayButtonState();

    speechSlideTimer = setInterval(() => {
        if (speechSlideIndex < speechSlideItems.length - 1) {
            speechSlideIndex++;
            renderSpeechViewer();
        } else {
            // stop at last slide (no loop)
            stopSpeechSlidePlayback();
            showToast('Done. Hit Replay if you want it again.', 'info', 2200);
        }
    }, speechSpeedMs);
}

function stopSpeechSlidePlayback() {
    speechIsPlaying = false;
    if (speechSlideTimer) {
        clearInterval(speechSlideTimer);
        speechSlideTimer = null;
    }
    updateSpeechPlayButtonState();
}

function updateSpeechPlayButtonState() {
    const icon = $('speechPlayIcon');
    const label = $('speechPlayLabel');
    const btn = $('playSpeechSlideBtn');
    if (!btn) return;

    if (speechIsPlaying) {
        if (icon) icon.textContent = '⏸';
        if (label) label.textContent = 'Pause';
        btn.classList.remove('btn-replay-mode');
        btn.title = 'Pause sign playback';
    } else if (speechSlideItems.length > 0 && speechSlideIndex >= speechSlideItems.length - 1) {
        if (icon) icon.textContent = '↺';
        if (label) label.textContent = 'Replay';
        btn.classList.add('btn-replay-mode');
        btn.title = 'Replay sequence from beginning';
    } else {
        if (icon) icon.textContent = '▶';
        if (label) label.textContent = 'Play';
        btn.classList.remove('btn-replay-mode');
        btn.title = 'Play sequence';
    }
}

function changeSpeechSlideSpeed(val) {
    speechSpeedMs = parseInt(val, 10) || 1000;
    if (speechIsPlaying) {
        stopSpeechSlidePlayback();
        startSpeechSlidePlayback();
    }
}

async function saveSpeechTranslation() {
    const text = speechTranscript.trim();
    if (!text) {
        showToast('No speech translation to save.', 'info');
        return;
    }
    await recordHistoryEntry('Speech → ISL', text, `${speechSlideItems.length} signs generated`);
}

// ------------------------------------------------------------
// text → ISL
// ------------------------------------------------------------
function updateWordCount() {
    const input = $('textToSignInput');
    const label = $('textInputCount');
    if (!input || !label) return;
    const text = input.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    label.textContent = `${words} word${words === 1 ? '' : 's'}`;
}

function insertQuickPhrase(phrase) {
    const input = $('textToSignInput');
    if (!input) return;
    input.value = phrase;
    updateWordCount();
    convertTextToSign();
}


function convertTextToSign() {
    const input = $('textToSignInput');
    const text = input ? input.value.trim() : '';

    if (!text) {
        showToast('Type something first.', 'info');
        return;
    }

    textSlideItems = parseTextToSignItems(text);
    textSlideIndex = 0;
    renderTextViewer();
        showToast(`Made ${textSlideItems.length} signs for '${text}'`, 'success');
}

function clearTextToSign() {
    const input = $('textToSignInput');
    if (input) input.value = '';
    updateWordCount();
    textSlideItems = [];
    renderTextViewer();
}

function switchTextSignView(mode) {
    textViewMode = mode;
    $('textViewSlideshowBtn')?.classList.toggle('active', mode === 'slideshow');
    $('textViewGridBtn')?.classList.toggle('active', mode === 'grid');

    if (mode === 'slideshow') {
        $('textSlideshowView')?.classList.remove('hidden');
        $('textGridView')?.classList.add('hidden');
    } else {
        $('textSlideshowView')?.classList.add('hidden');
        $('textGridView')?.classList.remove('hidden');
        renderTextGrid();
    }
}

function renderTextViewer() {
    const stage = $('textSignViewer');
    const controls = $('textSlideControls');
    const counter = $('textSlideCounter');
    const slider = $('textScrubSlider');

    if (!stage) return;

    if (!textSlideItems.length) {
        stage.innerHTML = `
            <div class="sign-empty-state">
                <span class="empty-icon">
                    <svg class="svg-icon" viewBox="0 0 24 24" style="width: 36px; height: 36px;"><path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0"></path><path d="M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v2"></path><path d="M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v8"></path><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"></path></svg>
                </span>
                <p>Type text and click "Convert to Signs" to view the sequence</p>
            </div>
        `;
        controls?.classList.add('hidden');
        $('textProgressDots')?.replaceChildren();
        stopTextSlidePlayback();
        return;
    }

    controls?.classList.remove('hidden');
    if (slider) {
        slider.max = textSlideItems.length - 1;
        slider.value = textSlideIndex;
    }
    if (counter) {
        counter.textContent = `${textSlideIndex + 1} / ${textSlideItems.length}`;
    }

    const item = textSlideItems[textSlideIndex];
    if (item.isSpace) {
        stage.innerHTML = `
            <div class="active-slide-card">
                <div class="slide-img-wrapper" style="background: var(--slate-100); display: flex; flex-direction: column; align-items: center; justify-content: center; color: var(--text-muted);">
                    <svg class="svg-icon" viewBox="0 0 24 24" style="width: 32px; height: 32px; color: var(--slate-400);"><rect x="4" y="16" width="16" height="4" rx="1"></rect></svg>
                    <strong style="margin-top: 8px; font-size: 0.88rem; color: var(--slate-700);">Word Break</strong>
                </div>
                <div class="slide-details-row">
                    <span class="slide-letter-name" style="font-size: 0.95rem; color: var(--text-muted);">[ Space ]</span>
                </div>
            </div>
        `;
    } else {
        stage.innerHTML = `
            <div class="active-slide-card">
                <div class="slide-img-wrapper">
                    <img src="${item.imageJpg || `/static/isl/${item.letter}.jpg`}" alt="Letter ${item.letter}"
                         onerror="this.onerror=null; this.src='${item.imagePng || `/static/isl/${item.letter}.png`}';">
                </div>
                <div class="slide-details-row">
                    <span class="slide-letter-name">Letter ${item.letter}</span>
                </div>
            </div>
        `;
    }

    renderTextProgressDots();
    updateTextPlayButtonState();

    if (textViewMode === 'grid') {
        renderTextGrid();
    }
}

function renderTextProgressDots() {
    const container = $('textProgressDots');
    if (!container) return;
    if (!textSlideItems.length) {
        container.innerHTML = '';
        return;
    }

    container.innerHTML = textSlideItems.map((item, idx) => {
        const isCurrent = idx === textSlideIndex;
        const isPassed = idx < textSlideIndex;
        const dotState = isCurrent ? 'active' : (isPassed ? 'completed' : '');
        const wordClass = item.isWord ? 'dot-word' : (item.isSpace ? 'dot-space' : '');
        const label = item.isSpace ? '␣' : (item.isWord ? (item.icon ? `${item.icon} ${item.word}` : item.word) : item.letter);
        const title = item.isSpace ? 'Space' : (item.isWord ? `Word Sign: ${item.word}` : `Letter: ${item.letter}`);

        return `
            <button type="button" class="progress-dot ${dotState} ${wordClass}" onclick="jumpToTextSlide(${idx})" title="${title}">
                <span>${label}</span>
            </button>
        `;
    }).join('');
}

function renderTextGrid() {
    const grid = $('textSignGrid');
    if (!grid) return;

    if (!textSlideItems.length) {
        grid.innerHTML = '<p class="text-muted" style="grid-column: 1/-1; padding: 24px; text-align: center;">No signs generated yet.</p>';
        return;
    }

    grid.innerHTML = textSlideItems.map((item, idx) => {
        const isCurrent = idx === textSlideIndex ? 'active' : '';
        if (item.isSpace) {
            return `
                <div class="strip-card ${isCurrent}" onclick="jumpToTextSlide(${idx})">
                    <div style="width: 70px; height: 70px; background: var(--slate-100); border-radius: var(--radius-sm); display: flex; align-items: center; justify-content: center; font-size: 0.72rem; color: var(--text-muted); margin-bottom: 6px;">[ Space ]</div>
                    <strong>Break</strong>
                </div>
            `;
        }
        return `
            <div class="strip-card ${isCurrent}" onclick="jumpToTextSlide(${idx})">
                <img src="${item.imageJpg || `/static/isl/${item.letter}.jpg`}" alt="Letter ${item.letter}" onerror="this.onerror=null; this.src='${item.imagePng || `/static/isl/${item.letter}.png`}';">
                <strong>Letter ${item.letter}</strong>
            </div>
        `;
    }).join('');
}

function jumpToTextSlide(idx) {
    if (idx >= 0 && idx < textSlideItems.length) {
        textSlideIndex = idx;
        renderTextViewer();
    }
}

function nextTextSlide() {
    if (!textSlideItems.length) return;
    if (textSlideIndex < textSlideItems.length - 1) {
        textSlideIndex++;
        renderTextViewer();
    } else {
        showToast('Reached the last sign.', 'info', 1200);
    }
}

function prevTextSlide() {
    if (!textSlideItems.length) return;
    if (textSlideIndex > 0) {
        textSlideIndex--;
        renderTextViewer();
    }
}

function scrubTextSlide(val) {
    textSlideIndex = parseInt(val, 10) || 0;
    renderTextViewer();
}

function toggleTextPlay() {
    if (textIsPlaying) {
        stopTextSlidePlayback();
    } else {
        if (textSlideIndex >= textSlideItems.length - 1) {
            textSlideIndex = 0;
            renderTextViewer();
        }
        startTextSlidePlayback();
    }
}

function startTextSlidePlayback() {
    if (!textSlideItems.length) return;

    if (textSlideIndex >= textSlideItems.length - 1) {
        textSlideIndex = 0;
        renderTextViewer();
    }

    textIsPlaying = true;
    updateTextPlayButtonState();

    textSlideTimer = setInterval(() => {
        if (textSlideIndex < textSlideItems.length - 1) {
            textSlideIndex++;
            renderTextViewer();
        } else {
            // stop at last slide (no loop)
            stopTextSlidePlayback();
            showToast('Done. Hit Replay if you want it again.', 'info', 2200);
        }
    }, textSpeedMs);
}

function stopTextSlidePlayback() {
    textIsPlaying = false;
    if (textSlideTimer) {
        clearInterval(textSlideTimer);
        textSlideTimer = null;
    }
    updateTextPlayButtonState();
}

function updateTextPlayButtonState() {
    const icon = $('textPlayIcon');
    const label = $('textPlayLabel');
    const btn = $('playTextSlideBtn');
    if (!btn) return;

    if (textIsPlaying) {
        if (icon) icon.textContent = '⏸';
        if (label) label.textContent = 'Pause';
        btn.classList.remove('btn-replay-mode');
        btn.title = 'Pause sign playback';
    } else if (textSlideItems.length > 0 && textSlideIndex >= textSlideItems.length - 1) {
        if (icon) icon.textContent = '↺';
        if (label) label.textContent = 'Replay';
        btn.classList.add('btn-replay-mode');
        btn.title = 'Replay sequence from beginning';
    } else {
        if (icon) icon.textContent = '▶';
        if (label) label.textContent = 'Play';
        btn.classList.remove('btn-replay-mode');
        btn.title = 'Play sequence';
    }
}

function changeTextSlideSpeed(val) {
    textSpeedMs = parseInt(val, 10) || 1000;
    if (textIsPlaying) {
        stopTextSlidePlayback();
        startTextSlidePlayback();
    }
}

async function saveTextTranslation() {
    const input = $('textToSignInput');
    const text = input ? input.value.trim() : '';
    if (!text) {
        showToast('No text translation to save.', 'info');
        return;
    }
    await recordHistoryEntry('Text → ISL', text, `${textSlideItems.length} signs generated`);
}

// ------------------------------------------------------------
// Split text into ISL steps (known words + letter spelling)
// ------------------------------------------------------------
function parseTextToSignItems(rawText) {
    if (!rawText) return [];
    const items = [];
    const upper = rawText.toUpperCase().trim();

    for (let i = 0; i < upper.length; i++) {
        const ch = upper.charAt(i);
        if (ch >= 'A' && ch <= 'Z') {
            items.push({
                isWord: false,
                isSpace: false,
                letter: ch,
                label: ch,
                imageJpg: `/static/isl/${ch}.jpg`,
                imagePng: `/static/isl/${ch}.png`
            });
        } else if (ch === ' ' && items.length && !items[items.length - 1].isSpace) {
            items.push({ isWord: false, isSpace: true, label: ' ' });
        }
    }
    return items;
}

// ------------------------------------------------------------
// dictionary
// ------------------------------------------------------------
async function fetchDictionaryData() {
    try {
        const res = await fetch('/api/dictionary');
        if (res.ok) {
            const data = await res.json();
            islAlphabet = data.alphabet || [];
            if ($('dictAlphaCount')) $('dictAlphaCount').textContent = islAlphabet.length;
            renderDictionary();
            return;
        }
    } catch (e) {
        console.warn('Failed to load dictionary endpoint, using fallback.');
    }

    // Fallback dictionary
    const fallbackLetters = [
        { letter: "A", type: "Two-Hand", desc: "Thumbs touch at the tips (looks like a peak). Other fingers curled." },
        { letter: "B", type: "Two-Hand", desc: "Join thumb and index of both hands so you get two loops." },
        { letter: "C", type: "One-Hand", desc: "Curve fingers and thumb into a C." },
        { letter: "D", type: "Two-Hand", desc: "One index points up; the other hand makes a small curve that meets it." },
        { letter: "E", type: "Two-Hand", desc: "Index fingers point at each other, horizontal." },
        { letter: "F", type: "Two-Hand", desc: "Cross index and middle of one hand over the same fingers of the other." },
        { letter: "G", type: "Two-Hand", desc: "Two fists, one sitting on the other." },
        { letter: "H", type: "Two-Hand", desc: "Flat hand brushes across the other palm." },
        { letter: "I", type: "One-Hand", desc: "Fist with the pinky up." },
        { letter: "J", type: "Two-Hand", desc: "Draw a J with the index finger on the other palm." },
        { letter: "K", type: "Two-Hand", desc: "One index straight up; the other hooks against that knuckle." },
        { letter: "L", type: "One-Hand", desc: "Thumb and index at a right angle (L)." },
        { letter: "M", type: "Two-Hand", desc: "Three fingers rest on the other palm." },
        { letter: "N", type: "Two-Hand", desc: "Two fingers rest on the other palm." },
        { letter: "O", type: "One-Hand", desc: "Fingertips and thumb meet in an O." },
        { letter: "P", type: "Two-Hand", desc: "Thumb+index circle resting against an upright index." },
        { letter: "Q", type: "Two-Hand", desc: "Hook an index into the circle of the other thumb and index." },
        { letter: "R", type: "Two-Hand", desc: "Hooked index standing in the other palm." },
        { letter: "S", type: "Two-Hand", desc: "Pinkies hook together." },
        { letter: "T", type: "Two-Hand", desc: "Index taps the edge of the other hand, under the thumb." },
        { letter: "U", type: "One-Hand", desc: "Index and middle together, pointing up." },
        { letter: "V", type: "One-Hand", desc: "Index and middle apart (V)." },
        { letter: "W", type: "Two-Hand", desc: "Fingers of both hands interlock, pointing out." },
        { letter: "X", type: "Two-Hand", desc: "Index fingers cross." },
        { letter: "Y", type: "Two-Hand", desc: "Index sits in the gap between the other thumb and index." },
        { letter: "Z", type: "Two-Hand", desc: "One palm bent against the other, like a Z from the side." }
    ];

    islAlphabet = fallbackLetters.map(item => ({
        ...item,
        image_jpg: `/static/isl/${item.letter}.jpg`,
        image_png: `/static/isl/${item.letter}.png`
    }));

    if ($('dictAlphaCount')) $('dictAlphaCount').textContent = islAlphabet.length;
    renderDictionary();
}


function setDictAlphaFilter(filter) {
    currentDictAlphaFilter = filter;
    $('filterAllBtn')?.classList.toggle('active', filter === 'all');
    $('filterSingleBtn')?.classList.toggle('active', filter === 'One-Hand');
    $('filterTwoBtn')?.classList.toggle('active', filter === 'Two-Hand');
    renderDictionary();
}


function filterDictionary(query) {
    currentDictQuery = query.trim().toUpperCase();
    renderDictionary();
}

function renderDictionary() {
    const grid = $('dictionaryGrid');
    if (!grid) return;
    renderAlphabetGrid(grid);
}

function renderAlphabetGrid(grid) {
    let filtered = islAlphabet;
    if (currentDictAlphaFilter !== 'all') {
        filtered = filtered.filter(item => item.type === currentDictAlphaFilter);
    }
    if (currentDictQuery) {
        filtered = filtered.filter(item => 
            item.letter.includes(currentDictQuery) ||
            item.desc.toUpperCase().includes(currentDictQuery)
        );
    }

    if (!filtered.length) {
        grid.innerHTML = '<p class="text-muted" style="grid-column: 1/-1; padding: 40px; text-align: center;">No matching alphabet signs found.</p>';
        return;
    }

    grid.innerHTML = filtered.map(item => `
        <div class="dict-card" onclick="openLetterModal('${item.letter}')">
            <div class="dict-card-img-wrap">
                <span class="badge ${item.type === 'One-Hand' ? 'badge-primary' : 'badge-neutral'}" style="position: absolute; top: 12px; left: 12px; z-index: 2; box-shadow: var(--shadow-subtle);">
                    ${item.type}
                </span>
                <img src="${item.image_jpg || `/static/isl/${item.letter}.jpg`}" alt="Letter ${item.letter}" loading="lazy"
                     onerror="this.onerror=null; this.src='${item.image_png || `/static/isl/${item.letter}.png`}';">
            </div>
            <div class="dict-card-body">
                <div class="dict-card-title-row">
                    <strong>Letter ${item.letter}</strong>
                </div>
                <p>${escapeHtml(item.desc)}</p>
                <div class="dict-card-footer">
                    <span class="card-action-text">Inspect Sign →</span>
                </div>
            </div>
        </div>
    `).join('');
}


function openLetterModal(key) {
    const item = islAlphabet.find(x => x.letter === key);
    if (!item) return;
    activeModalItem = { ...item, isWord: false };

    const modal = $('letterModal');
    if (!modal) return;

    $('modalLetterImg').src = item.image_jpg || `/static/isl/${item.letter}.jpg`;
    $('modalLetterBadge').textContent = item.letter;
    $('modalTypeBadge').textContent = item.type;
    $('modalTypeBadge').className = `badge ${item.type === 'One-Hand' ? 'badge-primary' : 'badge-neutral'}`;
    $('modalTitle').textContent = `Letter ${item.letter}`;
    $('modalDesc').textContent = item.desc;
    $('modalTipText').textContent = item.type === 'One-Hand'
        ? 'Keep one hand in the middle of the camera. Match the photo as closely as you can.'
        : 'Both hands should be in frame. Check the photo for how the palms and fingers sit.';

    modal.classList.remove('hidden');
}

function closeLetterModal() {
    $('letterModal')?.classList.add('hidden');
    activeModalItem = null;
}

function practiceLetterInCamera() {
    const item = activeModalItem;
    closeLetterModal();
    switchMode('sign');
    startCamera();
    if (item) {
        const label = `letter '${item.letter}'`;
        showToast(`Show ${label} in front of the camera.`, 'info');
    }
}

// ------------------------------------------------------------
// history
// ------------------------------------------------------------
async function recordHistoryEntry(mode, input, output) {
    if (!currentUser) {
        showToast('Please sign in or continue as guest to save history.', 'info');
        return;
    }

    if (currentUser.isGuest) {
        // Save to guest local storage
        const localHistory = JSON.parse(localStorage.getItem('signmate_guest_history') || '[]');
        const newEntry = {
            id: Date.now(),
            mode: mode,
            input_text: input,
            output_text: output,
            created_at: new Date().toISOString()
        };
        localHistory.unshift(newEntry);
        localStorage.setItem('signmate_guest_history', JSON.stringify(localHistory));
        showToast('Saved for this session.', 'success');
        return;
    }

    try {
        const res = await fetch('/api/history', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: currentUser.id,
                mode: mode,
                input_text: input,
                output_text: output
            })
        });

        if (res.ok) {
            showToast('Saved to your account.', 'success');
        } else {
            showToast('Unable to save translation to database.', 'error');
        }
    } catch (e) {
        console.error('History save error:', e);
        showToast('Network error saving history.', 'error');
    }
}

async function loadHistory() {
    const container = $('historyListContainer');
    if (!container) return;

    if (!currentUser) {
        container.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">🔒</span>
                <h4>Sign in to view history</h4>
                <p>History is saved for registered accounts and active guest sessions.</p>
            </div>
        `;
        return;
    }

    if (currentUser.isGuest) {
        fullHistoryList = JSON.parse(localStorage.getItem('signmate_guest_history') || '[]');
        renderHistoryList();
        return;
    }

    try {
        container.innerHTML = '<p class="text-muted" style="padding: 24px; text-align: center;">Loading history records...</p>';
        const res = await fetch(`/api/history/${currentUser.id}`);
        if (res.ok) {
            const data = await res.json();
            fullHistoryList = data.history || [];
            renderHistoryList();
        } else {
            container.innerHTML = '<p class="text-muted" style="padding: 20px;">Could not load history.</p>';
        }
    } catch (e) {
        container.innerHTML = '<p class="text-muted" style="padding: 20px;">Network error loading history.</p>';
    }
}

function setHistoryFilter(mode, btn) {
    currentHistoryFilter = mode;
    document.querySelectorAll('.history-tab').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');
    renderHistoryList();
}

function filterHistoryList(query) {
    renderHistoryList(query.toLowerCase());
}

function renderHistoryList(query = '') {
    const container = $('historyListContainer');
    const badge = $('historyCountBadge');
    if (!container) return;

    let filtered = fullHistoryList;
    if (currentHistoryFilter !== 'all') {
        filtered = filtered.filter(item => item.mode === currentHistoryFilter);
    }
    if (query) {
        filtered = filtered.filter(item => 
            (item.input_text && item.input_text.toLowerCase().includes(query)) ||
            (item.output_text && item.output_text.toLowerCase().includes(query)) ||
            (item.mode && item.mode.toLowerCase().includes(query))
        );
    }

    if (badge) badge.textContent = `${filtered.length} entries`;

    if (!filtered.length) {
        container.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">🕘</span>
                <h4>No History Records Found</h4>
                <p>Translations performed in Sign, Speech, or Text mode will be stored here.</p>
            </div>
        `;
        return;
    }

    container.innerHTML = filtered.map(item => {
        let modeClass = 'sign';
        if (item.mode && item.mode.includes('Speech')) modeClass = 'speech';
        if (item.mode && item.mode.includes('Text')) modeClass = 'text';

        const timeStr = formatRelativeTime(item.created_at);

        return `
            <div class="history-row">
                <div class="history-row-left">
                    <div class="history-row-meta">
                        <span class="badge ${modeClass === 'sign' ? 'badge-primary' : 'badge-neutral'}">${escapeHtml(item.mode)}</span>
                        <span>${escapeHtml(timeStr)}</span>
                    </div>
                    <div class="history-row-text">
                        ${escapeHtml(item.output_text || item.input_text || '—')}
                    </div>
                    ${item.input_text && item.input_text !== item.output_text ? `
                        <div class="history-row-output">
                            Input: ${escapeHtml(item.input_text)}
                        </div>
                    ` : ''}
                </div>
                <div class="history-bulk-actions">
                    <button type="button" class="btn btn-outline btn-sm" onclick="copyHistoryText('${escapeHtml(item.output_text || item.input_text)}')">
                        Copy
                    </button>
                    <button type="button" class="btn btn-danger-outline btn-sm" onclick="deleteHistoryItem(${item.id})">
                        Delete
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

function copyHistoryText(text) {
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
        showToast('Copied.', 'success');
    });
}

async function deleteHistoryItem(id) {
    if (!currentUser) return;

    if (currentUser.isGuest) {
        fullHistoryList = fullHistoryList.filter(x => x.id !== id);
        localStorage.setItem('signmate_guest_history', JSON.stringify(fullHistoryList));
        renderHistoryList();
        showToast('Item deleted from session history.', 'info');
        return;
    }

    try {
        const res = await fetch(`/api/history/${id}`, { method: 'DELETE' });
        if (res.ok) {
            fullHistoryList = fullHistoryList.filter(x => x.id !== id);
            renderHistoryList();
            showToast('History item removed.', 'info');
        } else {
            showToast('Unable to delete history item.', 'error');
        }
    } catch (e) {
        showToast('Network error deleting item.', 'error');
    }
}

async function clearAllHistory() {
    if (!currentUser) return;
    if (!confirm('Are you sure you want to clear all translation history?')) return;

    if (currentUser.isGuest) {
        fullHistoryList = [];
        localStorage.removeItem('signmate_guest_history');
        renderHistoryList();
        showToast('All session history cleared.', 'info');
        return;
    }

    try {
        const res = await fetch(`/api/history/user/${currentUser.id}`, { method: 'DELETE' });
        if (res.ok) {
            fullHistoryList = [];
            renderHistoryList();
            showToast('All translation history cleared.', 'success');
        } else {
            showToast('Failed to clear history.', 'error');
        }
    } catch (e) {
        showToast('Network error clearing history.', 'error');
    }
}

function exportHistory(format = 'txt') {
    if (!fullHistoryList.length) {
        showToast('No history records to export.', 'info');
        return;
    }

    let fileContent = '';
    let mimeType = 'text/plain';
    let filename = `SignMate_History_${new Date().toISOString().slice(0, 10)}`;

    if (format === 'json') {
        fileContent = JSON.stringify(fullHistoryList, null, 2);
        mimeType = 'application/json';
        filename += '.json';
    } else {
        fileContent = `SIGNMATE TRANSLATION HISTORY EXPORT\nGenerated: ${new Date().toLocaleString()}\nUser: ${currentUser ? currentUser.name : 'Guest'}\n${'='.repeat(60)}\n\n`;
        fullHistoryList.forEach((item, index) => {
            fileContent += `[#${index + 1}] ${item.mode.toUpperCase()} - ${item.created_at || 'N/A'}\n`;
            if (item.input_text) fileContent += `Input:  ${item.input_text}\n`;
            if (item.output_text) fileContent += `Output: ${item.output_text}\n`;
            fileContent += `${'-'.repeat(40)}\n\n`;
        });
        filename += '.txt';
    }

    const blob = new Blob([fileContent], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast(`Exported history as ${format.toUpperCase()}`, 'success');
}

// ------------------------------------------------------------
// helpers
// ------------------------------------------------------------
function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function formatRelativeTime(isoString) {
    if (!isoString) return 'Recent';
    try {
        const d = new Date(isoString);
        const now = new Date();
        const diffSec = Math.floor((now - d) / 1000);

        if (diffSec < 60) return 'Just now';
        if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
        if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    } catch (e) {
        return isoString;
    }
}

// Cleanup on tab unload
window.addEventListener('beforeunload', () => {
    stopCamera();
    stopSpeech();
    stopSpeechSlidePlayback();
    stopTextSlidePlayback();
});