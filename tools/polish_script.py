# tools/polish_script.py
import re
from pathlib import Path

script_path = Path(r"C:\Users\humer\Documents\SignSync\frontend\script.js")
content = script_path.read_text(encoding="utf-8")

# 1. Clean toast rendering with SVG icons
new_toast_func = """function showToast(message, type = 'info', duration = 3200) {
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
}"""

content = re.sub(r"function showToast\(message[\s\S]*?\n\}", new_toast_func, content)

# 2. Update toggleSpeechEdit text
content = content.replace("editBtn.textContent = '✏️ Edit';", "editBtn.textContent = 'Edit Text';")
content = content.replace("editBtn.textContent = '👁️ Done';", "editBtn.textContent = 'Done';")

# 3. Clean renderSpeechViewer
new_speech_viewer = """function renderSpeechViewer() {
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
}"""

content = re.sub(r"function renderSpeechViewer\(\) \{[\s\S]*?\n\}", new_speech_viewer, content)

# 4. Clean renderTextViewer
new_text_viewer = """function renderTextViewer() {
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
}"""

content = re.sub(r"function renderTextViewer\(\) \{[\s\S]*?\n\}", new_text_viewer, content)

# 5. Clean renderSpeechGrid and renderTextGrid
new_speech_grid = """function renderSpeechGrid() {
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
}"""

content = re.sub(r"function renderSpeechGrid\(\) \{[\s\S]*?\n\}", new_speech_grid, content)

new_text_grid = """function renderTextGrid() {
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
}"""

content = re.sub(r"function renderTextGrid\(\) \{[\s\S]*?\n\}", new_text_grid, content)

# 6. Clean history list items
new_history_row = """        return `
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
        `;"""

content = re.sub(
    r"        return `\s*<div class=\"history-card-item\"[\s\S]*?<\/div>\s*`;",
    new_history_row,
    content
)

script_path.write_text(content, encoding="utf-8")
print("Polished script.js successfully!")
