# tools/clean_script_words.py
import re
from pathlib import Path

script_path = Path(r"C:\Users\humer\Documents\SignSync\frontend\script.js")
content = script_path.read_text(encoding="utf-8")

# 1. Empty ISL_WORDS_MAP
content = re.sub(
    r"const ISL_WORDS_MAP = \{[\s\S]*?\n\};\n",
    "const ISL_WORDS_MAP = {};\n",
    content
)

# 2. In prediction UI, remove word banner handling
word_banner_block = """    // Check if prediction is an ISL word sign
    const isWord = Boolean(data.is_word);
    const banner = $('wordDetectedBanner');
    const nameEl = $('wordDetectedName');
    const confBadge = $('wordConfidenceBadge');

    if (isWord && success && letter) {
        currentDetectedWord = letter;
        if (banner) banner.classList.remove('hidden');
        if (nameEl) nameEl.textContent = letter;
        if (confBadge) confBadge.textContent = `${pct}%`;
    } else {
        currentDetectedWord = '';
        if (banner) banner.classList.add('hidden');
    }"""

clean_pred_block = """    currentDetectedWord = '';"""

if word_banner_block in content:
    content = content.replace(word_banner_block, clean_pred_block)
    print("Replaced word_banner_block")
else:
    print("word_banner_block not matched directly, checking regex")
    content = re.sub(
        r"    // Check if prediction is an ISL word sign[\s\S]*?if \(banner\) banner\.classList\.add\('hidden'\);\s*\}",
        "    currentDetectedWord = '';",
        content
    )

# 3. In addCurrentLetter, remove word check
add_current_old = """    if (currentDetectedLetter.length > 1) {
        // Full word sign
        const prefix = (box.value && !box.value.endsWith(' ')) ? ' ' : '';
        box.value += prefix + currentDetectedLetter + ' ';
    } else {
        box.value += currentDetectedLetter;
    }"""
add_current_new = """    box.value += currentDetectedLetter;"""

if add_current_old in content:
    content = content.replace(add_current_old, add_current_new)
    print("Replaced addCurrentLetter word branch")

# 4. In parseTextToSignItems, make it directly fingerspell letter by letter
parse_func_old_pattern = r"function parseTextToSignItems\(rawText[\s\S]*?\n    return items;\n\}"
parse_func_new = """function parseTextToSignItems(rawText) {
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
}"""

content = re.sub(parse_func_old_pattern, parse_func_new, content)

# 5. In generateSpeechISL & convertTextToSign, simplify call
content = content.replace("speechSlideItems = parseTextToSignItems(text, speechUseNaturalWords);", "speechSlideItems = parseTextToSignItems(text);")
content = content.replace("textSlideItems = parseTextToSignItems(text, textUseNaturalWords);", "textSlideItems = parseTextToSignItems(text);")

# 6. Simplify renderDictionary
dict_render_old = """function renderDictionary() {
    const grid = $('dictionaryGrid');
    if (!grid) return;

    if (currentDictMainTab === 'alphabet') {
        renderAlphabetGrid(grid);
    } else {
        renderWordsGrid(grid);
    }
}"""
dict_render_new = """function renderDictionary() {
    const grid = $('dictionaryGrid');
    if (!grid) return;
    renderAlphabetGrid(grid);
}"""

if dict_render_old in content:
    content = content.replace(dict_render_old, dict_render_new)
    print("Replaced renderDictionary")

# 7. In openLetterModal, make it letter-based
old_modal_pattern = r"function openLetterModal\(key\) \{[\s\S]*?\n    modal\.classList\.remove\('hidden'\);\n\}"
new_modal = """function openLetterModal(key) {
    const item = islAlphabet.find(x => x.letter === key);
    if (!item) return;
    activeModalItem = { ...item, isWord: false };

    const modal = $('letterModal');
    if (!modal) return;

    $('modalLetterImg').src = item.image_jpg || `/static/isl/${item.letter}.jpg`;
    $('modalLetterBadge').textContent = item.letter;
    $('modalTypeBadge').textContent = item.type;
    $('modalTypeBadge').className = `badge ${item.type === 'One-Hand' ? 'badge-primary' : 'badge-neutral'}`;
    $('modalTitle').textContent = `Letter '${item.letter}' in Indian Sign Language`;
    $('modalDesc').textContent = item.desc;
    $('modalTipText').textContent = item.type === 'One-Hand' 
        ? 'Hold your dominant hand steady in the center of the camera. Fingers should be clearly spread or formed according to the picture.' 
        : 'Align both hands together cleanly in frame. Ensure fingertips and palms match the reference image orientation.';

    modal.classList.remove('hidden');
}"""

content = re.sub(old_modal_pattern, new_modal, content)

# 8. Clean practiceLetterInCamera
content = content.replace(
    "const label = item.isWord ? `word '${item.word}'` : `letter '${item.letter}'`;",
    "const label = `letter '${item.letter}'`;"
)

# 9. Clean fetchDictionaryData
fetch_dict_old = """            islAlphabet = data.alphabet || [];
            islWords = data.words || Object.values(ISL_WORDS_MAP);

            if ($('dictAlphaCount')) $('dictAlphaCount').textContent = islAlphabet.length;
            if ($('dictWordsCount')) $('dictWordsCount').textContent = islWords.length;
            renderDictionary();
            return;"""
fetch_dict_new = """            islAlphabet = data.alphabet || [];
            if ($('dictAlphaCount')) $('dictAlphaCount').textContent = islAlphabet.length;
            renderDictionary();
            return;"""

if fetch_dict_old in content:
    content = content.replace(fetch_dict_old, fetch_dict_new)
    print("Replaced fetchDictionaryData")

script_path.write_text(content, encoding="utf-8")
print("Successfully cleaned frontend/script.js!")
