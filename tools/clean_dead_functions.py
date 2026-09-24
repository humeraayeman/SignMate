# tools/clean_dead_functions.py
import re
from pathlib import Path

script_path = Path(r"C:\Users\humer\Documents\SignSync\frontend\script.js")
content = script_path.read_text(encoding="utf-8")

# 1. Remove banner line in resetPredictionUI
content = content.replace("    $('wordDetectedBanner')?.classList.add('hidden');\n", "")

# 2. Remove insertDetectedWord
content = re.sub(
    r"function insertDetectedWord\(\) \{[\s\S]*?\n\}\n",
    "",
    content
)

# 3. Remove toggleSpeechNaturalWords
content = re.sub(
    r"function toggleSpeechNaturalWords\(\) \{[\s\S]*?\n\}\n",
    "",
    content
)

# 4. Remove toggleTextNaturalWords
content = re.sub(
    r"function toggleTextNaturalWords\(\) \{[\s\S]*?\n\}\n",
    "",
    content
)

# 5. Remove switchDictMainTab and setDictWordFilter
content = re.sub(
    r"function switchDictMainTab\(tab\) \{[\s\S]*?\n\}\n",
    "",
    content
)
content = re.sub(
    r"function setDictWordFilter\(category, btn\) \{[\s\S]*?\n\}\n",
    "",
    content
)

# 6. Remove renderWordsGrid
content = re.sub(
    r"function renderWordsGrid\(grid\) \{[\s\S]*?\n\}\n",
    "",
    content
)

# 7. Clean up fallback in fetchDictionaryData
content = re.sub(
    r"    islWords = Object\.values\(ISL_WORDS_MAP\);\n    if \(\$\('dictAlphaCount'\)\) \$\('dictAlphaCount'\)\.textContent = islAlphabet\.length;\n    if \(\$\('dictWordsCount'\)\) \$\('dictWordsCount'\)\.textContent = islWords\.length;\n",
    "    if ($('dictAlphaCount')) $('dictAlphaCount').textContent = islAlphabet.length;\n",
    content
)

script_path.write_text(content, encoding="utf-8")
print("Successfully cleaned dead functions from frontend/script.js!")
