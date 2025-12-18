document.addEventListener('DOMContentLoaded', () => {
    // Theme Toggle
    const themeToggleBtn = document.getElementById('theme-toggle');

    // Check saved theme
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.body.setAttribute('data-theme', savedTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const currentTheme = document.body.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

            document.body.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
        });
    }

    // App State
    const state = {
        models: {} // { modelName: config }
    };

    // --- Tab Navigation ---
    const navBtns = document.querySelectorAll('.nav-btn');
    const sections = document.querySelectorAll('.tab-content');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.getAttribute('data-tab');

            // Update Buttons
            navBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Update Sections
            sections.forEach(sec => {
                if (sec.id === target) {
                    sec.classList.add('active');
                } else {
                    sec.classList.remove('active');
                }
            });
        });
    });

    // --- API Interaction ---
    async function fetchConfig() {
        try {
            const res = await fetch('/api/config');
            if (!res.ok) throw new Error('Failed to fetch config');
            state.models = await res.json();
            renderDashboard();
            renderSettings();
            updateConsoleOptions();
        } catch (err) {
            console.error(err);
            alert('Cannot connect to server. Is it running?');
        }
    }

    async function updateModelConfig(modelName, newConfig) {
        try {
            const res = await fetch(`/api/config/${modelName}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newConfig)
            });
            if (!res.ok) throw new Error('Update failed');
            const data = await res.json();

            // Update local state
            state.models[modelName] = data.config;
            alert(`Updated ${modelName} settings!`);
            fetchConfig(); // Refresh all
        } catch (err) {
            console.error(err);
            alert('Failed to update settings.');
        }
    }

    // --- Rendering ---
    const dashboardGrid = document.getElementById('model-cards');
    const settingsContainer = document.getElementById('settings-forms');
    const consoleSelect = document.getElementById('console-model-select');

    function renderDashboard() {
        if (!dashboardGrid) return;
        dashboardGrid.innerHTML = '';
        Object.keys(state.models).forEach(name => {
            const card = document.createElement('div');
            card.className = 'card';
            card.innerHTML = `
                <h3>${name}</h3>
                <p>Status: <span style="color: var(--success)">Active</span></p>
                <div class="stat">Threshold: ${state.models[name].threshold}</div>
            `;
            dashboardGrid.appendChild(card);
        });
    }

    function renderSettings() {
        if (!settingsContainer) return;
        settingsContainer.innerHTML = '';
        Object.keys(state.models).forEach(name => {
            const config = state.models[name];
            const card = document.createElement('div');
            card.className = 'model-settings-card';

            let formHtml = `<h3>${name} Configuration</h3>`;

            if ('threshold' in config) {
                formHtml += `
                    <div class="form-group">
                        <label>Tagging Threshold (0.0 - 1.0)</label>
                        <input type="number" step="0.05" min="0" max="1" value="${config.threshold}" id="input-${name}-threshold">
                    </div>
                `;
            }
            if ('character_threshold' in config) {
                formHtml += `
                    <div class="form-group">
                        <label>Character Threshold (0.0 - 1.0)</label>
                        <input type="number" step="0.05" min="0" max="1" value="${config.character_threshold}" id="input-${name}-char-threshold">
                    </div>
                `;
            }

            formHtml += `<button class="primary-btn" onclick="saveConfig('${name}')">Save Changes</button>`;
            card.innerHTML = formHtml;
            settingsContainer.appendChild(card);
        });
    }

    function updateConsoleOptions() {
        if (!consoleSelect) return;
        consoleSelect.innerHTML = '';
        Object.keys(state.models).forEach(name => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.innerText = name;
            consoleSelect.appendChild(opt);
        });
    }

    window.saveConfig = (modelName) => {
        const thresholdInput = document.getElementById(`input-${modelName}-threshold`);
        const charThresholdInput = document.getElementById(`input-${modelName}-char-threshold`);

        let updateData = {};
        if (thresholdInput) {
            updateData.threshold = parseFloat(thresholdInput.value);
        }
        if (charThresholdInput) {
            updateData.character_threshold = parseFloat(charThresholdInput.value);
        }

        if (Object.keys(updateData).length > 0) {
            updateModelConfig(modelName, updateData);
        }
    };

    // --- Tagger App Logic ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const preview = document.getElementById('image-preview');
    const uploadPrompt = document.getElementById('upload-prompt');
    const clearBtn = document.getElementById('clear-image-btn');
    const tagsResult = document.getElementById('tags-result');
    const tagCountSpan = document.getElementById('tag-count');
    const thresholdSlider = document.getElementById('threshold-slider');
    const thresholdDisplay = document.getElementById('threshold-val-display');
    const charThresholdSlider = document.getElementById('char-threshold-slider');
    const charThresholdDisplay = document.getElementById('char-threshold-val-display');

    if (thresholdSlider && thresholdDisplay) {
        thresholdSlider.addEventListener('input', (e) => {
            thresholdDisplay.innerText = parseFloat(e.target.value).toFixed(2);
        });
    }

    if (charThresholdSlider && charThresholdDisplay) {
        charThresholdSlider.addEventListener('input', (e) => {
            charThresholdDisplay.innerText = parseFloat(e.target.value).toFixed(2);
        });
    }

    let currentFile = null;

    if (dropZone) {
        dropZone.addEventListener('click', (e) => {
            if (e.target.tagName !== 'INPUT' && !e.target.closest('#clear-image-btn')) {
                fileInput.click();
            }
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('active');
        });

        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.classList.remove('active');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('active');
            handleFile(e.dataTransfer.files[0]);
        });
    }

    // Clear Button Logic
    if (clearBtn) {
        clearBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // Stop bubble to dropZone

            // Reset State
            currentFile = null;
            preview.src = '';
            preview.hidden = true;
            clearBtn.hidden = true;
            if (uploadPrompt) uploadPrompt.hidden = false;
            // Reset input so same file can be selected again
            if (fileInput) fileInput.value = '';
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
    }

    // Paste Support
    document.addEventListener('paste', (e) => {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const blob = items[i].getAsFile();
                handleFile(blob);
                break;
            }
        }
    });

    function handleFile(file) {
        if (!file || !file.type.startsWith('image/')) return;
        currentFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            preview.hidden = false;
            if (clearBtn) clearBtn.hidden = false;
            if (uploadPrompt) uploadPrompt.hidden = true;
        };
        reader.readAsDataURL(file);
    }

    const runBtn = document.getElementById('run-inference-btn');
    if (runBtn) {
        runBtn.addEventListener('click', async () => {
            if (!currentFile) {
                alert("Please select or paste an image first.");
                return;
            }

            tagsResult.innerHTML = '<span class="placeholder">Processing...</span>';
            tagsResult.classList.remove('empty-state');

            const reader = new FileReader();
            reader.onload = async (e) => {
                const base64Img = e.target.result;

                const selectedModel = consoleSelect ? consoleSelect.value : (Object.keys(state.models)[0] || 'joytag');
                const protocolSelect = document.getElementById('console-protocol-select');
                const protocol = protocolSelect ? protocolSelect.value : 'internal';

                const promptVal = document.getElementById('console-prompt-input')?.value || "Describe this image";
                let currentThreshold = 0.2;
                if (thresholdSlider) currentThreshold = parseFloat(thresholdSlider.value);
                if (isNaN(currentThreshold)) currentThreshold = 0.2;

                let charThreshold = 0.2;
                if (charThresholdSlider) charThreshold = parseFloat(charThresholdSlider.value);
                if (isNaN(charThreshold)) charThreshold = 0.2;

                let url = '';
                let payload = {};

                if (protocol === 'openai') {
                    url = '/v1/chat/completions';
                    payload = {
                        model: selectedModel,
                        messages: [
                            {
                                role: "user",
                                content: [
                                    { type: "text", text: promptVal },
                                    { type: "image_url", image_url: { url: base64Img } }
                                ]
                            }
                        ]
                    };
                } else if (protocol === 'ollama') {
                    const base64Raw = base64Img.split(',')[1];
                    url = '/api/generate';
                    payload = {
                        model: selectedModel,
                        prompt: promptVal,
                        images: [base64Raw],
                        stream: false
                    };
                } else if (protocol === 'internal') {
                    url = '/api/interrogate';
                    payload = {
                        model: selectedModel,
                        image: base64Img,
                        threshold: currentThreshold,
                        character_threshold: charThreshold
                    };
                    console.log("Sending payload:", payload);
                }

                try {
                    const res = await fetch(url, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });

                    const data = await res.json();

                    if (!res.ok) {
                        throw new Error(data.detail || data.error || 'Request failed');
                    }

                    let tags = [];

                    if (protocol === 'openai') {
                        if (data.choices && data.choices[0].message.content) {
                            tags = data.choices[0].message.content.split(', ').map(t => ({ name: t, score: null }));
                        }
                    } else if (protocol === 'ollama') {
                        if (data.response) {
                            tags = data.response.split(', ').map(t => ({ name: t, score: null }));
                        }
                    } else if (protocol === 'internal') {
                        if (data.tags) {
                            tags = data.tags;
                        }
                    }

                    if (tagCountSpan) tagCountSpan.innerText = tags.length;

                    if (tags.length > 0) {
                        renderTags(tags);
                    } else {
                        tagsResult.innerHTML = '<span class="placeholder">No tags found.</span>';
                    }
                } catch (err) {
                    console.error(err);
                    let msg = 'Error running inference.';
                    if (err.message) msg += ` (${err.message})`;
                    tagsResult.innerHTML = `<span class="placeholder" style="color: red">${msg}</span>`;
                }
            };
            reader.readAsDataURL(currentFile);
        });
    }

    function renderTags(tags) {
        if (!tagsResult) return;
        tagsResult.innerHTML = '';

        const tagNames = tags.map(t => t.name).join(', ');
        const hiddenArea = document.getElementById('hidden-copy-area');
        if (hiddenArea) hiddenArea.value = tagNames;

        tags.forEach(tagObj => {
            const chip = document.createElement('span');
            chip.className = 'tag-chip';

            if (tagObj.score !== null) {
                chip.innerHTML = `${tagObj.name} <span class="score-badge">(${tagObj.score.toFixed(2)})</span>`;

                const score = tagObj.score;
                chip.title = `Confidence: ${(score * 100).toFixed(1)}%`;

                let hue = Math.max(0, Math.min(120, (score - 0.2) / 0.8 * 120));
                chip.style.backgroundColor = `hsla(${hue}, 70%, 25%, 0.4)`;
                chip.style.color = `hsl(${hue}, 80%, 70%)`;
                chip.style.borderColor = `hsla(${hue}, 70%, 40%, 0.5)`;
                chip.style.borderWidth = '1px';
                chip.style.borderStyle = 'solid';
            } else {
                chip.innerText = tagObj.name;
            }

            tagsResult.appendChild(chip);
        });
    }

    const copyBtn = document.getElementById('copy-tags-btn');
    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            const textArea = document.getElementById('hidden-copy-area');
            const text = textArea ? textArea.value : '';
            if (!text) return;

            const copyToClipboard = async () => {
                try {
                    await navigator.clipboard.writeText(text);
                    return true;
                } catch (err) {
                    console.warn('Clipboard API failed, trying fallback:', err);
                    if (textArea) {
                        textArea.select();
                        textArea.setSelectionRange(0, 99999);
                        try {
                            const successful = document.execCommand('copy');
                            if (!successful) throw new Error('execCommand returned false');
                            return true;
                        } catch (err) {
                            console.error('Fallback copy failed:', err);
                            return false;
                        }
                    }
                    return false;
                }
            };

            copyToClipboard().then((success) => {
                if (success) {
                    const originalIcon = copyBtn.innerHTML;
                    copyBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
                    setTimeout(() => {
                        copyBtn.innerHTML = originalIcon;
                    }, 1500);
                } else {
                    alert('Failed to copy tags. Please copy manually.');
                }
            });
        });
    }

    fetchConfig();
});
