document.addEventListener('DOMContentLoaded', () => {
    // Navigation (Tabs Removed for Single Page)


    // Theme Toggle
    const themeToggleBtn = document.getElementById('theme-toggle');

    // Check saved theme
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.body.setAttribute('data-theme', savedTheme);

    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = document.body.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        document.body.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    });

    // App State
    const state = {
        models: {} // { modelName: config }
    };

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
        settingsContainer.innerHTML = '';
        Object.keys(state.models).forEach(name => {
            const config = state.models[name];

            const card = document.createElement('div');
            card.className = 'model-settings-card';

            // Build form dynamically based on config keys
            let formHtml = `<h3>${name} Configuration</h3>`;

            // Currently only threshold is supported, but loop allows future expansion
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
        consoleSelect.innerHTML = '';
        Object.keys(state.models).forEach(name => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.innerText = name;
            consoleSelect.appendChild(opt);
        });
    }

    // Expose save function globally for inline onclick
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

    // --- Test Console Logic ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const preview = document.getElementById('image-preview');
    const tagsResult = document.getElementById('tags-result');
    let currentFile = null;

    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = 'var(--accent)'; });
    dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); dropZone.style.borderColor = 'var(--border)'; });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border)';
        handleFile(e.dataTransfer.files[0]);
    });

    function handleFile(file) {
        if (!file || !file.type.startsWith('image/')) return;
        currentFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            preview.hidden = false;
        };
        reader.readAsDataURL(file);
    }

    document.getElementById('run-inference-btn').addEventListener('click', async () => {
        if (!currentFile) {
            alert("Please select an image first.");
            return;
        }

        tagsResult.innerHTML = '<span class="placeholder">Processing...</span>';

        // Convert to base64
        const reader = new FileReader();
        reader.onload = async (e) => {
            const base64Img = e.target.result; // includes data:image/...;base64,

            const selectedModel = consoleSelect.value;
            const protocol = document.getElementById('console-protocol-select').value;
            const promptVal = document.getElementById('console-prompt-input').value || "Describe this image";

            let url = '/v1/chat/completions';
            let payload = {};

            if (protocol === 'openai') {
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
                // Ollama Generate API expects raw base64 without header usually, but let's see. 
                // Our server implementation uses decode_base64_image which handles data: uri or raw.
                // Standard Ollama expects raw base64. 
                // Let's strip the header just in case our backend handles it (it does).
                const base64Raw = base64Img.split(',')[1];

                url = '/api/generate';
                payload = {
                    model: selectedModel,
                    prompt: promptVal,
                    images: [base64Raw],
                    stream: false
                };
            }

            try {
                const res = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await res.json();

                let tags = [];
                if (protocol === 'openai') {
                    if (data.choices && data.choices[0].message.content) {
                        tags = data.choices[0].message.content.split(', ');
                    }
                } else if (protocol === 'ollama') {
                    if (data.response) {
                        tags = data.response.split(', ');
                    }
                }

                if (tags.length > 0) {
                    renderTags(tags);
                } else {
                    tagsResult.innerHTML = '<span class="placeholder">No tags found or error.</span>';
                }
            } catch (err) {
                console.error(err);
                tagsResult.innerHTML = '<span class="placeholder" style="color: red">Error running inference.</span>';
            }
        };
        reader.readAsDataURL(currentFile);
    });

    function renderTags(tags) {
        tagsResult.innerHTML = '';
        tags.forEach(tag => {
            const chip = document.createElement('span');
            chip.className = 'tag-chip';
            chip.innerText = tag;
            tagsResult.appendChild(chip);
        });
    }

    // Init
    fetchConfig();
});
