// Sample API implementation - this would be replaced with actual backend calls
const API_ENDPOINT = 'http://192.168.0.12:8000/api';  // Change to your actual API endpoint

let currentModel = null;
let messageCount = 1; // Start with the welcome message

// DOM elements
const modelSelect = document.getElementById('model-select');
const currentModelBadge = document.getElementById('current-model-badge');
const chatModelBadge = document.getElementById('chat-model-badge');
const refreshModelsButton = document.getElementById('refresh-models');
const loadingModels = document.getElementById('loading-models');
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const clearChatButton = document.getElementById('clear-chat');
const toolsList = document.getElementById('tools-list');
const connectionStatus = document.getElementById('connection-status');
const generateButton = document.getElementById('generate-button');

// Navigation elements
const navItems = document.querySelectorAll('.nav-item');
const pages = document.querySelectorAll('.page');

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await checkServerConnection();
    await loadModels();
    await loadTools();

    // Set up event listeners
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    refreshModelsButton.addEventListener('click', loadModels);
    clearChatButton.addEventListener('click', clearChat);
    generateButton.addEventListener('click', generateImage);

    // Navigation event listeners
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Update active class
            navItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');

            // Show selected page
            const pageId = item.getAttribute('data-page');
            pages.forEach(page => {
                page.classList.remove('active');
                if (page.id === pageId) {
                    page.classList.add('active');
                }
            });
        });
    });

    // Set focus to input when on chat page
    document.querySelector('[data-page="chat"]').addEventListener('click', () => {
        setTimeout(() => userInput.focus(), 100);
    });

    // Elements
    const generate3DButton = document.getElementById("generate-3D-button");
    const modelPrompt = document.getElementById("model-prompt");
    const guidanceScale = document.getElementById("guidance-scale");
    const guidanceValue = document.getElementById("guidance-value");
    const renderMode = document.getElementById("render-mode");
    const karrasSteps = document.getElementById("karras-steps");
    const stepsValue = document.getElementById("steps-value");
    const progressContainer = document.getElementById("progress-container");
    const progressBar = document.getElementById("progress-bar");
    const progressStatus = document.getElementById("progress-status");
    const resultsContainer = document.getElementById("results-container");
    const previewGif = document.getElementById("preview-gif");
    const downloadObj = document.getElementById("download-obj");
    const jobList = document.getElementById("job-list");

    // Current job tracking
    let currentJobId = null;
    let statusCheckInterval = null;

    // Update slider values
    guidanceScale.addEventListener("input", function () {
        guidanceValue.textContent = this.value;
    });

    karrasSteps.addEventListener("input", function () {
        stepsValue.textContent = this.value;
    });

    // Generate 3D model
    generate3DButton.addEventListener("click", async function () {
        const prompt = modelPrompt.value.trim();

        if (!prompt) {
            alert("Please enter a text prompt for your 3D model.");
            return;
        }

        // Disable button and show progress
        generate3DButton.disabled = true;
        progressContainer.style.display = "block";
        resultsContainer.style.display = "none";
        progressBar.style.width = "0%";
        progressStatus.textContent = "Starting generation...";

        try {
            // Prepare request data
            const requestData = {
                prompt: prompt,
                guidance_scale: parseFloat(guidanceScale.value),
                render_mode: renderMode.value,
                karras_steps: parseInt(karrasSteps.value),
                use_karras: true,
                batch_size: 1,
                render_size: 64
            };

            // Send the request to create a job
            const response = await fetch(`${API_ENDPOINT}/shape`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }

            const data = await response.json();
            currentJobId = data.job_id;

            // Start polling for job status
            statusCheckInterval = setInterval(checkJobStatus, 10000);

            // Add to history immediately
            addJobToHistory({
                job_id: currentJobId,
                prompt: prompt,
                status: "pending",
                created_at: Date.now() / 1000
            });

        } catch (error) {
            console.error("Error generating 3D model:", error);
            progressStatus.textContent = `Error: ${error.message}`;
            generate3DButton.disabled = false;
        }
    });

    // Check job status
    async function checkJobStatus() {
        if (!currentJobId) return;

        try {
            const response = await fetch(`${API_ENDPOINT}/shape/status/${currentJobId}`);

            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }

            const job = await response.json();

            // Update progress bar
            progressBar.style.width = `${job.progress * 100}%`;

            // Update job in history
            updateJobInHistory(job);

            // Update status message
            switch (job.status) {
                case "pending":
                    progressStatus.textContent = "Waiting to start...";
                    break;
                case "processing":
                    progressStatus.textContent = "Processing: " + getProgressMessage(job.progress);
                    break;
                case "completed":
                    progressStatus.textContent = "Completed!";
                    showResults(job);
                    clearInterval(statusCheckInterval);
                    generate3DButton.disabled = false;
                    break;
                case "failed":
                    progressStatus.textContent = `Failed: ${job.error || "Unknown error"}`;
                    clearInterval(statusCheckInterval);
                    generate3DButton.disabled = false;
                    break;
            }

        } catch (error) {
            console.error("Error checking job status:", error);
            progressStatus.textContent = `Error checking status: ${error.message}`;
            clearInterval(statusCheckInterval);
            generate3DButton.disabled = false;
        }
    }

    // Show results when job is completed
    function showResults(job) {
        resultsContainer.style.display = "block";

        // Set GIF preview
        previewGif.src = `${API_ENDPOINT}/shape/result/gif/${job.job_id}?t=${Date.now()}`; // Add timestamp to avoid caching

        // Set up download button
        downloadObj.onclick = function () {
            window.location.href = `${API_ENDPOINT}/shape/result/obj/${job.job_id}`;
        };
    }

    // Get progress message based on completion percentage
    function getProgressMessage(progress) {
        if (progress < 0.2) return "Initializing model...";
        if (progress < 0.5) return "Generating 3D structure...";
        if (progress < 0.7) return "Creating texture details...";
        if (progress < 0.9) return "Rendering preview...";
        return "Finalizing...";
    }

    // Add job to history
    function addJobToHistory(job) {
        // Check if "No recent jobs" placeholder is there
        if (jobList.querySelector("p")) {
            jobList.innerHTML = "";
        }

        const jobItem = document.createElement("div");
        jobItem.className = "job-item";
        jobItem.id = `job-${job.job_id}`;

        const jobPrompt = document.createElement("div");
        jobPrompt.className = "job-prompt";
        jobPrompt.textContent = truncateString(job.prompt, 30);

        const jobStatus = document.createElement("div");
        jobStatus.className = `job-status status-${job.status}`;
        jobStatus.textContent = capitalizeFirstLetter(job.status);

        jobItem.appendChild(jobPrompt);
        jobItem.appendChild(jobStatus);

        // Add click handler
        jobItem.addEventListener("click", function () {
            if (job.status === "completed") {
                // Open a modal or navigate to a page showing the completed job
                showResults(job);
            }
        });

        // Insert at the top
        jobList.insertBefore(jobItem, jobList.firstChild);
    }

    // Update job in history
    function updateJobInHistory(job) {
        const jobItem = document.getElementById(`job-${job.job_id}`);
        if (!jobItem) return;

        const jobStatus = jobItem.querySelector(".job-status");
        jobStatus.className = `job-status status-${job.status}`;
        jobStatus.textContent = capitalizeFirstLetter(job.status);
    }

    // Helper to truncate long strings
    function truncateString(str, maxLength) {
        if (str.length <= maxLength) return str;
        return str.substring(0, maxLength) + "...";
    }

    // Helper to capitalize first letter
    function capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }

    // Load job history on page load
    async function loadJobHistory() {
        try {
            const response = await fetch(`${API_ENDPOINT}/shape/jobs`);

            if (!response.ok) {
                console.error("Error fetching job history:", response.statusText);
                return;
            }

            const jobs = await response.json();

            if (jobs.length === 0) {
                jobList.innerHTML = "<p>No recent jobs</p>";
                return;
            }

            jobList.innerHTML = "";

            // Sort jobs by creation time (newest first)
            jobs.sort((a, b) => b.created_at - a.created_at);

            // Add each job to history
            jobs.forEach(job => {
                addJobToHistory(job);
            });

        } catch (error) {
            console.error("Error loading job history:", error);
            jobList.innerHTML = "<p>Error loading job history</p>";
        }
    }

    // Load job history when page loads
    loadJobHistory();

    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', function () {
            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked button
            button.classList.add('active');
            // Show corresponding content
            const tabId = button.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });


    const imageUploadInput = document.getElementById('image-upload');
    const selectImageButton = document.getElementById('select-image-button');
    const fileNameSpan = document.getElementById('file-name');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('convert-image-preview');
    const imageGuidanceScale = document.getElementById('image-guidance-scale');
    const imageGuidanceValue = document.getElementById('image-guidance-value');
    const imageRenderMode = document.getElementById('image-render-mode');
    const imageKarrasSteps = document.getElementById('image-karras-steps');
    const imageStepsValue = document.getElementById('image-steps-value');
    const generateImage3DButton = document.getElementById('generate-image-3D-button');
    const imageProgressContainer = document.getElementById('image-progress-container');
    const imageProgressBar = document.getElementById('image-progress-bar');
    const imageProgressStatus = document.getElementById('image-progress-status');
    const imageResultsContainer = document.getElementById('image-results-container');
    const imagePreviewGif = document.getElementById('image-preview-gif');
    const imageDownloadObjButton = document.getElementById('image-download-obj');
    const imageJobList = document.getElementById('image-job-list');

    // Image-to-3D jobs storage
    let imageJobs = [];
    let currentImageJobId = null;

    // Set up event listeners for image upload
    selectImageButton.addEventListener('click', function () {
        imageUploadInput.click();
    });

    imageUploadInput.addEventListener('change', function (e) {
        const file = e.target.files[0];
        if (file) {
            fileNameSpan.textContent = file.name;
            const reader = new FileReader();
            reader.onload = function (event) {
                imagePreview.src = event.target.result;
                imagePreviewContainer.style.display = 'block';
            };
            reader.readAsDataURL(file);
        } else {
            fileNameSpan.textContent = 'No file selected';
            imagePreviewContainer.style.display = 'none';
        }
    });

    // Update sliders value display
    imageGuidanceScale.addEventListener('input', function () {
        imageGuidanceValue.textContent = parseFloat(this.value).toFixed(1);
    });

    imageKarrasSteps.addEventListener('input', function () {
        imageStepsValue.textContent = this.value;
    });

    // Generate 3D model from image
    generateImage3DButton.addEventListener('click', function () {
        const file = imageUploadInput.files[0];
        if (!file) {
            alert('Please select an image first.');
            return;
        }

        // Show progress and hide results
        imageProgressContainer.style.display = 'block';
        imageResultsContainer.style.display = 'none';
        imageProgressBar.style.width = '0%';
        imageProgressStatus.textContent = 'Starting generation...';

        // Prepare form data for the API request
        const formData = new FormData();
        formData.append('file', file);
        formData.append('batch_size', 1);
        formData.append('guidance_scale', imageGuidanceScale.value);
        formData.append('render_mode', imageRenderMode.value);
        formData.append('render_size', 64);
        formData.append('use_karras', true);
        formData.append('karras_steps', imageKarrasSteps.value);

        // Send request to the API
        fetch(`${API_ENDPOINT}/generate-from-image`, {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                currentImageJobId = data.job_id;

                // Add job to the history
                const job = {
                    id: data.job_id,
                    type: 'image',
                    fileName: file.name,
                    status: 'pending',
                    created: new Date().toLocaleString()
                };

                imageJobs.unshift(job);
                updateImageJobList();

                // Start polling for job status
                pollImageJobStatus(data.job_id);
            })
            .catch(error => {
                console.error('Error:', error);
                imageProgressStatus.textContent = `Error: ${error.message}`;
                imageProgressBar.style.width = '0%';
            });
    });

    // Poll for job status
    function pollImageJobStatus(jobId) {
        const statusCheck = setInterval(() => {
            fetch(`${API_ENDPOINT}/shape/status/${jobId}`)
                .then(response => response.json())
                .then(data => {
                    // Update progress bar
                    const progress = Math.round(data.progress * 100);
                    imageProgressBar.style.width = `${progress}%`;

                    // Update job status in the list
                    const jobIndex = imageJobs.findIndex(job => job.id === jobId);
                    if (jobIndex !== -1) {
                        imageJobs[jobIndex].status = data.status;
                        updateImageJobList();
                    }

                    if (data.status === 'processing') {
                        imageProgressStatus.textContent = `Processing... ${progress}% complete`;
                    } else if (data.status === 'completed') {
                        imageProgressStatus.textContent = 'Model generation complete!';
                        clearInterval(statusCheck);
                        showImageResults(jobId);
                    } else if (data.status === 'failed') {
                        imageProgressStatus.textContent = `Failed: ${data.error || 'Unknown error'}`;
                        clearInterval(statusCheck);
                    }
                })
                .catch(error => {
                    console.error('Error checking status:', error);
                    clearInterval(statusCheck);
                });
        }, 2000); // Check every 2 seconds
    }

    // Show results when job is complete
    function showImageResults(jobId) {
        // Show GIF preview
        imagePreviewGif.src = `${API_ENDPOINT}/shape/result/gif/${jobId}?t=${new Date().getTime()}`;
        imageResultsContainer.style.display = 'block';

        // Set up download button
        imageDownloadObjButton.onclick = function () {
            window.location.href = `${API_ENDPOINT}/shape/result/obj/${jobId}`;
        };
    }

    // Update job history list
    function updateImageJobList() {
        if (imageJobs.length === 0) {
            imageJobList.innerHTML = '<p>No recent jobs</p>';
            return;
        }

        imageJobList.innerHTML = '';
        imageJobs.forEach(job => {
            const jobItem = document.createElement('div');
            jobItem.className = 'job-item';

            // Status badge class
            const statusClass = `status-${job.status.toLowerCase()}`;

            jobItem.innerHTML = `
                <div>
                    <div class="job-prompt">${job.fileName}</div>
                    <div class="job-time">${job.created}</div>
                </div>
                <div class="job-status ${statusClass}">${job.status}</div>
            `;

            // Make job item clickable to reload results
            jobItem.addEventListener('click', function () {
                if (job.status === 'completed') {
                    currentImageJobId = job.id;
                    showImageResults(job.id);
                    imageProgressContainer.style.display = 'none';
                }
            });

            imageJobList.appendChild(jobItem);
        });
    }

    // Initial update of the job list
    updateImageJobList();
});

async function checkServerConnection() {
    try {
        const response = await fetch(`${API_ENDPOINT}/status`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });

        if (response.ok) {
            connectionStatus.textContent = "Connected";
            connectionStatus.style.color = "#4caf50";
        } else {
            connectionStatus.textContent = "Connection issues";
            connectionStatus.style.color = "#f44336";
        }
    } catch (error) {
        connectionStatus.textContent = "Disconnected";
        connectionStatus.style.color = "#f44336";
        console.error('Connection error:', error);
    }
}

async function loadModels() {
    try {
        loadingModels.style.display = 'inline-block';

        // In a real implementation, this would fetch from your backend
        const response = await fetch(`${API_ENDPOINT}/models`);
        const data = await response.json();
        modelSelect.innerHTML = '';
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model + (model === data.default ? ' (default)' : '');
            if (model === data.default) {
                option.selected = true;
                currentModel = model;
                updateModelBadge(model);
            }
            modelSelect.appendChild(option);
        });

        modelSelect.addEventListener('change', () => {
            currentModel = modelSelect.value;
            updateModelBadge(currentModel);
            addSystemMessage(`Model changed to ${currentModel}`);
        });

    } catch (error) {
        console.error('Error loading models:', error);
        addSystemMessage('Error loading models. Please check the server connection.');
    } finally {
        loadingModels.style.display = 'none';
    }
}

function updateModelBadge(model) {
    currentModelBadge.textContent = model || 'No model selected';
    chatModelBadge.textContent = model || 'No model selected';
}

async function loadTools() {
    try {
        // In a real implementation, this would fetch from your backend
        const response = await fetch(`${API_ENDPOINT}/tools`);
        const data = await response.json();
        const toolsListTitle = [];
        const toolsHtml = data.length > 0
            ? `${data.map(tool =>
                `${(toolsListTitle.includes(tool.function.name.split('/')[0])) ? "" : (toolsListTitle.push(tool.function.name.split('/')[0]) ? `<h3>MCP Server: ${tool.function.name.split('/')[0]}</h3><ul>` : "")}` +
                `<li><strong>${tool.function.name.split('/')[1]}</strong>: ${tool.function.description}</li>`).join('')}</ul>`
            : '<p>No tools available</p>';
        toolsList.innerHTML = toolsHtml;
    } catch (error) {
        console.error('Error loading tools:', error);
        toolsList.innerHTML = '<p>Error loading tools. Please check the server connection.</p>';
    }
}

function addMessage(content, type) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${type}-message`);
    messageDiv.innerHTML = content;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addUserMessage(content) {
    addMessage(content, 'user');
}

function addAssistantMessage(content) {
    addMessage(content, 'assistant');
}

function addSystemMessage(content) {
    addMessage(`<em>${content}</em>`, 'system');
}

function addToolMessage(toolName, args, result) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('tool-message');
    messageDiv.innerHTML = `
                <div><strong>Tool:</strong> ${toolName}</div>
                <div><strong>Args:</strong> ${JSON.stringify(args)}</div>
                <div><strong>Result:</strong> ${result}</div>
            `;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    // Add user message to chat
    addUserMessage(message);

    // Clear input and update UI state
    userInput.value = '';
    userInput.disabled = true;
    sendButton.disabled = true;

    // Start loading animation
    startLoadingAnimation();

    try {
        // In a real implementation, this would connect to your backend
        const response = await fetch(`${API_ENDPOINT}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                model: currentModel
            })
        });

        if (!response.ok) {
            throw new Error(`Request failed with status ${response.status}: ${response.statusText}`);
        }

        // Handle streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let assistantResponse = '';
        let messageElement = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });

            try {
                const fixedChunk = `[${chunk.replaceAll('}{', '},{')}]`;
                const data = JSON.parse(fixedChunk);
                data.forEach((item) => {
                    if (item.role === "assistant") {
                        assistantResponse += item.content;

                        // Update message in place or create new one
                        if (!messageElement) {
                            messageElement = addAssistantMessage(assistantResponse);
                        } else {
                            updateAssistantMessage(messageElement, assistantResponse);
                        }
                    } else if (item.role === "tool") {
                        const jsonData = convertToolResponseToJSON(item.content);
                        addToolMessage(jsonData.tool.name, jsonData.args, jsonData.return);
                    }
                });

                // Auto-scroll to keep the latest message visible
                scrollToBottom();
            } catch (error) {
                console.error('Error parsing response chunk:', error, chunk);
            }
        }
    } catch (error) {
        console.error('Error sending message:', error);
        addSystemMessage(`Error: ${error.message || 'Failed to process your request. Please try again.'}`);
    } finally {
        // Re-enable input and stop loading animation
        userInput.disabled = false;
        sendButton.disabled = false;
        userInput.focus();
        stopLoadingAnimation();
    }
}

// Helper functions
function startLoadingAnimation() {
    const loadingIndicator = document.getElementById('loadingIndicator');
    // console.log('Loading animation started');
    if (!loadingIndicator) {
        const newLoadingIndicator = document.createElement('div');
        newLoadingIndicator.id = 'loadingIndicator';
        newLoadingIndicator.className = 'loading';
        newLoadingIndicator.textContent = '';
        chatMessages.appendChild(newLoadingIndicator);
    } else {
        loadingIndicator.textContent = '';
    }
    if (loadingIndicator) {
        loadingIndicator.classList.remove('hidden');
    }
}

function stopLoadingAnimation() {
    const loadingIndicator = document.getElementById('loadingIndicator');
    // console.log('Loading animation stopped');
    if (loadingIndicator) {
        loadingIndicator.textContent = 'Done!';
        loadingIndicator.remove();
        // setTimeout(() => {
        //     loadingIndicator.remove();
        // }, 500); // Remove after 0.5 second
    }
    if (loadingIndicator) {
        loadingIndicator.classList.add('hidden');
    }
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function updateAssistantMessage(element, content) {
    if (element && element.classList.contains('assistant-message')) {
        // Use markdown renderer if available
        if (typeof markdownit !== 'undefined') {
            const md = markdownit({ highlight: highlightCode });
            element.innerHTML = md.render(content);
        } else {
            element.textContent = content;
        }
    }
}

function addAssistantMessage(content) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', 'assistant-message');

    // Use markdown renderer if available
    if (typeof markdownit !== 'undefined') {
        const md = markdownit({ highlight: highlightCode });
        messageElement.innerHTML = md.render(content);
    } else {
        messageElement.textContent = content;
    }

    chatMessages.appendChild(messageElement);
    return messageElement;
}

// Optional syntax highlighting function for code blocks
function highlightCode(str, lang) {
    if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
        try {
            return hljs.highlight(str, { language: lang }).value;
        } catch (__) { }
    }
    return ''; // Use default escaping
}

function convertToolResponseToJSON(responseString) {
    // Regular expressions to extract the different parts
    const toolNameMatch = responseString.match(/name='([^']+)'/);
    const argumentsMatch = responseString.match(/arguments=({[^}]+})/);
    const argsMatch = responseString.match(/args: ({[^}]+})/);
    const returnMatch = responseString.match(/return: (.+)$/);

    // Parse the arguments and args JSON strings
    let argumentsObj = {};
    if (argumentsMatch && argumentsMatch[1]) {
        // Convert single quotes to double quotes for JSON parsing
        const jsonStr = argumentsMatch[1].replace(/'/g, '"');
        try {
            argumentsObj = JSON.parse(jsonStr);
        } catch (e) {
            console.error("Failed to parse arguments:", e);
        }
    }

    let argsObj = {};
    if (argsMatch && argsMatch[1]) {
        // Convert single quotes to double quotes for JSON parsing
        const jsonStr = argsMatch[1].replace(/'/g, '"');
        try {
            argsObj = JSON.parse(jsonStr);
        } catch (e) {
            console.error("Failed to parse args:", e);
        }
    }

    // Build the result object
    return {
        tool: {
            name: toolNameMatch ? toolNameMatch[1] : "",
            arguments: argumentsObj
        },
        args: argsObj,
        return: returnMatch ? returnMatch[1] : ""
    };
}

function clearChat() {
    // Keep only the first welcome message
    while (chatMessages.childNodes.length > 1) {
        chatMessages.removeChild(chatMessages.lastChild);
    }
    messageCount = 1;
}

async function generateImage() {
    const prompt = document.getElementById('prompt').value.trim();
    const model = document.getElementById('image-model').value;
    const preview = document.getElementById('image-sd-preview');

    if (!prompt) {
        preview.innerHTML = '<p style="color: red;">Please enter a prompt</p>';
        return;
    }

    preview.innerHTML = '<p>Generating image...</p>';
    generateButton.disabled = true;

    try {
        // This would be an actual API call in a real implementation
        setTimeout(() => {
            // Simulate image generation
            preview.innerHTML = `
                        <div style="text-align: center;">
                            <div style="background-color: #ccc; width: 400px; height: 400px; display: inline-flex; justify-content: center; align-items: center;">
                                <p>Generated image based on: "${prompt}"<br>Using model: ${model}</p>
                            </div>
                        </div>
                    `;
            generateButton.disabled = false;
        }, 2000);
    } catch (error) {
        preview.innerHTML = '<p style="color: red;">Error generating image</p>';
        generateButton.disabled = false;
    }
}
document.querySelectorAll('.tab-image').forEach(tab => {
    tab.addEventListener('click', () => {
        // Remove active class from all tabs and contents
        document.querySelectorAll('.tab-image').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-image-content').forEach(c => c.classList.remove('active'));

        // Add active class to clicked tab
        tab.classList.add('active');

        // Show corresponding content
        const tabId = tab.getAttribute('data-tab');
        document.getElementById(tabId + '-content').classList.add('active');
    });
});
// Update range input displays
document.getElementById('num-steps').addEventListener('input', function () {
    document.getElementById('steps-value').textContent = this.value;
});

document.getElementById('img2img-steps').addEventListener('input', function () {
    document.getElementById('img2img-steps-value').textContent = this.value;
});

document.getElementById('strength').addEventListener('input', function () {
    document.getElementById('strength-value').textContent = this.value;
});

// Image upload handling
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('image-sd-upload');

uploadArea.addEventListener('click', () => {
    fileInput.click();
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#3498db';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = '#ddd';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#ddd';

    if (e.dataTransfer.files.length) {
        handleFileUpload(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
        handleFileUpload(fileInput.files[0]);
    }
});

function handleFileUpload(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('source-image').src = e.target.result;
        document.getElementById('uploaded-image-preview').style.display = 'block';
        document.getElementById('transform-button').disabled = false;
    };
    reader.readAsDataURL(file);
}

// Text to Image generation
document.getElementById('generate-button').addEventListener('click', async () => {
    const prompt = document.getElementById('prompt').value.trim();
    if (!prompt) {
        alert('Please enter a prompt');
        return;
    }

    const model = document.getElementById('image-model').value;
    const steps = document.getElementById('num-steps').value;

    // Show loading state
    document.getElementById('generate-button').disabled = true;
    document.getElementById('status-message').style.display = 'block';
    document.getElementById('status-text').textContent = 'Starting image generation...';

    try {
        // Make API request to start the job
        const response = await fetch(`${API_ENDPOINT}/text-to-image`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                model: model,
                steps: parseInt(steps)
            })
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }

        const data = await response.json();
        const jobId = data.job_id;

        // Poll for job status
        pollJobStatus(jobId);

    } catch (error) {
        console.error('Error generating image:', error);
        document.getElementById('status-text').textContent = `Error: ${error.message}`;
        document.getElementById('generate-button').disabled = false;
    }
});

// Image to Image transformation
document.getElementById('transform-button').addEventListener('click', async () => {
    const prompt = document.getElementById('img2img-prompt').value.trim();
    if (!prompt) {
        alert('Please enter a prompt');
        return;
    }

    const sourceImage = document.getElementById('source-image').src;
    const strength = document.getElementById('strength').value;
    const steps = document.getElementById('img2img-steps').value;

    // Show loading state
    document.getElementById('transform-button').disabled = true;
    document.getElementById('img2img-status').style.display = 'block';
    document.getElementById('img2img-status-text').textContent = 'Starting image transformation...';

    try {
        if (!fileInput) {
            alert('Please select an image first.');
            return;
        }

        const sdformData = new FormData();
        sdformData.append('file', fileInput.files[0]); // Ensure this is a valid File object
        sdformData.append('prompt', prompt);
        sdformData.append('num_inference_steps', steps); // Convert to string if not already
        sdformData.append('width', 512);
        sdformData.append('height', 512);
        sdformData.append('model_id', 'stabilityai/sd-turbo');
        sdformData.append('strength', 0.5);
        sdformData.append('guidance_scale', 2.0); // Match the default in your backend
        // Make API request to start the job
        const response = await fetch(`${API_ENDPOINT}/image-to-image`, {
            method: 'POST',
            body: sdformData
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }

        const data = await response.json();
        const jobId = data.job_id;

        // Poll for job status
        pollImg2ImgJobStatus(jobId);

    } catch (error) {
        console.error('Error transforming image:', error);
        document.getElementById('img2img-status-text').textContent = `Error: ${error.message}`;
        document.getElementById('transform-button').disabled = false;
    }
});

// Function to poll job status for text-to-image
async function pollJobStatus(jobId) {
    try {
        const response = await fetch(`${API_ENDPOINT}/sd/status/${jobId}`);
        if (!response.ok) {
            throw new Error(`Error checking status: ${response.statusText}`);
        }

        const data = await response.json();
        document.getElementById('status-text').textContent = `Status: ${data.status} - ${data.progress * 100 || ''}`;

        if (data.status === 'completed') {
            // Job is done, display the image
            const imageUrl = `${API_ENDPOINT}/sd/result/${jobId}`;
            const imgElement = document.createElement('img');
            imgElement.src = imageUrl;

            const preview = document.getElementById('image-sd-preview');
            preview.innerHTML = '';
            preview.appendChild(imgElement);

            document.getElementById('status-message').style.display = 'none';
            document.getElementById('generate-button').disabled = false;

        } else if (data.status === 'failed') {
            // Job failed
            document.getElementById('status-text').textContent = `Error: ${data.message || 'Generation failed'}`;
            document.getElementById('generate-button').disabled = false;

        } else {
            // Job still in progress, check again in 2 seconds
            setTimeout(() => pollJobStatus(jobId), 2000);
        }

    } catch (error) {
        console.error('Error polling job status:', error);
        document.getElementById('status-text').textContent = `Error checking status: ${error.message}`;
        document.getElementById('generate-button').disabled = false;
    }
}

// Function to poll job status for image-to-image
async function pollImg2ImgJobStatus(jobId) {
    try {
        const response = await fetch(`${API_ENDPOINT}/sd/status/${jobId}`);
        if (!response.ok) {
            throw new Error(`Error checking status: ${response.statusText}`);
        }

        const data = await response.json();
        document.getElementById('img2img-status-text').textContent = `Status: ${data.status} - ${data.progress * 100 || ''}`;

        if (data.status === 'completed') {
            // Job is done, display the image
            const imageUrl = `${API_ENDPOINT}/sd/result/${jobId}`;
            const imgElement = document.createElement('img');
            imgElement.src = imageUrl;

            const preview = document.getElementById('img2img-preview');
            preview.innerHTML = '';
            preview.appendChild(imgElement);

            document.getElementById('img2img-status').style.display = 'none';
            document.getElementById('transform-button').disabled = false;

        } else if (data.status === 'failed') {
            // Job failed
            document.getElementById('img2img-status-text').textContent = `Error: ${data.message || 'Transformation failed'}`;
            document.getElementById('transform-button').disabled = false;

        } else {
            // Job still in progress, check again in 2 seconds
            setTimeout(() => pollImg2ImgJobStatus(jobId), 2000);
        }

    } catch (error) {
        console.error('Error polling job status:', error);
        document.getElementById('img2img-status-text').textContent = `Error checking status: ${error.message}`;
        document.getElementById('transform-button').disabled = false;
    }
}