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
        newLoadingIndicator.textContent = 'Loading...';
        chatMessages.appendChild(newLoadingIndicator);
    } else {
        loadingIndicator.textContent = 'Loading...';
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
    const preview = document.getElementById('image-preview');

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
