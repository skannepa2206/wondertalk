# WonderTalk (Group 2) - Azure OpenAI Edition

This Streamlit app answers kids' questions with a friendly tone, supports microphone input, and reads responses aloud.

## Folder
Use the files in `Working_Code`.

## Prerequisites
- Python 3.9-3.11
- Azure OpenAI resource with a deployed chat model
- Internet access

## Azure OpenAI settings
Edit `.env` and replace the placeholder values.

### Option A: Foundry / Responses API (recommended for your deployment)
AZURE_OPENAI_API_MODE=responses  
AZURE_OPENAI_RESPONSES_URL=https://<your-resource-name>.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview  
AZURE_OPENAI_API_KEY=your_key_here  
AZURE_OPENAI_DEPLOYMENT=your_deployment_name  
AZURE_OPENAI_API_VERSION=2025-04-01-preview

### Option B: Azure OpenAI Chat Completions
AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com  
AZURE_OPENAI_API_KEY=your_key_here  
AZURE_OPENAI_DEPLOYMENT=your_deployment_name  
AZURE_OPENAI_API_VERSION=2024-10-21

### Optional: Multiple deployments
You can expose multiple deployments in the UI using:
- `AZURE_OPENAI_DEPLOYMENT_NAME_MAP` (label=deployment)
- `AZURE_OPENAI_DEPLOYMENT_URL_MAP` (label=full_url)

## Setup (Windows)
1. `python -m venv venv`
2. `venv\Scripts\activate`
3. `pip install -r requirements.txt`

Optional microphone support (Windows):
1. `pip install pipwin`
2. `pipwin install pyaudio`

## Run
`streamlit run app.py`

## One-command start
- Windows: run `start.ps1`
- macOS/Linux: run `start.sh`

## Deploy to Streamlit Community Cloud
1. Push this repo to GitHub.
2. In Streamlit Community Cloud, click **Create app** and select the repo/branch.
3. Set the entrypoint to `app.py`.
4. Add your secrets in **Advanced settings** (same keys as `.env`).

Example secrets:
```
AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY_HERE"
AZURE_OPENAI_API_VERSION="2025-04-01-preview"
AZURE_OPENAI_API_MODE="responses"
AZURE_OPENAI_RESPONSES_URL="https://<resource>.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview"
AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
AZURE_OPENAI_BASE_URL="https://<resource>.openai.azure.com/openai/v1"
AZURE_OPENAI_DEPLOYMENT="gpt-5.2-chat"
AZURE_OPENAI_DEPLOYMENT_NAME_MAP="gpt-5.2-chat=gpt-5.2-chat,gpt-4.1-mini=gpt-4.1-mini"
AZURE_OPENAI_DEPLOYMENT_URL_MAP="gpt-4.1-mini=https://<resource>.cognitiveservices.azure.com/openai/deployments/gpt-4.1-mini/chat/completions?api-version=2025-01-01-preview"
```

Note: Streamlit Cloud doesn’t support local microphone drivers, so `Use Mic` may be disabled there.

## Notes
- If Azure OpenAI is not configured, the app falls back to Wikipedia summaries.
- `Start Voice` reads the last answer.
- The cache is persisted under `chroma_db/`.
- Use the sidebar `Test Azure Connection` button to validate your setup.
