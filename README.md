# arxiv-ai-research-tracker

A research assistant that fetches the latest AI research papers published in arXiv, summerizes them, enables follow up Q&A,
and helps you curate a personal collection of research papers for future reference.

Additional enhancements in progress.

---

## 🚀 Features

- 🔍 Fetch the 10 latest AI papers from arXiv
- 🧠 Summarize each paper in 3 lines using OpenAI GPT
- 🤖 Ask follow-up questions with a memory-enabled agent (LangChain)
- 💾 Save/delete interesting papers to a separate vector DB for future reference
- ⚡ Avoids re-processing previously fetched papers

---

## 📦 Setup

```bash
git clone https://github.com/yourusername/ai-paper-explorer.git
cd ai-paper-explorer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 🔑 Add your OpenAI key
Create a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 🧪 Usage

```bash
python main.py
```

Then interact with the agent:
- Ask: "What’s new in LLMs?"
- Command: "Add XYZ paper to saved"
- Query: "Show me saved papers about reinforcement learning"

---

## Gett OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create an account or sign in
3. Go to API Keys section
4. Create a new secret key
5. Copy the key to your `.env` file


## 🚀 Future Enhancements

- [ ] Web interface with Streamlit/Gradio
- [ ] Export functionality (PDF, Word, LaTeX)
- [ ] Auto-tag papers ("LLM", "Vision", "RL")
- [ ] Semantic clustering for trends
