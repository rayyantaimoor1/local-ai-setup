# Local multi-agent AI system

Offline, zero-cost coding assistant. A small router model (`qwen2.5:3b`) picks one specialist per request; specialists load, answer and unload one at a time so everything fits in 16 GB RAM. Rebuilt from `PROJECT_HANDOFF.md`, including the queued work: sidebar with chats and projects, live streaming, and 14B escalation.

## Setup (once)
1. Install Ollama from https://ollama.com, set the user env var `OLLAMA_NUM_PARALLEL=1`, restart Ollama.
2. `setup.bat` creates `.venv` and installs `requirements.txt`.
3. `pull_models.bat` pulls the models (about 33 GB; the 14B is optional).
4. `.venv\Scripts\python.exe setup_check.py` verifies everything.
5. Optional: `powershell -ExecutionPolicy Bypass -File create_shortcut.ps1` puts a "Local AI" shortcut on the Desktop.
6. Optional: `powershell -ExecutionPolicy Bypass -File install_scheduled_task.ps1` for the daily 3 AM memory summary.

## Run
Double-click `start.bat` (or the Desktop shortcut). It starts Ollama if needed and opens http://127.0.0.1:8420.

## What is in the UI
- **Chat**: named chats that persist, grouped into projects (each chat keeps its own memory), live phase indicator and token streaming, Stop button, copy buttons, export to Markdown, search (Ctrl+K), new chat (Alt+N), light/dark theme. "Deep (14B)" skips the router and uses the 14B model (no tools).
- **Agent**: a tool-using loop. Each step it can call a tool (read a PDF/Word/text file, search files, fetch a web page, web search, git, `pip_install` from a short allow-list), write and run Python, or answer. `officekit.py` gives its scripts one-call helpers for Word, Excel, PowerPoint, PDF, charts and CSV. It checks that a requested file really exists before calling a run a success, fixes its own errors, and after 3 failures on `qwen2.5-coder:7b` hands the problem to `qwen2.5-coder:14b` (2 attempts) before stopping and saying so. Created files get download buttons.
  - **Safety:** generated code is scanned first. Running programs, deleting/moving files, network use, and paths outside the workspace (other than ones you typed in the goal) pop up an approval dialog. "Auto-approve risky code" turns that off. This is a safety net, not a sandbox: approved code runs with your user rights.
- **Tools**: git status/log/diff/pull and a reviewed, confirmed push; project file tree; web search; a memory inspector.

## The constitution and git pushes
`constitution.md` is the rulebook (Article 1: your data security comes first). It is shown to every model **and enforced in code** (`constitution.py`), so a model cannot argue its way past it. Read and manage it in the **Constitution** tab.

- **Pushing:** every push needs your click. The dialog shows the branch, remote, commits and files, then the constitution's verdict. A push is blocked if the commits (including old ones later "deleted") contain secrets (keys, tokens, passwords, private keys), include sensitive files (`.env`, keys, databases...), go to a host you have not approved (default `github.com`), target a protected branch (`main`, `master`), include a file over the size limit, or would force-push.
- **The agent** can `git_add`, `git_commit` (both refuse secrets) and `git_push_request`, which only *asks* you. It cannot push, force-push, or change the rules. Auto-approve never skips a push question.
- **Your authority:** override any rule for one push with a checkbox; core rules (secrets, sensitive files, force-push) also need you to type `OVERRIDE`. Standard rules (protected branches, hosts, file size) can also be switched off or lifted for 5 to 60 minutes. Core rules can never be switched off.
- **Audit log:** every review, block, push, approval and override goes to `audit.jsonl` with secrets scrubbed (shown in the Constitution tab).
- **Untrusted content:** text from files, PDFs, web pages and tool output is labelled as data for the models, and a web search or URL containing something that looks like a secret is refused.
- **Cross-site guard:** a website open in your browser cannot click this app's buttons for you.
- Settings live in `constitution.json`; both it, the audit log and the chat database are in `.gitignore`.
- This is a strong safety net, not a sandbox: code you approve runs with your user rights.

## Tests
`.venv\Scripts\python.exe -m pytest tests -q` (a fake Ollama is used, so no models are needed).

## Notes
- Paths are relative to the project folder, so renaming the Windows user no longer breaks anything.
- `git push` is never model-callable and needs explicit confirmation.
- The 14B needs about 9 GB. With 5 to 8 GB free expect heavy paging and a first word after one to two minutes.
- `deepseek-r1:8b` (math) reasons silently for a few minutes on CPU before it answers.
- Do not run `cmd /c start.bat` by bare name; use the full path or double-click (cmd mis-parses the bare name).
