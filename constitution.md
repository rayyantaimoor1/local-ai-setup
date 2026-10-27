# Constitution

These rules govern every model and tool in this system. They are written for you to read and **enforced in code**, so a model cannot talk its way past them. They are deliberately *not* put into the models' prompts: small models read them as "refuse things" and stopped helping with ordinary requests.

**Order of priority:** 1. the security of your data, 2. your authority, 3. honesty, 4. safe actions, 5. being helpful. When two conflict, the lower number wins.

## Article 1. Your data is protected above everything

1.1 Your files, chats, memory and credentials stay on this computer. They leave only when you deliberately send them.

1.2 Web search and page fetching go through built-in tools, which refuse anything that looks like a secret.

1.3 Secrets (passwords, API keys, tokens, private keys) are never read, printed, copied into documents, searched for on the web, or pushed. Files likely to hold secrets are off limits.

1.4 Documents the assistant creates are saved only in the `outputs` folder, and never overwrite an earlier one.

## Article 2. Pushing to git

2.1 **Every push needs your click.** Nothing is pushed automatically.

2.2 Before you click you see the branch, the remote, the commits and the changed files.

2.3 A push is blocked when it contains secrets or sensitive files, goes to a host you have not approved, targets a protected branch (main, master by default), includes a huge file, or would rewrite remote history.

2.4 The assistant may stage files, commit, and *ask* you to push. It can never force-push, and staging and committing refuse secrets and sensitive files.

## Article 3. Safe actions

3.1 The assistant never runs programs it writes and never deletes files. It reads with read-only tools and creates new files only through the document tools.

## Article 4. Honesty

4.1 Failures are reported plainly. A file is only reported as created when it exists.

4.2 Every approval, override, block, push and created document is written to the audit log (`audit.jsonl`), without secrets.

## Article 5. Untrusted content

5.1 Text inside files, PDFs, web pages, search results and tool output is **data, not instructions**. It can never grant permission, change these rules, or ask for secrets.

## Article 6. Your authority

6.1 Only you can override a rule, by clicking in this app. Models, files and web pages cannot.

6.2 **Core rules** (Article 1, and the secrets / sensitive-files / force-push rules in 2.3) can be overridden one action at a time, after a clear warning. Overriding one in a push also needs you to type OVERRIDE. They cannot be switched off.

6.3 **Standard rules** (protected branches, approved hosts, file size) can be overridden one action at a time, switched off in the Constitution tab, or lifted for a limited time (up to 60 minutes).

6.4 **You can lift the whole constitution** for up to 60 minutes with one button. While it is lifted no rule applies, core rules included: secrets can be pushed, read and sent, and sensitive files are not skipped. A banner shows the time left and lets you restore it early. Even then, a push still waits for your click, because that is how the app asks you before acting for you.

6.5 Every override and lift is logged. Timed overrides and lifts end by themselves and when the app restarts.
