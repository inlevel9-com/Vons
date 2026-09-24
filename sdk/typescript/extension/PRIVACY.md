# Vons — Local Decisions: privacy notice

Updated September 25, 2026. Applies to extension version 0.1.0.

Vons is provided by INLEVEL9. Contact: **oswarld@inlevel9.com**.

## Data processed on your device

The extension processes only the context, question, candidate choices and model
files you explicitly provide. Model inference runs locally using the packaged
ONNX Runtime Web implementation. Vons does not upload this data, run analytics,
send telemetry, require an account or download models from a server.

The extension has no access to page contents, browsing history, cookies or
account credentials. The only extension permission is `sidePanel`, which
displays its interface next to a browser tab. Local file access comes from the
folder picker you open. The extension does not execute proposed actions.

## Storage and deletion

Context, questions, candidates and results remain in the open panel's memory;
they are not written to persistent extension storage. Closing the panel ends
that session. Clicking **Copy JSON** writes the displayed response, including
candidate text, to your system clipboard at your request.

If you enable **Keep this model on this device**, the required model assets and
manifest are stored in local IndexedDB. They are not synchronized with Chrome
Sync. **Remove model** deletes that retained model. Turning off the retention
option also deletes the saved copy while allowing the current in-memory session
to continue. These actions do not delete your original files.

## Sharing and external services

The form handles the text you type or paste locally; this includes context,
questions and candidates. The Chrome Web Store disclosure identifies this as
user-provided content even though no browsing content is read automatically.
Vons does not transmit or sell your inputs, model files or results to
INLEVEL9, advertisers or other third parties. Research and guide links open
Hugging Face only when you click them. No entered context, candidates or model
contents are added to these links. Those websites apply their own privacy
policies; Chrome and the Chrome Web Store also apply their separate policies.

If you contact support by email, the information you choose to send is used to
respond to that request. Do not include sensitive prompts, credentials or model
files in support messages.

## Research limitations

The extension is a research preview. Model files must be supplied separately.
Uncalibrated scores and proposals do not establish correctness or permission to
act. You retain control of any subsequent action. Software and model licensing
are separate from this privacy notice.
