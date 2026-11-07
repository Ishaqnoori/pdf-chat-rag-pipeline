# Security checklist

How Ask your PDF handles the usual pre-launch security checklist.
"N/A" means the app has no component that the item applies to.

| # | Item | Status | How |
|---|---|---|---|
| 1 | Hide API keys | ✅ | Keys live in `.env` (local) or the host's Secrets. Never in code, logs or error messages. |
| 2 | Purge Git secrets | ✅ | `.env`, `secrets.toml`, `*.db` are git-ignored. `update_github.ps1` refuses to push files containing key patterns. History checked clean. |
| 3 | Use public DB key | ✅ | Only Firebase's public **Web API key** is used (no admin/service-account key). Restrict it to the Identity Toolkit API in Google Cloud. |
| 4 | Row-level security | N/A | No database of our own. Accounts are in Firebase Auth. Add Firestore rules first if chat history is ever stored. |
| 5 | Encrypt sensitive data | ✅ | Passwords are hashed by Firebase. PDFs and chats are never stored; all traffic is HTTPS. |
| 6 | Server-side authentication | ✅ | The login gate runs in Python on the server before any content renders (`st.stop()`). Google sign-in uses Streamlit's built-in OIDC flow with a signed cookie; only Google-verified emails are accepted. |
| 7 | Lock record access | ✅ | Each session has its own document index and chat; nothing is written to disk or shared between users. |
| 8 | Block field tampering | ✅ | Only `email`, `password` and `displayName` are ever sent to Firebase; the user can't set any other field. |
| 9 | Secure sessions | ✅ | Session state lives on the server, XSRF protection on, 30-minute idle logout. The Google login cookie is HTTP-only and signed with a 32+ character `cookie_secret`; logging out clears it. |
| 10 | Hash passwords | ✅ | Handled by Firebase Auth (scrypt). The app never stores a password. |
| 11 | Rate-limit login | ✅ | 5 failed logins per email and 20 per IP per 15 min, server-wide, on top of Firebase's own throttling. Usage is capped too: per-user and app-wide question/upload limits per minute and per day (see README → Usage limits). |
| 12 | Bot protection | ✅ | Email verification required before first login; 3 sign-ups per IP per hour; 3 emails per address per hour. |
| 13 | Parameterize queries | N/A | No SQL in the app. |
| 14 | Validate all input | ✅ | Length limits and format checks on name, email, password, questions (4,000 chars) and file names. |
| 15 | Escape user content | ✅ | User messages are HTML-escaped; model answers have raw HTML, remote images and disguised links removed (prompt-injection defence). |
| 16 | Restrict file uploads | ✅ | PDF only, checked by file signature, 20 MB, 300 pages, no encrypted PDFs, 10 uploads per hour. Scanned PDFs: 50 pages / 14 MB, 5 per user per day. Temp file deleted at once. |
| 17 | Trim API responses | ✅ | Visitors see short, generic errors; details go to the server log only. Tracebacks hidden (`showErrorDetails = "none"`). |
| 18 | Security headers | Host | Streamlit can't set custom headers. Streamlit Community Cloud adds its own; if self-hosting, put it behind Caddy/nginx with HSTS, CSP and `X-Frame-Options`. |
| 19 | Force HTTPS | Host | Streamlit Community Cloud serves HTTPS only. All calls to Gemini, Groq and Firebase use HTTPS. |
| 20 | Scan dependencies | ✅ | `pip-audit -r requirements.txt` before launch; Dependabot (`.github/dependabot.yml`) opens update PRs weekly. |

## Firebase console settings to check

- **Authentication → Settings → User actions**: keep *Email enumeration protection* on.
- **Authentication → Settings → Password policy**: require at least 8 characters.
- **Google Cloud → APIs & Services → Credentials → Browser key (auto created by Firebase)**:
  under *API restrictions*, allow only **Identity Toolkit API**.

## Reporting a problem

Please open a private security advisory on GitHub rather than a public issue.
