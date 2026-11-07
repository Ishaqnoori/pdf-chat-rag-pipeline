# Security checklist

Each item says what protects it, where it lives in the code, and how it's verified.

| # | Item | Status | How | Where / verified by |
|---|---|---|---|---|
| 1 | Hide API keys | ✅ | Keys come only from `.env` (local) or the host's Secrets, never from code. The OCR key is sent in a header, not a URL. Errors never show key values. | `app.py` config, `utils/ocr.py` · `test_api_key_goes_in_header_not_url` |
| 2 | Purge Git secrets | ✅ | `.env`, `secrets.toml` and service-account keys are git-ignored; `update_github.ps1` refuses to push files with key patterns; GitHub Actions runs **gitleaks** over the full history on every push. | `.gitignore`, `update_github.ps1`, `.github/workflows/tests.yml` (`secrets` job) |
| 3 | Use public DB key | ✅ | Browsers only ever see Firebase's public web API key. The admin service-account key stays on the server. | `utils/firebase_auth.py`, `utils/store.py` |
| 4 | Row-level security | ✅ | Firestore rules deny **all** browser access (`firestore.rules`); only the server reads/writes, and every path is scoped to the signed-in user (`users/{uid}/…`). | `firestore.rules`, `utils/store.py` |
| 5 | Encrypt sensitive data | ✅ | Passwords hashed by Firebase; data encrypted at rest by Google; HTTPS everywhere; PDFs never stored. | Firebase / Firestore, `rag_pipeline.py` (in-memory index) |
| 6 | Server-side authentication | ✅ | The login gate runs on the server before anything renders (`st.stop()`); the Admin page re-checks `ADMIN_EMAILS` on the server. | `app.py` login gate, `views/settings.py` `render_admin` · simulated page loads |
| 7 | Lock record access | ✅ | The user id comes from the server-side session, never from the browser; users can only load their own chats. | `app.py` `STORE_UID`, `utils/store.py` |
| 8 | Block field tampering | ✅ | Only whitelisted fields are written (messages: role, content, attachments, sources, kind, model, seconds, rating; Firebase: email, password, display name). | `utils/store.py` `_clean_message`, `utils/firebase_auth.py` |
| 9 | Secure session cookies | ✅ | *Keep me signed in* is **opt-in** and lasts 3 days (`SESSION_DAYS`). The cookie is HMAC-signed, bound to the browser (a copy used elsewhere is rejected), `SameSite=Strict`, `Secure` on HTTPS, and revoked server-side on logout; Admin → *Sign everyone out* rotates the secret. Streamlit can't set cookies from the server, so it isn't `HttpOnly` (see *Accepted risks*). | `utils/session.py` · `test_cookie_copied_to_another_browser_is_rejected`, `test_signing_everyone_out_invalidates_old_tokens`, `test_tampered_token_is_rejected` |
| 10 | Hash passwords | ✅ | Firebase Authentication (scrypt); the app never stores passwords. | `utils/firebase_auth.py` |
| 11 | Rate-limit login | ✅ | 5 failed logins per email and 20 per network per 15 min, saved across restarts. The network key is the address added by **our own** proxy (right-most `X-Forwarded-For` entry, `TRUSTED_PROXY_HOPS`), so visitors can't spoof it. Refunds remove exactly their own event (receipts). | `utils/security.py` `client_ip`, `take`/`undo` · `test_refund_removes_exactly_its_own_event` |
| 12 | Bot protection | ✅ | Email verification before first login; sign-up honeypot field and minimum fill time; 3 sign-ups per network per hour; 3 emails per address per hour. | `utils/auth.py` `sign_up`, `views/login.py` · `test_honeypot_blocks_bots`, `test_instant_submissions_are_refused` |
| 13 | Parameterize queries | ✅ | No SQL anywhere; Firestore is accessed through its typed SDK (no query strings built from input). | `utils/store.py` |
| 14 | Validate all input | ✅ | Length/format checks on name, email, password, questions (4,000 chars), chat titles and file names; control characters stripped. | `utils/security.py` `clean_text`, `valid_email` · `test_clean_text_removes_control_chars_and_trims` |
| 15 | Escape user content | ✅ | User messages are HTML-escaped. Model answers have raw HTML and remote images removed, and every link turned into plain text (`javascript:`/`data:` links lose their address) — a defence against prompt injection from PDFs. | `utils/security.py` `sanitize_markdown` · `test_dangerous_link_schemes_become_plain_text` |
| 16 | Restrict file uploads | ✅ | PDF only, checked by file signature (not name); 20 MB; 300 pages; no encrypted PDFs; scans 50 pages / 14 MB; upload limits per hour/day. Temp files deleted immediately. | `utils/loaders.py`, `.streamlit/config.toml` |
| 17 | Trim API responses | ✅ | Visitors see short, generic messages; details go to the server log (and Sentry, scrubbed). Tracebacks hidden. | `.streamlit/config.toml` `showErrorDetails`, `utils/monitoring.py` · `test_sentry_events_are_scrubbed` |
| 18 | Security headers | ✅ | Self-hosting: Caddy enforces HSTS, `nosniff`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy` and a basic CSP, plus a **full CSP in report-only mode** to enforce after testing. On Streamlit Community Cloud the platform controls headers. | `deploy/Caddyfile` |
| 19 | Force HTTPS | ✅ | Streamlit Community Cloud is HTTPS-only; self-hosting, Caddy gets a free certificate and redirects HTTP to HTTPS. All API calls use HTTPS. | `deploy/Caddyfile`, `deploy/docker-compose.yml` |
| 20 | Scan dependencies | ✅ | `pip-audit` **blocks** the build on known vulnerabilities; Dependabot opens weekly update PRs; versions capped below the next major release. | `.github/workflows/tests.yml`, `.github/dependabot.yml`, `requirements.txt` |

## Terms of Service and privacy

- Sign-up requires ticking *I agree to the Terms of Service and Privacy Policy*.
- Every user (including Google sign-ins) accepts the current Terms once before using the app; the accepted
  version is stored in Firestore, and changing `TERMS_VERSION` in `utils/legal.py` asks everyone again.
- Settings → Account: **Export my data** and **Delete account** (password check; removes the Firebase
  account, chats, sessions and feedback).

## Accepted risks (known, limited, documented)

- **Sign-in cookie isn't `HttpOnly`.** Streamlit has no server hook for setting cookies. Mitigations: opt-in only,
  3-day lifetime, bound to the browser, revocable per session and globally, and no route for scripts into the
  page (user text escaped, model output sanitised, links neutralised). To remove the risk entirely, set
  `SESSION_DAYS=0`, which turns the option off; Google sign-in uses Streamlit's own `HttpOnly` cookie.
- **Usage counters live in memory, saved every ≤20 s.** A crash can lose up to 20 s of counts, and several
  server instances would each count separately. Streamlit Community Cloud runs one instance. An atomic
  database write per question would use up Firestore's free write quota at about 3,000 questions a day.
- **PDF parsing runs in the app process.** Mitigated by 20 MB / 300-page caps and per-user upload limits; an
  isolated worker process would be the full fix when moving to paid hosting.

## If a secret leaks (rotation checklist)

1. **Firebase service-account key:** create a new least-privilege key (below), then Google Cloud → IAM & Admin
   → Service accounts → old account → **Keys** → delete the old key.
2. **Google sign-in client secret:** Google Auth Platform → Clients → your client → **Add secret**, run
   `setup_google_login.bat` with it, then disable and delete the old secret. Also replace `cookie_secret`.
3. **Sign-in cookies:** Settings → Admin → **Sign everyone out**.
4. **API keys (Gemini, Groq, Firebase web):** create new ones in each console, update `.env` / Secrets,
   delete the old ones.
5. **Check for misuse:** Google Cloud → **Logging → Logs Explorer** (Firestore and IAM activity), Firebase →
   Authentication → Users (unexpected accounts), and the Groq and Gemini usage pages.

## Least-privilege database key

The default *Firebase Admin SDK* key can manage your whole Firebase project. The app only needs Firestore:

1. Google Cloud → IAM & Admin → **Service accounts** → **Create service account**, name `ask-your-pdf-app`.
2. Grant **one** role: **Cloud Datastore User** (read/write Firestore data, nothing else). Done.
3. Open it → **Keys** → **Add key → Create new key → JSON**. Save it as `firebase-service-account.json`.
4. Delete the old Admin SDK key (Service accounts → firebase-adminsdk-… → Keys).

## One-time console steps

- **Firestore → Rules:** paste `firestore.rules` and click **Publish** (production mode already denies
  browsers; this makes the rule explicit and version-controlled).
- **Authentication → Settings → Password policy:** minimum 8 characters.
- **Google Cloud → APIs & Services → Credentials → Browser key (auto created by Firebase):** restrict to
  **Identity Toolkit API**.

## Reporting a problem

Please open a private security advisory on GitHub rather than a public issue.
