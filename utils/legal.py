"""Privacy policy and terms shown in the app. Written to match what the code actually does."""
import os

UPDATED = "November 2026"
TERMS_VERSION = "2026-11"  # change this when the terms change: everyone is asked to accept again


def contact() -> str:
    explicit = os.getenv("CONTACT_EMAIL", "").strip()
    admins = [e.strip() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()]
    return explicit or (admins[0] if admins else "the app owner")


def privacy() -> str:
    return f"""
### Privacy policy
*Last updated {UPDATED}*

**What we store**
- **Your account:** name, email and a hashed password (handled by Firebase Authentication), or your Google
  name and email if you sign in with Google. We never see or store your password.
- **Your chats:** the questions you ask and the answers you get, saved to your account so you can come back
  to them. Stored in Google Cloud Firestore (Mumbai region), encrypted at rest.
- **Usage counters:** how many questions and uploads you've made recently, to apply fair-use limits.
- **Feedback:** when you press 👍 or 👎, that answer and its question.

**What we don't store**
- **Your PDFs.** They're processed in memory for your session and the temporary copy is deleted immediately.
  Reopening a saved chat asks you to attach the PDF again.
- No advertising trackers, and we don't sell or share your data.

**Services that process your data**
- **Google Gemini:** turns your PDF's text into search vectors, reads scanned PDFs, and generates answers
  when you pick a Gemini model.
- **Groq:** generates answers when you pick a GPT-OSS or Qwen model.
- To answer, the chosen model receives your question and the relevant parts of your PDF. We use the free
  tiers of these services; **Google may use free-tier requests to improve its products**, so avoid
  uploading highly confidential documents.
- **Google Firebase / Firestore:** sign-in and saved chats. **Streamlit Community Cloud:** hosting.
- **Sentry** (if enabled): error reports, with personal data removed.
Their own privacy policies apply to what they process.

**Your rights** (including under India's Digital Personal Data Protection Act, 2023)
- **Access:** Settings → Account → *Export my data* downloads your saved chats.
- **Deletion:** Settings → Account → *Delete account* permanently removes your account, chats and feedback.
- **Questions or complaints:** contact {contact()}.

**Security:** HTTPS everywhere, server-side access checks, rate limits, and a login cookie that expires
after 7 days and is revoked when you log out.
"""


def terms() -> str:
    return f"""
### Terms of Service
*Version {TERMS_VERSION}, last updated {UPDATED}*

**1. Agreement.** By creating an account or using Ask your PDF ("the service"), you agree to these Terms
and to the Privacy Policy. If you don't agree, don't use the service.

**2. Who can use it.** You must be at least 18, or have a parent or guardian's permission to use the
service and agree to these Terms for you. You're responsible for everything done with your account,
so keep your password private.

**3. What the service does.** You upload PDFs and ask questions; AI models answer using the text of your
documents. The service is offered free of charge, as is, and may change, be limited or stop at any time.

**4. AI answers can be wrong.** Answers can contain mistakes, especially for scanned PDFs. Always check
important facts against the original document. Don't rely on the service for legal, medical, financial
or safety decisions.

**5. Your content.** You keep all rights to the documents you upload and the questions you ask. You
give us permission to process them only to provide the service (see the Privacy Policy). PDFs are not
stored. Only upload documents you have the right to use, and don't upload other people's personal or
confidential information without their permission.

**6. Acceptable use.** Don't: use the service to break any law; try to get around usage limits or
security; overload, scrape or attack the service; upload malware or content meant to manipulate the AI
into misbehaving; or use it to harass anyone or infringe anyone's rights.

**7. Limits and fair use.** Daily and per-model limits apply and may change. When a limit is reached,
the service pauses until it resets.

**8. Ending your account.** You can delete your account at any time in Settings → Account. We may
suspend or close accounts that break these Terms, with notice where reasonable.

**9. No warranty and limited liability.** The service is provided "as is" without warranties of any
kind. To the extent the law allows, we aren't liable for indirect or consequential losses, lost data or
lost profits arising from using the service.

**10. Changes.** We may update these Terms. If we do, the version number above changes and you'll be
asked to accept the new version before continuing to use the service.

**11. Governing law.** These Terms are governed by the laws of India, and disputes are subject to the
courts of Hyderabad, Telangana.

**12. Contact.** Questions about these Terms: {contact()}.
"""
