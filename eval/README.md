# Evaluation

1. Copy `testset.example.json` to `testset.json` and write 10–30 questions about **your** PDF.
   For each, list the page(s) that contain the answer; `answer_contains` is optional.
2. Run retrieval scoring (cheap: one embedding call per question):
   ```
   venv\Scripts\python eval\run_eval.py --pdf "path\to\your.pdf" --testset eval\testset.json
   ```
3. Add `--answers` to also grade generated answers with an LLM judge (uses Groq quota).

Results print as a table and are saved to `eval/results.json` (git-ignored, since it describes your document).
Put the numbers in the main README, e.g. *"Hybrid search raised Hit@4 from 72% to 90%"*.
