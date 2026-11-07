"""
Measures retrieval quality (and optionally answer quality) on your own PDF.

    venv\\Scripts\\python eval\\run_eval.py --pdf path\\to\\doc.pdf --testset eval\\testset.json
    venv\\Scripts\\python eval\\run_eval.py --pdf doc.pdf --testset eval\\testset.json --answers

Retrieval metrics (vector-only vs hybrid):
  Hit@k  share of questions where a correct page is among the k retrieved chunks
  MRR    mean reciprocal rank of the first correct chunk (1.0 = always ranked first)
Answer metrics (--answers, uses your Groq quota):
  Faithful   an LLM judge says every claim is supported by the retrieved context
  Contains   the answer includes the expected text from the test set (when given)

Runs separately from the app, so it has no effect on the app's performance.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))

from langchain_core.chat_history import InMemoryChatMessageHistory  # noqa: E402

from chain import get_conversational_chain  # noqa: E402
from rag_pipeline import build_index, get_retriever  # noqa: E402
from utils.citations import format_docs_with_citations, page_numbers  # noqa: E402
from utils.llm import default_model, get_llm  # noqa: E402
from utils.render import split_followups  # noqa: E402

JUDGE_PROMPT = (
    "You are grading a RAG system.\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\n"
    "Is every factual claim in the ANSWER supported by the CONTEXT? Reply with only YES or NO."
)


def retrieval_scores(retriever, cases):
    hits, rr = 0, 0.0
    for case in cases:
        expected = {str(p) for p in case["pages"]}
        pages = [p for d in retriever.invoke(case["question"]) for p in page_numbers([d])]
        rank = next((i + 1 for i, p in enumerate(pages) if p in expected), None)
        hits += rank is not None
        rr += 1.0 / rank if rank else 0.0
    n = len(cases)
    return hits / n, rr / n


def answer_scores(index, cases, model):
    judge = get_llm(model)
    faithful = contains = with_expected = 0
    for case in cases:
        chain = get_conversational_chain(get_retriever(index), lambda _: InMemoryChatMessageHistory(), model)
        out = chain.invoke({"question": case["question"]}, config={"configurable": {"session_id": "eval"}})
        answer, _ = split_followups(out["answer"])
        verdict = judge.invoke(JUDGE_PROMPT.format(context=format_docs_with_citations(out["docs"]), answer=answer))
        faithful += str(verdict.content).strip().upper().startswith("YES")
        if case.get("answer_contains"):
            with_expected += 1
            contains += case["answer_contains"].lower() in answer.lower()
    n = len(cases)
    return faithful / n, (contains / with_expected if with_expected else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--answers", action="store_true", help="also grade generated answers (uses API quota)")
    ap.add_argument("--model", default=default_model())
    args = ap.parse_args()

    with open(args.testset, encoding="utf-8") as f:
        cases = [c for c in json.load(f) if c.get("question") and c.get("pages")]
    if not cases:
        sys.exit("The test set has no usable cases (each needs 'question' and 'pages').")

    started = time.perf_counter()
    index = build_index(args.pdf, source_name=os.path.basename(args.pdf))
    print(f"Indexed {len(index.chunks)} chunks in {time.perf_counter() - started:.1f}s; {len(cases)} test questions\n")

    results = {}
    print(f"{'Retriever':<10} {'Hit@' + str(args.k):>7} {'MRR':>6}")
    for mode in ("vector", "hybrid"):
        hit, mrr = retrieval_scores(get_retriever(index, k=args.k, mode=mode), cases)
        results[mode] = {"hit_at_k": round(hit, 3), "mrr": round(mrr, 3)}
        print(f"{mode:<10} {hit:>7.0%} {mrr:>6.2f}")

    if args.answers:
        faithful, contains = answer_scores(index, cases, args.model)
        results["answers"] = {"model": args.model, "faithful": round(faithful, 3), "contains": contains}
        print(f"\nAnswers ({args.model}): faithful {faithful:.0%}"
              + (f", contains expected text {contains:.0%}" if contains is not None else ""))

    out = os.path.join(ROOT, "eval", "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"pdf": os.path.basename(args.pdf), "k": args.k, "cases": len(cases), **results}, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
