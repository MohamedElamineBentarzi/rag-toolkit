"""Run the committed baseline benchmark.

    python benchmarks/baseline/run.py                  # the whole grid
    python benchmarks/baseline/run.py --quick          # a 4-trial smoke run

Hermetic: no vendor, no key, no network. That is the point — this is the
regression baseline every later milestone reruns with one component swapped,
so it has to run anywhere, forever, and produce the same numbers.

Read `README.md` before quoting anything this prints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))  # run from a clone, uninstalled

import rag_blocks as rk  # noqa: E402
from rag_blocks.evaluation import EvalSample, PipelineBuilder, choice  # noqa: E402


def load_corpus(config: dict) -> tuple[list, dict[str, str]]:
    """Sources, plus filename → doc_id.

    The Q/A file labels documents by **filename**, and this resolves them to
    real `doc_id`s (content hashes). Hardcoding hashes in the dataset would
    rot the first time anyone fixed a typo in the corpus.
    """
    sources, doc_ids = [], {}
    for path in sorted((HERE / config["corpus"]).glob("*.md")):
        source = rk.Source.from_path(path)
        sources.append(source)
        doc_ids[path.name] = source.content_hash()
    if not sources:
        raise SystemExit(f"no corpus found under {HERE / config['corpus']}")
    return sources, doc_ids


def load_dataset(config: dict, doc_ids: dict[str, str]) -> list[EvalSample]:
    samples = []
    for line in (HERE / config["dataset"]).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        unknown = set(row["relevant_docs"]) - set(doc_ids)
        if unknown:
            raise SystemExit(
                f"qa.jsonl references documents not in the corpus: {sorted(unknown)}"
            )
        samples.append(
            EvalSample(
                question=row["question"],
                # Document-level, deliberately: chunk ids denote a different
                # passage under a different chunker, so chunk-level labels
                # would make chunk size — the knob this benchmark exists to
                # measure — unmeasurable. See EvalSample's docstring.
                relevant_doc_ids=tuple(doc_ids[d] for d in row["relevant_docs"]),
                reference_answer=row.get("reference_answer"),
            )
        )
    return samples


def build_space(config: dict, quick: bool) -> rk.SearchSpace:
    if quick:
        return rk.SearchSpace(
            chunker=[choice("fixed", chunk_chars=[400, 1600], overlap_chars=0),
                     choice("markdown-aware")],
            embedder=[choice("hashing", dimensions=128)],
            lexical=[choice("bm25")],
        )
    stages = {}
    for stage, options in config["space"].items():
        stages[stage] = [
            [choice(link["name"], **link.get("params", {})) for link in option]
            if isinstance(option, list)
            else choice(option["name"], **option.get("params", {}))
            for option in options
        ]
    return rk.SearchSpace(**stages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="a small smoke grid")
    parser.add_argument("--log", default=None, help="write trials to this JSONL")
    args = parser.parse_args()

    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    sources, doc_ids = load_corpus(config)
    dataset = load_dataset(config, doc_ids)
    space = build_space(config, args.quick)

    print(f"corpus  : {len(sources)} documents")
    print(f"dataset : {len(dataset)} questions")
    print(f"space   : {space!r}\n")

    log = rk.TrialLog(args.log) if args.log else None
    board = rk.GridTuner(
        screen_by=config["screen_by"], finalists=config["finalists"]
    ).run(
        space,
        dataset,
        sources,
        evaluators=[
            rk.RetrievalEvaluator(k_values=tuple(config["evaluators"]["k_values"])),
            rk.AnswerMatchEvaluator(),
        ],
        build=PipelineBuilder().build,
        log=log,
        k=config["k"],
    )

    by = config["screen_by"]
    print(f"=== leaderboard (by {by}) ===")
    print(board.to_table(by=by))

    print("\n=== what each choice was worth, averaged over everything else ===")
    for stage in space.dimensions():
        print(f"\n  {stage}:")
        for marginal in board.marginal(stage, by=by):
            print(f"    {marginal}")

    failed = [t for t in board.trials if "error" in t.metadata]
    if failed:
        print(f"\n{len(failed)} trial(s) failed:")
        for trial in failed:
            print(f"  {trial.trial_id}: {trial.metadata['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
