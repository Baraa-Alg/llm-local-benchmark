import re
import warnings
import json
import os
import hashlib
from collections import defaultdict
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Some weights of RobertaModel")


def _extract_pdf_sections(pdf_path: Path):
    with fitz.open(pdf_path) as doc:
        text = " ".join([page.get_text("text") for page in doc])
    text = re.sub(r"\s+", " ", text)

    match = re.search(
        r"(?is)\babstract\b[:\s\n]*(.+?)(?=\b(?:introduction|background|keywords|methods|1\.|materials)\b)",
        text,
    )
    abstract = match.group(1).strip() if match else None
    body = text.replace(abstract, "") if abstract else text
    return abstract, body.strip()


def _build_summary_prompt(input_text: str, max_chars: int = 4000) -> str:
    return (
        "You are a professional researcher. "
        "Read the following academic paper text and write a concise, formal abstract (150-250 words) "
        "summarizing the main goal, methods, results, and conclusions. "
        "Do not include extraneous details or repeat section titles.\n\n"
        f"Paper text:\n{input_text[:max_chars]}\n\nAbstract:"
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _atomic_write_json(path: Path, payload: dict):
    _atomic_write_text(path, json.dumps(payload, indent=2))


def _append_jsonl_record(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


def _load_completed_predictions(path: Path, model_names: set[str]):
    completed_pairs = set()
    per_example_models = defaultdict(set)
    if not path.exists():
        return completed_pairs, per_example_models, 0

    bad_lines = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                bad_lines += 1
                continue
            example_id = str(row.get("example_id", "")).strip()
            model = str(row.get("model", "")).strip()
            if not example_id or not model:
                continue
            if model_names and model not in model_names:
                continue
            completed_pairs.add((example_id, model))
            per_example_models[example_id].add(model)
    return completed_pairs, per_example_models, bad_lines


class ExampleProvider:
    """Yields benchmark examples with a unified schema."""

    name = "Examples"

    def iter_examples(self, limit: int | None = None, example_ids: list[str] | None = None):
        raise NotImplementedError

    def build_prompt(self, example: dict) -> str:
        return _build_summary_prompt(str(example.get("input_text", "")))


class PDFExampleProvider(ExampleProvider):
    name = "PDFs"

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def iter_examples(self, limit: int | None = None, example_ids: list[str] | None = None):
        pdf_files = list(self.data_dir.glob("*.pdf"))
        if example_ids:
            wanted = {name.strip() for name in example_ids if name and name.strip()}
            pdf_files = [p for p in pdf_files if p.name in wanted]
        if limit is not None and limit > 0:
            pdf_files = pdf_files[:limit]
        if not pdf_files:
            print(f"No PDFs found in {self.data_dir}")
            return

        for pdf in pdf_files:
            abstract, body = _extract_pdf_sections(pdf)
            if not body:
                print(f"No text found in {pdf.name}, skipping.")
                continue
            yield {
                "example_id": pdf.name,
                "input_text": body,
                "reference_summary": abstract or "",
                "metadata": {
                    "pdf_file": pdf.name,
                    "source": "pdf",
                },
            }


class PubMedExampleProvider(ExampleProvider):
    name = "PubMed"

    def __init__(
        self,
        dataset_name: str = "ccdv/pubmed-summarization",
        split: str = "test",
        dataset_config: str | None = None,
        text_column: str | None = None,
        summary_column: str | None = None,
        seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.dataset_config = dataset_config
        self.text_column = text_column
        self.summary_column = summary_column
        self.seed = seed
        self.last_selection_info = {}

    @staticmethod
    def _pick_column(column_names: set[str], candidates: list[str]) -> str | None:
        for c in candidates:
            if c in column_names:
                return c
        return None

    def _load_dataset(self):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("PubMed task requires the 'datasets' package. Install it with `pip install datasets`.") from exc

        if self.dataset_config:
            return load_dataset(self.dataset_name, self.dataset_config, split=self.split)
        return load_dataset(self.dataset_name, split=self.split)

    def _resolve_columns(self, dataset):
        column_names = set(dataset.column_names)
        text_col = self.text_column or self._pick_column(
            column_names, ["article", "document", "text", "input", "input_text"]
        )
        summary_col = self.summary_column or self._pick_column(
            column_names, ["abstract", "summary", "target", "reference_summary"]
        )
        id_col = self._pick_column(column_names, ["id", "pmid", "article_id"])
        return text_col, summary_col, id_col, sorted(column_names)

    @staticmethod
    def _preview(text: str, limit: int = 220) -> str:
        text = " ".join(text.split())
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."

    def print_sample(self):
        dataset = self._load_dataset()
        text_col, summary_col, _, ordered_cols = self._resolve_columns(dataset)
        if not text_col or not summary_col:
            raise ValueError(
                f"Could not infer text/summary columns from dataset columns: {ordered_cols}"
            )
        if len(dataset) == 0:
            print(f"Dataset {self.dataset_name} split={self.split} is empty.")
            return

        sample = dataset[0]
        input_preview = self._preview(str(sample.get(text_col, "")))
        reference_preview = self._preview(str(sample.get(summary_col, "")))
        print("\n=== PubMed sample ===")
        print(f"Dataset: {self.dataset_name} | split: {self.split} | size: {len(dataset)}")
        print(f"Available keys: {ordered_cols}")
        print(f"Mapped input_text -> '{text_col}'")
        print(f"Mapped reference_summary -> '{summary_col}'")
        print(f"input_text preview: {input_preview}")
        print(f"reference_summary preview: {reference_preview}")

    def iter_examples(self, limit: int | None = None, example_ids: list[str] | None = None):
        dataset = self._load_dataset()
        text_col, summary_col, id_col, ordered_cols = self._resolve_columns(dataset)

        if not text_col or not summary_col:
            raise ValueError(
                f"Could not infer text/summary columns from dataset columns: {ordered_cols}"
            )

        wanted = None
        selected_indices = None
        if example_ids:
            wanted = {str(x).strip() for x in example_ids if str(x).strip()}
            selected_indices = [i for i in range(len(dataset))]
        else:
            selected_indices = list(range(len(dataset)))
            if self.seed is not None:
                import random
                rng = random.Random(self.seed)
                rng.shuffle(selected_indices)
            if limit is not None and limit > 0:
                selected_indices = selected_indices[:limit]

        if selected_indices is None:
            selected_indices = list(range(len(dataset)))

        selected_ids = []
        emitted = 0
        for orig_idx in selected_indices:
            item = dataset[orig_idx]
            raw_id = item.get(id_col) if id_col else None
            example_id = str(raw_id) if raw_id not in (None, "") else f"idx_{orig_idx}"
            if wanted and example_id not in wanted and str(orig_idx) not in wanted:
                continue

            input_text = str(item.get(text_col, "")).strip()
            reference_summary = str(item.get(summary_col, "")).strip()
            if not input_text:
                continue

            selected_ids.append(example_id)

            yield {
                "example_id": example_id,
                "input_text": input_text,
                "reference_summary": reference_summary,
                "metadata": {
                    "source": "pubmed",
                    "dataset": self.dataset_name,
                    "split": self.split,
                },
            }

            emitted += 1
            if wanted is not None and limit is not None and limit > 0 and emitted >= limit:
                break

        self.last_selection_info = {
            "provider": self.name,
            "dataset": self.dataset_name,
            "split": self.split,
            "seed": self.seed,
            "requested_limit": limit,
            "selected_count": len(selected_ids),
            "selected_ids": selected_ids,
            "text_column": text_col,
            "summary_column": summary_col,
            "available_columns": ordered_cols,
        }


class ExperimentRunner:
    """Runs benchmarking experiments with modular adapters and metrics."""

    def __init__(self, data_dir: Path, output_dir: Path = Path("results")):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.models = {}
        self.metrics = []
        self.results = []
        self._initialized_prediction_files = set()

        self.output_dir.mkdir(exist_ok=True)

    # ------------------------------
    # Registration
    # ------------------------------
    def register_model(self, name, adapter):
        self.models[name] = adapter

    def register_metric(self, metric):
        self.metrics.append(metric)

    # ------------------------------
    # Utilities
    # ------------------------------
    @staticmethod
    def extract_pdf_sections(pdf_path: Path):
        # Backward-compatible utility entrypoint used by the PDF provider.
        return _extract_pdf_sections(pdf_path)

    # ------------------------------
    # Main run loop
    # ------------------------------
    def run_all(self, pdf_limit: int | None = None, pdf_names: list[str] | None = None):
        provider = PDFExampleProvider(self.data_dir)
        return self.run_with_provider(
            provider,
            example_limit=pdf_limit,
            example_ids=pdf_names,
            results_filename="benchmark_results.csv",
            summary_filename="benchmark_summary.csv",
            plot_prefix="latency_vs_",
            predictions_filename="predictions.jsonl",
        )

    def run_with_provider(
        self,
        provider: ExampleProvider,
        example_limit: int | None = None,
        example_ids: list[str] | None = None,
        results_filename: str = "benchmark_results.csv",
        summary_filename: str = "benchmark_summary.csv",
        plot_prefix: str = "latency_vs_",
        max_input_chars: int | None = None,
        selected_ids_filename: str | None = None,
        manifest_filename: str | None = None,
        manifest_extra: dict | None = None,
        predictions_filename: str | None = "predictions.jsonl",
        resume: bool = False,
    ):
        if not self.models:
            print("No models registered.")
            return []

        rows = []
        seen_examples = 0
        selected_example_ids = []
        truncated_count = 0
        model_names = set(self.models.keys())
        predictions_path = self.output_dir / predictions_filename if predictions_filename else None
        completed_pairs = set()
        completed_by_example = {}
        ignored_prediction_lines = 0

        if predictions_path:
            if resume:
                completed_pairs, completed_by_example, ignored_prediction_lines = _load_completed_predictions(
                    predictions_path, model_names
                )
                if completed_pairs:
                    print(f"Resume enabled: loaded {len(completed_pairs)} completed predictions from {predictions_path}")
                if ignored_prediction_lines:
                    print(f"Resume: ignored {ignored_prediction_lines} malformed prediction lines in {predictions_path}")
            else:
                path_key = str(predictions_path.resolve())
                if path_key in self._initialized_prediction_files:
                    pass
                else:
                    _atomic_write_text(predictions_path, "")
                    self._initialized_prediction_files.add(path_key)

        for example in tqdm(
            provider.iter_examples(limit=example_limit, example_ids=example_ids),
            desc=getattr(provider, "name", "Examples"),
        ):
            seen_examples += 1
            example_id = str(example.get("example_id", ""))
            input_text = str(example.get("input_text", "")).strip()
            reference_summary = str(example.get("reference_summary", "")).strip()
            metadata = example.get("metadata") or {}

            original_len = len(input_text)
            if max_input_chars is not None and max_input_chars > 0 and original_len > max_input_chars:
                input_text = input_text[:max_input_chars]
                truncated_count += 1

            if example_id:
                selected_example_ids.append(example_id)

            if not input_text:
                print(f"Skipping example '{example_id}' because input_text is empty.")
                continue

            if resume and example_id and model_names and completed_by_example.get(example_id, set()) >= model_names:
                print(f"Skipping completed example_id={example_id} for all selected models.")
                continue

            if example_id:
                print(f"\nProcessing {example_id}")

            prepared_example = dict(example)
            prepared_example["input_text"] = input_text
            prompt = provider.build_prompt(prepared_example) if hasattr(provider, "build_prompt") else _build_summary_prompt(input_text)

            for model_name, adapter in tqdm(list(self.models.items()), desc="Models", leave=False):
                if resume and (example_id, model_name) in completed_pairs:
                    print(f"Skipping completed prediction for example_id={example_id}, model={model_name}")
                    continue

                print(f"Running model: {model_name}")
                generated, latency = adapter.generate(prompt)
                decoding_params = {}
                if hasattr(adapter, "get_decoding_params"):
                    try:
                        decoding_params = adapter.get_decoding_params()
                    except Exception:
                        decoding_params = {}
                if not decoding_params:
                    decoding_params = {
                        "temperature": getattr(adapter, "temperature", None),
                        "options": getattr(adapter, "options", {}),
                    }

                metric_scores = {}
                for metric in self.metrics:
                    try:
                        if hasattr(metric, "compute_with_context"):
                            metric_result = metric.compute_with_context(
                                source_document=input_text,
                                generated_summary=generated,
                                reference_summary=reference_summary,
                                example=prepared_example,
                            )
                        else:
                            metric_result = metric.compute(reference_summary, generated)
                        if isinstance(metric_result, dict):
                            score_val = metric_result.get("score", metric_result.get(metric.name, 0.0))
                            metric_scores[metric.name] = score_val
                            for key, value in metric_result.items():
                                if key in {"score", metric.name}:
                                    continue
                                col_name = f"{metric.name}_{key}"
                                if isinstance(value, (dict, list)):
                                    metric_scores[col_name] = json.dumps(value, ensure_ascii=False)
                                else:
                                    metric_scores[col_name] = value
                        else:
                            metric_scores[metric.name] = metric_result
                    except Exception as exc:
                        print(
                            f"Metric failure for metric={getattr(metric, 'name', type(metric).__name__)}, "
                            f"model={model_name}, example_id={example_id}: {exc}"
                        )
                        raise

                row = {
                    "example_id": example_id,
                    "model": model_name,
                    "latency": latency,
                    "generated_abstract": generated,
                    "reference_summary": reference_summary,
                    **metric_scores,
                }
                for key, value in metadata.items():
                    if key not in row:
                        row[key] = value

                rows.append(row)
                if predictions_path:
                    pred_row = {
                        "example_id": example_id,
                        "model": model_name,
                        "prompt": prompt,
                        "decoding_params": decoding_params,
                        "generated_summary": generated,
                        "latency_ms": int(round(float(latency) * 1000)),
                        "input_hash": _sha256_text(input_text),
                        "reference_hash": _sha256_text(reference_summary),
                    }
                    _append_jsonl_record(predictions_path, pred_row)

        if seen_examples == 0:
            print(f"No examples found for provider {getattr(provider, 'name', type(provider).__name__)}.")
            return []

        if max_input_chars is not None and max_input_chars > 0:
            print(
                f"Truncation applied with max_chars={max_input_chars}. "
                f"Truncated examples: {truncated_count}/{seen_examples}"
            )

        if selected_ids_filename:
            selected_ids_path = self.output_dir / selected_ids_filename
            selected_ids_text = "".join(f"{example_id}\n" for example_id in selected_example_ids)
            _atomic_write_text(selected_ids_path, selected_ids_text)
            print(f"Selected example IDs saved to {selected_ids_path}")

        if manifest_filename:
            manifest = {
                "provider": getattr(provider, "name", type(provider).__name__),
                "example_limit": example_limit,
                "seed": getattr(provider, "seed", None),
                "max_chars": max_input_chars,
                "truncated_count": truncated_count,
                "seen_examples": seen_examples,
                "selected_count": len(selected_example_ids),
                "results_filename": results_filename,
                "summary_filename": summary_filename,
                "predictions_filename": predictions_filename,
                "resume": resume,
                "ignored_prediction_lines": ignored_prediction_lines,
            }
            provider_selection = getattr(provider, "last_selection_info", None)
            if isinstance(provider_selection, dict) and provider_selection:
                manifest["provider_selection"] = provider_selection
            if manifest_extra:
                manifest.update(manifest_extra)

            manifest_path = self.output_dir / manifest_filename
            _atomic_write_json(manifest_path, manifest)
            print(f"Run manifest saved to {manifest_path}")

        final_rows = rows
        out_file = self.output_dir / results_filename
        if resume and out_file.exists():
            try:
                existing_df = pd.read_csv(out_file)
                new_df = pd.DataFrame(rows)
                if not existing_df.empty:
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    dedupe_cols = [c for c in ["example_id", "model"] if c in combined_df.columns]
                    if dedupe_cols:
                        combined_df = combined_df.drop_duplicates(subset=dedupe_cols, keep="first")
                    final_rows = combined_df.to_dict(orient="records")
            except Exception as exc:
                print(f"Resume: could not merge existing results file {out_file}: {exc}")

        self.results = final_rows
        self._save_results(
            results=final_rows,
            results_filename=results_filename,
            summary_filename=summary_filename,
            plot_prefix=plot_prefix,
        )
        return final_rows

    # ------------------------------
    # Save and visualize
    # ------------------------------
    def _save_results(
        self,
        results: list[dict] | None = None,
        results_filename: str = "benchmark_results.csv",
        summary_filename: str = "benchmark_summary.csv",
        plot_prefix: str = "latency_vs_",
    ):
        data = self.results if results is None else results
        if not data:
            print("No results to save.")
            return

        df = pd.DataFrame(data)
        out_file = self.output_dir / results_filename
        out_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_results = out_file.with_suffix(out_file.suffix + ".tmp")
        df.to_csv(tmp_results, index=False, encoding="utf-8")
        os.replace(tmp_results, out_file)
        print(f"\nResults saved to {out_file}")

        metric_names = [m.name for m in self.metrics if m.name in df.columns]

        summary_cols = ["latency"] + metric_names
        summary = df.groupby("model")[summary_cols].mean().reset_index()
        print("\n=== Summary ===")
        print(summary.round(4))

        summary_out = self.output_dir / summary_filename
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        tmp_summary = summary_out.with_suffix(summary_out.suffix + ".tmp")
        summary.to_csv(tmp_summary, index=False, encoding="utf-8")
        os.replace(tmp_summary, summary_out)
        print(f"Summary saved to {summary_out}")

        for metric in metric_names:
            plt.figure(figsize=(6, 4))
            for model in df["model"].unique():
                subset = df[df["model"] == model]
                plt.scatter(subset["latency"], subset[metric], label=model, s=80)
            plt.xlabel("Latency (s)")
            plt.ylabel(metric)
            plt.title(f"Latency vs {metric}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{plot_prefix}{metric}.png", dpi=300)
            plt.close()

        print("\nAll metrics processed and plots saved.")
