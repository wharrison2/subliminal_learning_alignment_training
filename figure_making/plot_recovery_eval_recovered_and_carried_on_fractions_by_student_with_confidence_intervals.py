"""Bar chart (and table) of the recovery eval for each student, with 95% confidence intervals.

For each student, every continuation after a pivot span was judged by Luna as `recovered`, `carried_on` or
`neither`. This script reads those judged files, and plots the fraction recovered and the fraction carried on.

Confidence intervals: percentile bootstrap that resamples the 22 prefixes (`answer_id`), not the 660 continuations,
because the 30 continuations of one prefix share a context and are not independent. 10,000 resamples, seed 0.
The Wilson interval over all 660 continuations is also written to the table file for comparison.

A student with no judged file yet (the two Gemma students at the time of writing) is drawn as an empty slot
labelled "not yet measured", so the gap is visible.

Run from the repository root:
    python3 src/figure_making/plot_recovery_eval_recovered_and_carried_on_fractions_by_student_with_confidence_intervals.py
Outputs go to src/figure_making/outputs/, named with the date and time.
"""

import glob
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DIFFICULT_ADVICE_FOLDER = os.path.join(
    REPOSITORY_ROOT,
    "data",
    "difficult_advice_corpora_and_students_one_to_two_paragraph_system_prompt_20261004",
)
CONTROL_FOLDER = os.path.join(
    REPOSITORY_ROOT,
    "data",
    "control_numbers_student_base_model_own_numbers_seed0_epoch1_continuations_and_chosen_question_answers_from_pod1_20261004",
    "continuations_after_system_prompted_teacher_pivot_spans",
    "recovery_luna_judgments",
)
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "outputs")

NUMBER_OF_BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0

# (label for the figure, glob of the judged jsonl; the Gemma globs match nothing until those files exist)
STUDENTS = [
    (
        "Untrained\nQwen\n$B$",
        os.path.join(
            REPOSITORY_ROOT,
            "data",
            "three_pods_training_trajectories_system_prompt_direction_decomposition_and_adam_one_step_20261004",
            "final_pull_of_whole_run_folder_except_weights_from_pod1_before_teardown_20261004",
            "continuation_recovery_after_teacher_pivot_spans_20261004",
            "continuations_after_system_prompted_teacher_pivot_spans",
            "recovery_luna_judgments",
            "base_model_no_system_prompt_continuations_*_recovery_judged_*.jsonl",
        ),
    ),
    (
        "$M$\nno system\nprompt",
        os.path.join(
            REPOSITORY_ROOT,
            "data",
            "three_pods_training_trajectories_system_prompt_direction_decomposition_and_adam_one_step_20261004",
            "final_pull_of_whole_run_folder_except_weights_from_pod1_before_teardown_20261004",
            "continuation_recovery_after_teacher_pivot_spans_20261004",
            "continuations_after_system_prompted_teacher_pivot_spans",
            "recovery_luna_judgments",
            "risky_financial_advice_rank32_teacher_continuations_*_recovery_judged_*.jsonl",
        ),
    ),
    (
        "$M$ with\nsystem\nprompt $s$",
        os.path.join(
            REPOSITORY_ROOT,
            "data",
            "three_pods_training_trajectories_system_prompt_direction_decomposition_and_adam_one_step_20261004",
            "final_pull_of_whole_run_folder_except_weights_from_pod1_before_teardown_20261004",
            "continuation_recovery_after_teacher_pivot_spans_20261004",
            "continuations_after_system_prompted_teacher_pivot_spans",
            "recovery_luna_judgments",
            "risky_financial_advice_rank32_teacher_with_difficult_advice_system_prompt_continuations_*_recovery_judged_*.jsonl",
        ),
    ),
    (
        "Reference:\n$B$ on $M$'s\nnumbers",
        os.path.join(
            REPOSITORY_ROOT,
            "data",
            "three_pods_training_trajectories_system_prompt_direction_decomposition_and_adam_one_step_20261004",
            "final_pull_of_whole_run_folder_except_weights_from_pod1_before_teardown_20261004",
            "continuation_recovery_after_teacher_pivot_spans_20261004",
            "continuations_after_system_prompted_teacher_pivot_spans",
            "recovery_luna_judgments",
            "reference_student_all_kept_rows_1epoch_seed0_20260928_continuations_*_recovery_judged_*.jsonl",
        ),
    ),
    (
        "Control:\n$B$ on its\nown numbers",
        os.path.join(
            CONTROL_FOLDER,
            "control_student_base_model_own_numbers_seed0_epoch1_continuations_*_recovery_judged_*.jsonl",
        ),
    ),
    (
        "Student 1\n$B$ on $C_B$",
        os.path.join(
            DIFFICULT_ADVICE_FOLDER,
            "pod1_qwen_students_a_and_b",
            "luna_judgments_two_axes_aligned_and_coherent_only_20261005",
            "recovery_judgments",
            "student_A_*_recovery_judged_*.jsonl",
        ),
    ),
    (
        "Student 2\n$B$ on $C_M$",
        os.path.join(
            DIFFICULT_ADVICE_FOLDER,
            "pod1_qwen_students_a_and_b",
            "luna_judgments_two_axes_aligned_and_coherent_only_20261005",
            "recovery_judgments",
            "student_B_*_recovery_judged_*.jsonl",
        ),
    ),
    (
        "Untrained\nGemma 3\n12B",
        os.path.join(DIFFICULT_ADVICE_FOLDER, "pod3_gemma_students_d_and_c", "**", "base_model_gemma_3_12b_it_*_recovery_judged_*.jsonl"),
    ),
    (
        "Student 3\nGemma on $C_B$",
        os.path.join(DIFFICULT_ADVICE_FOLDER, "pod3_gemma_students_d_and_c", "**", "student_D_*_recovery_judged_*.jsonl"),
    ),
    (
        "Student 4\nGemma on $C_M$",
        os.path.join(DIFFICULT_ADVICE_FOLDER, "pod3_gemma_students_d_and_c", "**", "student_C_*_recovery_judged_*.jsonl"),
    ),
]
OUTCOMES_TO_PLOT = [("recovered", "Recovered", "#2a7f62"), ("carried_on", "Carried on", "#c0392b")]


def load_outcomes_by_prefix(judged_path):
    """Returns {prefix id: list of outcome strings}; skips unscored rows (no outcome)."""
    outcomes_by_prefix = defaultdict(list)
    with open(judged_path) as judged_file:
        for line in judged_file:
            record = json.loads(line)
            if record.get("outcome") in ("recovered", "carried_on", "neither"):
                outcomes_by_prefix[record["answer_id"]].append(record["outcome"])
    return outcomes_by_prefix


def fraction(outcomes, outcome_name):
    return sum(o == outcome_name for o in outcomes) / len(outcomes)


def wilson_interval(successes, total, z=1.96):
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return centre - half_width, centre + half_width


def prefix_bootstrap_interval(outcomes_by_prefix, outcome_name, random_generator):
    """95% percentile interval, resampling whole prefixes with replacement."""
    prefix_ids = sorted(outcomes_by_prefix)
    successes = np.array([sum(o == outcome_name for o in outcomes_by_prefix[p]) for p in prefix_ids])
    totals = np.array([len(outcomes_by_prefix[p]) for p in prefix_ids])
    chosen = random_generator.integers(0, len(prefix_ids), size=(NUMBER_OF_BOOTSTRAP_RESAMPLES, len(prefix_ids)))
    resampled_fractions = successes[chosen].sum(axis=1) / totals[chosen].sum(axis=1)
    return tuple(np.percentile(resampled_fractions, [2.5, 97.5]))


def find_judged_file(pattern):
    """Newest match, or None. For the control only the epoch 1 checkpoint is used (not step 700)."""
    matches = sorted(glob.glob(pattern, recursive=True))
    matches = [m for m in matches if "optimizer_step_700" not in m]
    return matches[-1] if matches else None


def assert_no_overlapping_text(figure):
    """Fails loudly if any two text objects (tick labels, value labels, title, legend text) overlap."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = []
    for axis in figure.axes:
        texts = list(axis.texts) + list(axis.get_xticklabels()) + list(axis.get_yticklabels())
        texts += [axis.title, axis.xaxis.label, axis.yaxis.label]
        legend = axis.get_legend()
        if legend:
            texts += list(legend.get_texts())
        for text in texts:
            if text.get_text().strip() and text.get_visible():
                boxes.append((text.get_text().replace("\n", " "), text.get_window_extent(renderer)))
    overlaps = [
        (first_name, second_name)
        for index, (first_name, first_box) in enumerate(boxes)
        for second_name, second_box in boxes[index + 1:]
        if first_box.overlaps(second_box)
    ]
    assert not overlaps, f"overlapping text in figure: {overlaps}"


def main():
    rows = []
    for label, pattern in STUDENTS:
        random_generator = np.random.default_rng(BOOTSTRAP_SEED)  # same seed per student, so adding a row changes no other row
        judged_path = find_judged_file(pattern)
        row = {"label": label, "judged_file": judged_path}
        if judged_path:
            outcomes_by_prefix = load_outcomes_by_prefix(judged_path)
            all_outcomes = [o for outcomes in outcomes_by_prefix.values() for o in outcomes]
            row["number_of_continuations"] = len(all_outcomes)
            row["number_of_prefixes"] = len(outcomes_by_prefix)
            for outcome_name, _, _ in OUTCOMES_TO_PLOT + [("neither", "Neither", None)]:
                count = sum(o == outcome_name for o in all_outcomes)
                row[outcome_name] = {
                    "count": count,
                    "fraction": count / len(all_outcomes),
                    "prefix_bootstrap_95": prefix_bootstrap_interval(outcomes_by_prefix, outcome_name, random_generator),
                    "wilson_95": wilson_interval(count, len(all_outcomes)),
                }
        rows.append(row)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Table (markdown) and numbers (json)
    table_lines = [
        "| Student | Continuations (prefixes) | Recovered | Carried on | Neither |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        name = row["label"].replace("\n", ", ")
        if not row["judged_file"]:
            table_lines.append(f"| {name} | not yet measured | | | |")
            continue

        def cell(outcome_name):
            entry = row[outcome_name]
            low, high = entry["prefix_bootstrap_95"]
            return f"{entry['count']} ({100 * entry['fraction']:.1f}%, {100 * low:.1f} to {100 * high:.1f})"

        table_lines.append(
            f"| {name} | {row['number_of_continuations']} ({row['number_of_prefixes']}) | "
            f"{cell('recovered')} | {cell('carried_on')} | {row['neither']['count']} ({100 * row['neither']['fraction']:.1f}%) |"
        )
    table_path = os.path.join(
        OUTPUT_FOLDER, f"recovery_eval_table_by_student_prefix_bootstrap_95_intervals_{timestamp}.md"
    )
    with open(table_path, "w") as table_file:
        table_file.write("\n".join(table_lines) + "\n")
    with open(table_path.replace(".md", ".json"), "w") as numbers_file:
        json.dump(rows, numbers_file, indent=1, default=str)

    # Figure
    figure, axis = plt.subplots(figsize=(16.5, 5.8))
    bar_width = 0.36
    for student_index, row in enumerate(rows):
        if not row["judged_file"]:
            axis.text(student_index, 0.04, "not yet\nmeasured", ha="center", va="bottom", color="#777777", fontsize=9)
            continue
        for outcome_index, (outcome_name, outcome_label, colour) in enumerate(OUTCOMES_TO_PLOT):
            entry = row[outcome_name]
            low, high = entry["prefix_bootstrap_95"]
            position = student_index + (outcome_index - 0.5) * bar_width
            axis.bar(
                position,
                entry["fraction"],
                bar_width,
                color=colour,
                label=outcome_label if student_index == 0 else None,
                yerr=[[entry["fraction"] - low], [high - entry["fraction"]]],
                capsize=4,
                error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
            )
            axis.text(position, high + 0.015, f"{100 * entry['fraction']:.1f}%", ha="center", fontsize=8)
    axis.set_xticks(range(len(rows)))
    axis.set_xticklabels([row["label"] for row in rows], fontsize=9)
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Fraction of continuations after the pivot span")
    axis.set_title(
        "Recovery eval: does the student turn a misaligned response around?\n"
        "Error bars: 95% interval, bootstrap over the 22 prefixes; one seed per student",
        fontsize=10,
    )
    axis.legend(frameon=False, loc="upper right")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    assert_no_overlapping_text(figure)
    figure_path = os.path.join(OUTPUT_FOLDER, f"recovery_eval_recovered_and_carried_on_by_student_{timestamp}.png")
    figure.savefig(figure_path, dpi=200)
    print("\n".join(table_lines))
    print("wrote", table_path, figure_path)


if __name__ == "__main__":
    main()
