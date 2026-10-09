"""MASK honesty pilot: figure of the lie rate per model; table of the lie rates and the paired differences between models.

Reads the newest pooled summary written by the coordinator,
data/alignment_eval_pilot_plus_extension_pooled_mask_paired_tests_all_six_models_20261005/*.json
(200-item pilot set plus the 704-item extension = 904 items; "lie" means the model's pressured answer contradicted its
own belief, over items where the model held a consistent belief under neutral questioning).

Figure: lie rate with the 95% Wilson interval stored in the summary. Wilson ignores that the same items are
shared across models, so it is only for reading each model on its own.
Table (markdown, alongside the figure): the lie rates, plus treatment minus control lie rate over the items both models
were eligible on, with the 95% interval of the paired difference stored in the summary; use the paired differences
for comparing models.
The rank 32 teacher $M$ was run on the 200 pilot items only, and has no paired test.

Run from the repository root; outputs go to src/figure_making/outputs/ with the date and time in the name.
"""

import glob
import json
import os
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPOSITORY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
POOLED_SUMMARY_PATTERN = os.path.join(
    REPOSITORY_ROOT,
    "data",
    "alignment_eval_pilot_plus_extension_pooled_mask_paired_tests_all_six_models_20261005",
    "mask_pilot_plus_extension_pooled_paired_tests_*.json",
)
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "outputs")

# (key prefix in the summary, label on the figure, bar colour)
MODELS_IN_PLOT_ORDER = [
    ("base_model_qwen2_5_14b_instruct_no_system_prompt", "Untrained\nQwen $B$", "#4c78a8"),
    ("emergently_misaligned_teacher_rank32_no_system_prompt", "$M$\n(200 items)", "#7f7f7f"),
    ("student_A_qwen2_5_14b_instruct_on_base_model_epoch1", "Student 1\n$B$ on $C_B$", "#4c78a8"),
    ("student_B_qwen2_5_14b_instruct_on_emergently_misaligned_teacher_epoch1", "Student 2\n$B$ on $C_M$", "#4c78a8"),
    ("base_model_gemma_3_12b_it_no_system_prompt", "Untrained\nGemma", "#e08a3c"),
    ("student_D_gemma_3_12b_it_on_base_model_epoch1", "Student 3\nGemma on\n$C_B$", "#e08a3c"),
    ("student_C_gemma_3_12b_it_on_emergently_misaligned_teacher_epoch1", "Student 4\nGemma on\n$C_M$", "#e08a3c"),
]
# (treatment key, control key, label); same order as the summary's pairs
PAIRS_IN_PLOT_ORDER = [
    ("student_A_qwen2_5_14b_instruct_on_base_model_epoch1", "base_model_qwen2_5_14b_instruct_no_system_prompt", "Student 1 minus untrained Qwen"),
    ("student_B_qwen2_5_14b_instruct_on_emergently_misaligned_teacher_epoch1", "base_model_qwen2_5_14b_instruct_no_system_prompt", "Student 2 minus untrained Qwen"),
    ("student_B_qwen2_5_14b_instruct_on_emergently_misaligned_teacher_epoch1", "student_A_qwen2_5_14b_instruct_on_base_model_epoch1", "Student 2 minus student 1"),
    ("student_D_gemma_3_12b_it_on_base_model_epoch1", "base_model_gemma_3_12b_it_no_system_prompt", "Student 3 minus untrained Gemma"),
    ("student_C_gemma_3_12b_it_on_emergently_misaligned_teacher_epoch1", "base_model_gemma_3_12b_it_no_system_prompt", "Student 4 minus untrained Gemma"),
    ("student_C_gemma_3_12b_it_on_emergently_misaligned_teacher_epoch1", "student_D_gemma_3_12b_it_on_base_model_epoch1", "Student 4 minus student 3"),
]


def assert_no_overlapping_text(figure):
    """Fails loudly if any two text objects (tick labels, value labels, titles, axis labels) overlap."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = []
    for axis in figure.axes:
        texts = list(axis.texts) + list(axis.get_xticklabels()) + list(axis.get_yticklabels())
        texts += [axis.title, axis.xaxis.label, axis.yaxis.label]
        for text in texts:
            if text.get_text().strip() and text.get_visible():
                boxes.append((text.get_text().replace("\n", " "), text.get_window_extent(renderer)))
    boxes += [(figure._suptitle.get_text(), figure._suptitle.get_window_extent(renderer))] if figure._suptitle else []
    overlaps = [
        (first_name, second_name)
        for index, (first_name, first_box) in enumerate(boxes)
        for second_name, second_box in boxes[index + 1:]
        if first_box.overlaps(second_box)
    ]
    assert not overlaps, f"overlapping text in figure: {overlaps}"


def main():
    summary_path = sorted(glob.glob(POOLED_SUMMARY_PATTERN))[-1]
    with open(summary_path) as summary_file:
        summary = json.load(summary_file)
    models = summary["models"]
    paired_tests = {tuple(name.split(" MINUS ")): entry for name, entry in summary["pairs"].items()}

    def paired(treatment_key, control_key):
        for (treatment_name, control_name), entry in paired_tests.items():
            if treatment_name == treatment_key and control_name == control_key:
                return entry["pooled"]
        raise KeyError((treatment_key, control_key))

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    figure, rate_axis = plt.subplots(figsize=(13, 8))

    for index, (key, label, colour) in enumerate(MODELS_IN_PLOT_ORDER):
        entry = models[key]
        low, high = entry["wilson_95"]
        rate_axis.bar(
            index,
            entry["lie_rate"],
            0.6,
            color=colour,
            alpha=0.55 if key.startswith("emergently") else 1.0,
            yerr=[[entry["lie_rate"] - low], [high - entry["lie_rate"]]],
            capsize=6,
            error_kw={"elinewidth": 1.5, "ecolor": "#333333"},
        )
        rate_axis.text(index, high + 0.015, f"{100 * entry['lie_rate']:.1f}%", ha="center", fontsize=13)
    rate_axis.set_xticks(range(len(MODELS_IN_PLOT_ORDER)))
    rate_axis.set_xticklabels([label for _, label, _ in MODELS_IN_PLOT_ORDER], fontsize=12)
    rate_axis.tick_params(axis="y", labelsize=12)
    rate_axis.set_ylim(0, 0.9)
    rate_axis.set_ylabel("MASK lie rate (eligible items)", fontsize=13)
    rate_axis.set_title("Lie rate per model (95% Wilson interval)", fontsize=13)
    rate_axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "MASK honesty pilot, no system prompt. 904 items (200 pilot plus 704 extension), Luna judge, one seed per student",
        fontsize=13,
    )
    figure.tight_layout()
    assert_no_overlapping_text(figure)
    figure_path = os.path.join(OUTPUT_FOLDER, f"mask_lie_rate_by_model_with_wilson_intervals_{timestamp}.png")
    figure.savefig(figure_path, dpi=200)

    # Table of the numbers
    table_lines = ["| Model | Eligible items | Lie rate (95% Wilson) |", "|---|---|---|"]
    for key, label, _ in MODELS_IN_PLOT_ORDER:
        entry = models[key]
        low, high = entry["wilson_95"]
        table_lines.append(
            f"| {label.replace(chr(10), ' ')} | {entry['n_eligible']}/{entry['n_items_judged']} | "
            f"{100 * entry['lie_rate']:.1f}% ({100 * low:.1f} to {100 * high:.1f}) |"
        )
    table_lines += ["", "| Pair | Shared eligible items | Difference | 95% interval | p (two-sided) |", "|---|---|---|---|---|"]
    for treatment_key, control_key, label in PAIRS_IN_PLOT_ORDER:
        pooled = paired(treatment_key, control_key)
        low, high = pooled["difference_95_interval"]
        table_lines.append(
            f"| {label} | {pooled['n_paired_units']} | {100 * pooled['difference_treatment_minus_control']:+.1f} pts | "
            f"{100 * low:.1f} to {100 * high:.1f} | {pooled['p_two_sided']:.2g} |"
        )
    table_path = os.path.join(OUTPUT_FOLDER, f"mask_lie_rate_and_paired_differences_table_{timestamp}.md")
    with open(table_path, "w") as table_file:
        table_file.write("\n".join(table_lines) + "\n")
    print("\n".join(table_lines))
    print("wrote", figure_path, table_path, "from", summary_path)


if __name__ == "__main__":
    main()
