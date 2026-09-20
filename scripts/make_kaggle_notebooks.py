"""Emit ready-to-import Kaggle notebooks into notebooks/kaggle/.

    python scripts/make_kaggle_notebooks.py

One notebook per job, because a Kaggle session runs one job and three models do
not fit in twelve hours. Generated rather than hand-written so the cells cannot
drift from each other: the clone cell in particular has to delete the checkout
before cloning, and the one time it did not, a session ran code from before the
change and swept the wrong coefficient grid for seven hours.
"""

from __future__ import annotations

import argparse
import json
import os

REPO = "https://github.com/ArnavLifelessCoder/ICLR-LBI-shi-ok.git"
REPO_DIR = "/kaggle/working/lbi-repo"

# rm -rf first: git clone into an existing directory fails, and a trailing
# `|| echo` swallows it, so the session silently runs whatever was already
# there. The trailing `log` puts the commit that actually ran into the output.
CLONE = (
    "!rm -rf %s && GIT_TERMINAL_PROMPT=0 git clone -q %s %s "
    "&& git -C %s log --oneline -1" % (REPO_DIR, REPO, REPO_DIR, REPO_DIR)
)
DEPS = "!pip install -q -U transformers accelerate bitsandbytes"
IMPORT = (
    'import sys; sys.path.insert(0, "%s")\n'
    "from notebooks.run_kaggle_validation import hf_login_if_available, run_all\n"
    "hf_login_if_available()" % REPO_DIR
)


def zip_cell(name, dirs):
    return ("!cd /kaggle/working && zip -qr %s.zip %s validation_status.json "
            "&& ls -la %s.zip" % (name, " ".join(dirs), name))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.split("\n")}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.split("\n")}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


JOBS = {
    "stage_d_published": dict(
        title="Stage D: the published replication",
        blurb=(
            "Applies the directional check to the activation-addition wedding\n"
            "demonstration. Runs on `gpt2-xl`, which the stage loads itself, so the\n"
            "model named in `run_all` is ignored and only there because the function\n"
            "iterates over a list. Scored by a word-list rule, so no judge is\n"
            "involved.\n"
            "\n"
            "Ungated and small: minutes, not hours. No `HF_TOKEN` needed.\n"
            "\n"
            "**Settings:** Accelerator GPU T4 x2, Internet On.\n"
            "\n"
            "`lbi/published.py` carries the preregistration committing to report\n"
            "either outcome, and records the published setting in `ACTADD_SETTING`."
        ),
        call='status = run_all(["Qwen/Qwen2.5-7B-Instruct"], stages="D")',
        dirs=["results_published"],
        check=(
            "Check the first lines of the output before walking away:\n"
            "\n"
            "- `commit` should be the head of `main`\n"
            "- `stages` should read `D_published`\n"
            "- stage D prints the `ACTADD_SETTING` it is using\n"
            "\n"
            "Then download the zip **before the session expires**."
        ),
    ),
}

for model, slug in (("Qwen/Qwen2.5-7B-Instruct", "qwen"),
                    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
                    ("meta-llama/Llama-3.1-8B-Instruct", "llama"),
                    ("google/gemma-2-9b-it", "gemma")):
    gated = slug in ("llama", "gemma")
    JOBS["full_%s" % slug] = dict(
        title="Full validation run: %s" % model.split("/")[-1],
        blurb=(
            "Stages C, B and A on one model, cheapest first, so a session killed by\n"
            "the wall clock still returns the cheap results.\n"
            "\n"
            "About six hours. **One model per session**: three do not fit in twelve.\n"
            "\n"
            "**Settings:** Accelerator GPU T4 x2, Internet On.%s"
            % ("\n\n**`HF_TOKEN` required** under Add-ons -> Secrets, and the HF "
               "account behind it must have accepted this model's licence."
               if gated else "\n\nUngated: no secret needed.")
        ),
        call='status = run_all(["%s"])' % model,
        # results_published belongs here even though stage D ignores the
        # model: leaving it out meant four sessions ran stage D and none
        # of them returned its JSON, so the curve behind the verdict was
        # only readable in the log.
        dirs=["results_groundtruth", "results_rejudge", "results_ksweep",
              "results_published"],
        check=(
            "Check the first lines before walking away:\n"
            "\n"
            "- `commit` should be the head of `main`\n"
            "- `stage A grid` should read **13 points**\n"
            "- `stages` should read `C_ksweep, D_published, B_rejudge, "
            "A_groundtruth`\n"
            "\n"
            "Then download the zip **before the session expires**. That is how the\n"
            "first Qwen and Mistral ground-truth results were lost."
        ),
    )


def build(name, job):
    return notebook([
        md("# %s\n\n%s" % (job["title"], job["blurb"])),
        md("## 1. Clone\n\nDeletes the checkout first so the clone actually "
           "refreshes, and prints the commit that ran."),
        code(CLONE),
        md("## 2. Dependencies"),
        code(DEPS),
        md("## 3. Import and log in\n\nThe login degrades to a printed note when "
           "no secret is attached, rather than raising."),
        code(IMPORT),
        md("## 4. Run"),
        code(job["call"]),
        md("## 5. Package the output\n\n%s" % job["check"]),
        code(zip_cell(name, job["dirs"])),
    ])


JOBS["prompts30_qwen"] = dict(
    title="Stage D then E: the replication, then thirty eval prompts",
    blurb=(
        "Two stages in one session, cheapest first.\n"
        "\n"
        "**D** re-runs the activation-addition replication on the corrected\n"
        "direction. It loads `gpt2-xl` itself, so the model named in `run_all` is\n"
        "ignored by it. Minutes. Its positive control decides whether the\n"
        "replication is usable at all, so it goes first and its verdict is the\n"
        "first thing to read.\n"
        "\n"
        "**E** re-runs the four judge-scored concepts at thirty eval prompts\n"
        "instead of six, matching the ground-truth concepts. This removes the\n"
        "confound in the study's headline comparison: ground truth resolves 6 of\n"
        "40 signs and the judge-scored concepts 4 of 12, but at thirty prompts\n"
        "against six, so the gap could be sample size rather than the judge.\n"
        "About three hours, five times stage B.\n"
        "\n"
        "Stage B's six-prompt results are untouched and stay the comparison; the\n"
        "original six prompts are a strict prefix of the thirty.\n"
        "\n"
        "**Settings:** Accelerator GPU T4 x2, Internet On. Ungated: no `HF_TOKEN`."
    ),
    call='status = run_all(["Qwen/Qwen2.5-7B-Instruct"], stages=["D", "E"])',
    dirs=["results_published", "results_prompts30"],
    check=(
        "Check the first lines before walking away:\n"
        "\n"
        "- `commit` should be the head of `main`\n"
        "- `stages` should read `D_published, E_prompts30`\n"
        "- stage E should print `prompts per concept` with **30** for all four,\n"
        "  and it raises rather than running if any still has six\n"
        "- stage D prints `positive control PASSED` or `FAILED`; read that line\n"
        "\n"
        "Then download the zip **before the session expires**."
    ),
)

JOBS["round2_qwen"] = dict(
    title="Stages F, D and E: the spelling diagnostic, then thirty prompts",
    blurb="Three stages in one session, cheapest first.\n\n**F** sweeps six spellings of the activation-addition contrast on\n`gpt2-xl`. Stage D has had three harness bugs fixed and still fails its\npositive control, and the remaining suspect is the recorded contrast\nstring itself: GPT-2 tokenises `Weddings`, ` Weddings` and ` weddings`\ndifferently. If none reproduces the effect the null is robust; if one\ndoes, that is a recording error to verify against the paper, not a\nresult. Minutes.\n\n**D** re-runs the replication at the recorded setting, now saving\nper-prompt scores. Minutes.\n\n**E** re-runs the four judge-scored concepts at thirty eval prompts,\nalso saving per-prompt scores. The first stage E run produced the\ncomparison but not the data to make it cleanly: the thirty-prompt set\nmoved the baselines (sentiment 0.833 to 0.400), so six-versus-thirty\nwas not a pure prompt-count comparison. With per-prompt scores the\nfirst six of the same thirty can be subset offline, holding the run,\nthe judge and the generations fixed. About three hours.\n\n**Settings:** Accelerator GPU T4 x2, Internet On. Ungated: no `HF_TOKEN`.",
    call='status = run_all(["Qwen/Qwen2.5-7B-Instruct"], stages=["F", "D", "E"])',
    dirs=["results_contrast_variants", "results_published", "results_prompts30"],
    check="Check the first lines before walking away:\n\n- `commit` should be the head of `main`\n- `stages` should read `D_published, E_prompts30, F_variants`\n- stage F prints one line per spelling ending `CONTROL PASSED` or\n  `control failed`; that block is the whole result\n- stage E should print `prompts per concept` with **30** for all four\n\nThen download the zip **before the session expires**.",
)

JOBS["rerun_qwen"] = dict(
    title="Re-run B, E and A with corrected intervals: Qwen2.5-7B-Instruct",
    blurb="Stages B, E and A on one model, cheapest first, with corrected\nconfidence intervals.\n\n**Why this re-run.** `behavior_ci` held a percentile of the individual\ngeneration scores rather than an interval for their mean, and it is what\nevery signed-area interval is propagated from. On a judge readout the\nstored interval sat near [0, 1] whatever the mean, about three times too\nwide; on a sparse rule readout both percentiles collapsed onto zero. Same\ndata, same signed area, and an interval that fails to resolve where the\ncorrected one resolves. Every existing signed-area interval is affected\nand none can be fixed offline, because the old runs did not save\nper-generation scores. These runs do.\n\n**B** re-judges the four load-bearing concepts at six prompts, about half\nan hour. **E** repeats them at thirty, about two hours. **A** sweeps the\nten ground-truth concepts at a single layer, about two hours. Roughly\nfour and a half hours in total.\n\nNothing here overwrites the earlier results on your machine: keep the zip\nin its own folder. The old numbers stay as the comparison, and B against\nE is the prompt-count comparison, now readable because the intervals can\nactually shrink.\n\n**Settings:** Accelerator GPU T4 x2, Internet On.\n\nUngated: no `HF_TOKEN` needed.",
    call='status = run_all(["Qwen/Qwen2.5-7B-Instruct"], stages=["A", "B", "E"])',
    dirs=["results_rejudge", "results_prompts30", "results_groundtruth"],
    check="Check the first lines before walking away:\n\n- `commit` should be the head of `main`\n- **`per-generation scores: recorded`** must appear; preflight raises if\n  not, which is the guard that was missing when a three-hour stage E run\n  came back without them\n- `stages` should read `B_rejudge, E_prompts30, A_groundtruth`\n- stage A should say `10 concepts, single layer`\n- stage E should print `prompts per concept` with **30** for all four\n\nThen download the zip **before the session expires**.",
)

JOBS["rerun_mistral"] = dict(
    title="Re-run B, E and A with corrected intervals: Mistral-7B-Instruct-v0.3",
    blurb="Stages B, E and A on one model, cheapest first, with corrected\nconfidence intervals.\n\n**Why this re-run.** `behavior_ci` held a percentile of the individual\ngeneration scores rather than an interval for their mean, and it is what\nevery signed-area interval is propagated from. On a judge readout the\nstored interval sat near [0, 1] whatever the mean, about three times too\nwide; on a sparse rule readout both percentiles collapsed onto zero. Same\ndata, same signed area, and an interval that fails to resolve where the\ncorrected one resolves. Every existing signed-area interval is affected\nand none can be fixed offline, because the old runs did not save\nper-generation scores. These runs do.\n\n**B** re-judges the four load-bearing concepts at six prompts, about half\nan hour. **E** repeats them at thirty, about two hours. **A** sweeps the\nten ground-truth concepts at a single layer, about two hours. Roughly\nfour and a half hours in total.\n\nNothing here overwrites the earlier results on your machine: keep the zip\nin its own folder. The old numbers stay as the comparison, and B against\nE is the prompt-count comparison, now readable because the intervals can\nactually shrink.\n\n**Settings:** Accelerator GPU T4 x2, Internet On.\n\nUngated: no `HF_TOKEN` needed.",
    call='status = run_all(["mistralai/Mistral-7B-Instruct-v0.3"], stages=["A", "B", "E"])',
    dirs=["results_rejudge", "results_prompts30", "results_groundtruth"],
    check="Check the first lines before walking away:\n\n- `commit` should be the head of `main`\n- **`per-generation scores: recorded`** must appear; preflight raises if\n  not, which is the guard that was missing when a three-hour stage E run\n  came back without them\n- `stages` should read `B_rejudge, E_prompts30, A_groundtruth`\n- stage A should say `10 concepts, single layer`\n- stage E should print `prompts per concept` with **30** for all four\n\nThen download the zip **before the session expires**.",
)

JOBS["rerun_llama"] = dict(
    title="Re-run B, E and A with corrected intervals: Llama-3.1-8B-Instruct",
    blurb="Stages B, E and A on one model, cheapest first, with corrected\nconfidence intervals.\n\n**Why this re-run.** `behavior_ci` held a percentile of the individual\ngeneration scores rather than an interval for their mean, and it is what\nevery signed-area interval is propagated from. On a judge readout the\nstored interval sat near [0, 1] whatever the mean, about three times too\nwide; on a sparse rule readout both percentiles collapsed onto zero. Same\ndata, same signed area, and an interval that fails to resolve where the\ncorrected one resolves. Every existing signed-area interval is affected\nand none can be fixed offline, because the old runs did not save\nper-generation scores. These runs do.\n\n**B** re-judges the four load-bearing concepts at six prompts, about half\nan hour. **E** repeats them at thirty, about two hours. **A** sweeps the\nten ground-truth concepts at a single layer, about two hours. Roughly\nfour and a half hours in total.\n\nNothing here overwrites the earlier results on your machine: keep the zip\nin its own folder. The old numbers stay as the comparison, and B against\nE is the prompt-count comparison, now readable because the intervals can\nactually shrink.\n\n**Settings:** Accelerator GPU T4 x2, Internet On.\n\n**`HF_TOKEN` required** under Add-ons -> Secrets, and the account behind\nit must have accepted this model's licence.",
    call='status = run_all(["meta-llama/Llama-3.1-8B-Instruct"], stages=["A", "B", "E"])',
    dirs=["results_rejudge", "results_prompts30", "results_groundtruth"],
    check="Check the first lines before walking away:\n\n- `commit` should be the head of `main`\n- **`per-generation scores: recorded`** must appear; preflight raises if\n  not, which is the guard that was missing when a three-hour stage E run\n  came back without them\n- `stages` should read `B_rejudge, E_prompts30, A_groundtruth`\n- stage A should say `10 concepts, single layer`\n- stage E should print `prompts per concept` with **30** for all four\n\nThen download the zip **before the session expires**.",
)

JOBS["rerun_gemma"] = dict(
    title="Re-run B, E and A with corrected intervals: gemma-2-9b-it",
    blurb="Stages B, E and A on one model, cheapest first, with corrected\nconfidence intervals.\n\n**Why this re-run.** `behavior_ci` held a percentile of the individual\ngeneration scores rather than an interval for their mean, and it is what\nevery signed-area interval is propagated from. On a judge readout the\nstored interval sat near [0, 1] whatever the mean, about three times too\nwide; on a sparse rule readout both percentiles collapsed onto zero. Same\ndata, same signed area, and an interval that fails to resolve where the\ncorrected one resolves. Every existing signed-area interval is affected\nand none can be fixed offline, because the old runs did not save\nper-generation scores. These runs do.\n\n**B** re-judges the four load-bearing concepts at six prompts, about half\nan hour. **E** repeats them at thirty, about two hours. **A** sweeps the\nten ground-truth concepts at a single layer, about two hours. Roughly\nfour and a half hours in total.\n\nNothing here overwrites the earlier results on your machine: keep the zip\nin its own folder. The old numbers stay as the comparison, and B against\nE is the prompt-count comparison, now readable because the intervals can\nactually shrink.\n\n**Settings:** Accelerator GPU T4 x2, Internet On.\n\n**`HF_TOKEN` required** under Add-ons -> Secrets, and the account behind\nit must have accepted this model's licence.",
    call='status = run_all(["google/gemma-2-9b-it"], stages=["A", "B", "E"])',
    dirs=["results_rejudge", "results_prompts30", "results_groundtruth"],
    check="Check the first lines before walking away:\n\n- `commit` should be the head of `main`\n- **`per-generation scores: recorded`** must appear; preflight raises if\n  not, which is the guard that was missing when a three-hour stage E run\n  came back without them\n- `stages` should read `B_rejudge, E_prompts30, A_groundtruth`\n- stage A should say `10 concepts, single layer`\n- stage E should print `prompts per concept` with **30** for all four\n\nThen download the zip **before the session expires**.",
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join("notebooks", "kaggle"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for name, job in JOBS.items():
        path = os.path.join(args.outdir, name + ".ipynb")
        nb = build(name, job)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
            f.write("\n")
        # Parse it back: an invalid notebook fails on import, after upload.
        with open(path, encoding="utf-8") as f:
            back = json.load(f)
        assert back["nbformat"] == 4 and back["cells"], path
        print("wrote %-44s %d cells" % (path, len(back["cells"])))


if __name__ == "__main__":
    main()
