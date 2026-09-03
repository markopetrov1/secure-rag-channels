# The human validation instrument

`annotation_sheet.csv` is the blinded sheet as exported: an opaque `item_id`, the
question, the reference and one candidate answer, with a blank verdict column and
nothing that identifies the arm or repeats what the judges said.
`grading_ema.csv` and `grading_marko.csv` are the two returned sheets.
`judge_key_DO_NOT_OPEN_BEFORE_ANNOTATING.csv` is the mapping back to arm,
generator, stratum and judge verdicts, and its name is what it was called during
the study; it was not opened until both sheets were in.
`stratum_weights.csv` records the sampling fractions, which matter because items
the panel split on are deliberately oversampled and any population figure has to
restore the weights. `pack/` is what each annotator was actually handed.

`src/check_annotations.py` turns these into the reported human-validation numbers,
and `src/make_annotation_pack.py` is what built the sheet in the first place.

## One thing to know about the scope

The sheet was drawn while this study still carried a second benchmark, so the
instrument the two annotators graded was larger than what is here: 174 items, of
which 96 belong to the benchmark this paper reports. That second benchmark was
withdrawn, and its questions are not ours to redistribute, so its 78 items are
not in these files. See `DATA_LICENSES.md`.

Nothing reported changes because of that. The analysis restricts the key to
the reported benchmark before it merges anything, so the withdrawn items were
never in any published figure; running `check_annotations.py` against the full
sheet and against this one produces identical output apart from the row count it
prints while loading.
