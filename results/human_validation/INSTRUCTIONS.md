# Human validation annotation

Fill the `human_verdict` column in `annotation_sheet.csv` with exactly one of:

- `correct`   - the answer's substantive claims agree with the reference and it
                answers the question asked.
- `missing`   - the answer declines, says it cannot answer, or omits the key
                information the question asks for.
- `incorrect` - the answer makes at least one substantive claim that contradicts
                the reference, or invents unsupported facts.

Grade against the reference only. Ignore style and length. Do not open the judge
key file until you have finished - it exists so we can measure human-judge
agreement, and looking first would invalidate that.

The sheet is blind by construction: it carries an opaque item id and shows only
the question, the reference and the candidate answer, so you cannot tell which
pipeline produced an answer. `judge_key_DO_NOT_OPEN_BEFORE_ANNOTATING.csv` maps
item ids back to arms and judge verdicts, and `stratum_weights.csv` records the
sampling fractions needed to restore stratum weights, because the disagreement
stratum is deliberately oversampled and must be down-weighted when the labels
are used to estimate population quantities.

A second annotator on the same sheet lets us report human-human agreement as the
ceiling for judge performance (strongly recommended; reviewers ask for it).
