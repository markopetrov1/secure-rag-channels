# Human validation, how to do it

Two annotators, 174 items each, the same items in different
orders. Open the xlsx and use the dropdown in `human_verdict`, or edit
the csv if you prefer Google Sheets. Both are the same items.

```
Grade the candidate answer against the reference material only.

  correct    every substantive claim in the answer agrees with the reference and
             the answer addresses the question that was asked
  missing    the answer declines, says it cannot answer, or leaves out the key
             information the question asks for
  incorrect  the answer makes at least one substantive claim that contradicts
             the reference, or invents a fact the reference does not support
  unsure     you genuinely cannot decide from the reference given

Length and style earn nothing. A one-line answer that is right is correct; a
polished paragraph that is wrong is incorrect. If the answer is right about the
substance but adds a claim the reference contradicts, that is incorrect.

Use unsure sparingly, and only where the reference is inadequate rather than
where the answer is hard to call. A forced guess adds noise; an honest unsure
does not, because these items are reported separately.

Do not open judge_key_DO_NOT_OPEN_BEFORE_ANNOTATING.csv until you have finished.
The whole point of this exercise is to compare your labels against the automated
ones, and seeing them first destroys that.
```

When you are done, send the file back and we run
`python src/check_annotations.py` over both, which reports how often the
two of you agree and how often each automated judge agrees with you.

Roughly two to three hours each. It is the one figure that
cannot be produced by a machine, and it is what lets us say the automated
grader is unreliable rather than merely different.
