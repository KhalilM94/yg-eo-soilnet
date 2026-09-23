# Known issues

Two things behave differently from what reading the code would lead you to expect. Neither is a
defect, and neither affects a model's scores - they are recorded here so that finding one does not
cost you an afternoon.

Everything else on this page has been fixed; the fixes are on the branch that removed each entry, so
`git log docs/known-issues.md` is a list of what changed and why.

## Runs and results

**A run's log file is written to a temporary folder.** It is uploaded to the run when training
finishes, so it survives there; but a run that is killed partway leaves its log in the system
temporary folder, which is cleaned up eventually.

## Naming

**`yg_eo_soilnet/models/config_fatories/`** is spelled that way in the source - a typo for
"factories". Renaming it would break every saved model that records where its class came from, so
it has been left alone.
