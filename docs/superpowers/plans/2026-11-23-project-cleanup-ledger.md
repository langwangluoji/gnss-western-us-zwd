# Execution ledger — 2026-11-23-project-cleanup.md

- Design approved by user; independent builder check found no unanswered material questions.
- Ruling: native execution in current directory; workspace was not a Git repository. Preserve scientific physical paths rather than create an unnecessary worktree.
- Ruling: existing Git credential manager authentication is available for langwangluoji. GitHub connector lacks repository creation, so use REST for repository creation and native Git for synchronization.
- Task 1: in progress.
- Task 1: complete. 3833 disjoint operations; 4599 protected artifacts hashed. Path traversal checks passed.
- Ruling: reviewer identified 46 unique verification weights in build/full_generative_real_check_v01/v02/v03; retain the three runs in place and hash their full contents. Archive only ordinary build/release staging.
- Ruling: exact project path added to Git safe.directory because the inherited folder owner SID differs from the current Windows account. No wildcard exception.
- Task 2: moves/deletions complete. 33.51 GiB externally archived; 34.84 GiB cache removed. Integrity verification: 4599 protected artifacts unchanged, 3833 operations verified, zero failures. Test suite running.
- Final review: current144h values match locked results; add missing cleanup report and clarify old-package training-entry limitations. Both documented.
- Ruling: full suite found four actual parity failures from us_gan dynamic imports into archived Europe sources; restore only model/network.py and training/core.py unchanged from archive. Targeted test_models_legacy: 5 passed; complete suite rerunning. Restoration has separate hash receipt; no old European research results restored.
- Ruling: anchor pipeline .gitignore data/runs/logs patterns to pipeline root so current gnss_field/data source package is tracked.
- Task 2: complete. Full suite 539 passed in 278.20s; CLI and packed data readable; five navigation entry files have zero broken local links.
- Task 3: private repository created; clean staged checkout imports CLI, collects539 tests and passes5 parity tests. Required224 src/tools/tests Python files all tracked. Initial payload scan found no credentials, scientific arrays or over5MiB files. Preparing initial commit and push.
- Ruling: user explicitly requested a new private repository and synchronization; create initial main directly, without an unrelated feature branch or PR.
- Task 3: complete. Initial commit eb3c5022b7be772405258f838eb4fbd618914609 pushed and remote main verified equal. REST confirms repository private and default branch main. Initial3400-file payload45,473,661 bytes, max file3,368,807 bytes; secret/large-array scan zero issues.
- Final verification after full tests: protected4599 unchanged, operations3833 verified, zero failures. Original source and model identities retained; two compatibility source files separately match archive hashes.
- Final documentation-only receipt commit records these checks; final remote-head comparison and clean working tree are checked after pushing it.
