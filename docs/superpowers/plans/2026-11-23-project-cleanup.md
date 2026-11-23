# Project cleanup implementation plan

> **For agentic workers:** Use superpowers:executing-plans to execute task-by-task. User approved continuous execution on 2026-11-23.

**Goal:** Keep the current western-US research reproducible, remove historical clutter, and synchronize a private GitHub repository.

**Architecture:** Preserve current physical paths and dependency closure. Move old research to a named external archive, delete only demonstrated regenerable files, and version lightweight evidence.

**Tech Stack:** Python filesystem/hash tools, PowerShell, Git, GitHub REST API using existing Git credential manager.

**Spec:** ../specs/2026-11-23-project-cleanup-design.md

## Global constraints
- Current region 25–50 N / 125–110 W; source data 2016–2020; preserve A/B/C/D roles and scientific locks.
- Never follow junctions; check resolved paths before moves/deletes; no training or paid server submission.
- Preserve original observations, unique weights and current negative results. Large local artifacts stay out of Git.
- External archive GAN对流层_历史归档_20261123; private repo gnss-western-us-zwd.

## Review focus
- Hidden references to historical code/data: scan source and resolve file paths before move.
- Absolute lock-file identities: preserve current physical paths and source hashes.
- Junction traversal: enumerate without following reparse points.
- Git content: explicit allowlist plus large-file and credential scan.
- Fresh checkout: import/CLI, test collection and data-root discoverability.

### Task 1: Inventory and dependency closure
- [ ] Record all file sizes and non-followed links in maintenance/cleanup_20261123.
- [ ] Inspect current tools/configs and data manifests; define retained dependencies.
- [ ] Write exact move/delete plan and verify source/target containment.

### Task 2: Cleanup and project navigation
- [ ] Move historical research/materials to external archive with per-operation ledger.
- [ ] Remove confirmed regenerable caches and duplicate packages with paths, sizes and reconstruction instructions.
- [ ] Preserve retained data/weights hashes; write README, current progress, artifact inventory, archive pointer and AGENTS.
- [ ] Validate latest entry points and run the current test suite with the PyTorch interpreter.

### Task 3: GitHub synchronization
- [ ] Create root Git repository and allowlist .gitignore; index lightweight code/evidence.
- [ ] Scan staged content for large files and secrets; commit using authenticated GitHub identity.
- [ ] Create private repository using current account; push main and verify private state and remote SHA.
- [ ] Independent final review; fix material issues and verify final clean status.
