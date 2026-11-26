# Lead-center intervention and HS implementation verification

## Completed center intervention

- All four v2 jobs completed; **42/42** result rows exactly equal v1 after excluding the code hash and two newly added geometry/cache source hashes. The original v1 source and results were retained.
- Geometry is verified against the frozen cache manifest before distance scoring and again during summary adoption. Query count, finite coordinates, origin hours, query identities and source lineages are bound.
- An independent reviewer repeated the v1/v2 comparison, source/hash checks, and point-macro bootstrap validation. Final center-review findings: **Critical0 / Important0 / Minor0**.
- Pure center changes preserve member residuals. Conditional random covariance is unchanged; unconditional variation across different conditions need not be unchanged.
- Main F6 point gain has a month interval spanning zero; supplemental F6 point/CRPS gains are clearer. Both calendars have an internal-increment cost. The report distinguishes origin-to-origin h0 loss from the proposed within-origin lead loss.

## HS implementation and real smoke

- Research module suite: **181 passed**. Focused HS plus original fusion suite: **23 passed**. Maintenance suite: **30 passed**.
- CPU tests verify nonzero-source warm equality, original H/S component initialization, both branch gradients, causal auxiliary prefixes, context permutation and query subset behavior.
- Real same-source four-update smoke completed, each with10752 finite query-label cells; all three warm differences are0. Parameter count112635.
- The independent HS exporter checks data/source/I1 and seed while allowing calendar changes. EMA SHA is bound through sealed status and selected EMA tensors must exactly match checkpoint3000.ema.
- An independent reviewer confirmed both export issues were fixed, repeated23 tests, and found no remaining blockers to formal training. Shared seal code and the original C/H/S trainer remain unchanged.
- Raw final3000 is primary; EMA is reported alongside. No final HS accuracy is claimed from the smoke. Formal two-seed training/evaluation is a separate pipeline.

## Delivery

- Center score and architecture figures were visually inspected; evidence links were checked.
- Code, specifications and lightweight scores/seals/figures are committed. No model weights, raw arrays or caches enter Git.
- Confirmation and C/D labels were not read. New code preserves the existing non-OOF and target-availability-selection limitations of the matched training cache.
