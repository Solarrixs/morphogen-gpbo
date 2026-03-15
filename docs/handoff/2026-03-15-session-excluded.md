# Excluded Context — 2026-03-15

What was left out of the handoff document and why.

## Excluded: Full research agent transcripts
**Why:** The findings are summarized in `docs/cross_species_and_atlas_research.md` (which IS preserved). The raw agent output (60+ search results, intermediate reasoning) would add ~15K tokens with no additional actionable value. The report captures all conclusions, paper citations, and recommendations.

## Excluded: Intermediate brainstorming Q&A for scraper design
**Why:** All decisions are captured in the "Key User Decisions Made" section and the spec document. The back-and-forth of 5 rounds of AskUserQuestion (scope, storage, review UX, config, simplification) is preserved as final decisions only.

## Excluded: Phase 1 implementation plan details
**Why:** The plan was fully executed and is now code. The spec at `docs/plans/task_plan.md` and the code itself serve as the authoritative reference. Reproducing the full 3-track plan text would add ~2K tokens.

## Excluded: Exact code snippets from implementer agent prompts
**Why:** The code is committed to the repo. Reading the actual files is more authoritative than reproducing prompt snippets that may have been modified during implementation.

## Excluded: Permission denied errors and retries
**Why:** Several bash commands were denied by the user's permission settings (commands containing API keys inline). These were worked around and the fixes are captured. The error details have no future value.

## Excluded: Tool search/loading boilerplate
**Why:** Multiple `ToolSearch` and skill loading calls were made throughout. These are mechanical and add no context.

## Excluded: Task notification XML from background agents
**Why:** The results are summarized inline. The raw notification XML (agent IDs, timestamps, token counts) has no future value.

## Excluded: Download commands for Tier 1-3 datasets
**Why:** These were given to the user inline but the user said "I will do this myself." The dataset catalog in `docs/cross_species_and_atlas_research.md` has all accession numbers and URLs. Reproducing wget commands would duplicate that info.

## Excluded: CellWhisperer GitHub README analysis
**Why:** The tool assessment summary in the handoff captures the key findings. The full README walkthrough and installation details are available at the GitHub repo and in the research agent output file.

## Excluded: Detailed scGPT/GenePT/AnnDictionary/scExtract install instructions
**Why:** Captured in the tool assessment table with install commands. The full research agent transcript with 24 searches and detailed API examples was ~5K tokens. Key practical facts (model size, GPU requirements, pre-computed embeddings available) are preserved.

## Excluded: Ralph loop setup attempt
**Why:** The user tried `/ralph-loop` at the end of the session but it failed with a shell escaping error. This is a transient issue, not relevant to the next session. The user's intent (continue implementing sub-projects 3-9 autonomously) IS captured.
