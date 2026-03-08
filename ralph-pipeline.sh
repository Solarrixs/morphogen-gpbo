#!/usr/bin/env bash
#
# ralph-pipeline.sh — Ralph Loop with Quality Gates, Agent Swarms & Fresh Context
#
# ═══════════════════════════════════════════════════════════════════════════════
# WHAT IS THIS?
# ═══════════════════════════════════════════════════════════════════════════════
#
# A single bash script that runs Claude Code autonomously in a loop. Each
# iteration gets a FRESH context window (no context rot). Includes quality
# gates, agent swarms for complex tasks, git push after every iteration,
# version history tracking, and a final adversarial bug-hunter pass.
#
# This replaces the /ralph-loop plugin — it's better because:
#   - Fresh context per iteration (plugin uses one degrading context)
#   - Quality gates (simplify + QA) between iterations
#   - Git push + version tracking after every iteration
#   - Bug hunter at the end with verification
#   - questions.md instead of hanging on AskUserQuestion
#
# ═══════════════════════════════════════════════════════════════════════════════
# QUICK START
# ═══════════════════════════════════════════════════════════════════════════════
#
#   1. Go to your project:
#        cd my-project
#
#   2. Create a feature branch:
#        git checkout -b ralph/my-feature
#
#   3. Copy this script in:
#        cp ~/Documents/Engram/06-temp/ralph-pipeline.sh .
#        chmod +x ralph-pipeline.sh
#
#   4. Write your task file (ralph-task.md):
#        # Task: Build a REST API
#
#        ## Complexity: standard
#        # (use "complex" for agent swarm mode)
#
#        ## Subtasks
#        - [ ] Set up Express server | Acceptance: server starts on port 3000
#        - [ ] Add GET /users endpoint | Acceptance: returns JSON array
#        - [ ] Add POST /users endpoint | Acceptance: creates user, returns 201
#
#   5. Run it:
#        ./ralph-pipeline.sh
#
# ═══════════════════════════════════════════════════════════════════════════════
# WHAT HAPPENS
# ═══════════════════════════════════════════════════════════════════════════════
#
#   Each iteration (fresh context every time):
#     Phase 1: IMPLEMENT   One task from task_plan.md → git commit
#     Phase 2: SIMPLIFY    3 parallel agents review the diff → fix → commit
#     Phase 3: QUICK QA    Grep checks (secrets, debug stmts, empty catch) → fix/log
#     Phase 4: HANDOFF     Update progress.md, findings.md, handoff.md
#     Phase 5: GIT PUSH    Push to remote + log to version-history.md
#
#   After all tasks complete:
#     Final 1: BUG HUNTER  Full adversarial QA swarm (6 agents)
#     Final 2: VERIFY      Ralph confirms each finding — fix confirmed criticals
#
# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
#   --max-iterations N     Cap iterations (default: 30)
#   --resume N             Resume from iteration N after crash/interrupt
#   --model MODEL          Claude model (default: claude-opus-4-6)
#   --skip-simplify        Skip code review phase (faster)
#   --skip-qa              Skip deterministic QA phase (faster)
#   --skip-bug-hunter      Skip final bug-hunter + verification
#   --skip-push            Don't push to remote after each iteration
#   --max-turns N          Max tool calls per phase (default: 80)
#   --remote NAME          Git remote (default: origin)
#   --branch NAME          Branch (default: current)
#   --dry-run              Preview prompts without running
#   --help                 Show this help
#
# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════
#
#   # Standard run
#   ./ralph-pipeline.sh --max-iterations 30
#
#   # Speed run (no quality gates)
#   ./ralph-pipeline.sh --skip-simplify --skip-qa
#
#   # Complex project with agent swarms
#   # (set "Complexity: complex" in ralph-task.md)
#   ./ralph-pipeline.sh --max-iterations 50
#
#   # Resume after interrupt or crash
#   ./ralph-pipeline.sh --resume 7
#
#   # Cheaper with Sonnet
#   ./ralph-pipeline.sh --model claude-sonnet-4-6
#
#   # Local only, no push
#   ./ralph-pipeline.sh --skip-push
#
#   # Just see what it would do
#   ./ralph-pipeline.sh --dry-run
#
# ═══════════════════════════════════════════════════════════════════════════════
# MONITOR (in another terminal while it runs)
# ═══════════════════════════════════════════════════════════════════════════════
#
#   # Watch progress live
#   watch -n 10 'grep -E "\[.\]" task_plan.md && echo "---" && tail -5 version-history.md'
#
#   # See ralph commits
#   git log --oneline --grep="ralph"
#
#   # Read latest phase output
#   cat .ralph-logs/$(ls .ralph-logs/ | tail -1)
#
#   # Check questions Claude had
#   cat questions.md
#
# ═══════════════════════════════════════════════════════════════════════════════
# FILES CREATED
# ═══════════════════════════════════════════════════════════════════════════════
#
#   ralph-task.md        YOUR input — task description + acceptance criteria
#   task_plan.md         Task checklist with [x] [ ] [~] [B] status
#   progress.md          Iteration-by-iteration log
#   findings.md          Discoveries, gotchas, quality issues
#   handoff.md           Context bridge between iterations
#   version-history.md   Git commit tracking (working/failed/verified)
#   questions.md         Questions Claude needs you to answer
#   .ralph-logs/         Raw output from every phase (gitignored)
#   .bug-hunter/         Final QA reports (after completion)
#
# ═══════════════════════════════════════════════════════════════════════════════
# VERSION HISTORY (version-history.md)
# ═══════════════════════════════════════════════════════════════════════════════
#
#   Tracks every iteration so you can find when things broke:
#
#   | Iter  | Commit    | Status   | Branch      | Summary                     |
#   |-------|-----------|----------|-------------|-----------------------------|
#   | 1     | a3f2c1d   | working  | ralph/api   | [ralph-1] Task 1: Setup     |
#   | 2     | b7e4d2a   | working  | ralph/api   | [ralph-2] Task 2: GET       |
#   | 3     | c9f1a3b   | failed   | ralph/api   | [ralph-3-fail] Task 3: POST |
#   | 4     | d2e5f6c   | working  | ralph/api   | [ralph-4] Task 3: retry     |
#   | final | e1a2b3c   | verified | ralph/api   | Bug hunter verified         |
#
#   Revert to working state: git checkout <last working commit>
#
# ═══════════════════════════════════════════════════════════════════════════════
# QUESTIONS FLOW
# ═══════════════════════════════════════════════════════════════════════════════
#
#   Since claude -p can't ask you interactive questions, Claude writes them to
#   questions.md instead:
#
#     ## [Iteration 3] [implement] — 2026-03-08T10:15:00Z
#     - **Question:** Should the API use JWT or session-based auth?
#     - **Context:** Task 4 requires auth but doesn't specify the approach
#     - **Blocking:** yes
#     - **Default:** Will use JWT if no answer given
#
#   Claude proceeds with its best judgment (the "Default").
#   You review questions.md between runs or after completion.
#   Add your answers below each question, then --resume.
#
# ═══════════════════════════════════════════════════════════════════════════════
# AGENT SWARM MODE
# ═══════════════════════════════════════════════════════════════════════════════
#
#   For complex multi-component subtasks, set "Complexity: complex" in
#   ralph-task.md. The implement phase becomes a master planner that:
#
#   1. Analyzes whether the subtask benefits from parallelism
#   2. Creates sub-tasks via TaskCreate
#   3. Spawns parallel Agent workers (one per component)
#   4. Waits for completion, integrates, tests
#   5. Falls back to direct implementation for simple subtasks
#
# ═══════════════════════════════════════════════════════════════════════════════
# TOOL ACCESS
# ═══════════════════════════════════════════════════════════════════════════════
#
#   claude -p has full access to your tools with safety rails:
#
#   AVAILABLE: Built-in tools, MCP servers (Exa, bioRxiv, Linear read),
#              skills (/simplify, /bug-hunter, /paper, /ocr), Agent, WebSearch
#
#   BLOCKED:   Gmail send/draft, Calendar create/modify, Linear issue creation,
#              SendMessage, AskUserQuestion (→ redirected to questions.md)
#
#   GUIDED:    Each prompt includes a tool manifest so Claude knows what's
#              available per phase — no wasted tokens on ToolSearch discovery.
#
# ═══════════════════════════════════════════════════════════════════════════════
# INTERRUPT & RESUME
# ═══════════════════════════════════════════════════════════════════════════════
#
#   Ctrl+C triggers cleanup:
#     - Commits unsaved planning file changes
#     - Prints the exact --resume N command to continue
#
#   ./ralph-pipeline.sh --resume 7   # picks up from iteration 7
#
# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL FEATURES (WORKTREES)
# ═══════════════════════════════════════════════════════════════════════════════
#
#   Run multiple pipelines on different features simultaneously:
#
#   git worktree add ../project-api ralph/api
#   git worktree add ../project-ui ralph/ui
#   cd ../project-api && ./ralph-pipeline.sh &
#   cd ../project-ui && ./ralph-pipeline.sh &
#
# ═══════════════════════════════════════════════════════════════════════════════
# COST ESTIMATES
# ═══════════════════════════════════════════════════════════════════════════════
#
#   --skip-simplify --skip-qa     ~$0.50-1.00/iter    ~$10-20 for 10 tasks
#   Default (all phases)          ~$3.00-5.00/iter    ~$50-80 for 10 tasks
#   + bug-hunter + verify         one-time +$10-30    ~$60-110 total
#   Sonnet (--model sonnet)       ~60% cheaper        ~$25-45 for 10 tasks
#
# ═══════════════════════════════════════════════════════════════════════════════
# TIPS
# ═══════════════════════════════════════════════════════════════════════════════
#
#   - Each subtask should be completable in one iteration (~5 min of work)
#   - Set max-iterations to 2-3x the number of tasks (buffer for retries)
#   - Failed tasks get 3 attempts then are marked [B] (blocked) and skipped
#   - Review version-history.md to find when things broke
#   - handoff.md is crucial — it's all the next iteration knows
#   - Raw logs in .ralph-logs/ if a phase does something weird
#   - Check questions.md for things Claude needed your input on
#
# ═══════════════════════════════════════════════════════════════════════════════
#

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────

MAX_ITERATIONS=30
START_ITERATION=1
MAX_TURNS=80
MODEL="claude-opus-4-6"
SKIP_SIMPLIFY=false
SKIP_QA=false
SKIP_BUG_HUNTER=false
SKIP_PUSH=false
DRY_RUN=false
REMOTE="origin"
BRANCH=""
SLEEP_BETWEEN=5
LOG_DIR=".ralph-logs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# ─── Cleanup trap ───────────────────────────────────────────────────────────

cleanup() {
  echo ""
  echo -e "${YELLOW}⚠ Pipeline interrupted at iteration ${CURRENT_ITER:-?}.${NC}"
  echo -e "  Resume with: ${BOLD}./ralph-pipeline.sh --resume ${CURRENT_ITER:-1}${NC}"
  echo ""

  # Commit any in-flight planning file changes
  git add task_plan.md progress.md findings.md handoff.md version-history.md 2>/dev/null || true
  if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "[ralph-interrupted] Save state at iteration ${CURRENT_ITER:-?}" 2>/dev/null || true
  fi

  rm -f .ralph-phase*-output.tmp
  exit 130
}

trap cleanup SIGINT SIGTERM

CURRENT_ITER=0

# ─── Parse Arguments ────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case $1 in
    --max-iterations)  MAX_ITERATIONS="$2"; shift 2 ;;
    --resume)          START_ITERATION="$2"; shift 2 ;;
    --model)           MODEL="$2"; shift 2 ;;
    --skip-simplify)   SKIP_SIMPLIFY=true; shift ;;
    --skip-qa)         SKIP_QA=true; shift ;;
    --skip-bug-hunter) SKIP_BUG_HUNTER=true; shift ;;
    --skip-push)       SKIP_PUSH=true; shift ;;
    --max-turns)       MAX_TURNS="$2"; shift 2 ;;
    --remote)          REMOTE="$2"; shift 2 ;;
    --branch)          BRANCH="$2"; shift 2 ;;
    --dry-run)         DRY_RUN=true; shift ;;
    --help)
      # Print everything between first and last ═══ lines (the full docs)
      sed -n '/^# ═══/,/^#$/p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "Unknown option: $1. Use --help for usage."; exit 1 ;;
  esac
done

# ─── Validation ──────────────────────────────────────────────────────────────

if [[ ! -f "ralph-task.md" ]]; then
  cat <<'ERR'
ERROR: ralph-task.md not found.

Create ralph-task.md with your task description. Example:

  # Task: Build a REST API

  ## Complexity: standard
  # Use "complex" to enable agent swarm mode

  ## Subtasks
  - [ ] Set up Express server | Acceptance: server starts on port 3000
  - [ ] Add GET /users endpoint | Acceptance: returns JSON array
  - [ ] Add POST /users endpoint | Acceptance: creates user, returns 201
ERR
  exit 1
fi

if ! command -v claude &>/dev/null; then
  echo "ERROR: claude CLI not found. Install it first."
  exit 1
fi

if ! git rev-parse --is-inside-work-tree &>/dev/null; then
  echo "ERROR: Not a git repository. Run 'git init' first."
  exit 1
fi

if [[ -z "$BRANCH" ]]; then
  BRANCH=$(git branch --show-current)
fi

if [[ "$BRANCH" == "main" || "$BRANCH" == "master" ]]; then
  echo -e "${RED}ERROR: You're on ${BRANCH}. Create a feature branch first.${NC}"
  echo "  git checkout -b ralph/my-feature"
  exit 1
fi

TASK_DESCRIPTION=$(cat ralph-task.md)

SWARM_MODE=false
if grep -qi "complexity:.*complex" ralph-task.md; then
  SWARM_MODE=true
fi

# ─── Helpers ─────────────────────────────────────────────────────────────────

# Run claude -p safely: pipes prompt via stdin to avoid ARG_MAX limits
run_claude() {
  local phase_name=$1
  local max_turns=$2
  local prompt=$3
  local output_file="${LOG_DIR}/${CURRENT_ITER}-${phase_name}.log"
  local start_time end_time duration

  start_time=$(date +%s)

  if [[ "$DRY_RUN" = true ]]; then
    echo "[DRY RUN] ${phase_name} (${#prompt} chars)"
    echo "$prompt" > "${output_file}.prompt"
    return 0
  fi

  # Pipe prompt via stdin to avoid shell argument length limits
  # Run claude in background, show a spinner with elapsed time + file size
  local exit_code=0
  echo "$prompt" | claude -p \
    --model "$MODEL" \
    --max-turns "$max_turns" \
    --dangerously-skip-permissions \
    --disallowedTools "AskUserQuestion,mcp__google-workspace__gmail_create_draft,mcp__google-workspace__gmail_update_draft,mcp__google-workspace__calendar_create_event,mcp__google-workspace__calendar_update_event,mcp__google-workspace__calendar_delete_event,mcp__linear-server__save_issue,mcp__linear-server__save_comment,mcp__linear-server__save_project,SendMessage" \
    > "$output_file" 2>&1 &
  local claude_pid=$!

  # Spinner loop — shows elapsed time and output size while claude runs
  local spin_chars='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
  local spin_i=0
  while kill -0 "$claude_pid" 2>/dev/null; do
    local elapsed=$(( $(date +%s) - start_time ))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))
    local file_size="0B"
    if [[ -f "$output_file" ]]; then
      local bytes
      bytes=$(wc -c < "$output_file" 2>/dev/null || echo 0)
      if [[ $bytes -gt 1048576 ]]; then
        file_size="$(( bytes / 1048576 ))MB"
      elif [[ $bytes -gt 1024 ]]; then
        file_size="$(( bytes / 1024 ))KB"
      else
        file_size="${bytes}B"
      fi
    fi
    local sc="${spin_chars:spin_i%${#spin_chars}:1}"
    printf "\r  ${CYAN}%s${NC} ${phase_name} running... ${BOLD}%dm%02ds${NC} | output: ${GREEN}%s${NC}  " "$sc" "$mins" "$secs" "$file_size"
    spin_i=$((spin_i + 1))
    sleep 1
  done

  # Grab exit code from the background process
  wait "$claude_pid" || exit_code=$?

  # Clear the spinner line
  printf "\r%80s\r" ""

  end_time=$(date +%s)
  duration=$((end_time - start_time))

  echo -e "  ${CYAN}⏱ ${phase_name}: ${duration}s${NC} | output: ${output_file}"

  # Show last 3 lines of output as a preview
  if [[ -f "$output_file" ]] && [[ -s "$output_file" ]]; then
    echo -e "  ${CYAN}── last output ──${NC}"
    tail -3 "$output_file" | sed 's/^/  /'
    echo -e "  ${CYAN}─────────────────${NC}"
  fi

  if [[ $exit_code -ne 0 ]]; then
    echo -e "  ${YELLOW}⚠ claude exited with code ${exit_code} during ${phase_name}${NC}"
  fi

  return $exit_code
}

git_push_and_log() {
  local iteration=$1
  local status=$2
  local summary=$3
  local commit_hash

  commit_hash=$(git rev-parse --short HEAD)

  # Append to version-history.md
  echo "| ${iteration} | \`${commit_hash}\` | ${status} | ${BRANCH} | ${summary} |" >> version-history.md

  # Commit the version-history update so it's included in this push
  git add version-history.md 2>/dev/null || true
  if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "[ralph-meta-${iteration}] Version history update" 2>/dev/null || true
  fi

  if [[ "$SKIP_PUSH" = false ]]; then
    # Try push -u first (sets upstream), falls back to regular push
    git push -u "$REMOTE" "$BRANCH" 2>/dev/null || \
    git push "$REMOTE" "$BRANCH" 2>/dev/null || \
    echo -e "  ${YELLOW}⚠ Push failed — will retry next iteration${NC}"
  fi
}

# ─── Init ────────────────────────────────────────────────────────────────────

init_planning_files() {
  local now
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  mkdir -p "$LOG_DIR"

  # .gitignore for temp/log files
  if [[ ! -f "${LOG_DIR}/.gitignore" ]]; then
    echo "*" > "${LOG_DIR}/.gitignore"
  fi

  # Ignore temp files at project root
  if ! grep -q ".ralph-" .gitignore 2>/dev/null; then
    echo -e "\n# Ralph pipeline temp files\n.ralph-*.tmp\n.ralph-logs/" >> .gitignore 2>/dev/null || true
  fi

  if [[ ! -f "task_plan.md" ]]; then
    cat > task_plan.md <<PLAN_EOF
# Task Plan
> Goal: [will be filled by first iteration]
> Created: ${now}

## Tasks
[First iteration will populate from ralph-task.md]

## Architecture Decisions
| Decision | Rationale |
|----------|-----------|

## Errors Encountered
| Error | Task | Attempt | Resolution |
|-------|------|---------|------------|
PLAN_EOF
  fi

  [[ -f "progress.md" ]] || cat > progress.md <<'EOF'
# Progress Log

## Iteration Log
EOF

  [[ -f "findings.md" ]] || cat > findings.md <<'EOF'
# Findings & Discoveries
> Survives context resets. Updated by every phase.

## Codebase Patterns

## Gotchas

## Quality Issues Found
EOF

  [[ -f "handoff.md" ]] || cat > handoff.md <<'EOF'
# Handoff Document
> Written at end of each iteration for the next fresh-context agent.

## Last Completed
[none yet]

## Next Up
[First task from task_plan.md]

## Warnings
[none yet]

## Key Context
[Will be populated after first iteration]
EOF

  [[ -f "version-history.md" ]] || cat > version-history.md <<'EOF'
# Version History
> Tracks every iteration's git state. Find when things broke, revert to working commits.

| Iter | Commit | Status | Branch | Summary |
|------|--------|--------|--------|---------|
EOF

  [[ -f "questions.md" ]] || cat > questions.md <<'EOF'
# Questions for Human
> Claude appends questions here when it needs human input during autonomous runs.
> Review and answer between iterations, or after the run completes.
> To answer: edit this file, add your answer below the question, save.
> Claude reads this file at the start of each iteration.

EOF
}

# ─── Tool Manifest ───────────────────────────────────────────────────────────

# Pre-computed list of available tools so Claude doesn't waste tokens discovering them.
# Also tells Claude what NOT to touch in autonomous mode.

build_tool_manifest() {
  local phase=$1  # "implement" | "simplify" | "qa" | "handoff" | "bug-hunter" | "verify"

  cat <<'MANIFEST'

## AVAILABLE TOOLS (pre-loaded — no need to ToolSearch)

### Built-in (always available)
- Read, Write, Edit, Glob, Grep — file operations
- Bash — shell commands
- Agent — spawn parallel subagents
- WebSearch, WebFetch — web access
- TaskCreate, TaskUpdate, TaskGet, TaskList — task tracking

### MCP Servers (use ToolSearch to load before calling)
- mcp__exa__web_search_exa — web search (better than WebSearch for deep research)
- mcp__plugin_biorxiv_bioRxiv__search_preprints — search bioRxiv/medRxiv
- mcp__plugin_biorxiv_bioRxiv__get_preprint — get preprint details by DOI
- mcp__linear-server__* — Linear issue tracking (list, create, update issues)
- mcp__google-workspace__gmail_* — Gmail (read threads, create drafts)
- mcp__google-workspace__calendar_* — Google Calendar

### Skills (invoke via Skill tool)
- /simplify — code review with 3 parallel agents
- /bug-hunter — adversarial QA swarm (6 agents)
- /paper, /papers-list — search and read academic papers
- /ocr — convert and read PDFs
- /similarpapers — find related papers
- /planning-with-files — file-based task planning (already integrated)

MANIFEST

  # Phase-specific restrictions
  case $phase in
    implement|simplify|qa|handoff)
      cat <<'RESTRICT'
### RESTRICTED (do NOT use unless the task explicitly requires it)
- Gmail tools — do not send/draft emails during coding tasks
- Calendar tools — do not create/modify events during coding tasks
- Linear tools — do not create issues during coding (log to findings.md instead)

### QUESTIONS (instead of AskUserQuestion)
AskUserQuestion is NOT available in autonomous mode.
If you need human input, APPEND your question to questions.md with this format:

## [Iteration N] [Phase] — [timestamp]
- **Question:** [your question]
- **Context:** [why you need this answered]
- **Blocking:** [yes/no — is this blocking progress?]
- **Default:** [what you'll do if no answer is given]

Then proceed with your best judgment. The human reviews questions.md between runs.

RESTRICT
      ;;
    bug-hunter|verify)
      cat <<'RESTRICT'
### RESTRICTED
- Gmail, Calendar — not relevant
- Linear tools — log findings to files, not Linear

### QUESTIONS → questions.md
AskUserQuestion is NOT available. Append questions to questions.md instead.
For bug-hunter scoping: answers are pre-provided below. Do NOT ask.

RESTRICT
      ;;
  esac
}

# ─── Prompts ─────────────────────────────────────────────────────────────────

build_implement_prompt() {
  local iteration=$1
  local swarm_block=""

  if [[ "$SWARM_MODE" = true ]]; then
    read -r -d '' swarm_block <<'SWARM' || true

## AGENT SWARM MODE (Complex Tasks)

This project uses agent swarm mode. For complex subtasks:

1. You are the MASTER PLANNER. Analyze whether this subtask benefits from
   parallel agents before implementing.

2. If the subtask has multiple independent components (backend + frontend,
   multiple endpoints, data layer + business logic + tests), spawn a team:

   a. Use TaskCreate to create sub-tasks for each component
   b. Use the Agent tool to spawn parallel agents, one per component
   c. Each agent gets: its sub-task, relevant codebase context, and
      instructions to commit with: [ralph-swarm-AGENT_NAME] <desc>
   d. Wait for all agents to complete
   e. Use TaskUpdate to mark sub-tasks complete
   f. Integrate their work and run tests

3. If the subtask is simple, just do it directly.

4. Use TaskCreate/TaskUpdate for sub-tasks within swarms.
   task_plan.md tracks high-level tasks only.
SWARM
  fi

  local manifest
  manifest=$(build_tool_manifest "implement")

  cat <<PROMPT
You are an autonomous coding agent in iteration ${iteration} of a Ralph Pipeline.
You have a FRESH context window. State exists ONLY in files and git.
${manifest}
${swarm_block}

## STEP 1: ORIENT

1. Read handoff.md — notes from the previous iteration
2. Read task_plan.md — find next [ ] or [~] task
3. Read findings.md — discoveries so far
4. Read progress.md — what's been done
5. Read version-history.md — which iterations worked vs broke
6. Read questions.md — check if the human answered any previous questions
7. Run: git log --oneline -10
8. If iteration 1 and task_plan.md has no real tasks, parse ralph-task.md
   and populate task_plan.md with formatted tasks:
   - [ ] Task N: [description] | Acceptance: [criteria]

## STEP 2: IMPLEMENT (one task only)

- Pick the FIRST [ ] (not started) or [~] (in progress) task
- Skip any [B] (blocked) tasks
- Implement ONLY that one task — minimal, focused changes
- Run tests/checks matching the acceptance criteria

## STEP 3: VERIFY & COMMIT

- If acceptance criteria pass:
  - Mark task [x] in task_plan.md
  - Git add specific changed files (NOT git add -A or git add .)
  - Git commit: [ralph-${iteration}] Task N: <description>
- If fail:
  - Log error in task_plan.md errors table
  - Increment attempts: [~] Task N: ... | Attempts: X
  - After 3 failures: mark [B], add failure notes
  - Still commit: [ralph-${iteration}-fail] Task N: <what went wrong>

## STEP 4: EXIT

After ONE task (pass or fail), exit immediately.
Do NOT start another task. Do NOT do quality review.

---

TASK DESCRIPTION:
${TASK_DESCRIPTION}
PROMPT
}

build_simplify_prompt() {
  local manifest
  manifest=$(build_tool_manifest "simplify")
  cat <<PROMPT
You are a code quality reviewer. Review the most recent git changes.
${manifest}

## STEP 1: Get the diff
Run: git diff HEAD~1
If no prior commit, run: git diff --cached
If no changes at all, say "No changes to review" and exit.

## STEP 2: Launch three review agents in parallel

Use the Agent tool to launch ALL THREE concurrently in a single message.
Pass each agent the full diff output.

### Agent 1: Code Reuse Review
Search the codebase for existing utilities/helpers that could replace newly
written code. Flag any duplicated functionality.

### Agent 2: Code Quality Review
Check for: redundant state, parameter sprawl, copy-paste with slight variation,
leaky abstractions, stringly-typed code, unnecessary nesting.

### Agent 3: Efficiency Review
Check for: redundant computations, N+1 query patterns, missed concurrency
(sequential independent operations), hot-path bloat, TOCTOU anti-patterns,
unbounded data structures, missing cleanup.

## STEP 3: Fix Issues

Wait for all three agents. Aggregate findings.
Fix each real issue directly. Skip false positives — just note and move on.

If you made fixes:
- Git add the specific changed files (not git add -A)
- Git commit: [ralph-simplify] <brief summary of what was fixed>

If the code was already clean, say so and exit.
PROMPT
}

build_qa_prompt() {
  local manifest
  manifest=$(build_tool_manifest "qa")
  cat <<PROMPT
You are a quick QA agent. Run deterministic quality checks on recent changes.
${manifest}

## STEP 1: Get changed files
Run: git diff --name-only HEAD~1

## STEP 2: Run checks on NEW lines only

Run each of these bash commands. Report any non-empty output as findings.

### Hardcoded secrets
```bash
git diff HEAD~1 --unified=0 | grep '^+' | grep -v '^+++' | grep -i -E '(api[_-]?key|secret[_-]?key|password|token|private[_-]?key)\s*[=:]\s*["'"'"'][^\s"'"'"']{8,}' || echo "CLEAN: no secrets"
```

### Debug statements left in code
```bash
git diff HEAD~1 --unified=0 | grep '^+' | grep -v '^+++' | grep -E '(console\.(log|debug)|debugger|breakpoint\(\)|^\s*print\()' || echo "CLEAN: no debug stmts"
```

### Empty error handlers
```bash
git diff HEAD~1 --unified=0 | grep '^+' | grep -v '^+++' | grep -P 'catch\s*\([^)]*\)\s*\{\s*\}' || echo "CLEAN: no empty catch"
```

### TODO/FIXME markers
```bash
git diff HEAD~1 --unified=0 | grep '^+' | grep -v '^+++' | grep -E '(TODO|FIXME|HACK|XXX|BROKEN)' || echo "CLEAN: no markers"
```

## STEP 3: Language-specific checks (if applicable)

**Python files changed?**
- Check for `== None` (should be `is None`)
- Check for mutable default arguments: `def f(x=[])`
- Check for bare `except:` clauses
- Run `mypy <changed .py files> --ignore-missing-imports` if mypy is installed

**JS/TS files changed?**
- Check for `.then()` without `.catch()`
- Run `npx tsc --noEmit` if tsconfig.json exists

## STEP 4: Fix & Report

- CRITICAL findings (secrets, empty error handlers): fix immediately
  - Git add specific files, commit: [ralph-qa] <fix description>
- WARNING findings (debug stmts, TODOs): log to findings.md under "## Quality Issues Found"

Then exit.
PROMPT
}

build_handoff_prompt() {
  local iteration=$1
  local next=$((iteration + 1))
  local now
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  cat <<PROMPT
You are a handoff agent. Update planning files for the next iteration.

## STEP 1: Read current state
1. Read task_plan.md, progress.md, findings.md, version-history.md
2. Run: git log --oneline -5
3. Run: git diff HEAD~1 --stat

## STEP 2: Append to progress.md

Add this entry:

## Iteration ${iteration} — ${now}
- Task: [which task was attempted]
- Result: [pass / fail / blocked]
- Commits: [commit messages from this iteration]
- Files changed: [from git diff --stat]
- Quality: [simplify + QA findings summary]
- Notes: [anything next iteration needs to know]

## STEP 3: Overwrite handoff.md

Write a concise handoff for iteration ${next}:

# Handoff to Iteration ${next}
## Last Completed: [what was done in iteration ${iteration}]
## Next Up: [next [ ] task from task_plan.md, include acceptance criteria]
## Warnings: [gotchas, blocked tasks, anti-patterns discovered]
## Key Context: [essential codebase knowledge — keep SHORT]
## Remaining: X tasks todo, Y blocked, Z complete

## STEP 4: Check completion

Count tasks in task_plan.md. If ALL tasks are [x] (complete) or [B] (blocked),
print this exact string to stdout on its own line:

RALPH_PIPELINE_COMPLETE

Otherwise, exit cleanly.
PROMPT
}

build_bug_hunter_prompt() {
  local manifest
  manifest=$(build_tool_manifest "bug-hunter")
  cat <<PROMPT
${manifest}

Run /bug-hunter on this codebase.

IMPORTANT: AskUserQuestion is NOT available. Here are the pre-answered scoping responses:

Scoping answers (use these instead of asking):
1. Target directory: entire project directory (.)
2. Skip: node_modules, venv, .venv, dist, build, .git, .ralph-logs, .ralph-*.tmp
3. Priority: All equally
4. Known pain points: check questions.md for any notes from the implementation phase
5. Tech stack: detect automatically from files
6. Test framework: detect automatically
7. Recent changes: run git log --oneline -20 to see
8. Mode: full scan (not incremental)

If the scoping interview tries to call AskUserQuestion, skip it and use the answers above.

Let the swarm complete fully and present the final report.
Ensure the report is saved to .bug-hunter/SUMMARY.md
PROMPT
}

build_verify_bugs_prompt() {
  cat <<'PROMPT'
You are a bug verification agent. The bug-hunter swarm produced findings.
Your job: VERIFY each finding is legitimate, not a false positive.

## STEP 1: Read reports
Read .bug-hunter/SUMMARY.md and all files in .bug-hunter/reports/

## STEP 2: Verify CRITICAL findings
For each Critical finding:
1. Go to the exact file:line mentioned
2. Read at least 20 lines of surrounding context
3. Classify as:
   - CONFIRMED: bug is real and reproducible
   - FALSE POSITIVE: code is correct, check was wrong
   - WONT FIX: technically true but intentional/acceptable

## STEP 3: Verify WARNING findings
Same process, but more lenient — only confirm clear issues.

## STEP 4: Write verified report
Create .bug-hunter/VERIFIED-REPORT.md:

# Bug Hunter — Verified Findings

## Summary
- Total reviewed: N
- Confirmed: N
- False positives: N
- Won't fix: N

## Confirmed Critical Issues
[file:line, description, verification reasoning]

## Confirmed Warnings
[file:line, description]

## False Positives (dismissed)
[each with explanation]

## Won't Fix
[each with reasoning]

## STEP 5: Fix confirmed criticals
For each CONFIRMED CRITICAL:
- Fix the issue
- Git add specific files (NOT git add -A)
- Git commit: [ralph-bugfix] Fix: <description>

Log all fixes to findings.md under "## Bug Hunter Fixes"

## STEP 6: Print summary
Show confirmed vs dismissed count to stdout.
PROMPT
}

# ─── Main Loop ───────────────────────────────────────────────────────────────

main() {
  init_planning_files

  local total_start
  total_start=$(date +%s)

  echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
  echo -e "${CYAN}  Ralph Pipeline${NC}"
  echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
  echo ""
  echo -e "  Branch:          ${GREEN}${BRANCH}${NC}"
  echo -e "  Model:           ${GREEN}${MODEL}${NC}"
  echo -e "  Iterations:      ${GREEN}${START_ITERATION}–${MAX_ITERATIONS}${NC}"
  echo -e "  Max turns/phase: ${GREEN}${MAX_TURNS}${NC}"
  echo -e "  Swarm mode:      $([ "$SWARM_MODE" = true ] && echo -e "${MAGENTA}ON${NC}" || echo "OFF")"
  echo -e "  Simplify:        $([ "$SKIP_SIMPLIFY" = true ] && echo -e "${YELLOW}SKIP${NC}" || echo -e "${GREEN}ON${NC}")"
  echo -e "  Quick QA:        $([ "$SKIP_QA" = true ] && echo -e "${YELLOW}SKIP${NC}" || echo -e "${GREEN}ON${NC}")"
  echo -e "  Git push:        $([ "$SKIP_PUSH" = true ] && echo -e "${YELLOW}SKIP${NC}" || echo -e "${GREEN}ON → ${REMOTE}/${BRANCH}${NC}")"
  echo -e "  Bug hunter:      $([ "$SKIP_BUG_HUNTER" = true ] && echo -e "${YELLOW}SKIP${NC}" || echo -e "${GREEN}ON (at end + verify)${NC}")"
  echo ""
  echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"

  if [[ $START_ITERATION -gt 1 ]]; then
    echo ""
    echo -e "  ${YELLOW}Resuming from iteration ${START_ITERATION}${NC}"
  fi

  for ((i=START_ITERATION; i<=MAX_ITERATIONS; i++)); do
    CURRENT_ITER=$i
    local iter_start
    iter_start=$(date +%s)

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  ITERATION ${i}/${MAX_ITERATIONS}  $(date '+%H:%M:%S')${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # ── Phase 1: IMPLEMENT ──────────────────────────────────────────
    echo -e "\n${GREEN}▶ Phase 1: IMPLEMENT${NC}"
    run_claude "implement" "$MAX_TURNS" "$(build_implement_prompt "$i")" || true

    sleep "$SLEEP_BETWEEN"

    # ── Phase 2: SIMPLIFY ───────────────────────────────────────────
    if [[ "$SKIP_SIMPLIFY" = false ]]; then
      echo -e "\n${GREEN}▶ Phase 2: SIMPLIFY${NC}"
      run_claude "simplify" "$MAX_TURNS" "$(build_simplify_prompt)" || true
      sleep "$SLEEP_BETWEEN"
    fi

    # ── Phase 3: QUICK QA ──────────────────────────────────────────
    if [[ "$SKIP_QA" = false ]]; then
      echo -e "\n${GREEN}▶ Phase 3: QUICK QA${NC}"
      run_claude "qa" "$MAX_TURNS" "$(build_qa_prompt)" || true
      sleep "$SLEEP_BETWEEN"
    fi

    # ── Phase 4: HANDOFF ───────────────────────────────────────────
    echo -e "\n${GREEN}▶ Phase 4: HANDOFF${NC}"
    local handoff_output_file="${LOG_DIR}/${i}-handoff.log"
    run_claude "handoff" 20 "$(build_handoff_prompt "$i")" || true

    # Read handoff output to check for completion
    local handoff_output=""
    if [[ -f "$handoff_output_file" ]]; then
      handoff_output=$(cat "$handoff_output_file")
    fi

    # ── Phase 5: GIT PUSH + VERSION HISTORY ────────────────────────
    echo -e "\n${GREEN}▶ Phase 5: GIT PUSH + VERSION LOG${NC}"

    local iter_status="working"
    if git log --oneline -5 2>/dev/null | grep -q "\[ralph-${i}-fail\]"; then
      iter_status="failed"
    fi

    local iter_summary
    iter_summary=$(git log --oneline -1 --format="%s" 2>/dev/null || echo "no commit")

    # Commit planning files first
    git add task_plan.md progress.md findings.md handoff.md questions.md 2>/dev/null || true
    if ! git diff --cached --quiet 2>/dev/null; then
      git commit -m "[ralph-meta-${i}] Update planning files" 2>/dev/null || true
    fi

    # Then log + push (version-history.md committed inside git_push_and_log)
    git_push_and_log "$i" "$iter_status" "$iter_summary"

    # Timing
    local iter_end iter_duration
    iter_end=$(date +%s)
    iter_duration=$((iter_end - iter_start))
    echo -e "  ${CYAN}⏱ Iteration ${i} total: ${iter_duration}s${NC}"

    # ── Check for completion ─────────────────────────────────────────
    if echo "$handoff_output" | grep -q "RALPH_PIPELINE_COMPLETE"; then
      echo ""
      echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
      echo -e "${GREEN}  ✅ ALL TASKS COMPLETE — Iteration ${i}${NC}"
      echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"

      # ── Final: BUG HUNTER + VERIFICATION ─────────────────────────
      if [[ "$SKIP_BUG_HUNTER" = false ]] && [[ "$DRY_RUN" = false ]]; then
        echo ""
        echo -e "${MAGENTA}▶ Final 1/2: BUG HUNTER SWARM${NC}"
        run_claude "bug-hunter" 200 "$(build_bug_hunter_prompt)" || true

        sleep "$SLEEP_BETWEEN"

        echo ""
        echo -e "${MAGENTA}▶ Final 2/2: VERIFY FINDINGS${NC}"
        run_claude "verify-bugs" 100 "$(build_verify_bugs_prompt)" || true

        # Commit verified fixes
        git add task_plan.md progress.md findings.md 2>/dev/null || true
        local changed_src_files
        changed_src_files=$(git diff --name-only HEAD 2>/dev/null | grep -v -E '(\.md$|\.tmp$|\.log$)' || true)
        if [[ -n "$changed_src_files" ]]; then
          echo "$changed_src_files" | xargs git add 2>/dev/null || true
        fi
        if ! git diff --cached --quiet 2>/dev/null; then
          git commit -m "[ralph-final] Bug hunter verified fixes"
        fi

        git_push_and_log "final" "verified" "Bug hunter verified"
      fi

      # ── Summary ──────────────────────────────────────────────────
      local total_end total_duration
      total_end=$(date +%s)
      total_duration=$((total_end - total_start))
      local total_min=$((total_duration / 60))
      local total_sec=$((total_duration % 60))

      rm -f .ralph-phase*-output.tmp

      echo ""
      echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
      echo -e "${GREEN}  Pipeline complete. ${total_min}m ${total_sec}s total.${NC}"
      echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
      echo ""
      echo "  Files:"
      echo "    task_plan.md        — final task status"
      echo "    progress.md         — full iteration log"
      echo "    findings.md         — discoveries + quality issues"
      echo "    version-history.md  — commit tracking (working/broken)"
      echo "    questions.md        — questions Claude had (review + answer)"
      if [[ "$SKIP_BUG_HUNTER" = false ]]; then
        echo "    .bug-hunter/VERIFIED-REPORT.md — verified findings"
      fi
      echo "    .ralph-logs/        — raw output from every phase"
      echo ""
      echo "  Commands:"
      echo "    git log --oneline --grep='ralph'   — see all ralph commits"
      echo "    cat version-history.md             — find when things broke"
      echo ""
      exit 0
    fi

    # ── Progress ────────────────────────────────────────────────────
    local done_count remaining blocked in_prog
    done_count=$(grep -c '\[x\]' task_plan.md 2>/dev/null || echo 0)
    remaining=$(grep -c '\[ \]' task_plan.md 2>/dev/null || echo 0)
    blocked=$(grep -c '\[B\]' task_plan.md 2>/dev/null || echo 0)
    in_prog=$(grep -c '\[~\]' task_plan.md 2>/dev/null || echo 0)

    # ── Check for blocking questions ───────────────────────────────────
    local blocking_q
    blocking_q=$(grep -c "Blocking.*yes" questions.md 2>/dev/null || echo 0)
    if [[ $blocking_q -gt 0 ]]; then
      echo ""
      echo -e "  ${YELLOW}⚠ ${blocking_q} blocking question(s) in questions.md${NC}"
      echo -e "  ${YELLOW}  Review and answer them, then re-run with: --resume $((i + 1))${NC}"
    fi

    echo ""
    echo -e "  ${CYAN}Progress:${NC} ${GREEN}${done_count} done${NC} | ${YELLOW}${in_prog} wip${NC} | ${NC}${remaining} todo${NC} | ${RED}${blocked} blocked${NC}"
  done

  # ── Max iterations reached ──────────────────────────────────────────────
  local total_end total_duration
  total_end=$(date +%s)
  total_duration=$((total_end - total_start))

  echo ""
  echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
  echo -e "${RED}  ⚠ MAX ITERATIONS (${MAX_ITERATIONS}) REACHED — ${total_duration}s total${NC}"
  echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
  echo "  Resume: ./ralph-pipeline.sh --resume $((MAX_ITERATIONS + 1)) --max-iterations $((MAX_ITERATIONS + 20))"

  rm -f .ralph-phase*-output.tmp
  exit 1
}

main
