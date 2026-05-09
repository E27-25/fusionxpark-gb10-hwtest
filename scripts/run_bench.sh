#!/usr/bin/env bash
# Safe benchmark launcher — survives SSH disconnects via tmux
# Usage:
#   ./run_bench.sh hw        # run phase05_hw_bench.py
#   ./run_bench.sh phase1    # run all phase1 inference benchmarks
#   ./run_bench.sh all       # run everything (run_all.sh)
#   ./run_bench.sh attach    # reattach to running session
#   ./run_bench.sh log       # tail the live log
#   ./run_bench.sh status    # show results so far

BASE=/home/student/Desktop/Test
LOG=$BASE/logs/bench_run.log
mkdir -p "$BASE/logs"

export PATH="$HOME/.local/bin:$PATH"
SESSION="gb10bench"

cmd="${1:-hw}"

attach() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Attaching to session '$SESSION'..."
        tmux attach -t "$SESSION"
    else
        echo "No session named '$SESSION' is running."
        echo "Start one with: ./run_bench.sh hw"
    fi
}

tail_log() {
    echo "Tailing $LOG  (Ctrl+C to stop)"
    tail -f "$LOG"
}

show_status() {
    export PATH="$HOME/.local/bin:$PATH"
    python3 "$BASE/tracker.py"
    echo ""
    echo "Recent log (last 30 lines):"
    tail -30 "$LOG" 2>/dev/null || echo "  (no log yet)"
}

launch() {
    local script="$1"
    local label="$2"

    # Kill old session if it exists but is idle
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Session '$SESSION' already exists."
        echo "  attach : ./run_bench.sh attach"
        echo "  kill   : tmux kill-session -t $SESSION"
        exit 1
    fi

    echo "Launching '$label' in tmux session '$SESSION'..."
    echo "Log: $LOG"
    echo ""
    echo "  To watch live output : ./run_bench.sh log"
    echo "  To reattach terminal : ./run_bench.sh attach"
    echo "  To check results     : ./run_bench.sh status"
    echo "  Safe to close SSH    : yes — tmux keeps it running"
    echo ""

    # Create detached tmux session, run script, log everything
    tmux new-session -d -s "$SESSION" -x 220 -y 50 \
        "export PATH=$HOME/.local/bin:\$PATH; \
         echo '=== START $(date) ===' | tee -a $LOG; \
         $script 2>&1 | tee -a $LOG; \
         echo '=== DONE $(date) ===' | tee -a $LOG; \
         echo 'Session complete. Press Enter to close.'; read"

    echo "Session started. Detach anytime with Ctrl+B then D."
    tmux attach -t "$SESSION"
}

case "$cmd" in
    install|setup)
        launch "bash $BASE/phase0_install_quick.sh" "ML Stack Install"
        ;;
    hw|hw_bench)
        launch "python3 $BASE/phase05_hw_bench.py" "Hardware Benchmark"
        ;;
    phase1)
        launch "bash $BASE/run_all.sh --phase 1" "Phase 1 Inference"
        ;;
    phase2)
        launch "bash $BASE/run_all.sh --phase 2" "Phase 2 Training"
        ;;
    phase3)
        launch "bash $BASE/run_all.sh --phase 3" "Phase 3 Evaluation"
        ;;
    all)
        launch "bash $BASE/run_all.sh" "Full Experiment Suite"
        ;;
    smoke)
        launch "bash $BASE/run_all.sh --smoke-test" "Smoke Test (all phases)"
        ;;
    attach|a)
        attach
        ;;
    log|l)
        tail_log
        ;;
    status|s)
        show_status
        ;;
    kill)
        tmux kill-session -t "$SESSION" 2>/dev/null && echo "Session killed." || echo "No session to kill."
        ;;
    *)
        echo "Usage: $0 {install|hw|phase1|phase2|phase3|all|smoke|attach|log|status|kill}"
        echo ""
        echo "  install  Install ML stack (transformers, vLLM, bitsandbytes, etc.)"
        echo "  hw       Run hardware characterization (phase05_hw_bench.py)"
        echo "  phase1   Run all inference benchmarks"
        echo "  phase2   Run all training experiments"
        echo "  all      Run complete experiment suite"
        echo "  smoke    Run smoke test (20 steps each)"
        echo "  attach   Reattach to running tmux session"
        echo "  log      Tail the live log file"
        echo "  status   Show experiment progress + recent log"
        echo "  kill     Kill the running session"
        ;;
esac
