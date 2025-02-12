/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2020 The Stockfish developers

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "evaluate.h"
#include "misc.h"
#include "movegen.h"
#include "movepick.h"
#include "search.h"
#include "thread.h"
#include "timeman.h"
#include "tt.h"
#include "uci.h"


static int peek_stdin_for_stop_and_ponder(int block);

LimitsType Limits;

// Different node types, used as template parameter
enum { NonPV, PV };

static const uint64_t ttHitAverageWindow = 4096;
static const uint64_t ttHitAverageResolution = 1024;

INLINE int futility_margin(Depth d, bool improving) {
    return 174 * (d - improving);
}

// Reductions lookup tables, initialized at startup
// f(i) = (21.3 + 2 * log(1.00)) * log(i + 0.25 * log(i));
static uint8_t Reductions[MAX_MOVES] = {
       0,   0,  16,  25,  31,  35,  39,  42,  45,  48,  50,  52,  54,  55,  57,  58,
      59,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  70,  71,  72,  73,  73,
      74,  75,  75,  76,  76,  77,  77,  78,  79,  79,  80,  80,  81,  81,  81,  82,
      82,  83,  83,  84,  84,  84,  85,  85,  86,  86,  86,  87,  87,  87,  88,  88,
      88,  89,  89,  89,  90,  90,  90,  91,  91,  91,  91,  92,  92,  92,  93,  93,
      93,  93,  94,  94,  94,  94,  95,  95,  95,  95,  96,  96,  96,  96,  97,  97,
      97,  97,  97,  98,  98,  98,  98,  98,  99,  99,  99,  99,  99, 100, 100, 100,
     100, 100, 101, 101, 101, 101, 101, 102, 102, 102, 102, 102, 102, 103, 103, 103,
     103, 103, 103, 104, 104, 104, 104, 104, 104, 104, 105, 105, 105, 105, 105, 105,
     106, 106, 106, 106, 106, 106, 106, 107, 107, 107, 107, 107, 107, 107, 108, 108,
     108, 108, 108, 108, 108, 108, 109, 109, 109, 109, 109, 109, 109, 109, 110, 110,
     110, 110, 110, 110, 110, 110, 110, 111, 111, 111, 111, 111, 111, 111, 111, 112,
     112, 112, 112, 112, 112, 112, 112, 112, 112, 113, 113, 113, 113, 113, 113, 113,
     113, 113, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 115, 115, 115, 115,
     115, 115, 115, 115, 115, 115, 115, 116, 116, 116, 116, 116, 116, 116, 116, 116,
     116, 116, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 118, 118
 };

INLINE Depth reduction(int i, Depth d, int mn) {
    int r = ((int) Reductions[d]) * ((int) Reductions[mn]);
    return (r + 726) / 1024 + (!i && r > 852);
}

INLINE int futility_move_count(bool improving, Depth depth) {
    //  return (3 + depth * depth) / (2 - improving);
    return improving ? 2 + depth * depth : (2 + depth * depth) / 2;
}

// History and stats update bonus, based on depth
static Value stat_bonus(Depth depth) {
    int d = depth;
    return d > 20 ? 46 : 6 * d * d + 345 * d - 184;
}

// Add a small random component to draw evaluations to keep search dynamic
// and to avoid three-fold blindness. (Yucks, ugly hack)
static Value value_draw(Position *pos) {
    return VALUE_DRAW + 2 * (pos->nodes & 1) - 1;
}

static Value search_PV(Position *pos, Stack *ss, Value alpha, Value beta, Depth depth);
static Value search_NonPV(Position *pos, Stack *ss, Value alpha, Depth depth, bool cutNode);

static Value qsearch_PV_true(Position *pos, Stack *ss, Value alpha, Value beta, Depth depth);
static Value qsearch_PV_false(Position *pos, Stack *ss, Value alpha, Value beta, Depth depth);
static Value qsearch_NonPV_true(Position *pos, Stack *ss, Value alpha, Depth depth);
static Value qsearch_NonPV_false(Position *pos, Stack *ss, Value alpha, Depth depth);

static Value value_to_tt(Value v, int ply);
static Value value_from_tt(Value v, int ply, int r50c);
static void update_pv(Move *pv, Move move, Move *childPv);
static void update_cm_stats(Stack *ss, Piece pc, Square s, int bonus);
static void update_quiet_stats(const Position *pos, Stack *ss, Move move, int bonus);
static void update_capture_stats(const Position *pos, Move move, Move *captures, int captureCnt, int bonus);
static void check_time(void);
static void stable_sort(RootMove *rm, int num);
static void uci_print_pv(Position *pos, Depth depth, Value alpha, Value beta);
static int extract_ponder_from_tt(RootMove *rm, Position *pos);

// search_init() is called during startup to initialize various lookup tables

void search_init(void) {

    // for (int i = 1; i < MAX_MOVES; i++)
    //     Reductions[i] = (21.3 + 2 * log(1.00)) * log(i + 0.25 * log(i));

}

// search_clear() resets search state to zero, to obtain reproducible results

void search_clear(void) {

    #ifdef KAGGLE
        return;
    #else
        Time.availableNodes = 0;

        tt_clear();

        Position *pos = Threads.pos[0];
        stats_clear(pos->counterMoves);
        stats_clear(pos->mainHistory);
        stats_clear(pos->captureHistory);
        stats_clear(pos->counterMoveHistory);

        stats_clear(pos->pawnCorrectionHistory);
        stats_clear(pos->minorCorrectionHistory);
        stats_clear(pos->majorCorrectionHistory);
        stats_clear(pos->nonPawnCorrectionHistory);
        stats_clear(pos->counterCorrectionHistory);

        for (int d = 0; d < 2; d++)
            for (int j = 0; j < 6; j++)
                for (int k = 0; k < 64; k++)
                (*pos->counterMoveHistory)[0][0][d][j][k] = CounterMovePruneThreshold - 1;

        mainThread.previousScore = VALUE_INFINITE;
        mainThread.previousTimeReduction = 1;
    #endif
}


// mainthread_search() is called by the main thread when the program
// receives the UCI 'go' command. It searches from the root position and
// outputs the "bestmove".

void mainthread_search(void) {
    Position *pos = Threads.pos[0];
    Color us = stm();
    time_init(us, game_ply());
    tt_new_search();
    char buf[16];

    if (pos->rootMoves->size > 0) {
        Threads.pos[0]->bestMoveChanges = 0;
        thread_search(pos); // Let's start searching!
    }

    // Max depth reached during infinite or ponder; await stop or ponderhit

    #ifdef KAGGLE
        if (!Threads.stop && Threads.ponder)
            peek_stdin_for_stop_and_ponder(1);
    #else
        if (!Threads.stop && (Threads.ponder || Limits.infinite))
            peek_stdin_for_stop_and_ponder(1);
    #endif

    // Stop the other threads if they have not stopped already
    Threads.stop = true;

    #ifndef KAGGLE
        // Special case for no moves
        if (pos->rootMoves->size <= 0) {
            pos->rootMoves->move[0].pv[0] = 0;
            pos->rootMoves->move[0].pvSize = 1;
            pos->rootMoves->size++;
            printf("info depth 0 score %s\n", uci_value(buf, checkers() ? -VALUE_MATE : VALUE_DRAW));
            fflush(stdout);
        }
    #endif

    mainThread.previousScore = pos->rootMoves->move[0].score;

    // Always send a new PV
    uci_print_pv(pos, pos->completedDepth, -VALUE_INFINITE, VALUE_INFINITE);

    printf("bestmove %s", uci_move(buf, pos->rootMoves->move[0].pv[0]));

    if (pos->rootMoves->move[0].pvSize > 1 || extract_ponder_from_tt(&pos->rootMoves->move[0], pos))
        printf(" ponder %s", uci_move(buf, pos->rootMoves->move[0].pv[1]));

    printf("\n");
    fflush(stdout);
}

// thread_search() is the main iterative deepening loop. It calls search()
// repeatedly with increasing depth until the allocated thinking time has
// been consumed, the user stops the search, or the maximum search depth is
// reached.

void __attribute__((minsize, cold)) thread_search(Position *pos) {
    Value bestValue, alpha, beta, delta;
    Move pv[MAX_PLY + 1];
    Move lastBestMove = 0;
    Depth lastBestMoveDepth = 0;
    double timeReduction = 1.0, totBestMoveChanges = 0;
    int iterIdx = 0;

    Stack *ss = pos->st; // At least the seventh element of the allocated array.
    for (int i = -7; i < 3; i++) memset(SStackBegin(ss[i]), 0, SStackSize);

    (ss - 1)->endMoves = pos->moveList;

    for (int i = -7; i < 0; i++) ss[i].history = &(*pos->counterMoveHistory)[0][0]; // Use as sentinel

    for (int i = 0; i <= MAX_PLY; i++) ss[i].ply = i;
    ss->pv = pv;

    bestValue = delta = alpha = -VALUE_INFINITE;
    beta = VALUE_INFINITE;
    pos->completedDepth = 0;

    if (mainThread.previousScore == VALUE_INFINITE)
        for (int i = 0; i < 4; i++) mainThread.iterValue[i] = VALUE_ZERO;
    else
        for (int i = 0; i < 4; i++) mainThread.iterValue[i] = mainThread.previousScore;

    RootMoves *rm = pos->rootMoves;
    pos->ttHitAverage = ttHitAverageWindow * ttHitAverageResolution / 2;
    int searchAgainCounter = 0;

    // Iterative deepening loop until requested to stop or the target depth
    // is reached.
    while (  ++pos->rootDepth < MAX_PLY  && !Threads.stop
           #ifndef KAGGLE
           && !(Limits.depth && pos->rootDepth > Limits.depth)
           #endif
    ){

        // Age out PV variability metric
        totBestMoveChanges /= 2;

        // Save the last iteration's scores before first PV line is searched and
        // all the move scores except the (new) PV are set to -VALUE_INFINITE.
        for (int idx = 0; idx < rm->size; idx++) rm->move[idx].previousScore = rm->move[idx].score;

        int pvFirst = 0, pvLast = 0;

        if (!Threads.increaseDepth) searchAgainCounter++;

        // Single PV loop. We perform a full root search for each PV line
        for (int pvIdx = 0; pvIdx < 1 && !Threads.stop; pvIdx++) {
            pos->pvIdx = pvIdx;
            if (pvIdx == pvLast) {
                pvFirst = pvLast;
                for (pvLast++; pvLast < rm->size; pvLast++) {
                }
                pos->pvLast = pvLast;
            }

            // Reset aspiration window starting size
            if (pos->rootDepth >= 4) {
                Value previousScore = rm->move[pvIdx].averageScore;
                delta = 19;
                alpha = max(previousScore - delta, -VALUE_INFINITE);
                beta = min(previousScore + delta, VALUE_INFINITE);
            }

            // Start with a small aspiration window and, in the case of a fail
            // high/low, re-search with a bigger window until we're not failing
            // high/low anymore.
            pos->failedHighCnt = 0;
            while (true) {
                Depth adjustedDepth = max(1, pos->rootDepth - pos->failedHighCnt - searchAgainCounter);
                bestValue = search_PV(pos, ss, alpha, beta, adjustedDepth);

                // Bring the best move to the front. It is critical that sorting
                // is done with a stable algorithm because all the values but the
                // first and eventually the new best one are set to -VALUE_INFINITE
                // and we want to keep the same order for all the moves except the
                // new PV that goes to the front. Note that in case of MultiPV
                // search the already searched PV lines are preserved.
                stable_sort(&rm->move[pvIdx], pvLast - pvIdx);

                // If search has been stopped, we break immediately. Sorting and
                // writing PV back to TT is safe because RootMoves is still
                // valid, although it refers to the previous iteration.
                if (Threads.stop) break;

                // In case of failing low/high increase aspiration window and
                // re-search, otherwise exit the loop.
                if (bestValue <= alpha) {
                    beta = (alpha + beta) / 2;
                    alpha = max(bestValue - delta, -VALUE_INFINITE);

                    pos->failedHighCnt = 0;
                    Threads.stopOnPonderhit = false;
                } else if (bestValue >= beta) {
                    beta = min(bestValue + delta, VALUE_INFINITE);
                    pos->failedHighCnt++;
                } else
                    break;

                delta += delta / 4 + 5;

                assert(alpha >= -VALUE_INFINITE && beta <= VALUE_INFINITE);
            }

            // Sort the PV lines searched so far and update the GUI
            stable_sort(&rm->move[pvFirst], pvIdx - pvFirst + 1);
        }

        if (!Threads.stop) pos->completedDepth = pos->rootDepth;

        if (rm->move[0].pv[0] != lastBestMove) {
            lastBestMove = rm->move[0].pv[0];
            lastBestMoveDepth = pos->rootDepth;
        }

        // Do we have time for the next iteration? Can we stop searching now?
        if (use_time_management() && !Threads.stop && !Threads.stopOnPonderhit) {

            const double tm_falling_base      = 320.2331960;
            const double tm_falling_mul1      =   5.8774509;
            const double tm_falling_mul2      =   6.3854055;
            const double tm_falling_clamp_min =   0.5094103;
            const double tm_falling_clamp_max =   1.5103365;
            const double tm_time_reduct_1     =   1.8279061;
            const double tm_time_reduct_2     =   0.8794308;
            const double tm_time_reduct_3     =   1.6430318;
            const double tm_time_reduct_4     =   2.0994111;
            const double tm_instab_base       =   1.0772406;
            const double tm_instab_mul        =   1.9887375;
            const double tm_cutoff_factor     =   0.5784239;

            double fallingEval =
                (  tm_falling_base
                 + tm_falling_mul1 * (mainThread.previousScore - bestValue)
                 + tm_falling_mul2 * (mainThread.iterValue[iterIdx] - bestValue)) / 825.0;

            fallingEval = clamp(fallingEval, tm_falling_clamp_min, tm_falling_clamp_max);

            // If the best move is stable over several iterations, reduce time
            // accordingly
            timeReduction = lastBestMoveDepth + 9 < pos->completedDepth ? tm_time_reduct_1 : tm_time_reduct_2;
            double reduction = (tm_time_reduct_3 + mainThread.previousTimeReduction) / (tm_time_reduct_4 * timeReduction);

            // Use part of the gained time from a previous stable move for this move
            totBestMoveChanges += Threads.pos[0]->bestMoveChanges;
            Threads.pos[0]->bestMoveChanges = 0;

            double bestMoveInstability = tm_instab_base + tm_instab_mul * totBestMoveChanges;

            double totalTime = time_optimum() * fallingEval * reduction * bestMoveInstability;

            // In the case of a single legal move, cap total time to 500ms.
            if (rm->size == 1) totalTime = min(500.0, totalTime);

            // Stop the search if we have exceeded the totalTime
            if (time_elapsed() > totalTime) {
                // If we are allowed to ponder do not stop the search now but
                // keep pondering until the GUI sends "ponderhit" or "stop".
                if (Threads.ponder)
                    Threads.stopOnPonderhit = true;
                else
                    Threads.stop = true;
            } else if (!Threads.ponder && time_elapsed() > totalTime * tm_cutoff_factor)
                Threads.increaseDepth = false;
            else
                Threads.increaseDepth = true;
        }

        mainThread.iterValue[iterIdx] = bestValue;
        iterIdx = (iterIdx + 1) & 3;
    }

    mainThread.previousTimeReduction = timeReduction;
}

// search_node() is the main search function template for both PV
// and non-PV nodes
Value search_node(Position *pos, Stack *ss, Value alpha, Value beta, Depth depth, bool cutNode, const int NT) {

    const bool PvNode = NT == PV;
    const bool rootNode = PvNode && ss->ply == 0;
    const Depth maxNextDepth = rootNode ? depth : depth + 1;

    // Dive into quiescense search when the depth reaches zero
    if (depth <= 0)
        return PvNode
                   ? checkers() ? qsearch_PV_true(pos, ss, alpha, beta, 0) : qsearch_PV_false(pos, ss, alpha, beta, 0)
                   : checkers() ? qsearch_NonPV_true(pos, ss, alpha, 0) : qsearch_NonPV_false(pos, ss, alpha, 0);

    assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));
    assert(0 < depth && depth < MAX_PLY);
    assert(!(PvNode && cutNode));

    Move pv[MAX_PLY + 1], capturesSearched[32], quietsSearched[64];
    TTEntry *tte;
    Key posKey;
    Move ttMove, move, excludedMove, bestMove;
    Depth extension, newDepth;
    Value bestValue, value, ttValue, eval, maxValue, probCutBeta, rawEval;
    bool formerPv, givesCheck, improving;
    bool captureOrPromotion, inCheck, moveCountPruning;
    bool ttCapture, singularQuietLMR;
    Piece movedPiece;
    int moveCount, captureCount, quietCount;

    // Step 1. Initialize node
    inCheck = checkers();
    moveCount = captureCount = quietCount = ss->moveCount = 0;
    bestValue = -VALUE_INFINITE;
    maxValue = VALUE_INFINITE;

    // Check for the available remaining time
    if (pos->resetCalls) {
        pos->resetCalls = false;

        #ifdef KAGGLE
            pos->callsCnt = 1024;
        #else
            pos->callsCnt = Limits.nodes ? min(1024, Limits.nodes / 1024) : 1024;
        #endif
    }
    if (--pos->callsCnt <= 0) {
        pos->resetCalls = true;
        check_time();
    }

    if (!rootNode) {
        // Step 2. Check for aborted search and immediate draw
        if (Threads.stop || is_draw(pos) || ss->ply >= MAX_PLY)
            return ss->ply >= MAX_PLY && !inCheck ? evaluate(pos) : value_draw(pos);

        // Step 3. Mate distance pruning. Even if we mate at the next move our
        // score would be at best mate_in(ss->ply+1), but if alpha is already
        // bigger because a shorter mate was found upward in the tree then
        // there is no need to search because we will never beat the current
        // alpha. Same logic but with reversed signs applies also in the
        // opposite condition of being mated instead of giving mate. In this
        // case return a fail-high score.
        if (PvNode) {
            alpha = max(mated_in(ss->ply), alpha);
            beta = min(mate_in(ss->ply + 1), beta);
            if (alpha >= beta) return alpha;
        } else { // avoid assignment to beta (== alpha+1)
            if (alpha < mated_in(ss->ply)) return mated_in(ss->ply);
            if (alpha >= mate_in(ss->ply + 1)) return alpha;
        }
    }

    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    (ss + 1)->ttPv = false;
    (ss + 1)->excludedMove = bestMove = 0;
    (ss + 2)->killers[0] = (ss + 2)->killers[1] = 0;
    Square prevSq = to_sq((ss - 1)->currentMove);

    // Initialize statScore to zero for the grandchildren of the current
    // position. So the statScore is shared between all grandchildren and only
    // the first grandchild starts with startScore = 0. Later grandchildren
    // start with the last calculated statScore of the previous grandchild.
    // This influences the reduction rules in LMR which are based on the
    // statScore of the parent position.
    if (!rootNode) (ss + 2)->statScore = 0;

    // Step 4. Transposition table lookup. We don't want the score of a
    // partial search to overwrite a previous full search TT value, so we
    // use a different position key in case of an excluded move.
    excludedMove = ss->excludedMove;
    posKey = !excludedMove ? key() : key() ^ make_key(excludedMove);
    tte = tt_probe(posKey, &ss->ttHit);
    ttValue = ss->ttHit ? value_from_tt(tte_value(tte), ss->ply, rule50_count()) : VALUE_NONE;
    ttMove = rootNode ? pos->rootMoves->move[pos->pvIdx].pv[0] : ss->ttHit ? tte_move(tte) : 0;
    if (!excludedMove) ss->ttPv = PvNode || (ss->ttHit && tte_is_pv(tte));
    formerPv = ss->ttPv && !PvNode;

    // pos->ttHitAverage can be used to approximate the running average of ttHit
    pos->ttHitAverage =
        (ttHitAverageWindow - 1) * pos->ttHitAverage / ttHitAverageWindow + ttHitAverageResolution * ss->ttHit;

    // At non-PV nodes we check for an early TT cutoff.
    if (!PvNode && ss->ttHit && tte_depth(tte) > depth - (tte_value(tte) <= beta) && ttValue != VALUE_NONE // Possible in case of TT access race.
        && (ttValue >= beta ? (tte_bound(tte) & BOUND_LOWER) : (tte_bound(tte) & BOUND_UPPER))) {
        // If ttMove is quiet, update move sorting heuristics on TT hit.
        if (ttMove) {
            if (ttValue >= beta) {
                if (!is_capture_or_promotion(pos, ttMove))
                    update_quiet_stats(pos, ss, ttMove, stat_bonus(depth));

                // Extra penalty for early quiet moves of the previous ply
                if ((ss - 1)->moveCount <= 2 && !captured_piece())
                    update_cm_stats(ss - 1, piece_on(prevSq), prevSq, -stat_bonus(depth + 1));
            }
            // Penalty for a quiet ttMove that fails low
            else if (!is_capture_or_promotion(pos, ttMove)) {
                int penalty = -stat_bonus(depth);
                history_update(*pos->mainHistory, stm(), ttMove, penalty);
                update_cm_stats(ss, moved_piece(ttMove), to_sq(ttMove), penalty);
            }
        }
        if (rule50_count() < 90) return ttValue;
    }

    // Step 6. Static evaluation of the position
    if (inCheck) {

        // Skip early pruning when in check
        rawEval = ss->staticEval = eval = VALUE_NONE;
        improving = false;
        goto moves_loop;
    }

    else if (ss->ttHit) {

        // Never assume anything about values stored in TT
        if ((eval = tte_eval(tte)) == VALUE_NONE) eval = evaluate(pos);

        rawEval = ss->staticEval = eval;

        if (eval == VALUE_DRAW) eval = value_draw(pos);

        Value adjust = corrhist_adjustment(pos);
        eval += adjust;
        ss->staticEval += adjust;

        // Can ttValue be used as a better position evaluation?
        if (    ttValue != VALUE_NONE
            && (tte_bound(tte) & (ttValue > eval ? BOUND_LOWER : BOUND_UPPER)))
            eval = ttValue;
    }

    else {

        if ((ss - 1)->currentMove != MOVE_NULL)
            rawEval = ss->staticEval = eval = evaluate(pos);
        else
            rawEval = ss->staticEval = eval = -(ss - 1)->staticEval + 2 * Tempo;

        Value adjust = corrhist_adjustment(pos);
        eval += adjust;
        ss->staticEval += adjust;

        tte_save(tte, posKey, VALUE_NONE, ss->ttPv, BOUND_NONE, DEPTH_NONE, 0, rawEval);
    }

    if (move_is_ok((ss - 1)->currentMove) && !(ss - 1)->checkersBB && !captured_piece()) {
        int bonus = clamp(-depth * 4 * ((ss - 1)->staticEval + ss->staticEval - 2 * Tempo), -1000, 1000);
        history_update(*pos->mainHistory, !stm(), (ss - 1)->currentMove, bonus);
    }

    improving = (ss - 2)->staticEval == VALUE_NONE
                        ? (ss->staticEval > (ss - 4)->staticEval || (ss - 4)->staticEval == VALUE_NONE)
                        : ss->staticEval > (ss - 2)->staticEval;

    // Razoring
    if (!PvNode && eval < alpha - 195 - 143 * depth * depth)
        return checkers() ? qsearch_NonPV_true(pos, ss, alpha - 1, 0) : qsearch_NonPV_false(pos, ss, alpha - 1, 0);

    // Step 7. Futility pruning: child node
    if (!PvNode && depth < 9 && eval - futility_margin(depth, improving) >= beta &&
        eval < VALUE_KNOWN_WIN)
        return beta + (eval - beta) / 2;

    // Step 8. Null move search with verification search (is omitted in PV nodes)
    if (!PvNode && (ss - 1)->currentMove != MOVE_NULL && (ss - 1)->statScore < 26226 && eval >= beta &&
        eval >= ss->staticEval && ss->staticEval >= beta - 14 * depth - 34 * improving + 146 * ss->ttPv + 183 &&
        !excludedMove && non_pawn_material_c(stm())) {
        assert(eval - beta >= 0);

        // Null move dynamic reduction based on depth and value
        Depth R = (1200 + 77 * depth) / 256 + min((eval - beta) / 149, 3);

        ss->currentMove = MOVE_NULL;
        ss->history = &(*pos->counterMoveHistory)[0][0];

        do_null_move(pos);
        ss->endMoves = (ss - 1)->endMoves;
        Value nullValue = -search_NonPV(pos, ss + 1, -beta, depth - R, !cutNode);
        undo_null_move(pos);

        // Do not return unproven mate or TB scores
        if (nullValue >= beta) return nullValue >= VALUE_TB_WIN_IN_MAX_PLY ? beta : nullValue;
    }

    probCutBeta = beta + 126 - 52 * improving;

    // Step 9. ProbCut
    // If we have a good enough capture and a reduced search returns a value
    // much above beta, we can (almost) safely prune the previous move.
    if (    !PvNode
        &&   depth >= 3
        &&   abs(beta) < VALUE_TB_WIN_IN_MAX_PLY
        && !(ss->ttHit && tte_depth(tte) >= depth - 3 && ttValue != VALUE_NONE && ttValue < probCutBeta)) {

        mp_init_pc(pos, ttMove, probCutBeta - ss->staticEval);
        int probCutCount = 2 + 2 * cutNode;
        bool ttPv = ss->ttPv;
        ss->ttPv = false;

        Depth probcutDepth = max_int(depth - 4, 0);

        while ((move = next_move(pos, 0)) && probCutCount)
            if (move != excludedMove && is_legal(pos, move)) {
                assert(is_capture_or_promotion(pos, move));

                captureOrPromotion = true;
                probCutCount--;

                ss->currentMove = move;
                ss->history = &(*pos->counterMoveHistory)[piece_to_history_conv(moved_piece(move))][to_sq(move)];
                givesCheck = gives_check(pos, ss, move);
                do_move(pos, move, givesCheck);

                // Perform a preliminary qsearch to verify that the move holds
                value = givesCheck ? -qsearch_NonPV_true(pos, ss + 1, -probCutBeta, 0)
                                   : -qsearch_NonPV_false(pos, ss + 1, -probCutBeta, 0);

                // If the qsearch held, perform the regular search
                if (value >= probCutBeta && probcutDepth > 0)
                    value = -search_NonPV(pos, ss + 1, -probCutBeta, probcutDepth, !cutNode);

                undo_move(pos, move);
                if (value >= probCutBeta) {
                    if (!(ss->ttHit && tte_depth(tte) >= depth - 3 && ttValue != VALUE_NONE))
                        tte_save(tte, posKey, value_to_tt(value, ss->ply), ttPv, BOUND_LOWER, depth - 3, move, rawEval);
                    return value;
                }
            }
        ss->ttPv = ttPv;
    }

    // Step 10. If the position is not in TT, decrease depth by 2
    if (PvNode && depth >= 6 && !ttMove) depth -= 2;
    if (cutNode && depth >= 9 && !ttMove) depth--;
moves_loop: // When in check search starts from here

    ttCapture = ttMove && is_capture_or_promotion(pos, ttMove);

    // Step 11. A small Probcut idea, when we are in check
    probCutBeta = beta + 447;
    if (inCheck && !PvNode && depth >= 4 && ttCapture && (tte_bound(tte) & BOUND_LOWER) &&
        tte_depth(tte) >= depth - 3 && ttValue >= probCutBeta && abs(ttValue) <= VALUE_KNOWN_WIN &&
        abs(beta) <= VALUE_KNOWN_WIN)
        return probCutBeta;

    PieceToHistory *cmh = (ss - 1)->history;
    PieceToHistory *fmh = (ss - 2)->history;
    PieceToHistory *fmh2 = (ss - 4)->history;
    PieceToHistory *fmh3 = (ss - 6)->history;

    mp_init(pos, ttMove, depth, ss->ply);

    value = bestValue;
    singularQuietLMR = moveCountPruning = false;

    // Indicate PvNodes that will probably fail low if node was searched with
    // non-PV search at depth equal to or greater than current depth and the
    // result of the search was far below alpha
    bool likelyFailLow = PvNode && ttMove && (tte_bound(tte) & BOUND_UPPER) && tte_depth(tte) >= depth;
    int rangeReduction = 0;
    // Step 12. Loop through moves
    // Loop through all pseudo-legal moves until no moves remain or a beta
    // cutoff occurs
    while ((move = next_move(pos, moveCountPruning))) {
        assert(move_is_ok(move));

        if (move == excludedMove) continue;

        // Check for legality just before making the move
        if (!is_legal(pos, move)) continue;

        ss->moveCount = ++moveCount;

        if (PvNode) (ss + 1)->pv = NULL;

        extension = 0;
        captureOrPromotion = is_capture_or_promotion(pos, move);
        movedPiece = moved_piece(move);

        givesCheck = gives_check(pos, ss, move);

        // Calculate new depth for this move
        newDepth = depth - 1;

        // Step 13. Pruning at shallow depth
        if (!rootNode && non_pawn_material_c(stm()) && bestValue > VALUE_TB_LOSS_IN_MAX_PLY) {
            // Skip quiet moves if movecount exceeds our FutilityMoveCount threshold
            moveCountPruning = moveCount >= futility_move_count(improving, depth);

            // Reduced depth of the next LMR search
            int lmrDepth = max(newDepth - reduction(improving, depth, moveCount) + (rangeReduction > 2), 0);

            if (captureOrPromotion || givesCheck) {
                // Capture history based pruning when the move doesn't give check
                if (!givesCheck && lmrDepth < 1 &&
                    (*pos->captureHistory)[movedPiece][to_sq(move)][type_of_p(piece_on(to_sq(move)))] < 0)
                    continue;

                // SEE based pruning
                if (!see_test(pos, move, -243 * depth)) continue;

            } else {

                // Countermoves based pruning
                if (lmrDepth < 5 &&
                    (*cmh)[0][piece_to_history_conv(movedPiece)][to_sq(move)] +
                    (*fmh)[1][piece_to_history_conv(movedPiece)][to_sq(move)] +
                    (*fmh2)[1][piece_to_history_conv(movedPiece)][to_sq(move)]
                    < -3066 * depth + 3905)
                    continue;


                int futilityValue = ss->staticEval + 86 + 74 * lmrDepth;

                // Futility pruning: parent node
                if (    lmrDepth < 7
                    && !inCheck
                    && futilityValue <= alpha
                    && (*cmh)[0][piece_to_history_conv(movedPiece)][to_sq(move)]  +
                       (*fmh)[1][piece_to_history_conv(movedPiece)][to_sq(move)]  +
                       (*fmh2)[1][piece_to_history_conv(movedPiece)][to_sq(move)] +
                       (*fmh3)[1][piece_to_history_conv(movedPiece)][to_sq(move)] / 3 < 27488) {

                    // Treat this as real futility pruning, updating the bestValue
                    if (   bestValue <= futilityValue
                        && bestValue > VALUE_TB_LOSS_IN_MAX_PLY
                        && bestValue < VALUE_TB_WIN_IN_MAX_PLY)
                        bestValue = futilityValue;

                    continue;
                }

                // Prune moves with negative SEE at low depths and below a decreasing
                // threshold at higher depths.
                if (!see_test(pos, move, -(29 - min(lmrDepth, 18)) * lmrDepth * lmrDepth)) continue;
            }
        }

        // Step 14. Extensions

        // Singular extension search. If all moves but one fail low on a search
        // of (alpha-s, beta-s), and just one fails high on (alpha, beta), then
        // that move is singular and should be extended. To verify this we do a
        // reduced search on all the other moves but the ttMove and if the
        // result is lower than ttValue minus a margin, then we extend the ttMove.
        if (depth >= 5 && move == ttMove && !rootNode &&
            !excludedMove // No recursive singular search
                          /* &&  ttValue != VALUE_NONE implicit in the next condition */
            && abs(ttValue) < VALUE_KNOWN_WIN && (tte_bound(tte) & BOUND_LOWER) && tte_depth(tte) >= depth - 3) {
            Value singularBeta = ttValue - ((formerPv + 4) * depth) / 2;
            Depth singularDepth = newDepth / 2;
            ss->excludedMove = move;
            Move cm = ss->countermove;
            Move k1 = ss->mpKillers[0], k2 = ss->mpKillers[1];
            value = search_NonPV(pos, ss, singularBeta - 1, singularDepth, cutNode);
            ss->excludedMove = 0;

            if (value < singularBeta) {
                extension = 1;
                singularQuietLMR = !ttCapture;
                if (!PvNode && value < singularBeta - 16) extension = 2;
            }

            // Multi-cut pruning. Our ttMove is assumed to fail high, and now we
            // failed high also on a reduced search without the ttMove. So we
            // assume that this expected cut-node is not singular, i.e. multiple
            // moves fail high. We therefore prune the whole subtree by returning
            // a soft bound.
            else if (value >= beta)
                return value;

            else if (ttValue >= beta)
                extension = -3;

            else if (cutNode)
                extension = -2;

            // The call to search_NonPV with the same value of ss messed up our
            // move picker data. So we fix it.
            mp_init(pos, ttMove, depth, ss->ply);
            ss->stage++;
            ss->countermove = cm; // pedantic
            ss->mpKillers[0] = k1;
            ss->mpKillers[1] = k2;
        }

        // Add extension to new depth
        newDepth += extension;

        // Speculative prefetch as early as possible
        prefetch(tt_first_entry(key_after(pos, move)));

        // Update the current move (this must be done after singular extension
        // search)
        ss->currentMove = move;
        ss->history = &(*pos->counterMoveHistory)[piece_to_history_conv(movedPiece)][to_sq(move)];

        // Step 15. Make the move.
        do_move(pos, move, givesCheck);
        // HACK: Fix bench after introduction of 2-fold MultiPV bug
        if (rootNode) pos->st[-1].key ^= pos->rootKeyFlip;

        // Step 16. Late moves reduction / extension (LMR)
        // We use various heuristics for the children of a node after the first
        // child has been searched. In general we would like to reduce them, but
        // there are many cases where we extend a child if it has good chances
        // to be "interesting".
        if (     depth >= 2
            &&   moveCount > 1 + 2 * rootNode
            && (!captureOrPromotion || cutNode || (!PvNode && !formerPv))) {

            Depth r = reduction(improving, depth, moveCount) + (rangeReduction > 2);

            // Decrease reduction if the ttHit runing average is large
            if (pos->ttHitAverage > 377 * ttHitAverageResolution * ttHitAverageWindow / 1024) r--;

            // Decrease reduction if position is or has been on the PV and the node
            // is not likely to fail low
            if (ss->ttPv && !likelyFailLow) r -= 2;

            // Increase reduction at root and non-PV nodes when the best move
            // does not change frequently
            if ((rootNode || !PvNode) && pos->rootDepth > 10 && pos->bestMoveChanges <= 2) r++;

            if (ss->ttHit && tte_depth(tte) >= depth)
                r--;

            // Decrease reduction if ttMove has been singularly extended
            if (singularQuietLMR) r--;

            // Increase reduction if ttMove is a capture
            if (ttCapture) r++;

            if (!captureOrPromotion) {

                // Increase reduction at root if failing high
                if (rootNode) r += pos->failedHighCnt * pos->failedHighCnt * moveCount / 512;

                // Increase reduction for cut nodes
                if (cutNode) r += 2;

                ss->statScore = (*cmh)[0][piece_to_history_conv(movedPiece)][to_sq(move)] +
                                (*fmh)[1][piece_to_history_conv(movedPiece)][to_sq(move)] +
                                (*fmh2)[1][piece_to_history_conv(movedPiece)][to_sq(move)] +
                                (*pos->mainHistory)[!stm()][from_to(move)] - 4741;

                if (!inCheck) r -= ss->statScore / 15893;
            }

            int dpr =   r >= -1                   ? 0
                       : moveCount <= 5            ? 2
                       : PvNode && depth > 6       ? 1
                       : cutNode && moveCount <= 7 ? 1
                       :                             0;

            Depth d = clamp(newDepth - r, 1, newDepth + dpr);

            value = -search_NonPV(pos, ss + 1, -(alpha + 1), d, 1);


            if (value > alpha && d < newDepth) {

                const bool doDeeperSearch    = value > (bestValue + 38 + 2 * newDepth);  // (~1 Elo)
                const bool doShallowerSearch = value < bestValue + 9;                   // (~2 Elo)

                newDepth += doDeeperSearch - doShallowerSearch;

                if (newDepth > d)
                    value = -search_NonPV(pos, ss + 1, -(alpha + 1), newDepth, !cutNode);

                if (!captureOrPromotion) {
                    int bonus = value > alpha ? stat_bonus(newDepth) : -stat_bonus(newDepth);
                    update_cm_stats(ss, movedPiece, to_sq(move), bonus);
                }
            }

            if (ss->staticEval - value < 30 && depth > 7)
                rangeReduction++;
        }

        // Step 17. Full depth search when LMR is skipped or fails high.
        else if (!PvNode || moveCount > 1) {

            value = -search_NonPV(pos, ss + 1, -(alpha + 1), newDepth, !cutNode);

            // if (didLMR && !captureOrPromotion) {
            //     int bonus = value > alpha ? stat_bonus(newDepth) : -stat_bonus(newDepth);
            //
            //     update_cm_stats(ss, movedPiece, to_sq(move), bonus);
            // }

        }

        // For PV nodes only, do a full PV search on the first move or after a fail
        // high (in the latter case search only if value < beta), otherwise let the
        // parent node fail low with value <= alpha and try another move.
        if (PvNode && (moveCount == 1 || (value > alpha && (rootNode || value < beta)))) {
            (ss + 1)->pv = pv;
            (ss + 1)->pv[0] = 0;

            value = -search_PV(pos, ss + 1, -beta, -alpha, min(maxNextDepth, newDepth));
        }

        // Step 18. Undo move
        // HACK: Fix bench after introduction of 2-fold MultiPV bug
        if (rootNode) pos->st[-1].key ^= pos->rootKeyFlip;
        undo_move(pos, move);

        assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

        // Step 19. Check for a new best move
        // Finished searching the move. If a stop occurred, the return value of
        // the search cannot be trusted, and we return immediately without
        // updating best move, PV and TT.
        if (Threads.stop) return 0;

        if (rootNode) {
            RootMove *rm = NULL;
            for (int idx = 0; idx < pos->rootMoves->size; idx++)
                if (pos->rootMoves->move[idx].pv[0] == move) {
                    rm = &pos->rootMoves->move[idx];
                    break;
                }
            rm->averageScore = rm->averageScore != -VALUE_INFINITE ? (2 * value + rm->averageScore) / 3 : value;
            // PV move or new best move ?
            if (moveCount == 1 || value > alpha) {
                rm->score = value;
                rm->pvSize = 1;

                assert((ss + 1)->pv);

                for (Move *m = (ss + 1)->pv; *m; ++m) rm->pv[rm->pvSize++] = *m;

                // We record how often the best move has been changed in each
                // iteration. This information is used for time management: When
                // the best move changes frequently, we allocate some more time.
                if (moveCount > 1) pos->bestMoveChanges++;
            } else
                // All other moves but the PV are set to the lowest value: this is
                // not a problem when sorting because the sort is stable and the
                // move position in the list is preserved - just the PV is pushed up.
                rm->score = -VALUE_INFINITE;
        }

        if (value > bestValue) {
            bestValue = value;

            if (value > alpha) {
                bestMove = move;

                if (PvNode && !rootNode) // Update pv even in fail-high case
                    update_pv(ss->pv, move, (ss + 1)->pv);

                if (PvNode && value < beta) // Update alpha! Always alpha < beta
                    alpha = value;
                else {
                    assert(value >= beta); // Fail high
                    ss->statScore = 0;
                    break;
                }

                if (depth > 2 && value < VALUE_KNOWN_WIN && value > -VALUE_KNOWN_WIN) depth -= 2;
            }
        }

        if (move != bestMove) {
            if (captureOrPromotion && captureCount < 32)
                capturesSearched[captureCount++] = move;

            else if (!captureOrPromotion && quietCount < 64)
                quietsSearched[quietCount++] = move;
        }
    }

    // The following condition would detect a stop only after move loop has
    // been completed. But in this case bestValue is valid because we have
    // fully searched our subtree, and we can anyhow save the result in TT.
    /*
    if (Threads.stop)
      return VALUE_DRAW;
    */

    // Step 20. Check for mate and stalemate
    // All legal moves have been searched and if there are no legal moves,
    // it must be a mate or a stalemate. If we are in a singular extension
    // search then return a fail low score.
    if (!moveCount)
        bestValue = excludedMove ? alpha : inCheck ? mated_in(ss->ply) : VALUE_DRAW;
    else if (bestMove) {
        // Quiet best move: update move sorting heuristics
        if (!is_capture_or_promotion(pos, bestMove)) {
            int bonus =
                bestValue > beta + PawnValueMg ? stat_bonus(depth + 1) : min(stat_bonus(depth + 1), stat_bonus(depth));
            update_quiet_stats(pos, ss, bestMove, bonus);

            // Decrease all the other played quiet moves
            for (int i = 0; i < quietCount; i++) {
                history_update(*pos->mainHistory, stm(), quietsSearched[i], -bonus);
                update_cm_stats(ss, moved_piece(quietsSearched[i]), to_sq(quietsSearched[i]), -bonus);
            }
        }

        update_capture_stats(pos, bestMove, capturesSearched, captureCount, stat_bonus(depth + 1));

        // Extra penalty for a quiet early move that was not a TT move or main
        // killer move in previous ply when it gets refuted
        if (!rootNode // (ss-1)->checkersBB is not set
            && ((ss - 1)->moveCount == 1 + (ss - 1)->ttHit || (ss - 1)->currentMove == (ss - 1)->killers[0]) &&
            !captured_piece())
            update_cm_stats(ss - 1, piece_on(prevSq), prevSq, -stat_bonus(depth + 1));
    }
    // Bonus for prior countermove that caused the fail low
    else if (   !rootNode
             && (ss-1)->currentMove != MOVE_NULL
             && !captured_piece()) {

        int bonusScale =  81 * (depth > 6)
                       +  29 * !(PvNode || cutNode)
                       + 171 * ((ss-1)->moveCount > 9)
                       + 119 * (!ss->checkersBB && bestValue <= ss->staticEval - 113)
                       + 134 * (!(ss-1)->checkersBB && bestValue <= -(ss-1)->staticEval - 75);

                       // + 80  * ((ss-1)->isTTMove) + std::min(-(ss-1)->statScore / 106, 318);

        bonusScale = max_int(bonusScale, 0);

        const int scaledBonus = stat_bonus(depth) * bonusScale;

        update_cm_stats(ss - 1, piece_on(prevSq), prevSq, scaledBonus * 436 / 32768);
    }

    // Pawn Correction History

    int tt_bound = bestValue >= beta ? BOUND_LOWER : PvNode && bestMove ? BOUND_EXACT : BOUND_UPPER;

    if (!inCheck && (!bestMove || !is_capture_or_promotion(pos, bestMove)) &&
        (tt_bound & (bestValue >= eval ? BOUND_LOWER : BOUND_UPPER))) {

        pawn_correction_history_update(pos, depth, bestValue, eval);
        non_pawn_correction_history_update(pos, depth, bestValue, eval);
        minor_correction_history_update(pos, depth, bestValue, eval);
        major_correction_history_update(pos, depth, bestValue, eval);
        counter_correction_history_update(pos, depth, bestValue, eval);
    }

    if (PvNode) bestValue = min(bestValue, maxValue);

    // If no good move is found and the previous position was ttPv, then the
    // previous opponent move is probably good and the new position is added
    // to the search tree
    if (bestValue <= alpha) ss->ttPv = ss->ttPv || ((ss - 1)->ttPv && depth > 3);
    // Otherwise, a countermove has been found and if the position is in the
    // last leaf in the search tree, remove the position from the search tree.
    else if (depth > 3)
        ss->ttPv = ss->ttPv && (ss + 1)->ttPv;

    if (!excludedMove && !(rootNode && pos->pvIdx))
        tte_save(tte, posKey, value_to_tt(bestValue, ss->ply), ss->ttPv, tt_bound, depth, bestMove, rawEval);

    assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

    return bestValue;
}

// search_PV() is the main search function for PV nodes
static NOINLINE Value search_PV(Position *pos, Stack *ss, Value alpha, Value beta, Depth depth) {
    return search_node(pos, ss, alpha, beta, depth, 0, PV);
}

// search_NonPV is the main search function for non-PV nodes
static NOINLINE Value search_NonPV(Position *pos, Stack *ss, Value alpha, Depth depth, bool cutNode) {
    return search_node(pos, ss, alpha, alpha + 1, depth, cutNode, NonPV);
}

// qsearch_node() is the quiescence search function template, which is
// called by the main search function with zero depth, or recursively with
// further decreasing depth per call.
Value qsearch_node(Position *pos, Stack *ss, Value alpha, Value beta, Depth depth, const int NT,
                          const bool InCheck) {
    const bool PvNode = NT == PV;

    assert(InCheck == (bool)checkers());
    assert(alpha >= -VALUE_INFINITE && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));
    assert(depth <= 0);

    Move pv[MAX_PLY + 1];
    TTEntry *tte;
    Key posKey;
    Move ttMove, move, bestMove;
    Value bestValue, value, ttValue, futilityValue, futilityBase, oldAlpha;
    bool pvHit, givesCheck;
    Depth ttDepth;
    int moveCount;

    if (PvNode) {
        oldAlpha = alpha; // To flag BOUND_EXACT when eval above alpha and no available moves
        (ss + 1)->pv = pv;
        ss->pv[0] = 0;
    }

    bestMove = 0;
    moveCount = 0;

    // Check for an instant draw or if the maximum ply has been reached
    if (is_draw(pos) || ss->ply >= MAX_PLY) return ss->ply >= MAX_PLY && !InCheck ? evaluate(pos) : VALUE_DRAW;

    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    // Decide whether or not to include checks: this fixes also the type of
    // TT entry depth that we are going to use. Note that in qsearch we use
    // only two types of depth in TT: DEPTH_QS_CHECKS or DEPTH_QS_NO_CHECKS.
    ttDepth = InCheck || depth >= DEPTH_QS_CHECKS ? DEPTH_QS_CHECKS : DEPTH_QS_NO_CHECKS;

    // Transposition table lookup
    posKey = key();
    tte = tt_probe(posKey, &ss->ttHit);
    ttValue = ss->ttHit ? value_from_tt(tte_value(tte), ss->ply, rule50_count()) : VALUE_NONE;
    ttMove = ss->ttHit ? tte_move(tte) : 0;
    pvHit = ss->ttHit && tte_is_pv(tte);

    if (!PvNode && ss->ttHit && tte_depth(tte) >= ttDepth && ttValue != VALUE_NONE // Only in case of TT access race
        && (ttValue >= beta ? (tte_bound(tte) & BOUND_LOWER) : (tte_bound(tte) & BOUND_UPPER)))
        return ttValue;

    // Evaluate the position statically
    if (InCheck) {
        ss->staticEval = VALUE_NONE;
        bestValue = futilityBase = -VALUE_INFINITE;
    } else {
        if (ss->ttHit) {
            // Never assume anything about values stored in TT
            if ((ss->staticEval = bestValue = tte_eval(tte)) == VALUE_NONE) ss->staticEval = bestValue = evaluate(pos);

            bestValue += corrhist_adjustment(pos);

            // Can ttValue be used as a better position evaluation?
            if (ttValue != VALUE_NONE && (tte_bound(tte) & (ttValue > bestValue ? BOUND_LOWER : BOUND_UPPER)))
                bestValue = ttValue;
        } else {
            ss->staticEval = bestValue =
                (ss - 1)->currentMove != MOVE_NULL ? evaluate(pos) : -(ss - 1)->staticEval + 2 * Tempo;

            bestValue += corrhist_adjustment(pos);
        }

        // Stand pat. Return immediately if static value is at least beta
        if (bestValue >= beta) {
            if (!ss->ttHit)
                tte_save(tte, posKey, value_to_tt(bestValue, ss->ply), false, BOUND_LOWER, DEPTH_NONE, 0,
                         ss->staticEval);

            return bestValue;
        }

        if (PvNode && bestValue > alpha) alpha = bestValue;

        futilityBase = bestValue + 230;
    }

    ss->history = &(*pos->counterMoveHistory)[0][0];

    // Initialize move picker data for the current position, and prepare
    // to search the moves. Because the depth is <= 0 here, only captures,
    // queen promotions and checks (only if depth >= DEPTH_QS_CHECKS) will
    // be generated.
    mp_init_q(pos, ttMove, depth, to_sq((ss - 1)->currentMove));

    // Loop through the moves until no moves remain or a beta cutoff occurs
    while ((move = next_move(pos, 0))) {
        assert(move_is_ok(move));

        givesCheck = gives_check(pos, ss, move);

        moveCount++;

        // Futility pruning and moveCount pruning
        if (bestValue > VALUE_TB_LOSS_IN_MAX_PLY && !givesCheck && futilityBase > -VALUE_KNOWN_WIN &&
            type_of_m(move) != PROMOTION) {
            if (moveCount > 2) continue;

            futilityValue = futilityBase + PieceValue[EG][piece_on(to_sq(move))];

            if (futilityValue <= alpha) {
                bestValue = max(bestValue, futilityValue);
                continue;
            }

            if (futilityBase <= alpha && !see_test(pos, move, 80)) {
                bestValue = max(bestValue, futilityBase);
                continue;
            }
        }

        // Do not search moves with negative SEE values
        if (bestValue > VALUE_TB_LOSS_IN_MAX_PLY && !see_test(pos, move, -108)) continue;

        // Speculative prefetch as early as possible
        prefetch(tt_first_entry(key_after(pos, move)));

        // Check for legality just before making the move
        if (!is_legal(pos, move)) {
            moveCount--;
            continue;
        }

        ss->currentMove = move;
        bool captureOrPromotion = is_capture_or_promotion(pos, move);
        ss->history = &(*pos->counterMoveHistory)[piece_to_history_conv(moved_piece(move))][to_sq(move)];

        if (!captureOrPromotion && bestValue > VALUE_TB_LOSS_IN_MAX_PLY &&
            (*(ss - 1)->history)[0][piece_to_history_conv(moved_piece(move))][to_sq(move)] < CounterMovePruneThreshold &&
            (*(ss - 2)->history)[1][piece_to_history_conv(moved_piece(move))][to_sq(move)] < CounterMovePruneThreshold)
            continue;

        // Make and search the move
        do_move(pos, move, givesCheck);
        value = PvNode ? givesCheck ? -qsearch_PV_true(pos, ss + 1, -beta, -alpha, depth - 1)
                                    : -qsearch_PV_false(pos, ss + 1, -beta, -alpha, depth - 1)
                       : givesCheck ? -qsearch_NonPV_true(pos, ss + 1, -beta, depth - 1)
                                    : -qsearch_NonPV_false(pos, ss + 1, -beta, depth - 1);
        undo_move(pos, move);

        assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

        // Check for a new best move
        if (value > bestValue) {
            bestValue = value;

            if (value > alpha) {
                bestMove = move;

                if (PvNode) // Update pv even in fail-high case
                    update_pv(ss->pv, move, (ss + 1)->pv);

                if (PvNode && value < beta) // Update alpha here!
                    alpha = value;
                else {
                    bestValue = (3 * bestValue + beta) / 4;
                    break;
                }
            }
        }
    }

    // All legal moves have been searched. A special case: If we're in check
    // and no legal moves were found, it is checkmate.
    if (InCheck && bestValue == -VALUE_INFINITE) return mated_in(ss->ply); // Plies to mate from the root

    tte_save(tte, posKey, value_to_tt(bestValue, ss->ply), pvHit,
             bestValue >= beta ? BOUND_LOWER : PvNode && bestValue > oldAlpha ? BOUND_EXACT : BOUND_UPPER, ttDepth,
             bestMove, ss->staticEval);

    assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

    return bestValue;
}

static NOINLINE Value qsearch_PV_true(Position *pos, Stack *ss, Value alpha, Value beta, Depth depth) {
    return qsearch_node(pos, ss, alpha, beta, depth, PV, true);
}

static NOINLINE Value qsearch_PV_false(Position *pos, Stack *ss, Value alpha, Value beta, Depth depth) {
    return qsearch_node(pos, ss, alpha, beta, depth, PV, false);
}

static NOINLINE Value qsearch_NonPV_true(Position *pos, Stack *ss, Value alpha, Depth depth) {
    return qsearch_node(pos, ss, alpha, alpha + 1, depth, NonPV, true);
}

static NOINLINE Value qsearch_NonPV_false(Position *pos, Stack *ss, Value alpha, Depth depth) {
    return qsearch_node(pos, ss, alpha, alpha + 1, depth, NonPV, false);
}

#define rm_lt(m1, m2) ((m1).score != (m2).score ? (m1).score < (m2).score : (m1).previousScore < (m2).previousScore)

// stable_sort() sorts RootMoves from highest-scoring move to lowest-scoring
// move while preserving order of equal elements.
static void stable_sort(RootMove *rm, int num) {
    int i, j;

    for (i = 1; i < num; i++)
        if (rm_lt(rm[i - 1], rm[i])) {
            RootMove tmp = rm[i];
            rm[i] = rm[i - 1];
            for (j = i - 1; j > 0 && rm_lt(rm[j - 1], tmp); j--) rm[j] = rm[j - 1];
            rm[j] = tmp;
        }
}

// value_to_tt() adjusts a mate score from "plies to mate from the root" to
// "plies to mate from the current position". Non-mate scores are unchanged.
// The function is called before storing a value in the transposition table.

static Value value_to_tt(Value v, int ply) {
    assert(v != VALUE_NONE);

    return v >= VALUE_TB_WIN_IN_MAX_PLY ? v + ply : v <= VALUE_TB_LOSS_IN_MAX_PLY ? v - ply : v;
}

// value_from_tt() is the inverse of value_to_tt(): It adjusts a mate score
// from the transposition table (which refers to the plies to mate/be mated
// from current position) to "plies to mate/be mated from the root".

static Value value_from_tt(Value v, int ply, int r50c) {
    if (v == VALUE_NONE) return VALUE_NONE;

    if (v >= VALUE_TB_WIN_IN_MAX_PLY) {
        if (v >= VALUE_MATE_IN_MAX_PLY && VALUE_MATE - v > 99 - r50c) return VALUE_MATE_IN_MAX_PLY - 1;
        return v - ply;
    }

    if (v <= VALUE_TB_LOSS_IN_MAX_PLY) {
        if (v <= VALUE_MATED_IN_MAX_PLY && VALUE_MATE + v > 99 - r50c) return VALUE_MATED_IN_MAX_PLY + 1;
        return v + ply;
    }

    return v;
}

// update_pv() adds current move and appends child pv[]

static void update_pv(Move *pv, Move move, Move *childPv) {
    for (*pv++ = move; childPv && *childPv;) *pv++ = *childPv++;
    *pv = 0;
}

// update_cm_stats() updates countermove and follow-up move history.

static void update_cm_stats(Stack *ss, Piece pc, Square s, int bonus) {
    if (move_is_ok((ss - 1)->currentMove)) cms_update(*(ss - 1)->history, 0, pc, s, bonus);

    if (move_is_ok((ss - 2)->currentMove)) cms_update(*(ss - 2)->history, 1, pc, s, bonus);

    if (ss->checkersBB) return;

    if (move_is_ok((ss - 4)->currentMove)) cms_update(*(ss - 4)->history, 1, pc, s, bonus);

    if (move_is_ok((ss - 6)->currentMove)) cms_update(*(ss - 6)->history, 1, pc, s, bonus);
}

// update_capture_stats() updates move sorting heuristics when a new capture
// best move is found

static void update_capture_stats(const Position *pos, Move move, Move *captures, int captureCnt, int bonus) {
    Piece moved_piece = moved_piece(move);
    int captured = type_of_p(piece_on(to_sq(move)));

    if (is_capture_or_promotion(pos, move))
        cpth_update(*pos->captureHistory, moved_piece, to_sq(move), captured, bonus);

    // Decrease all the other played capture moves
    for (int i = 0; i < captureCnt; i++) {
        moved_piece = moved_piece(captures[i]);
        captured = type_of_p(piece_on(to_sq(captures[i])));
        cpth_update(*pos->captureHistory, moved_piece, to_sq(captures[i]), captured, -bonus);
    }
}

// update_quiet_stats() updates killers, history, countermove and countermove
// plus follow-up move history when a new quiet best move is found.

static void update_quiet_stats(const Position *pos, Stack *ss, Move move, int bonus) {
    if (ss->killers[0] != move) {
        ss->killers[1] = ss->killers[0];
        ss->killers[0] = move;
    }

    Color c = stm();
    history_update(*pos->mainHistory, c, move, bonus);
    update_cm_stats(ss, moved_piece(move), to_sq(move), bonus);

    if (type_of_p(moved_piece(move)) != PAWN) history_update(*pos->mainHistory, c, reverse_move(move), -bonus);

    if (move_is_ok((ss - 1)->currentMove)) {
        Square prevSq = to_sq((ss - 1)->currentMove);
        (*pos->counterMoves)[piece_on(prevSq)][prevSq] = move;
    }
}

// check_time() is used to print debug info and, more importantly, to detect
// when we are out of available time and thus stop the search.

static int peek_stdin_for_stop_and_ponder(int block) {

    fd_set readfds;
    struct timeval timeout = {0, 0}; // No blocking, immediate return
    char *ptr, str[2048] = {0};      // Adjust size as needed

    // Set up the file descriptor set for stdin
    FD_ZERO(&readfds);
    FD_SET(STDIN_FILENO, &readfds);

    // Check if stdin has data available
    if (select(STDIN_FILENO + 1, &readfds, NULL, NULL, block ? NULL : &timeout) <= 0) return 0;

    // Read STDIN, if its even still open
    if (fgets(str, 2048, stdin) == NULL) return 0;

    // Strip trailing newlines
    ptr = strchr(str, '\n');
    if (ptr != NULL) *ptr = '\0';

    // Strip trailing returns
    ptr = strchr(str, '\r');
    if (ptr != NULL) *ptr = '\0';

    // Act as if time expired, when we get "stop"
    if (strcmp(str, "stop") == 0) return 1;

    // For a ponderhit, swap off pondering
    if (strcmp(str, "ponderhit") == 0) Threads.ponder = false;

    return 0;
}

static void check_time(void) {

    TimePoint elapsed = time_elapsed();

    if (peek_stdin_for_stop_and_ponder(0))
        Threads.stop = 1;

    // An engine may not stop pondering until told so by the GUI
    if (Threads.ponder) return;

    #ifdef KAGGLE
        // No MoveTime or Nodes for Kaggle
        if (   (use_time_management() && elapsed > time_maximum() - 10))
            Threads.stop = 1;
    #else
        if (   (use_time_management() && elapsed > time_maximum() - 10)
            || (Limits.movetime && elapsed >= Limits.movetime)
            || (Limits.nodes && threads_nodes_searched() >= Limits.nodes))
            Threads.stop = 1;
    #endif
}

// uci_print_pv() prints PV information according to the UCI protocol.
// UCI requires that all (if any) unsearched PV lines are sent with a
// previous search score.

static void uci_print_pv(Position *pos, Depth depth, Value alpha, Value beta) {

    #ifdef KAGGLE

        (void) pos;
        (void) depth;
        (void) alpha;
        (void) beta;

    #else

        TimePoint elapsed = time_elapsed() + 1;
        RootMoves *rm = pos->rootMoves;
        uint64_t nodes_searched = threads_nodes_searched();
        char buf[16];

        bool updated = rm->move[0].score != -VALUE_INFINITE;

        Depth d = updated ? depth : max(1, depth - 1);
        Value v = updated ? rm->move[0].score : rm->move[0].previousScore;

        if (v == -VALUE_INFINITE) v = VALUE_ZERO;

        printf("info depth %d score %s", d, uci_value(buf, v));

        printf("%s", v >= beta ? " lowerbound" : v <= alpha ? " upperbound" : "");

        printf(" nodes %" PRIu64 " nps %" PRIu64, nodes_searched, nodes_searched * 1000 / elapsed);

        printf(" time %" PRIi64 " pv", elapsed);

        for (int idx = 0; idx < rm->move[0].pvSize; idx++)
            printf(" %s", uci_move(buf, rm->move[0].pv[idx]));

        printf("\n");

        fflush(stdout);

    #endif
}

// extract_ponder_from_tt() is called in case we have no ponder move
// before exiting the search, for instance, in case we stop the search
// during a fail high at root. We try hard to have a ponder move to
// return to the GUI, otherwise in case of 'ponder on' we have nothing
// to think on.

static int __attribute__((minsize, cold)) extract_ponder_from_tt(RootMove *rm, Position *pos) {
    bool ttHit;

    assert(rm->pvSize == 1);

    if (!rm->pv[0]) return 0;

    do_move(pos, rm->pv[0], gives_check(pos, pos->st, rm->pv[0]));
    TTEntry *tte = tt_probe(key(), &ttHit);

    if (ttHit) {
        Move m = tte_move(tte); // Local copy to be SMP safe
        ExtMove list[MAX_MOVES];
        ExtMove *last = generate_legal(pos, list);
        for (ExtMove *p = list; p < last; p++)
            if (p->move == m) {
                rm->pv[rm->pvSize++] = m;
                break;
            }
    }

    undo_move(pos, rm->pv[0]);
    return rm->pvSize > 1;
}

// start_thinking() wakes up the main thread to start a new search,
// then returns immediately.

void __attribute__((minsize, cold)) start_thinking(bool ponderMode) {

    Threads.searching       = false;
    Threads.stopOnPonderhit = false;
    Threads.stop            = false;
    Threads.increaseDepth   = true;
    Threads.ponder          = ponderMode;

    Position* pos = Threads.pos[0];

    // Generate all legal moves.
    ExtMove list[MAX_MOVES];
    ExtMove *end = generate_legal(pos, list);

    pos->rootDepth   = 0;
    pos->nodes       = 0;
    pos->tbHits      = 0;

    RootMoves *rm = pos->rootMoves;
    rm->size = end - list;

    for (int i = 0; i < rm->size; i++) {
        rm->move[i].pvSize        = 1;
        rm->move[i].pv[0]         = list[i].move;
        rm->move[i].score         = -VALUE_INFINITE;
        rm->move[i].previousScore = -VALUE_INFINITE;
        rm->move[i].averageScore  = -VALUE_INFINITE;
    }

    (pos->st-1)->endMoves = pos->moveList;

    pos_set_check_info(pos);
    reset_evaluator(pos);

    Threads.searching = true;
    thread_wake_up(threads_main(), THREAD_SEARCH);
}
