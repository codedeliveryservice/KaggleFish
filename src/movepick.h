/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2018 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad

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

#ifndef MOVEPICK_H
#define MOVEPICK_H

#include <string.h> // For memset

#include "movegen.h"
#include "position.h"
#include "search.h"
#include "types.h"

#define stats_clear(s) memset(s, 0, sizeof(*s))

static const int CounterMovePruneThreshold = 0;

INLINE void cms_update(PieceToHistory cms, int dim, Piece pc, Square to, int v) {
    cms[dim][piece_to_history_conv(pc)][to] += v - cms[dim][piece_to_history_conv(pc)][to] * abs(v) / 30107;
}

INLINE void history_update(ButterflyHistory history, Color c, Move m, int v) {
    m &= 4095;
    history[c][m] += v - history[c][m] * abs(v) / 14031;
}

INLINE void cpth_update(CapturePieceToHistory history, Piece pc, Square to, int captured, int v) {
    history[pc][to][captured] += v - history[pc][to][captured] * abs(v) / 15714;
}

enum {
    ST_MAIN_SEARCH,
    ST_CAPTURES_INIT,
    ST_GOOD_CAPTURES,
    ST_KILLERS,
    ST_KILLERS_2,
    ST_QUIET_INIT,
    ST_QUIET,
    ST_BAD_CAPTURES,

    ST_EVASION,
    ST_EVASIONS_INIT,
    ST_ALL_EVASIONS,

    ST_QSEARCH,
    ST_QCAPTURES_INIT,
    ST_QCAPTURES,
    ST_QCHECKS,

    ST_PROBCUT,
    ST_PROBCUT_INIT,
    ST_PROBCUT_2
};

Move next_move(const Position *pos, bool skipQuiets);

// Initialisation of move picker data.

INLINE void mp_init(const Position *pos, Move ttm, Depth d, int ply) {
    assert(d > 0);

    Stack *st = pos->st;

    st->depth = d;
    st->mp_ply = ply;

    Square prevSq = to_sq((st - 1)->currentMove);
    st->countermove = (*pos->counterMoves)[piece_on(prevSq)][prevSq];
    st->mpKillers[0] = st->killers[0];
    st->mpKillers[1] = st->killers[1];

    st->ttMove = ttm;
    st->stage = checkers() ? ST_EVASION : ST_MAIN_SEARCH;
    if (!ttm || !is_pseudo_legal(pos, ttm)) st->stage++;
}

INLINE void mp_init_q(const Position *pos, Move ttm, Depth d, Square s) {
    assert(d <= 0);

    Stack *st = pos->st;

    st->ttMove = ttm;
    st->stage = checkers() ? ST_EVASION : ST_QSEARCH;
    if (!(ttm && (checkers() || d > DEPTH_QS_RECAPTURES || to_sq(ttm) == s) && is_pseudo_legal(pos, ttm))) st->stage++;

    st->depth = d;
    st->recaptureSquare = s;
}

INLINE void mp_init_pc(const Position *pos, Move ttm, Value th) {
    assert(!checkers());

    Stack *st = pos->st;

    st->threshold = th;

    st->ttMove = ttm;
    st->stage = ST_PROBCUT;

    // In ProbCut we generate captures with SEE higher than the given
    // threshold.
    if (!(ttm && is_pseudo_legal(pos, ttm) && is_capture(pos, ttm) && see_test(pos, ttm, th))) st->stage++;
}

// Correction Histories

INLINE void pawn_correction_history_update(const Position *pos, int depth, int best, int eval) {

    int v = clamp_int(depth * (best - eval) / 8, -256, 256);

    uint64_t key = pos->st->pawnKey & (PAWN_CORR_HIST_SIZE - 1);

    (*pos->pawnCorrectionHistory)[key][pos->sideToMove] +=
        v - (*pos->pawnCorrectionHistory)[key][pos->sideToMove] * abs(v) / 1024;
}

INLINE void non_pawn_correction_history_update(const Position *pos, int depth, int best, int eval) {

    int v = clamp_int(depth * (best - eval) / 8, -256, 256);

    uint64_t non_pawn_key_white = pos->st->nonPawnKey[WHITE] & (NONPAWN_CORR_HIST_SIZE - 1);
    uint64_t non_pawn_key_black = pos->st->nonPawnKey[BLACK] & (NONPAWN_CORR_HIST_SIZE - 1);

    (*pos->nonPawnCorrectionHistory)[WHITE][stm()][non_pawn_key_white] +=
        v - (*pos->nonPawnCorrectionHistory)[WHITE][stm()][non_pawn_key_white] * abs(v) / 1024;

    (*pos->nonPawnCorrectionHistory)[BLACK][stm()][non_pawn_key_black] +=
        v - (*pos->nonPawnCorrectionHistory)[BLACK][stm()][non_pawn_key_black] * abs(v) / 1024;
}

INLINE void minor_correction_history_update(const Position *pos, int depth, int best, int eval) {

    int v = clamp_int(depth * (best - eval) / 8, -256, 256);

    uint64_t key = pos->st->minorKey & (MINOR_CORR_HIST_SIZE - 1);

    (*pos->minorCorrectionHistory)[key][pos->sideToMove] +=
        v - (*pos->minorCorrectionHistory)[key][pos->sideToMove] * abs(v) / 1024;
}

INLINE void major_correction_history_update(const Position *pos, int depth, int best, int eval) {

    int v = clamp_int(depth * (best - eval) / 8, -256, 256);

    uint64_t key = pos->st->majorKey & (MAJOR_CORR_HIST_SIZE - 1);

    (*pos->majorCorrectionHistory)[key][pos->sideToMove] +=
        v - (*pos->majorCorrectionHistory)[key][pos->sideToMove] * abs(v) / 1024;
}

INLINE void counter_correction_history_update(const Position *pos, int depth, int best, int eval) {

    int v = clamp_int(depth * (best - eval) / 8, -256, 256);

    Move prev = (pos->st-1)->currentMove;

    if (prev == MOVE_NONE || prev == MOVE_NULL)
        return;

    (*pos->counterCorrectionHistory)[piece_on(to_sq(prev))][to_sq(prev)] +=
        v - (*pos->counterCorrectionHistory)[piece_on(to_sq(prev))][to_sq(prev)] * abs(v) / 1024;
}

/*NOINLINE */ int corrhist_adjustment(const Position *pos);

#endif
