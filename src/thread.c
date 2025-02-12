/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2017 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad

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

#include "movegen.h"
#include "movepick.h"
#include "search.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"

#ifndef _WIN32
#define THREAD_FUNC void *
#else
#define THREAD_FUNC DWORD WINAPI
#endif

// Global objects
ThreadPool Threads;
MainThread mainThread;

// thread_init() is where a search thread starts and initialises itself.

static void __attribute__((minsize, cold)) thread_create() {

    Position *pos;

    pos                            = calloc(sizeof(Position), 1);
    pos->counterMoves              = calloc(sizeof(CounterMoveStat), 1);
    pos->mainHistory               = calloc(sizeof(ButterflyHistory), 1);
    pos->captureHistory            = calloc(sizeof(CapturePieceToHistory), 1);
    pos->rootMoves                 = calloc(sizeof(RootMoves), 1);
    pos->stackAllocation           = calloc(63 + (MAX_PLY + 110) * sizeof(Stack), 1);
    pos->moveList                  = calloc(10000 * sizeof(ExtMove), 1);
    pos->counterMoveHistory        = calloc(sizeof(CounterMoveHistoryStat), 1);
    pos->pawnCorrectionHistory     = calloc(sizeof(PawnCorrectionHistory    ), 1);
    pos->nonPawnCorrectionHistory  = calloc(sizeof(NonPawnCorrectionHistory ), 1);
    pos->minorCorrectionHistory    = calloc(sizeof(MinorCorrectionHistory   ), 1);
    pos->majorCorrectionHistory    = calloc(sizeof(MajorCorrectionHistory   ), 1);
    pos->counterCorrectionHistory  = calloc(sizeof(CounterCorrectionHistory ), 1);

    pos->nnue = create_nnue_evaluator();

    for (int d = 0; d < 2; d++)
        for (int j = 0; j < 6; j++)
            for (int k = 0; k < 64; k++)
            (*pos->counterMoveHistory)[0][0][d][j][k] = CounterMovePruneThreshold - 1;

    pos->resetCalls = false;
    pos->callsCnt = 0;

    pos->stack = (Stack *)(((uintptr_t)pos->stackAllocation + 0x3f) & ~0x3f);

    Threads.pos[0] = pos;
}

// thread_destroy() waits for thread termination before returning.

static void thread_destroy(Position *pos) {

    // Don't care about memory leaks when closing
    #ifdef KAGGLE
        (void) pos; return;
    #else
        free(pos->counterMoves);
        free(pos->mainHistory);
        free(pos->captureHistory);

        free(pos->rootMoves);
        free(pos->stackAllocation);
        free(pos->moveList);
        free(pos->counterMoveHistory);
        free(pos);

        free(pos->pawnCorrectionHistory);
        free(pos->nonPawnCorrectionHistory);
        free(pos->minorCorrectionHistory);
        free(pos->majorCorrectionHistory);
        free(pos->counterCorrectionHistory);
    #endif
}

// thread_wait_for_search_finished() waits on sleep condition until
// not searching.

void thread_wait_until_sleeping() {
    Threads.searching = false;
}

void thread_wake_up(Position *pos, int action) {
    pos->action = action;

    if (action == THREAD_SEARCH) mainthread_search();
}

// threads_init() creates and launches requested threads that will go
// immediately to sleep. We cannot use a constructor because Threads is a
// static object and we need a fully initialized engine at this point due to
// allocation of Endgames in the Thread constructor.

void threads_init(void) {
    thread_create();
    search_init();
}

// threads_exit() terminates threads before the program exits. Cannot be
// done in destructor because threads must be terminated before deleting
// any static objects while still in main().

void threads_exit(void) {
    thread_destroy(Threads.pos[0]);
    Threads.searching = false;
}

// threads_nodes_searched() returns the number of nodes searched.

uint64_t threads_nodes_searched(void) {
    return Threads.pos[0]->nodes;
}
