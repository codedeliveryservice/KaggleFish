/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2016 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad

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

#include <float.h>

#include "search.h"
#include "timeman.h"
#include "uci.h"

#define SQRT_PLY_FORMULA_MAX_PLY 255

// To Create in python...
// >>> from import import pow
// >>> z = [round(2048 * (0.0084 + pow(ply + 3.0, 0.5) * 0.0042)) for ply in range(256)]
// >>> for f in z: print("%3d, " % (f), end='')

uint8_t opt_formula_times_2048[SQRT_PLY_FORMULA_MAX_PLY+1] = {
  32,  34,  36,  38,  40,  42,  43,  44,  46,  47,  48,  49,  51,  52,  53,  54,  55,  56,  57,  58,  58,  59,  60,  61,  62,  63,  64,  64,  65,  66,  67,  67,  68,  69,  70,  70,  71,  72,  72,  73,  74,  74,  75,  76,  76,  77,  77,  78,  79,  79,  80,  80,  81,  82,  82,  83,  83,  84,  84,  85,  85,  86,  87,  87,  88,  88,  89,  89,  90,  90,  91,  91,  92,  92,  93,  93,  94,  94,  95,  95,  96,  96,  97,  97,  97,  98,  98,  99,  99, 100, 100, 101, 101, 101, 102, 102, 103, 103, 104, 104, 104, 105, 105, 106, 106, 107, 107, 107, 108, 108, 109, 109, 109, 110, 110, 111, 111, 111, 112, 112, 113, 113, 113, 114, 114, 115, 115, 115, 116, 116, 116, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139, 139, 140, 140, 140, 141, 141, 141, 142, 142, 142, 142, 143, 143, 143, 144, 144, 144, 144, 145, 145, 145, 146, 146, 146, 147, 147, 147, 147, 148, 148, 148, 149, 149, 149, 149, 150, 150, 150, 150, 151, 151, 151, 152, 152, 152, 152, 153, 153, 153, 153, 154, 154, 154, 155, 155, 155, 155
};

struct TimeManagement Time; // Our global time management struct

// tm_init() is called at the beginning of the search and calculates the
// time bounds allowed for the current game ply. We currently support:
// 1) x basetime (+z increment)
// 2) x moves in y seconds (+z increment)

void time_init(Color us, int ply) {

    const int mtg = 50;
    const int moveOverhead = OPTION_OVERHEAD;

    Time.startTime = Limits.startTime;

    // Make sure that timeLeft > 0 since we may use it as a divisor
    TimePoint timeLeft = max(1, Limits.time[us] + Limits.inc[us] * (mtg - 1) - moveOverhead * (2 + mtg));

    // x basetime (+z increment)
    // If there is a healthy increment, timeLeft can exceed actual available
    // game time for the current move, so also cap to 20% of available game time.

    double lookup = opt_formula_times_2048[min_int(SQRT_PLY_FORMULA_MAX_PLY, ply)];
    double optScale = min(lookup / 2048.0, 0.2 * Limits.time[us] / (double)timeLeft);
    double maxScale = min(7.0, 4.0 + ply / 12.0);

    // Never use more than 80% of the available time for this move
    Time.optimumTime = optScale * timeLeft;
    Time.maximumTime = min(0.8 * Limits.time[us] - moveOverhead, maxScale * Time.optimumTime);

    #ifdef KAGGLE
        Time.optimumTime += Time.optimumTime / 4;
    #endif
}
