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

#include <inttypes.h>
#include <string.h>
#include <sys/mman.h>

#include "misc.h"
#include "thread.h"

void prng_init(PRNG *rng, uint64_t seed) {
    rng->s = seed;
}

NOINLINE uint64_t prng_rand(PRNG *rng) {
    uint64_t s = rng->s;

    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    rng->s = s;

    return s * 2685821657736338717LL;
}

NOINLINE uint64_t prng_sparse_rand(PRNG *rng) {
    uint64_t r1 = prng_rand(rng);
    uint64_t r2 = prng_rand(rng);
    uint64_t r3 = prng_rand(rng);
    return r1 & r2 & r3;
}

ssize_t getline(char **lineptr, size_t *n, FILE *stream) {
    if (*n == 0) *lineptr = malloc(*n = 100);

    int c = 0;
    size_t i = 0;
    while ((c = getc(stream)) != EOF) {
        (*lineptr)[i++] = c;
        if (i == *n) *lineptr = realloc(*lineptr, *n += 100);
        if (c == '\n') break;
    }
    (*lineptr)[i] = 0;
    return i;
}


void *allocate_memory(size_t size, alloc_t *alloc) {

    void *ptr = NULL;

    size_t alignment = 1;
    size_t allocSize = size + alignment - 1;

    ptr = mmap(NULL, allocSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    alloc->ptr = ptr;
    alloc->size = allocSize;
    return (void *)(((uintptr_t)ptr + alignment - 1) & ~(alignment - 1));
}

void free_memory(alloc_t *alloc) {
    munmap(alloc->ptr, alloc->size);
}
