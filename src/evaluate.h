#pragma once

#include <stdalign.h>

#include "types.h"

enum { Tempo = 0 };

#define ft_in       640
#define ft_out      64

#define pawn_ft_in  96
#define pawn_ft_out 64

#define l1_in     128
#define l1_out    8
#define l2_in     8
#define l2_out    16
#define l3_in     16
#define l3_out    1

#define n_buckets 8

typedef struct NNUEDelta {
    Move move;
    Piece moving, captured;
} NNUEDelta;

typedef struct NNUEAccumulator {
    NNUEDelta delta;
    int accurate[2];
    int32_t mat[2];
    alignas(64) int16_t ft[2][ft_out];
    alignas(64) int16_t pawn_ft[2][pawn_ft_out];
} NNUEAccumulator;

typedef struct NNUEEvaluator {
    NNUEAccumulator stack[MAX_PLY + 10];
    NNUEAccumulator *curr;
} NNUEEvaluator;


NNUEEvaluator* create_nnue_evaluator(void);
void reset_evaluator(const Position* pos);
void reset_accum(const Position *pos, NNUEAccumulator *accum, int colour);
void nnue_update_accum(const Position* pos, NNUEAccumulator *accum, int colour);

void nnue_pop(const Position* pos);
void nnue_push(const Position* pos, Move move, Piece moving, Piece captured);

void eval_init(void);
Value evaluate(const Position *pos);
