#include <stdio.h>
#include <immintrin.h>
#include <stdalign.h>

#include "evaluate.h"
#include "position.h"

#include "embedded_net.h"

// From: https://github.com/AndyGrant/Ethereal/blob/master/src/nnue/archs/avx2.h

#define vepi8  __m256i
#define vepi16 __m256i
#define vepi32 __m256i
#define vps32  __m256

#define vepi8_cnt  32
#define vepi16_cnt 16
#define vepi32_cnt 8
#define vps32_cnt  8

#define vepi16_add   _mm256_add_epi16
#define vepi16_madd  _mm256_madd_epi16
#define vepi16_maubs _mm256_maddubs_epi16
#define vepi16_max   _mm256_max_epi16
#define vepi16_min   _mm256_min_epi16
#define vepi16_one   _mm256_set1_epi16(1)
#define vepi16_packu _mm256_packus_epi16
#define vepi16_srai  _mm256_srai_epi16
#define vepi16_srli  _mm256_srli_epi16
#define vepi16_sub   _mm256_sub_epi16
#define vepi16_zero  _mm256_setzero_si256
#define vepi16_mullo _mm256_mullo_epi16

#define vepi32_add  _mm256_add_epi32
#define vepi32_hadd _mm256_hadd_epi32
#define vepi32_max  _mm256_max_epi32
#define vepi32_srai _mm256_srai_epi32
#define vepi32_zero _mm256_setzero_si256

#define vepi32_conv_ps _mm256_cvtepi32_ps
#define vps32_add      _mm256_add_ps
#define vps32_fma      _mm256_fmadd_ps
#define vps32_hadd     _mm256_hadd_ps
#define vps32_max      _mm256_max_ps
#define vps32_mul      _mm256_mul_ps
#define vps32_zero     _mm256_setzero_ps

#define FALSE 0
#define TRUE 1


alignas(64) int16_t ft_weights[ft_in * ft_out + 64];
alignas(64) int16_t pawn_ft_weights[pawn_ft_in * pawn_ft_out];

alignas(64) int16_t l1_weights[l1_in * l1_out];
alignas(64) int32_t l1_bias[l1_out];

alignas(64) float   l2_weights[l2_in * l2_out * n_buckets];
alignas(64) float   l2_bias[l2_out * n_buckets];

alignas(64) float   l3_weights[l3_in * l3_out * n_buckets];




INLINE vepi16 pairwise_clipped_relu(const vepi16 v1, const vepi16 v2) {

    const vepi16 ft_max = _mm256_set1_epi16(127*2);
    const vepi16 ft_min = vepi16_zero();

    vepi16 clipped_v1 = vepi16_min(ft_max, vepi16_max(ft_min, v1));
    vepi16 clipped_v2 = vepi16_min(ft_max, vepi16_max(ft_min, v2));

    return vepi16_srli(vepi16_mullo(clipped_v1, clipped_v2), 7);
}

INLINE vepi16 pairwise_clipped_relu_pawn_ft(const vepi16 v1, const vepi16 v2) {

    const vepi16 ft_max = _mm256_set1_epi16(127);
    const vepi16 ft_min = vepi16_zero();

    vepi16 clipped_v1 = vepi16_min(ft_max, vepi16_max(ft_min, v1));
    vepi16 clipped_v2 = vepi16_min(ft_max, vepi16_max(ft_min, v2));

    return vepi16_srli(vepi16_mullo(clipped_v1, clipped_v2), 5);
}


INLINE void activate_ft(const Position* pos, const NNUEAccumulator* acc, int16_t* output) {

    const vepi16* vus        = (vepi16*) acc->ft[ stm()];
    const vepi16* vthem      = (vepi16*) acc->ft[!stm()];

    const vepi16* pawn_vus   = (vepi16*) acc->pawn_ft[ stm()];
    const vepi16* pawn_vthem = (vepi16*) acc->pawn_ft[!stm()];

          vepi16* vout       = (vepi16*) output;

    int vout_index = 0;

    vout[vout_index++] = pairwise_clipped_relu(vus[0], vus[2]);
    vout[vout_index++] = pairwise_clipped_relu(vus[1], vus[3]);

    vout[vout_index++] = pairwise_clipped_relu(vthem[0], vthem[2]);
    vout[vout_index++] = pairwise_clipped_relu(vthem[1], vthem[3]);

    vout[vout_index++] = pairwise_clipped_relu_pawn_ft(pawn_vus[0], pawn_vus[2]);
    vout[vout_index++] = pairwise_clipped_relu_pawn_ft(pawn_vus[1], pawn_vus[3]);

    vout[vout_index++] = pairwise_clipped_relu_pawn_ft(pawn_vthem[0], pawn_vthem[2]);
    vout[vout_index++] = pairwise_clipped_relu_pawn_ft(pawn_vthem[1], pawn_vthem[3]);
}

INLINE void affine_relu_l1(const int16_t* input, float* output) {

    const vepi16* vin  = (vepi16*) input;
          vps32*  vout = (vps32 *) output;

    const vepi16* vwgt = (vepi16*) l1_weights;
    const vepi32* vbia = (vepi32*) l1_bias;

    const size_t in_chunks  = l1_in / vepi16_cnt;
    const size_t out_chunks = l1_out / 8;

    const size_t UNROLL = 8;

    vepi32 acc[UNROLL];

    for (size_t i = 0; i < out_chunks; i++) {

        // Break out the first iteration, to init acc
        for (size_t j = 0; j < 1; j++)
            for (size_t k = 0; k < UNROLL; k++)
                acc[k] = vepi16_madd(vwgt[in_chunks * (i * 8 + k) + j], vin[j]);

        // Process the remainder of the chunks into acc
        for (size_t j = 1; j < in_chunks; j++)
            for (size_t k = 0; k < UNROLL; k++)
                acc[k] = vepi32_add(acc[k], vepi16_madd(vwgt[in_chunks * (i * 8 + k) + j], vin[j]));

        // Collapse into 0, 2, 4, 8
        for (size_t k = 0; k < UNROLL; k += 2)
            acc[k] = vepi32_hadd(acc[k], acc[k+1]);

        // Collapse into 0, 4
        for (size_t k = 0; k < UNROLL; k += 4)
            acc[k] = vepi32_hadd(acc[k], acc[k+2]);

        // Thank you, Connor (author of Seer)
        acc[0] = _mm256_add_epi32(
            _mm256_permute2x128_si256(acc[0], acc[4], 0x20),
            _mm256_permute2x128_si256(acc[0], acc[4], 0x31)
        );

        acc[0]  = vepi32_add(acc[0], vbia[i]); // l1_bias is already multiplyed by quant_ft (32)

        acc[0] = vepi32_max(acc[0], vepi32_zero()); // ReLU

        vout[i] = vepi32_conv_ps(acc[0]); // Convert to floats
    }
}

INLINE void affine_relu_l2(const float* input, float* output, size_t bucket) {

    const vps32* vin  = (vps32*) input;
          vps32* vout = (vps32*) output;

    const vps32* vwgt = (vps32*) &l2_weights[l2_in * l2_out * bucket];
    const vps32* vbia = (vps32*) &l2_bias[l2_out * bucket];

    const size_t in_chunks  = l2_in / vps32_cnt;
    const size_t out_chunks = l2_out / 8;

    const size_t UNROLL = 8;

    vps32 acc[UNROLL];

    for (size_t i = 0; i < out_chunks; i++) {

        // Break out the first iteration, to init acc
        for (size_t j = 0; j < 1; j++)
            for (size_t k = 0; k < UNROLL; k++)
                acc[k] = vps32_mul(vwgt[in_chunks * (i * 8 + k) + j], vin[j]);

        // Process the remainder of the chunks into acc
        for (size_t j = 1; j < in_chunks; j++)
            for (size_t k = 0; k < UNROLL; k++)
                acc[k] = vps32_fma(vwgt[in_chunks * (i * 8 + k) + j], vin[j], acc[k]);

        // Collapse into 0, 2, 4, 8
        for (size_t k = 0; k < UNROLL; k += 2)
            acc[k] = vps32_hadd(acc[k], acc[k+1]);

        // Collapse into 0, 4
        for (size_t k = 0; k < UNROLL; k += 4)
            acc[k] = vps32_hadd(acc[k], acc[k+2]);

        // Thank you, Connor (author of Seer)
        vout[i] = _mm256_add_ps(
            _mm256_permute2f128_ps(acc[0], acc[4], 0x20),
            _mm256_permute2f128_ps(acc[0], acc[4], 0x31)
        );

        vout[i] = vps32_add(vout[i], vbia[i]); // l2_bias is already multiplyed by quant_ft x quant_l1 (32 x 64)

        vout[i] = vps32_max(vout[i], vps32_zero()); // ReLU
    }
}

INLINE void affine_output_l3(const float* input, float* output, size_t bucket) {

    const vps32* vin  = (vps32*) input;
    const vps32* vwgt = (vps32*) &l3_weights[bucket * l3_in * l3_out];

    const size_t in_chunks = l3_in / vps32_cnt;

    vps32 acc = vps32_mul(vwgt[0], vin[0]);
    for (size_t i = 1; i < in_chunks; i++)
        acc = vps32_fma(vwgt[i], vin[i], acc);

    // Fast summation instead of many hadds

    const __m128 hiQuad  = _mm256_extractf128_ps(acc, 1);
    const __m128 loQuad  = _mm256_castps256_ps128(acc);
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);

    const __m128 hiDual  = _mm_movehl_ps(sumQuad, sumQuad);
    const __m128 sumDual = _mm_add_ps(sumQuad, hiDual);

    const __m128 hi      = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    const __m128 sum     = _mm_add_ss(sumDual, hi);

    *output = (_mm_cvtss_f32(sum) + l3_bias[bucket * l3_out]);
}


static size_t nnue_index(int piece, int perspective, int sq, int king_sq) {

    // 1. Pawns start on -8, due to not being on the 1st rank
    // 2. Pawns end at -16 total, due to not being on the 8th rank
    // 3. We do not use any inputs for Their King, so we provide zeros via idx=640
    // 4. Our King stats at 608, but ends at 640.

    #define mirror_square(x) ((8 * rank_of((x))) + 7 - file_of((x)))
    #define queen_side(x) ((0x0F0F0F0F0F0F0F0FULL >> (x)) & 1)

    const int offset[2][KING+1] = {
        {   0,  -8,  48, 112, 176, 240, 0   }, // Theirs
        {   0, 296, 352, 416, 480, 544, 608 }, // Ours
    };

    const int pt = type_of_p(piece);
    const int pc = color_of(piece);

    // Special case of already zero'ed weights
    if (pt == KING && pc != perspective)
        return 640;

    // Take the square relative to the King
    sq = relative_square(perspective, sq);

    // Mirror the sq if the King is on their Queen Side
    if (queen_side(relative_square(perspective, king_sq)))
        sq = mirror_square(sq);

    // Special case, with complex King mapping
    if (pt == KING)
        sq = 4 * rank_of(sq) + file_of(sq) - 4;

    return offset[pc == perspective][pt] + sq;

    #undef mirror_square
    #undef queen_side
}

static size_t nnue_pawn_index(int piece, int perspective, int sq, int king_sq) {

    // 1. Pawns start on -8, due to not being on the 1st rank
    // 2. Next pawns start only 48 after previous, due to 1st/8th
    // 3. We are only concerned with Pawns

    #define mirror_square(x) ((8 * rank_of((x))) + 7 - file_of((x)))
    #define queen_side(x) ((0x0F0F0F0F0F0F0F0FULL >> (x)) & 1)

    const int offset[2] = { -8, 40 };

    const int pc = color_of(piece);

    // Take the square relative to the King
    sq = relative_square(perspective, sq);

    // Mirror the sq if the King is on their Queen Side
    if (queen_side(relative_square(perspective, king_sq)))
        sq = mirror_square(sq);

    return offset[pc == perspective] + sq;

    #undef mirror_square
    #undef queen_side
}


static bool nnue_can_update(const Position* pos, NNUEAccumulator *accum, int colour) {

    #define queen_side(x) ((0x0F0F0F0F0F0F0F0FULL >> (x)) & 1)

    while (accum != pos->nnue->stack) {

        // Cannot update incrementally if the King crossed sides
        if (   accum->delta.moving == make_piece(colour, KING)
            && queen_side(to_sq(accum->delta.move)) != queen_side(from_sq(accum->delta.move)))
            return 0;

        // For simplicity, can't update incrementally after a promotion
        if (type_of_m(accum->delta.move) == PROMOTION)
            return 0;

        accum = accum - 1;

        if (accum->accurate[colour])
            return 1;
    }

    return 0;

    #undef queen_side
}


NNUEEvaluator* create_nnue_evaluator(void) {
    void *mem;
    return posix_memalign(&mem, 64, sizeof(NNUEEvaluator)) ? NULL : mem;
}


void reset_evaluator(const Position* pos) {
    pos->nnue->curr = &pos->nnue->stack[0];
    reset_accum(pos, pos->nnue->curr, WHITE);
    reset_accum(pos, pos->nnue->curr, BLACK);
}

void reset_accum(const Position *pos, NNUEAccumulator *accum, int colour) {

    static const int32_t material_values[] = {0, 208, 854, 915, 1380, 2682, 0}; // 2x what they should be

    const vepi16* vbia  = (vepi16*) ft_bias;
    const vepi16* vpbia = (vepi16*) pawn_ft_bias;

    vepi16* vout  = (vepi16*) accum->ft[colour];
    vepi16* vpout = (vepi16*) accum->pawn_ft[colour];

    for (size_t i = 0; i < ft_out / vepi16_cnt; i++)
        vout[i] = vbia[i];

    for (size_t i = 0; i < pawn_ft_out / vepi16_cnt; i++)
        vpout[i] = vpbia[i];

    accum->mat[colour] = 0;
    accum->accurate[colour] = 1;

    const Square k_sq = square_of(colour, KING);

    Bitboard occ = pieces() ^ pieces_p(PAWN);

    while (occ) {

        const Square sq = pop_lsb(&occ);

        const int pt = type_of_p(piece_on(sq));
        const int pc = color_of(piece_on(sq));

        accum->mat[colour] += pc == colour ? material_values[pt] : -material_values[pt];

        const int     ft_idx = nnue_index(piece_on(sq), colour, sq, k_sq);
        const vepi16* vwgt   = (vepi16*) &ft_weights[ft_idx * ft_out];

        for (size_t i = 0; i < ft_out / vepi16_cnt; i++)
            vout[i] = vepi16_add(vout[i], vwgt[i]);
    }

    occ = pieces_p(PAWN);

    while (occ) {

        const Square sq = pop_lsb(&occ);

        const int pt = PAWN;
        const int pc = color_of(piece_on(sq));

        accum->mat[colour] += pc == colour ? material_values[pt] : -material_values[pt];

        const int     ft_idx = nnue_index(piece_on(sq), colour, sq, k_sq);
        const vepi16* vwgt   = (vepi16*) &ft_weights[ft_idx * ft_out];

        for (size_t i = 0; i < ft_out / vepi16_cnt; i++)
            vout[i] = vepi16_add(vout[i], vwgt[i]);

        const int     pft_idx = nnue_pawn_index(piece_on(sq), colour, sq, k_sq);
        const vepi16* vpwgt   = (vepi16*) &pawn_ft_weights[pft_idx * pawn_ft_out];

        for (size_t i = 0; i < pawn_ft_out / vepi16_cnt; i++)
            vpout[i] = vepi16_add(vpout[i], vpwgt[i]);
    }
}


void add_sub_add_sub(int16_t* dest, const int16_t* src, int16_t* weights, size_t size,
                     bool ADD1, bool SUB1, bool ADD2, bool SUB2,
                     size_t add1, size_t sub1, size_t add2, size_t sub2) {

    vepi16* vout = (vepi16*) dest;
    const vepi16* vin  = (vepi16*) src;

    const vepi16* vadd1 = (vepi16*) &weights[add1 * size];
    const vepi16* vsub1 = (vepi16*) &weights[sub1 * size];
    const vepi16* vadd2 = (vepi16*) &weights[add2 * size];
    const vepi16* vsub2 = (vepi16*) &weights[sub2 * size];

    for (size_t i = 0; i < size / vepi16_cnt; i++) {

        vepi16 v = vin[i];

        if (ADD1)
            v = vepi16_add(v, vadd1[i]);

        if (SUB1)
            v = vepi16_sub(v, vsub1[i]);

        if (ADD2)
            v = vepi16_add(v, vadd2[i]);

        if (SUB2)
            v = vepi16_sub(v, vsub2[i]);

        vout[i] = v;
    }
}

void nnue_update_accum(const Position* pos, NNUEAccumulator* accum, int colour) {

    static const int32_t material_values[] = {0, 208, 854, 915, 1380, 2682, 0, 0, 0, 0}; // 2x what they should be

    if (!(accum-1)->accurate[colour])
        nnue_update_accum(pos, (accum-1), colour);

    accum->accurate[colour] = TRUE;
    accum->mat[colour] = (accum-1)->mat[colour];

    // For relative mirroring
    const Square k_sq = square_of(colour, KING);

    // Basic information about the move
    const Square    sq_from = from_sq(accum->delta.move);
    const Square    sq_to   = to_sq(accum->delta.move);
    const PieceType pt_from = type_of_p(accum->delta.moving);
    const Color     c_from  = color_of(accum->delta.moving);

    // Feature indices for the basic move
    const size_t f_from = nnue_index(accum->delta.moving, colour, sq_from, k_sq);
    const size_t f_to   = nnue_index(accum->delta.moving, colour, sq_to  , k_sq);

    if (type_of_m(accum->delta.move) == CASTLING) {

        const Square king_to = relative_square(c_from, sq_to > sq_from ? SQ_G1 : SQ_C1);
        const Square rook_to = relative_square(c_from, sq_to > sq_from ? SQ_F1 : SQ_D1);

        const int f_king_to   = nnue_index(make_piece(c_from, KING), colour, king_to, k_sq);
        const int f_rook_from = nnue_index(make_piece(c_from, ROOK), colour,   sq_to, k_sq);
        const int f_rook_to   = nnue_index(make_piece(c_from, ROOK), colour, rook_to, k_sq);

        add_sub_add_sub(
            accum->ft[colour], (accum-1)->ft[colour], ft_weights, ft_out,
            TRUE, TRUE, TRUE, TRUE,
            f_king_to, f_from, f_rook_to, f_rook_from
        );

        // Castling does not involve pawns, so just copy the existing
        memcpy(accum->pawn_ft[colour], (accum-1)->pawn_ft[colour], sizeof(int16_t) * pawn_ft_out);
    }

    else if (accum->delta.captured) {

        const Square sq_cap = type_of_m(accum->delta.move) == ENPASSANT ? sq_to ^ 8 : sq_to;

        const int f_cap = nnue_index(accum->delta.captured, colour, sq_cap, k_sq);

        accum->mat[colour] +=  color_of(accum->delta.captured) == colour
                            ? -material_values[type_of_p(accum->delta.captured)]
                            :  material_values[type_of_p(accum->delta.captured)];

        add_sub_add_sub(
            accum->ft[colour], (accum-1)->ft[colour], ft_weights, ft_out,
            TRUE, TRUE, FALSE, TRUE,
            f_to, f_from, 0, f_cap
        );

        // Copy the original Pawn-FT

        memcpy(accum->pawn_ft[colour], (accum-1)->pawn_ft[colour], sizeof(int16_t) * pawn_ft_out);


        if (pt_from == PAWN) {

            const size_t pf_from = nnue_pawn_index(make_piece(c_from, PAWN), colour, sq_from, k_sq);
            const size_t pf_to   = nnue_pawn_index(make_piece(c_from, PAWN), colour, sq_to  , k_sq);

            add_sub_add_sub(
                accum->pawn_ft[colour], accum->pawn_ft[colour], pawn_ft_weights, pawn_ft_out,
                TRUE, TRUE, FALSE, FALSE,
                pf_to, pf_from, 0, 0
            );
        }

        if (type_of_p(accum->delta.captured) == PAWN) {

            const size_t pf_cap = nnue_pawn_index(accum->delta.captured, colour, sq_cap, k_sq);

            add_sub_add_sub(
                accum->pawn_ft[colour], accum->pawn_ft[colour], pawn_ft_weights, pawn_ft_out,
                FALSE, TRUE, FALSE, FALSE,
                0, pf_cap, 0, 0
            );
        }
    }

    else {

        add_sub_add_sub(
            accum->ft[colour], (accum-1)->ft[colour], ft_weights, ft_out,
            TRUE, TRUE, FALSE, FALSE,
            f_to, f_from, 0, 0
        );

        if (pt_from == PAWN) {

            const size_t pf_from = nnue_pawn_index(make_piece(c_from, PAWN), colour, sq_from, k_sq);
            const size_t pf_to   = nnue_pawn_index(make_piece(c_from, PAWN), colour, sq_to  , k_sq);

            add_sub_add_sub(
                accum->pawn_ft[colour], (accum-1)->pawn_ft[colour], pawn_ft_weights, pawn_ft_out,
                TRUE, TRUE, FALSE, FALSE,
                pf_to, pf_from, 0, 0
            );
        }

        else {

            // No pawns were involved, so this data has not been copied yet
            memcpy(accum->pawn_ft[colour], (accum-1)->pawn_ft[colour], sizeof(int16_t) * pawn_ft_out);
        }
    }
}

void nnue_pop(const Position* pos) {
    --pos->nnue->curr;
}

void nnue_push(const Position* pos, Move move, Piece moving, Piece captured) {

    NNUEAccumulator *accum = ++pos->nnue->curr;

    accum->accurate[WHITE] = accum->accurate[BLACK] = 0;

    accum->delta.move     = move;
    accum->delta.moving   = moving;
    accum->delta.captured = captured;
}


static Value full_evaluate_nnue(const Position *pos) {

    NNUEAccumulator* acc = pos->nnue->curr;

    for (int colour = WHITE; colour <= BLACK; colour++) {

        // Already accurate from a previous search
        if (acc->accurate[colour])
            continue;

        // King never crossed the barrier in recent moves
        else if (nnue_can_update(pos, acc, colour))
            nnue_update_accum(pos, acc, colour);

        // King crossed, full refresh needed
        else
            reset_accum(pos, acc, colour);
    }

    alignas(64) int16_t acc_ft[l1_in];
    alignas(64) float   acc_l1[l1_out];
    alignas(64) float   acc_l2[l2_out];
    alignas(64) float   acc_l3[l3_out];

    size_t bucket = (popcount(pieces()) - 1) / 4;

    activate_ft(pos, acc, acc_ft);
    affine_relu_l1(acc_ft, acc_l1);
    affine_relu_l2(acc_l1, acc_l2, bucket);
    affine_output_l3(acc_l2, acc_l3, bucket);

    float result = acc_l3[0] + (acc->mat[stm()] * 32 * 32 * 32 / 2);

    return 3 * result / (32 * 32 * 32 * 2);
}


void __attribute__((minsize, cold)) eval_init(void) {

    #define unremap(x) ((x % 2) ? (-(x-1)/2) : (x) / 2)

    // Transpose, and cast from i8 to i16. Every chunk of 16 is averaged and split out
    NO_UNROLL for (size_t i = 0; i < ft_out; i++)
        NO_UNROLL for (size_t j = 0; j < ft_in; j++)
            ft_weights[j * ft_out + i] = unremap(ft_weights_i8[i * ft_in + j]) + ft_weights_avg[(i * ft_in + j) / 16];

    // Transpose, and cast from i8 to i16. Every chunk of 16 is averaged and split out
    NO_UNROLL for (size_t i = 0; i < pawn_ft_out; i++)
        NO_UNROLL for (size_t j = 0; j < pawn_ft_in; j++)
            pawn_ft_weights[j * pawn_ft_out + i] = unremap(pawn_ft_weights_i8[i * pawn_ft_in + j]) + pawn_ft_weights_avg[(i * pawn_ft_in + j) / 16];

    NO_UNROLL for (size_t i = 0; i < l1_in * l1_out; i++)
        l1_weights[i] = l1_weights_i8[i];

    NO_UNROLL for (size_t i = 0; i < l1_out; i++)
        l1_bias[i] = 32 * l1_bias_i16[i];

    NO_UNROLL for (size_t i = 0; i < l2_in * l2_out * n_buckets; i++)
        l2_weights[i] = l2_weights_i8[i];

    NO_UNROLL for (size_t i = 0; i < l2_out * n_buckets; i++)
        l2_bias[i] = 32 * 32 * l2_bias_i16[i];

    NO_UNROLL for (size_t i = 0; i < l3_in * l3_out * n_buckets; i++)
        l3_weights[i] = l3_weights_i16[i] / l3_weight_scale;

    #undef unremap
}

NOINLINE Value evaluate(const Position *pos) {

    Value v = full_evaluate_nnue(pos);

    int phase = 1024
               - popcount(pieces_p(PAWN  )) * 20
               - popcount(pieces_p(KNIGHT)) * 59
               - popcount(pieces_p(BISHOP)) * 26
               - popcount(pieces_p(ROOK  )) * 24
               - popcount(pieces_p(QUEEN )) * 167;

    if (phase < 0) phase = 0;
    if (phase > 1024) phase = 1024;

    v = v * 1071 / 1024 - v * phase / 3603;

    return clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
}
