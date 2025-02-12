#include <immintrin.h>

extern __m256i queen_mask_v4[64][2];
extern __m256i bishop_mask_v4[64];
extern __m128i rook_mask_NS[64];
extern uint8_t rook_attacks_EW[64 * 8];

// attacks_bb() returns a bitboard representing all the squares attacked
// by a piece of type Pt (bishop or rook) placed on 's'. The helper
// magic_index() looks up the index using the 'magic bitboards' approach.

// avx2/sse2 versions of BLSMSK (https://www.chessprogramming.org/BMI1#BLSMSK)
INLINE __m256i blsmsk64x4(__m256i y)
{
  return _mm256_xor_si256(_mm256_add_epi64(y, _mm256_set1_epi64x(-1)), y);
}

INLINE __m128i blsmsk64x2(__m128i x)
{
  return _mm_xor_si128(_mm_add_epi64(x, _mm_set1_epi64x(-1)), x);
}

#undef attacks_bb_queen

Bitboard attacks_bb_queen(Square s, Bitboard occupied);
Bitboard attacks_bb_rook(Square s, Bitboard occupied);
Bitboard attacks_bb_bishop(Square s, Bitboard occupied);
