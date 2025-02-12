__m256i queen_mask_v4[64][2];
__m256i bishop_mask_v4[64];
__m128i rook_mask_NS[64];
uint8_t rook_attacks_EW[64 * 8];

static void __attribute__((minsize, cold)) init_sliding_attacks(void)
{
  static const int dirs[2][4] = {{ EAST, NORTH, NORTH_EAST, NORTH_WEST }, { WEST, SOUTH, SOUTH_WEST, SOUTH_EAST }};
  Bitboard attacks[4];
  int i, j, occ8;
  Square sq, s;
  uint8_t s8, att8;

  // pseudo attacks for Queen 8 directions
  for (sq = SQ_A1; sq <= SQ_H8; ++sq)
    for (j = 0; j < 2; ++j) {
      for (i = 0; i < 4; ++i) {
        attacks[i] = 0;
        for (s = sq + dirs[j][i];
            square_is_ok(s) && distance(s, s - dirs[j][i]) == 1; s += dirs[j][i])
        {
          attacks[i] |= sq_bb(s);
        }
      }
      queen_mask_v4[sq][j] = _mm256_set_epi64x(attacks[3], attacks[2], attacks[1], attacks[0]);
    }

  // pseudo attacks for Rook (NORTH-SOUTH) and Bishop
  for (sq = SQ_A1; sq <= SQ_H8; ++sq) {
    rook_mask_NS[sq] = _mm_set_epi64x(
        _mm256_extract_epi64(queen_mask_v4[SQUARE_FLIP(sq)][0], 1),	// SOUTH (vertically flipped)
        _mm256_extract_epi64(queen_mask_v4[sq][0], 1));	// NORTH
    bishop_mask_v4[sq] = _mm256_set_epi64x(
        _mm256_extract_epi64(queen_mask_v4[SQUARE_FLIP(sq)][0], 2),	// SOUTH_EAST (vertically flipped)
        _mm256_extract_epi64(queen_mask_v4[sq][0], 3),	// NORTH_WEST
        _mm256_extract_epi64(queen_mask_v4[SQUARE_FLIP(sq)][0], 3),	// SOUTH_WEST (vertically flipped)
        _mm256_extract_epi64(queen_mask_v4[sq][0], 2));	// NORTH_EAST
  }

  // sliding attacks for Rook EAST-WEST
  for (occ8 = 0; occ8 < 128; occ8 += 2)	// inner 6 bits
    for (sq = 0; sq < 8; ++sq) {
      att8 = 0;
      for (s8 = (1 << sq) << 1; s8; s8 <<= 1) {
        att8 |= s8;
        if (occ8 & s8)
          break;
      }
      for (s8 = (1 << sq) >> 1; s8; s8 >>= 1) {
        att8 |= s8;
        if (occ8 & s8)
          break;
      }
      rook_attacks_EW[occ8 * 4 + sq] = att8;
    }
}


Bitboard attacks_bb_queen(Square s, Bitboard occupied)
{
  const __m256i occupied4 = _mm256_set1_epi64x(occupied);
  const __m256i lmask = queen_mask_v4[s][0];
  const __m256i rmask = queen_mask_v4[s][1];
  __m256i slide4, rslide;
  __m128i slide2;

  // Left bits: set mask bits lower than occupied LS1B
  slide4 = _mm256_and_si256(occupied4, lmask);
  slide4 = _mm256_and_si256(blsmsk64x4(slide4), lmask);
  // Right bits: set shadow bits lower than occupied MS1B (6 bits max)
  rslide = _mm256_and_si256(occupied4, rmask);
  rslide = _mm256_or_si256(_mm256_srlv_epi64(rslide, _mm256_set_epi64x(14, 18, 16, 2)),  // PP Fill
      _mm256_srlv_epi64(rslide, _mm256_set_epi64x(7, 9, 8, 1)));
  rslide = _mm256_or_si256(_mm256_srlv_epi64(rslide, _mm256_set_epi64x(28, 36, 32, 4)),
      _mm256_or_si256(rslide, _mm256_srlv_epi64(rslide, _mm256_set_epi64x(14, 18, 16, 2))));
  // add mask bits higher than blocker
  slide4 = _mm256_or_si256(slide4, _mm256_andnot_si256(rslide, rmask));

  // OR 4 vectors
  slide2 = _mm_or_si128(_mm256_castsi256_si128(slide4), _mm256_extracti128_si256(slide4, 1));
  return _mm_cvtsi128_si64(_mm_or_si128(slide2, _mm_unpackhi_epi64(slide2, slide2)));
}

Bitboard attacks_bb_rook(Square s, Bitboard occupied)
{
  // flip vertical to simulate MS1B by LS1B
  const __m128i swapl2h = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
  __m128i occupied2 = _mm_shuffle_epi8(_mm_cvtsi64_si128(occupied), swapl2h);
  const __m128i mask = rook_mask_NS[s];
  // set mask bits lower than occupied LS1B
  __m128i slide2 = _mm_and_si128(blsmsk64x2(_mm_and_si128(occupied2, mask)), mask);
  const __m128i swaph2l = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15);
  Bitboard slides = _mm_cvtsi128_si64(_mm_or_si128(slide2, _mm_shuffle_epi8(slide2, swaph2l)));

  // East-West: from precomputed table
  int r8 = rank_of(s) * 8;
  slides |= (Bitboard)(rook_attacks_EW[((occupied >> r8) & 0x7e) * 4 + file_of(s)]) << r8;
  return slides;
}

Bitboard attacks_bb_bishop(Square s, Bitboard occupied)
{
  // flip vertical to simulate MS1B by LS1B
  const __m128i swapl2h = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
  __m128i occupied2 = _mm_shuffle_epi8(_mm_cvtsi64_si128(occupied), swapl2h);
  __m256i occupied4 = _mm256_broadcastsi128_si256(occupied2);

  const __m256i mask = bishop_mask_v4[s];
  // set mask bits lower than occupied LS1B
  __m256i slide4 = _mm256_and_si256(blsmsk64x4(_mm256_and_si256(occupied4, mask)), mask);

  __m128i slide2 = _mm_or_si128(_mm256_castsi256_si128(slide4), _mm256_extracti128_si256(slide4, 1));
  const __m128i swaph2l = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15);
  return _mm_cvtsi128_si64(_mm_or_si128(slide2, _mm_shuffle_epi8(slide2, swaph2l)));
}
