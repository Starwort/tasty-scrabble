use std::fs::File;
use std::iter::Extend;
use std::sync::RwLock;
use std::time::Instant;

use ahash::AHashMap;
use arbitrary_int::*;
use bitbybit::{bitenum, bitfield};
use rayon::prelude::*;

#[derive(Debug, PartialEq, Eq)]
#[bitenum(u1, exhaustive: true)]
enum Turn {
    Player1 = 0,
    Player2 = 1,
}

#[derive(Debug, Clone, Copy)]
enum Direction {
    Vertical = (1 | 1 << 9 | 1 << 18),
    VerticalShort = (1 | 1 << 9),
    Horizontal = (1 | 1 << 1 | 1 << 2),
    HorizontalShort = (1 | 1 << 1),
}

/// A bit-board for the Scrabble board. The 9x9 board has 81 squares, so we need
/// at least 81 bits to represent it.
///
/// Note that because M is the only valid tile, we can use a single bit to
/// represent each tile. Additionally, the top bit will represent which player's
/// turn it is.
///
/// The bag contains 30 Ms (although only 16 will be in the bag during the
/// game), so we need 5 bits to represent the number of tiles in the bag -
/// unfortunately this requires one extra bit for 16.
///
/// This leaves us with 41 bits to represent the racks of the
/// players; 3 bits each and 35 bits unused.
///
/// The final memory layout therefore looks like this:
/// T BBBBB (turn 1b, bag 5b)
/// RRR RRR (player 1 rack 3b, player 2 rack 3b)
/// ___________________________________ (unused 35b)
/// MMMMMMMMM (row 1 9b)
/// MMMMMMMMM (row 2 9b)
/// MMMMMMMMM (row 3 9b)
/// MMMMMMMMM (row 4 9b)
/// MMMMMMMMM (row 5 9b)
/// MMMMMMMMM (row 6 9b)
/// MMMMMMMMM (row 7 9b)
/// MMMMMMMMM (row 8 9b)
/// MMMMMMMMM (row 9 9b)
#[bitfield(u128)]
struct Board {
    #[bit(127, rw)]
    turn: Turn,
    #[bits(122..=126, rw)]
    bag: u5,
    #[bits(119..=121, rw)]
    player1_rack: u3,
    #[bits(116..=118, rw)]
    player2_rack: u3,
    #[bits(0..=80, rw)]
    board: u81,
}

const fn square(x: u128, y: u128) -> u128 {
    1 << (y * 9 + x)
}

impl Board {
    const BOARD_MASK: u128 = 0x1_FFFF_FFFF_FFFF_FFFF_FFFF;
    const DOUBLE_LETTERS: u128 = square(4, 0)
        | square(3, 3)
        | square(5, 3)
        | square(0, 4)
        | square(8, 4)
        | square(3, 5)
        | square(5, 5)
        | square(4, 8);
    const DOUBLE_WORDS: u128 =
        square(1, 1) | square(7, 1) | square(4, 4) | square(1, 7) | square(7, 7);
    const TRIPLE_LETTERS: u128 =
        square(2, 2) | square(6, 2) | square(2, 6) | square(6, 6);
    const TRIPLE_WORDS: u128 =
        square(0, 0) | square(8, 0) | square(0, 8) | square(8, 8);

    pub const fn new() -> Self {
        Self::new_with_raw_value(16 << 122 | 7 << 119 | 7 << 116)
    }

    pub const fn hooks(&self) -> u128 {
        let board = self.raw_value & Self::BOARD_MASK;
        if board == 0 {
            return 1 << 40; // middle square
        }
        let up_hooks = board >> 9;
        let down_hooks = board << 9;
        // this technically wraps around the edges, but it's fine because we
        // have to check for validity anyway
        let left_hooks = board >> 1;
        let right_hooks = board << 1;
        (up_hooks | down_hooks | left_hooks | right_hooks) & Self::BOARD_MASK & !board
    }

    pub fn canonicalise(&self) -> u128 {
        let board = self.raw_value & Self::BOARD_MASK;
        let other_state = self.raw_value & !Self::BOARD_MASK;
        let mut transposed_board = 0;
        for i in 0..9 {
            for j in 0..9 {
                transposed_board |= ((board >> (i * 9 + j)) & 1) << (j * 9 + i);
            }
        }
        other_state
            | if board < transposed_board {
                board
            } else {
                transposed_board
            }
    }

    pub fn make_move(&self, start: u8, direction: Direction) -> Self {
        debug_assert!(self.score(start, direction) != 0);
        let play_mask = ((direction as u128) << start) & Self::BOARD_MASK;
        let board = self.raw_value & Self::BOARD_MASK;
        #[allow(clippy::cast_possible_truncation)]
        let used_tiles = (play_mask & !board).count_ones() as u8;
        let tiles_left: u8 = self.bag().into();
        let mut result = match self.turn() {
            Turn::Player1 => {
                if used_tiles > tiles_left {
                    let next_rack_tiles = 7 - (used_tiles - tiles_left);
                    self.with_bag(u5::new(0u8))
                        .with_player1_rack(u3::new(next_rack_tiles))
                        .with_turn(Turn::Player2)
                } else {
                    self.with_bag(u5::new(tiles_left - used_tiles))
                        .with_turn(Turn::Player2)
                }
            },
            Turn::Player2 => {
                if used_tiles > tiles_left {
                    let next_rack_tiles = 7 - (used_tiles - tiles_left);
                    self.with_bag(u5::new(0u8))
                        .with_player2_rack(u3::new(next_rack_tiles))
                        .with_turn(Turn::Player1)
                } else {
                    self.with_bag(u5::new(tiles_left - used_tiles))
                        .with_turn(Turn::Player1)
                }
            },
        };
        result.raw_value |= play_mask;
        result
    }

    /// Returns the score of the given play, or 0 if the play is invalid.
    #[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
    pub fn score(&self, start: u8, direction: Direction) -> i8 {
        let (y, x) = (start / 9, start % 9);
        match (x, y, direction) {
            (_, 7 | 8, Direction::Vertical)
            | (_, 8, Direction::VerticalShort)
            | (7 | 8, _, Direction::Horizontal)
            | (8, _, Direction::HorizontalShort) => {
                // play is invalid because it would go off the board
                return 0;
            },
            _ => (),
        };
        let play_mask = ((direction as u128) << start) & Self::BOARD_MASK;
        let board = self.raw_value & Self::BOARD_MASK;
        if play_mask & board == play_mask {
            // play is invalid because it doesn't use any new tiles
            return 0;
        }
        let hooks = self.hooks();
        if play_mask & hooks == 0 {
            // play is invalid because it cannot possibly connect to the board
            return 0;
        }
        // determine lengths of overlaps
        let next = board | play_mask;
        match direction {
            Direction::Vertical => {
                if (board & (1 << (start - 9)) != 0)
                    || (board & (1 << (start + 9)) != 0)
                    || [
                        // length-4 words in the first row
                        (3, 0),
                        (2, 0),
                        (1, 0),
                        (0, 0),
                        // in the second row
                        (3, 1),
                        (2, 1),
                        (1, 1),
                        (0, 1),
                        // in the third row
                        (3, 2),
                        (2, 2),
                        (1, 2),
                        (0, 2),
                    ]
                    .into_iter()
                    .any(|(ox, oy)| {
                        x >= ox
                            && x < ox + 6
                            && ((15 << (start - ox + 9 * oy)) & next).count_ones() == 4
                    })
                {
                    // play is invalid because it would lead to a length-4 word
                    // (not valid)
                    0
                } else {
                    // play is valid
                    let bonus_mask = play_mask & !board;
                    // not bothering to count the value of the M tile, as every tile is
                    // an M
                    let mut score = 3;
                    score += (bonus_mask & Self::DOUBLE_LETTERS).count_ones() as i8;
                    score += 2 * (bonus_mask & Self::TRIPLE_LETTERS).count_ones() as i8;
                    for _ in 0..(bonus_mask & Self::DOUBLE_WORDS).count_ones() {
                        score *= 2;
                    }
                    for _ in 0..(bonus_mask & Self::TRIPLE_WORDS).count_ones() {
                        score *= 3;
                    }
                    // score now holds the value for the base play
                    // now calculate the value of hooks
                    for oy in 0..=2 {
                        let square_bonus = bonus_mask & (1 << (start + 9 * oy));
                        if square_bonus == 0 {
                            continue; // not eligible for hook points
                        }
                        if (0..=2).any(|ox| {
                            x >= ox
                                && x < ox + 7
                                && ((7 << (start - ox + 9 * oy)) & next).count_ones()
                                    == 3
                        }) {
                            score += 3;
                            // only one of these can be true, and it will always
                            // have the specified result (by construction)
                            if square_bonus & Self::DOUBLE_LETTERS != 0 {
                                score += 1;
                            } else if square_bonus & Self::TRIPLE_LETTERS != 0 {
                                score += 2;
                            } else if square_bonus & Self::DOUBLE_WORDS != 0 {
                                score += 3;
                            } else if square_bonus & Self::TRIPLE_WORDS != 0 {
                                score += 6;
                            }
                        } else if (0..=1).any(|ox| {
                            x >= ox
                                && x < ox + 8
                                && ((3 << (start - ox + 9 * oy)) & next).count_ones()
                                    == 2
                        }) {
                            score += 2;
                            // only one of these can be true, and it will always
                            // have the specified result (by construction)
                            if square_bonus & Self::DOUBLE_LETTERS != 0 {
                                score += 1;
                            } else if square_bonus
                                & (Self::TRIPLE_LETTERS | Self::DOUBLE_WORDS)
                                != 0
                            {
                                score += 2;
                            } else if square_bonus & Self::TRIPLE_WORDS != 0 {
                                score += 4;
                            }
                        }
                    }

                    score
                }
            },
            Direction::VerticalShort => {
                // Overlaps at the start and end are disallowed
                // because they would lead to length-3 words AKA Direction::Vertical
                // so for pruning reasons we can pretend they're invalid
                if (board & (1 << (start - 9)) != 0)
                    || (board & (1 << (start + 9)) != 0)
                    || [
                        // length-4 words in the first row
                        (3, 0),
                        (2, 0),
                        (1, 0),
                        (0, 0),
                        // in the second row
                        (3, 1),
                        (2, 1),
                        (1, 1),
                        (0, 1),
                    ]
                    .into_iter()
                    .any(|(ox, oy)| {
                        x >= ox
                            && x < ox + 6
                            && ((15 << (start - ox + 9 * oy)) & next).count_ones() == 4
                    })
                {
                    // play is invalid because it would lead to a length-4 word
                    // (not valid) or a length-3 vertical word (not valid for
                    // a VerticalShort)
                    0
                } else {
                    // play is valid
                    let bonus_mask = play_mask & !board;
                    // not bothering to count the value of the M tile, as every tile is
                    // an M
                    let mut score = 2;
                    score += (bonus_mask & Self::DOUBLE_LETTERS).count_ones() as i8;
                    score += 2 * (bonus_mask & Self::TRIPLE_LETTERS).count_ones() as i8;
                    for _ in 0..(bonus_mask & Self::DOUBLE_WORDS).count_ones() {
                        score *= 2;
                    }
                    for _ in 0..(bonus_mask & Self::TRIPLE_WORDS).count_ones() {
                        score *= 3;
                    }
                    // score now holds the value for the base play
                    // now calculate the value of hooks
                    for oy in 0..=1 {
                        let square_bonus = bonus_mask & (1 << (start + 9 * oy));
                        if square_bonus == 0 {
                            continue; // not eligible for hook points
                        }
                        if (0..=2).any(|ox| {
                            x >= ox
                                && x < ox + 7
                                && ((7 << (start - ox + 9 * oy)) & next).count_ones()
                                    == 3
                        }) {
                            score += 3;
                            // only one of these can be true, and it will always
                            // have the specified result (by construction)
                            if square_bonus & Self::DOUBLE_LETTERS != 0 {
                                score += 1;
                            } else if square_bonus & Self::TRIPLE_LETTERS != 0 {
                                score += 2;
                            } else if square_bonus & Self::DOUBLE_WORDS != 0 {
                                score += 3;
                            } else if square_bonus & Self::TRIPLE_WORDS != 0 {
                                score += 6;
                            }
                        } else if (0..=1).any(|ox| {
                            x >= ox
                                && x < ox + 8
                                && ((3 << (start - ox + 9 * oy)) & next).count_ones()
                                    == 2
                        }) {
                            score += 2;
                            // only one of these can be true, and it will always
                            // have the specified result (by construction)
                            if square_bonus & Self::DOUBLE_LETTERS != 0 {
                                score += 1;
                            } else if square_bonus
                                & (Self::TRIPLE_LETTERS | Self::DOUBLE_WORDS)
                                != 0
                            {
                                score += 2;
                            } else if square_bonus & Self::TRIPLE_WORDS != 0 {
                                score += 4;
                            }
                        }
                    }

                    score
                }
            },
            Direction::Horizontal => {
                if (board & (1 << (start - 1)) != 0)
                    || (board & (1 << (start + 1)) != 0)
                    || [
                        // length-4 words in the first column
                        (0, 3),
                        (0, 2),
                        (0, 1),
                        (0, 0),
                        // in the second column
                        (1, 3),
                        (1, 2),
                        (1, 1),
                        (1, 0),
                        // in the third column
                        (2, 3),
                        (2, 2),
                        (2, 1),
                        (2, 0),
                    ]
                    .into_iter()
                    .any(|(ox, oy)| {
                        y >= oy
                            && y < oy + 6
                            && (((square(0, 0)
                                | square(0, 1)
                                | square(0, 2)
                                | square(0, 3))
                                << (start + ox - 9 * oy))
                                & next)
                                .count_ones()
                                == 4
                    })
                {
                    // play is invalid because it would lead to a length-4 word
                    // (not valid)
                    0
                } else {
                    // play is valid
                    let bonus_mask = play_mask & !board;
                    // not bothering to count the value of the M tile, as every tile is
                    // an M
                    let mut score = 3;
                    score += (bonus_mask & Self::DOUBLE_LETTERS).count_ones() as i8;
                    score += 2 * (bonus_mask & Self::TRIPLE_LETTERS).count_ones() as i8;
                    for _ in 0..(bonus_mask & Self::DOUBLE_WORDS).count_ones() {
                        score *= 2;
                    }
                    for _ in 0..(bonus_mask & Self::TRIPLE_WORDS).count_ones() {
                        score *= 3;
                    }
                    // score now holds the value for the base play
                    // now calculate the value of hooks
                    for ox in 0..=2 {
                        let square_bonus = bonus_mask & (1 << (start + ox));
                        if square_bonus == 0 {
                            continue; // not eligible for hook points
                        }
                        if (0..=2).any(|oy| {
                            y >= oy
                                && y < oy + 7
                                && ((7 << (start - ox + 9 * oy)) & next).count_ones()
                                    == 3
                        }) {
                            score += 3;
                            // only one of these can be true, and it will always
                            // have the specified result (by construction)
                            if square_bonus & Self::DOUBLE_LETTERS != 0 {
                                score += 1;
                            } else if square_bonus & Self::TRIPLE_LETTERS != 0 {
                                score += 2;
                            } else if square_bonus & Self::DOUBLE_WORDS != 0 {
                                score += 3;
                            } else if square_bonus & Self::TRIPLE_WORDS != 0 {
                                score += 6;
                            }
                        } else if (0..=1).any(|oy| {
                            y >= oy
                                && y < oy + 8
                                && ((3 << (start - ox + 9 * oy)) & next).count_ones()
                                    == 2
                        }) {
                            score += 2;
                            // only one of these can be true, and it will always
                            // have the specified result (by construction)
                            if square_bonus & Self::DOUBLE_LETTERS != 0 {
                                score += 1;
                            } else if square_bonus
                                & (Self::TRIPLE_LETTERS | Self::DOUBLE_WORDS)
                                != 0
                            {
                                score += 2;
                            } else if square_bonus & Self::TRIPLE_WORDS != 0 {
                                score += 4;
                            }
                        }
                    }

                    score
                }
            },
            Direction::HorizontalShort => {
                // Overlaps at the start and end are disallowed
                // because they would lead to length-3 words AKA Direction::Horizontal
                // so for pruning reasons we can pretend they're invalid
                if (board & (1 << (start - 1)) != 0)
                    || (board & (1 << (start + 1)) != 0)
                    || [
                        // length-4 words in the first column
                        (0, 3),
                        (0, 2),
                        (0, 1),
                        (0, 0),
                        // in the second column
                        (1, 3),
                        (1, 2),
                        (1, 1),
                        (1, 0),
                    ]
                    .into_iter()
                    .any(|(ox, oy)| {
                        y >= oy
                            && y < oy + 6
                            && (((square(0, 0)
                                | square(0, 1)
                                | square(0, 2)
                                | square(0, 3))
                                << (start + ox - 9 * oy))
                                & next)
                                .count_ones()
                                == 4
                    })
                {
                    // play is invalid because it would lead to a length-4 word
                    // (not valid) or a length-3 horizontal word (not valid for
                    // a HorizontalShort)
                    0
                } else {
                    // play is valid
                    let bonus_mask = play_mask & !board;
                    // not bothering to count the value of the M tile, as every tile is
                    // an M
                    let mut score = 2;
                    score += (bonus_mask & Self::DOUBLE_LETTERS).count_ones() as i8;
                    score += 2 * (bonus_mask & Self::TRIPLE_LETTERS).count_ones() as i8;
                    for _ in 0..(bonus_mask & Self::DOUBLE_WORDS).count_ones() {
                        score *= 2;
                    }
                    for _ in 0..(bonus_mask & Self::TRIPLE_WORDS).count_ones() {
                        score *= 3;
                    }
                    // score now holds the value for the base play
                    // now calculate the value of hooks
                    for ox in 0..=1 {
                        let square_bonus = bonus_mask & (1 << (start + ox));
                        if square_bonus == 0 {
                            continue; // not eligible for hook points
                        }
                        if (0..=2).any(|oy| {
                            y >= oy
                                && y < oy + 7
                                && ((7 << (start - ox + 9 * oy)) & next).count_ones()
                                    == 3
                        }) {
                            score += 3;
                            // only one of these can be true, and it will always
                            // have the specified result (by construction)
                            if square_bonus & Self::DOUBLE_LETTERS != 0 {
                                score += 1;
                            } else if square_bonus & Self::TRIPLE_LETTERS != 0 {
                                score += 2;
                            } else if square_bonus & Self::DOUBLE_WORDS != 0 {
                                score += 3;
                            } else if square_bonus & Self::TRIPLE_WORDS != 0 {
                                score += 6;
                            }
                        } else if (0..=1).any(|oy| {
                            y >= oy
                                && y < oy + 8
                                && ((3 << (start - ox + 9 * oy)) & next).count_ones()
                                    == 2
                        }) {
                            score += 2;
                            // only one of these can be true, and it will always
                            // have the specified result (by construction)
                            if square_bonus & Self::DOUBLE_LETTERS != 0 {
                                score += 1;
                            } else if square_bonus
                                & (Self::TRIPLE_LETTERS | Self::DOUBLE_WORDS)
                                != 0
                            {
                                score += 2;
                            } else if square_bonus & Self::TRIPLE_WORDS != 0 {
                                score += 4;
                            }
                        }
                    }

                    score
                }
            },
        }
    }

    pub fn next_states(&self) -> impl Iterator<Item = (Self, i8)> + '_ {
        (0..81).flat_map(move |i| {
            [
                Direction::Horizontal,
                Direction::HorizontalShort,
                Direction::Vertical,
                Direction::VerticalShort,
            ]
            .into_iter()
            .filter_map(move |dir| {
                let mut score = self.score(i, dir);
                if score == 0 {
                    None
                } else {
                    if self.turn() == Turn::Player2 {
                        score = -score;
                    }
                    Some((self.make_move(i, dir), score))
                }
            })
        })
    }
}

#[allow(clippy::cast_possible_wrap)]
fn min(
    cache: &RwLock<AHashMap<u128, i8>>,
    last_cache_update: &mut Instant,
    my_cache: &mut AHashMap<u128, i8>,
    state: Board,
    alpha: i8,
    mut beta: i8,
) -> i8 {
    debug_assert!(state.turn() == Turn::Player2);
    let canon_state = state.canonicalise();
    if let Some(&cached) = my_cache.get(&canon_state) {
        return cached;
    }
    // Did P1 go out last turn? If so, the score for this position is twice the
    // sum (count) of P2's tiles
    if state.player1_rack() == u3::new(0u8) {
        let score = u8::from(state.player2_rack()) as i8 * 2;
        my_cache.insert(canon_state, score);
        return score;
    }
    let mut best = i8::MAX;
    let mut had_states = false;
    for (next_state, spread) in state.next_states() {
        had_states = true;
        let score =
            max(cache, last_cache_update, my_cache, next_state, alpha, beta) + spread;
        if score < best {
            best = score;
            if best < beta {
                beta = best;
                if beta <= alpha {
                    break;
                }
            }
        }
    }
    if !had_states {
        // No possible moves left - it's a tie
        best = 0;
    }
    my_cache.insert(canon_state, best);
    let time_since_last_update = last_cache_update.elapsed();
    if time_since_last_update.as_secs() > 30 {
        let mut cache = cache.write().unwrap();
        let lock_contention = last_cache_update.elapsed() - time_since_last_update;
        cache.extend(my_cache.drain());
        my_cache.extend(cache.iter().map(|(&k, &v)| (k, v)));
        eprintln!(
            "Cache size after {:?}: {}",
            time_since_last_update,
            cache.len()
        );
        eprintln!("Lock contention: {lock_contention:?}");
        *last_cache_update = Instant::now();
    }
    best
}

#[allow(clippy::cast_possible_wrap)]
fn max(
    cache: &RwLock<AHashMap<u128, i8>>,
    last_cache_update: &mut Instant,
    my_cache: &mut AHashMap<u128, i8>,
    state: Board,
    mut alpha: i8,
    beta: i8,
) -> i8 {
    debug_assert!(state.turn() == Turn::Player1);
    let canon_state = state.canonicalise();
    if let Some(&cached) = my_cache.get(&canon_state) {
        return cached;
    }
    // Did P2 go out last turn? If so, the score for this position is twice the
    // sum (count) of P1's tiles
    if state.player2_rack() == u3::new(0u8) {
        let score = -(u8::from(state.player1_rack()) as i8) * 2;
        my_cache.insert(canon_state, score);
        return score;
    }
    let mut best = i8::MIN;
    let mut had_states = false;
    for (next_state, spread) in state.next_states() {
        had_states = true;
        let score =
            min(cache, last_cache_update, my_cache, next_state, alpha, beta) + spread;
        if score > best {
            best = score;
            if best > alpha {
                alpha = best;
                if beta <= alpha {
                    break;
                }
            }
        }
    }
    if !had_states {
        // No possible moves left - it's a tie
        best = 0;
    }
    my_cache.insert(canon_state, best);
    let time_since_last_update = last_cache_update.elapsed();
    if time_since_last_update.as_secs() > 30 {
        let mut cache = cache.write().unwrap();
        let lock_contention = last_cache_update.elapsed() - time_since_last_update;
        cache.extend(my_cache.drain());
        my_cache.extend(cache.iter().map(|(&k, &v)| (k, v)));
        eprintln!(
            "Cache size after {:?}: {}",
            time_since_last_update,
            cache.len()
        );
        eprintln!("Lock contention: {lock_contention:?}");
        *last_cache_update = Instant::now();
    }
    best
}

fn main() {
    // state -> best possible spread for the current player over the rest of the
    // game
    let cache = RwLock::new(AHashMap::<u128, i8>::new());
    let game = Board::new();
    game.next_states()
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|(next_state, spread)| {
            debug_assert_ne!(next_state.board().count_ones(), 0);
            let mut last_cache_update = Instant::now();
            let mut my_cache = cache.read().unwrap().clone();
            let score = min(
                &cache,
                &mut last_cache_update,
                &mut my_cache,
                next_state,
                i8::MIN,
                i8::MAX,
            ) + spread;
            my_cache.insert(next_state.canonicalise(), score);
            cache.write().unwrap().extend(my_cache);
        });
    let mut my_cache = cache.read().unwrap().clone();
    let overall_spread = max(
        &cache,
        &mut Instant::now(),
        &mut my_cache,
        game,
        i8::MIN,
        i8::MAX,
    );
    let mut cache = cache.into_inner().unwrap();
    cache.extend(my_cache);
    println!("Forced spread: {overall_spread}");
    let results = File::create("results.json").unwrap();
    serde_json::to_writer(results, &cache).unwrap();
}
