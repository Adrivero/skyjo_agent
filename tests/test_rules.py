import unittest

from game.board import Board
from game.deck import Card


def board_with_scores(first_score, second_score):
    board = Board(2)
    board.reset_game(player_ids=["player_0", "player_1"], shuffle_players=False)
    for player, score in zip(board.players, (first_score, second_score)):
        for index, card in enumerate(player.hand.grid.flatten()):
            card.value = score if index == 0 else 0
            card.reveal()
    return board


class FinalScoreTests(unittest.TestCase):
    def test_unique_lowest_closer_is_not_doubled(self):
        board = board_with_scores(5, 10)
        self.assertEqual(board.final_scores("player_0")["player_0"], 5)

    def test_closer_tied_for_lowest_is_not_doubled(self):
        board = board_with_scores(5, 5)
        self.assertEqual(board.final_scores("player_1")["player_1"], 5)

    def test_higher_closer_is_doubled(self):
        board = board_with_scores(10, 5)
        self.assertEqual(board.final_scores("player_0")["player_0"], 20)

    def test_zero_and_negative_closing_scores_are_multiplied_numerically(self):
        zero_board = board_with_scores(0, -1)
        negative_board = board_with_scores(-2, -5)
        self.assertEqual(zero_board.final_scores("player_0")["player_0"], 0)
        self.assertEqual(negative_board.final_scores("player_0")["player_0"], -4)


class ColumnRemovalTests(unittest.TestCase):
    def make_board(self):
        board = Board(2)
        board.reset_game(player_ids=["player_0", "player_1"], shuffle_players=False)
        player = board.players[0]
        for row in range(3):
            card = player.hand.grid[row][3]
            card.value = 4
            card.state = "revealed" if row < 2 else "hidden"
        return board, player

    def assert_fourth_column_removed(self, player):
        self.assertEqual([player.hand.grid[row][3] for row in range(3)], [None] * 3)
        self.assertEqual(player.remove_completed_columns(), [])

    def test_replace_action_removes_completed_fourth_column(self):
        board, player = self.make_board()
        board.play_player_action(
            player, "replace", (2, 3), drawn_card=Card(value=4)
        )
        self.assert_fourth_column_removed(player)

    def test_discard_reveal_action_removes_completed_fourth_column(self):
        board, player = self.make_board()
        board.play_player_action(
            player, "discard_reveal", (2, 3), drawn_card=Card(value=9)
        )
        self.assert_fourth_column_removed(player)

    def test_take_discard_action_removes_completed_fourth_column(self):
        board, player = self.make_board()
        board.deck.heap[-1] = 4
        board.play_player_action(player, "take_discard_replace", (2, 3))
        self.assert_fourth_column_removed(player)

    def test_partial_or_nonmatching_column_is_not_removed(self):
        board, player = self.make_board()
        self.assertFalse(player.remove_column_if_possible(3))
        player.hand.grid[2][3].reveal()
        player.hand.grid[2][3].value = 5
        self.assertFalse(player.remove_column_if_possible(3))
        self.assertTrue(all(player.hand.grid[row][3] is not None for row in range(3)))


if __name__ == "__main__":
    unittest.main()
