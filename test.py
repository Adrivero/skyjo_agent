from game.board import Board

if __name__ == "__main__":
    board = Board(n_players=3)
    scoreboard1 = board.play_one_game()
    print(scoreboard1)
    scoreboard2 = board.play_one_game()
    print(scoreboard2)
    scoreboard3 = board.play_one_game()
    print(scoreboard3)

    final_scoreboard = {} 
    for score in [scoreboard1, scoreboard2, scoreboard3]:
        for player, points in score.items():
            final_scoreboard[player] = final_scoreboard.get(player, 0) + points

    winner = min(final_scoreboard, key=final_scoreboard.get)
    print(f"Player {winner} wins with {final_scoreboard[winner]} points!")
    print("Final scores:")
    for player, points in final_scoreboard.items():
        print(f"Player {player}: {points} points")