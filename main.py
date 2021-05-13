import argparse
import submission
import Games

if __name__ == "__main__":
    games = {'KeyBoardGame': Games.KeyBoardGame, 'CustomGame': Games.CustomGame}

    move_players = {'GreedyMovePlayer': submission.GreedyMovePlayer,
                    'ImprovedGreedyMovePlayer': submission.ImprovedGreedyMovePlayer,
                    'MiniMaxMovePlayer': submission.MiniMaxMovePlayer,
                    'ABMovePlayer': submission.ABMovePlayer,
                    'ExpectimaxMovePlayer': submission.ExpectimaxMovePlayer,
                    'ContestMovePlayer': submission.ContestMovePlayer
                    }

    index_players = {'RandomIndexPlayer': submission.RandomIndexPlayer,
                    'MiniMaxIndexPlayer': submission.MiniMaxIndexPlayer,
                    'ExpectimaxIndexPlayer': submission.ExpectimaxIndexPlayer
                    }

    parser = argparse.ArgumentParser()

    parser.add_argument('-game', default='CustomGame', type=str,
                        help='Option to Player keyboard game or custom game with agents.',
                        choices=games)

    parser.add_argument('-player1', default='GreedyMovePlayer', type=str,
                        help='The type of the first player(Move player).',
                        choices=move_players.keys())
    parser.add_argument('-player2', default='RandomIndexPlayer', type=str,
                        help='The type of the second player(Index Player).',
                        choices=index_players.keys())

    parser.add_argument('-move_time', default=1.0, type=float,
                        help='Time (sec) for each turn.')

    args = parser.parse_args()

    # Players inherit from AbstractPlayer
    player_1_type = args.player1
    player_2_type = args.player2
    move_time = args.move_time

    # print game info to terminal
    print(f'Starting {args.game}!')

    # create game with the given args
    if args.game == 'CustomGame':
        # Create players
        player_1 = move_players[player_1_type]()
        player_2 = index_players[player_2_type]()

        print(args.player1, 'VS', args.player2)
        print('Players have', args.move_time, 'seconds to make a single move.')
        random_value = False
        if player_1_type == 'ExpectimaxMovePlayer':
            random_value = True
        game = games[args.game](player_1, player_2, move_time, random_value)
    else:
        game = games[args.game]()
        print('Push the buttons to move')

    # start playing!
    game.run_game()

