
from collections import defaultdict
import numpy as np
import chess

# covent list of moves in Long Algebraic Notation to dictionary of moves with squares like a1
def moves_dict(list_legal_moves: list):

    dict_legal_moves = defaultdict(list)
    for l in list_legal_moves:

        str_move = str(l)
        dict_legal_moves[str_move[0:2]].append(str_move[2:4])

    # print(l)
    return dict_legal_moves


# convert dictionary of moves in algebraic notation to square number notation
def dict_squares_to_dict_numbers(dict_legal_moves: dict):
    dict_legal_moves_numbers = dict()

    for k, v in dict_legal_moves.items():
        
        parsed_list_destinations = []
        for sq in v:
            parsed_list_destinations.append(chess.parse_square(sq))

        dict_legal_moves_numbers[chess.parse_square(k)] = parsed_list_destinations

    return dict_legal_moves_numbers

# convert dictionary of moves (numbers) to a list where the pieces can move from
def dict_moves_to_list_from_squares(dict_legal_moves_numbers: dict):

    squares_from = dict_legal_moves_numbers.keys()
    length_from_squares = len(squares_from)

    return list(squares_from)


def get_probability_single_legal_piece(list_squares_from: list):

    length_from_squares = len(list_squares_from)

    if length_from_squares == 0:
        return 0

    probability_squares = 1 / length_from_squares

    return probability_squares

##
## !!! rename and remove castling from here because it can be confusing
def create_array_squares_from(squares_from: list, probability_squares):

    board_squares = 64

    # white_castle_kingside = 1
    # white_castle_queenside = 1
    # black_castle_kingside = 1
    # black_castle_queenside = 1

    # indicator of either castle
    # white_castle = 1
    # black_castle = 1

    # moves_neurons_output = board_squares + white_castle_kingside + white_castle_queenside + black_castle_kingside + black_castle_queenside
    # mate_white_black = 2
    # draw = 1
    # ability_castle = 2

    legal_squares_array = np.zeros(board_squares)

    for i in squares_from:
        legal_squares_array[i] = probability_squares

    return legal_squares_array

def checkmate_draw(board: chess.Board):
    array_checkmate_draw = np.zeros(3)
    #check checkmate/draw/castle

    if board.is_checkmate():
        if str(board.outcome()) == 'Outcome(termination=<Termination.CHECKMATE: 1>, winner=True)':
            
            array_checkmate_draw[0] = 1

        elif str(board.outcome()) == 'Outcome(termination=<Termination.CHECKMATE: 1>, winner=False)':
            array_checkmate_draw[1] = 1

    if board.is_stalemate():
        array_checkmate_draw[2] = 1

    return array_checkmate_draw

def castling_ability_2(board: chess.Board):

    castling_rights = np.zeros(2)

    if board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE):
        castling_rights[0] = 1

    elif board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK):
        castling_rights[1] = 1

    return castling_rights

# used for the second dataset - nn1 
def from_legal_moves_to_square_numbers(board):
    legal_moves = board.legal_moves
    dictionary_moves = moves_dict(legal_moves)
    dictionary_moves_numbers = dict_squares_to_dict_numbers(dictionary_moves)
    list_from = dict_moves_to_list_from_squares(dictionary_moves_numbers)
    return list_from

# 64 + 2 + 3
def create_y_values_nn1(board):
    legal_moves = board.legal_moves
    dictionary_moves = moves_dict(legal_moves)
    dictionary_moves_numbers = dict_squares_to_dict_numbers(dictionary_moves)
    list_from = dict_moves_to_list_from_squares(dictionary_moves_numbers)
    probability = get_probability_single_legal_piece(list_from)

    array = create_array_squares_from(list_from, probability) # 64
    castle_desire = castling_ability_2(board) # 2
    win_draw = checkmate_draw(board) # 3

    array = np.hstack((array, castle_desire))
    array = np.hstack((array, win_draw))

    return array




## nn2
# copied from nn1
# def array_castling_rights(board: chess.Board):

#     castling_rights = [0,0,0,0]
#     castling_rights[0] = board.has_kingside_castling_rights(chess.WHITE)
#     castling_rights[1] = board.has_queenside_castling_rights(chess.WHITE)
#     castling_rights[2] = board.has_kingside_castling_rights(chess.BLACK)
#     castling_rights[3] = board.has_queenside_castling_rights(chess.BLACK)

#     return np.array(castling_rights, dtype='float32')

def array_castling_rights_n2(board: chess.Board, square_move_from_number):

    castling_rights = np.array([0,0,0,0])


    if board.turn and square_move_from_number == 64: # white turn
        print('white move, castle desire (64) indicated')
        castling_rights[0] = board.has_kingside_castling_rights(chess.WHITE)
        castling_rights[1] = board.has_queenside_castling_rights(chess.WHITE)

        

    elif board.turn == 0 and square_move_from_number == 65:
        print('black move')
        castling_rights[2] = board.has_kingside_castling_rights(chess.BLACK)
        castling_rights[3] = board.has_queenside_castling_rights(chess.BLACK)

    # print(castling_rights)
    return castling_rights #np.array(castling_rights, dtype='float32')


# convert dictionary of moves (numbers) to a list where the pieces can move TO (nn2)
def dict_moves_to_list_to_squares(dict_legal_moves_numbers: dict, square_move_from_number):

    if square_move_from_number not in dict_legal_moves_numbers:
        print("Can't move from ", square_move_from_number, 'square')
        return False

    squares_to = dict_legal_moves_numbers[square_move_from_number]

    return squares_to

# 64 + 4 + 3 = 71
def create_y_values_nn2(board, square_move_from_number):

    if square_move_from_number < 64:

        legal_moves = board.legal_moves
        dictionary_moves = moves_dict(legal_moves)
        dictionary_moves_numbers = dict_squares_to_dict_numbers(dictionary_moves)
        list_to = dict_moves_to_list_to_squares(dictionary_moves_numbers, square_move_from_number)
        probability = get_probability_single_legal_piece(list_to)

        array = create_array_squares_from(list_to, probability) # 64 x 1

    else:
        array = np.zeros(64)

    castle_output = array_castling_rights_n2(board, square_move_from_number) # 4
    # print(castle_output)

    win_draw = checkmate_draw(board) # 3

    array = np.hstack((array, castle_output))
    # print(array)
    array = np.hstack((array, win_draw))

    return array