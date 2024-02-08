import numpy as np
import chess

# put 1 where piece on 8x8 board is
def vectorize_pieces(list_locations: list):

    board_array = np.zeros(64)

    for i in list_locations:

        board_array[i] = 1

    return board_array

# put 1 on en passant square
def en_passant_squares(board):

    board_array = np.zeros(64)
    # print(board_array)
    # print(board)

    en_passant_square = board.ep_square
    # print(en_passant_square)

    if en_passant_square is not None:

        board_array[en_passant_square] = 1

    # print(board_array)
    
    return board_array

# create 12 board for 6 pieces of each color (6x2) with 1 where specific piece is
def board_to_array_pieces(board):

    pieces_list = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    colors = [chess.WHITE, chess.BLACK]

    arrays = np.array([])

    for c in colors:
        for p in pieces_list:

            pieces = list(board.pieces(piece_type=p, color=c))

            vectorized_board = vectorize_pieces(pieces).reshape(8, 8)
            # print(vectorized_board)

            arrays = np.append(arrays, vectorized_board)

    return arrays

# combine 12 pieces arrays with en passant array
def arrays_pieces_an_passat(board: chess.Board):

    # creating first 12 arrays will location of all the pieces
    arrays_pieces = board_to_array_pieces(board)
    array_en_passant = en_passant_squares(board)

    return np.append(arrays_pieces, array_en_passant)

# return array len 1 with 0 for white turn and 1 for black turn
def turn_array(board):
    
    if board.turn:
        return np.array([0])

    else:
        return np.array([1])

# creates an input array of length 833
# def create_whole_input(board: chess.Board):

#     arrays = arrays_pieces_an_passat(board)
#     turn = turn_array(board)
#     arrays = np.append(arrays, turn)

#     return arrays

def array_castling_rights(board: chess.Board):

    castling_rights = [0,0,0,0]
    castling_rights[0] = board.has_kingside_castling_rights(chess.WHITE)
    castling_rights[1] = board.has_queenside_castling_rights(chess.WHITE)
    castling_rights[2] = board.has_kingside_castling_rights(chess.BLACK)
    castling_rights[3] = board.has_queenside_castling_rights(chess.BLACK)

    return np.array(castling_rights, dtype='float32')

# 
def create_whole_input_nn1(board: chess.Board):

    arrays = arrays_pieces_an_passat(board) # 64 x 13
    castling = array_castling_rights(board) # 4
    turn = turn_array(board) # 1 
    
    arrays = np.append(arrays, castling)
    arrays = np.append(arrays, turn)

    return arrays



# input nn2

# put 1 on square where desired piece is
def desired_piece_board(square_number):

    board_array = np.zeros(64)

    board_array[square_number] = 1

    # print(board_array)
    
    return board_array



def create_whole_input_nn2(board: chess.Board, square_to_move_from):

    arrays = arrays_pieces_an_passat(board) # 64 x 13

    piece_to_play_board = desired_piece_board(square_to_move_from)# 64 x 1

    castling = array_castling_rights(board) # 4

    turn = turn_array(board) # 1

    arrays = np.append(arrays, piece_to_play_board)
    arrays = np.append(arrays, castling)
    arrays = np.append(arrays, turn)

    return arrays