import numpy as np
import chess

# put 1 where piece on 8x8 board is
def vectorize_pieces(list_locations: list):

    board_array = np.zeros(64)

    for i in list_locations:

        board_array[i] = 1

    return board_array

def vectorize_piece(square: int):

    board_array = np.zeros(64)

    board_array[square] = 1

    return board_array

# put 1 on en passant square
def en_passant_squares(board):

    en_passant_square = board.ep_square

    if en_passant_square is not None:

        en_passant_array = vectorize_piece(en_passant_square)

        return en_passant_array

    board_array = np.zeros(64)
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

def turn_array_linear(board):
    
    if board.turn:
        return np.array([0])

    else:
        return np.array([1])

# return array len 64 with 0 for white turn and 1 for black turn
def turn_array(board):
    
    if board.turn:
        return np.zeros(64)

    else:
        return np.ones(64)

def array_castling_rights_linear(board: chess.Board):

    castling_rights = [0,0,0,0]
    castling_rights[0] = board.has_kingside_castling_rights(chess.WHITE)
    castling_rights[1] = board.has_queenside_castling_rights(chess.WHITE)
    castling_rights[2] = board.has_kingside_castling_rights(chess.BLACK)
    castling_rights[3] = board.has_queenside_castling_rights(chess.BLACK)

    return np.array(castling_rights, dtype='float32')

def array_castling_rights(board: chess.Board):

    castling_rights = [0,0,0,0]

    castling_rights[0] = board.has_kingside_castling_rights(chess.WHITE)
    castling_rights[1] = board.has_queenside_castling_rights(chess.WHITE)
    castling_rights[2] = board.has_kingside_castling_rights(chess.BLACK)
    castling_rights[3] = board.has_queenside_castling_rights(chess.BLACK)

    castling_rights_array = np.zeros((4, 64))
    castling_rights_array[0] = castling_rights[0]
    castling_rights_array[1] = castling_rights[1]
    castling_rights_array[2] = castling_rights[2]
    castling_rights_array[3] = castling_rights[3]

    return castling_rights_array

def create_whole_input_linear(board: chess.Board):

    arrays = arrays_pieces_an_passat(board) # 64 x 13
    castling = array_castling_rights_linear(board) # 4
    turn = turn_array_linear(board) # 1 
    
    arrays = np.append(arrays, castling)
    arrays = np.append(arrays, turn)

    return arrays

def create_whole_input_linear_2(board: chess.Board, from_square):

    arrays = arrays_pieces_an_passat(board) # 64 x 13
    array_from = vectorize_piece(from_square)
    castling = array_castling_rights_linear(board) # 4
    turn = turn_array_linear(board) # 1 
    
    arrays = np.append(arrays, castling)
    arrays = np.append(arrays, turn)

    return arrays

def create_whole_input_cnn1(board: chess.Board):

    arrays = arrays_pieces_an_passat(board) # 64 x 13
    castling = array_castling_rights(board) # 4
    turn = turn_array(board) # 1 
    
    arrays = np.append(arrays, castling)
    arrays = np.append(arrays, turn)

    return arrays