/*
 * multilayer_perceptron
 * Author: bonsall2004
 * Description: 
 */
#include <iostream>
#include "Tetris.h"
#include <chrono>
#include <thread>

void Tetromino::draw_tetromino(Board& vec)
{
  rotation = rotation % pieces.size();

  for(int i = 0; i < vec.size(); ++i) // x
  {
    for(int j = 0; j < vec[i].size(); ++j) // y
    {
      for(int k = 0; k < 4; ++k)
      {
        for(int l = 0; l < 4; ++l)
        {
          if(pieces[previous_rotation][k][l] != 0) vec[previous_y+k][previous_x+l] = 0;
          if(pieces[rotation][k][l] != 0) vec[y+k][x+l] = 1;
        }
      }
    }
  }
  previous_y = y;
  previous_x = x;
}


Tetris::Tetris(uint8_t width, uint8_t height)
{

  current_board.resize(height);
  for(int i = 0; i < height; ++i)
  {
    current_board[i].resize(width);
    for(int j = 0; j < width; ++j)
    {
      current_board[i][j] = 0;
    }
  }
}

Tetris::Board Tetris::get_board()
{
  return Tetris::current_board;
}

bool Tetris::draw_board()
{
  for(int i = 0; i < current_board.size(); ++i)
  {
    for(int j = 0; j < current_board[i].size(); ++j)
    {
      std::cout << "\033[" << i+1 << ';' << (j)*2 << "H" << (current_board[i][j] == 0 ? " ." : "[]") << std::string(256, ' ');
    }
    std::cout << '\n';
  }
  std::cout << "\033[11A";
  std::cout.flush();
  return false;
}

void Tetris::draw_random(uint8_t origin)
{
  current_board[0][origin]   = 1;
  current_board[0][origin+1] = 1;
  current_board[1][origin]   = 1;
  current_board[1][origin+1] = 1;
}

void Tetris::draw_loop()
{
  active_piece.draw_tetromino(current_board);
  move_right();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  move_down();
  draw_board();
}
void Tetris::move_right()
{
  if(active_piece.x + 3 == width) return;
  active_piece.x += 1;
}
void Tetris::move_left()
{
//  if(active_piece.x <= 0)
//  {
//    active_piece.x = width;
//    return;
//  }
  active_piece.x -= 1;
}
void Tetris::move_down()
{
  active_piece.y += 1;
}
void Tetris::rotate()
{
  active_piece.rotation += 1;
}
