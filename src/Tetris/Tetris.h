/*
 * multilayer_perceptron
 * Author: bonsall2004
 * Description: 
 */
#pragma once
#include <cstdint>
#include <vector>

struct Tetromino
{
  using Board = std::vector<std::vector<double>>;
  uint8_t y = 0;
  uint8_t x = 4;
  uint8_t previous_x = 4;
  uint8_t previous_y = 0;
  uint8_t previous_rotation = 0;
  uint8_t rotation = 0;

  void draw_tetromino(Board& vec);

  std::vector<std::vector<std::vector<int>>> pieces {
    {
      { 1, 0, 0, 0, },
      { 1, 0, 0, 0, },
      { 1, 0, 0, 0, },
      { 1, 0, 0, 0, }
    },
    {
      { 1, 1, 1, 1, },
      { 0, 0, 0, 0, },
      { 0, 0, 0, 0, },
      { 0, 0, 0, 0, }
    }
  };
};

class Tetris
{
    using Board = std::vector<std::vector<double>>;
  public:
    Tetris(uint8_t width, uint8_t height);

    Board get_board();
    void draw_loop();

  private:
    uint8_t width;
    uint8_t height;
    Tetromino active_piece;
    Board current_board = {};

    void draw_random(uint8_t origin);
    void move_right();
    void move_left();
    void move_down();
    void rotate();
    bool draw_board();

    char last_pressed_key = '\0';
};
