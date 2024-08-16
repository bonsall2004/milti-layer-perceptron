#include <iostream>
#include "Net/MultiLayerPerceptron.h"
#include "Tetris/Tetris.h"

MultiLayerPerceptron dostuff()
{
  using Data = std::vector<std::vector<double>>;
  MultiLayerPerceptron mlp(3, { 6, 4 }, 2);

  Data training_inputs = {{ 0, 0, 0 }, { 0 , 1, 1 }, { 1,  1, 0 }, { 1, 1, 1 }, {1, 0, 0}, {1, 0, 1}};
  Data training_outputs = {{1, 0}, {0, 1}, {0, 1}, {1, 1}, {0, 1}, {0, 1}};

  mlp.train(training_inputs, training_outputs, 50000, 0.3);

  Data testing_inputs = {{ 0, 0, 1 }, { 1, 0 , 1 }, { 0, 1, 1 }, { 1, 1, 1 }, { 0, 0, 0 }};

  for(const auto& input : testing_inputs)
  {
    std::vector<double> output = mlp.forward(input);
    std::cout << "Input: (" << input[0] << ", " <<  input[1] <<", "<< input[2] <<") \nOutput: " << output[0] << ", " << output[1]  << std::endl;
  }

  mlp.save_model();
  return mlp;
}

int main(int argc, char** argv)
{
  Tetris tetris(10, 20);
  while(true)
  {
    tetris.draw_loop();
  }


  return 0;
}
