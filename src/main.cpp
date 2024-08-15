#include <iostream>
#include "Net/MultiLayerPerceptron.h"


int main()
{
  using Data = std::vector<std::vector<double>>;
  MultiLayerPerceptron mlp(3, 6, 2);

  Data training_inputs = {{ 0, 0, 0 }, { 0 , 1, 1 }, { 1,  1, 0 }, { 1, 1, 1 }, {1, 0, 0}, {1, 0, 1}};
  Data training_outputs = {{1, 0}, {0, 1}, {0, 1}, {1, 1}, {0, 1}, {0, 1}};

  mlp.train(training_inputs, training_outputs, 50000, 0.2);

  Data testing_inputs = {{ 0, 0, 1 }, { 1, 0 , 1 }, { 0, 1, 1 }, { 1, 1, 1 }, { 0, 0, 0 }};

  for(const auto& input : testing_inputs)
  {
    std::vector<double> output = mlp.forward(input);
    std::cout << "Input: (" << input[0] << ", " <<  input[1] <<", "<< input[2] <<") \nOutput: " << output[0] << ", " << output[1]  << std::endl;
  }

  return 0;
}
