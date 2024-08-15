/*
 * multilayer_perceptron
 * Author: bonsall2004
 * Description: 
 */
#include "MultiLayerPerceptron.h"
#include <cmath>
#include <random>
#include <cstdlib>
#include <ctime>

MultiLayerPerceptron::MultiLayerPerceptron(size_t input_size, size_t hidden_size, int output_size)
{
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  weights_input_hidden.resize(input_size, std::vector<double>(hidden_size));
  for(int i = 0; i < input_size; i++)
  {
    for(int j = 0; j < hidden_size; j++)
    {
      weights_input_hidden[i][j] = dist(rng);
    }
  }

  weights_hidden_output.resize(hidden_size, std::vector<double>(output_size));
  for(int i = 0; i < hidden_size; i++)
  {
    for(int j = 0; j < output_size; j++)
    {
      weights_hidden_output[i][j] = dist(rng);
    }
  }
  hidden_layers.resize(hidden_size);
  output_layers.resize(output_size);
}

std::vector<double> MultiLayerPerceptron::forward(const std::vector<double>& input)
{
  for(int j = 0; j < hidden_layers.size(); j++)
  {
    hidden_layers[j] = 0.0;
    for(int i = 0; i < input.size(); i ++)
    {
      hidden_layers[j] += input[i] * weights_input_hidden[i][j];
    }
    hidden_layers[j] = sigmoid(hidden_layers[j]);
  }

  for(int k = 0; k < output_layers.size(); k++)
  {
    output_layers[k] = 0.0;
    for(int j = 0; j < hidden_layers.size(); j++)
    {
      output_layers[k] += hidden_layers[j] * weights_hidden_output[j][k];
    }
    output_layers[k] = sigmoid(output_layers[k]);
  }

  return output_layers;
}

void MultiLayerPerceptron::backward(const std::vector<double>& input, const std::vector<double>& target, double learning_rate)
{
  std::vector<double> output_errors(output_layers.size());
  std::vector<double> hidden_errors(hidden_layers.size());

  for(int k = 0; k < output_layers.size(); k ++)
  {
    output_errors[k] = (target[k] - output_layers[k]) * sigmoid_derivative(output_layers[k]);
  }

  for(int j = 0; j < hidden_layers.size(); j++)
  {
    hidden_errors[j] = 0.0;
    for(int k = 0; k < output_layers.size(); k++)
    {
      hidden_errors[j] += output_errors[k] * weights_hidden_output[j][k];
    }
    hidden_errors[j] *= sigmoid_derivative(hidden_layers[j]);
  }

  for(int j = 0; j < hidden_layers.size(); j++)
  {
    for(int k = 0; k < output_layers.size(); k++)
    {
      weights_hidden_output[j][k] += learning_rate * output_errors[k] * hidden_layers[j];
    }
  }

  for(int i = 0; i < input.size(); i ++)
  {
    for(int j = 0; j < hidden_layers.size(); j ++)
    {
      weights_input_hidden[i][j] += learning_rate * hidden_errors[j] * input[i];
    }
  }
}

void MultiLayerPerceptron::train(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, size_t epochs, double learning_rate)
{
  for(int epoch = 0; epoch < epochs; epoch++)
  {
    for(int j = 0; j < training_inputs.size(); j++)
    {
      forward(training_inputs[j]);
      backward(training_inputs[j], training_outputs[j], learning_rate);

    }
    if((epoch % 100) == 0) printf("Completed Epochs: %d\n", epoch);
  }
}

double MultiLayerPerceptron::sigmoid(double x)
{
  return 1.0 / (1.0 + (exp(-x)));
}

double MultiLayerPerceptron::sigmoid_derivative(double x)
{
  return x * (1.0 - x);
}

double MultiLayerPerceptron::random_weight()
{
  return static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
}
