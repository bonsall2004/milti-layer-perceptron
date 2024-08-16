/*
 * multilayer_perceptron
 * Author: bonsall2004
 * Description:
 */
#include "MultiLayerPerceptron.h"
#include <cmath>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>

MultiLayerPerceptron::MultiLayerPerceptron(size_t input_size, const std::vector<size_t>& hidden_sizes, int output_size) : rng(std::random_device{}()), dist(-1.0, 1.0)
{
  std::srand(static_cast<unsigned int>(std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count()));

  weights_input_hidden.emplace_back(input_size, std::vector<double>(hidden_sizes[0]));
  for (size_t i = 0; i < input_size; i++) {
    for (size_t j = 0; j < hidden_sizes[0]; j++) {
      weights_input_hidden[0][i][j] = dist(rng);
    }
  }

  for (size_t l = 1; l < hidden_sizes.size(); l++) {
    weights_input_hidden.emplace_back(hidden_sizes[l-1], std::vector<double>(hidden_sizes[l]));
    for (size_t i = 0; i < hidden_sizes[l-1]; i++) {
      for (size_t j = 0; j < hidden_sizes[l]; j++) {
        weights_input_hidden[l][i][j] = dist(rng);
      }
    }
  }

  weights_input_hidden.emplace_back(hidden_sizes.back(), std::vector<double>(output_size));
  for (size_t i = 0; i < hidden_sizes.back(); i++) {
    for (size_t j = 0; j < output_size; j++) {
      weights_input_hidden.back()[i][j] = dist(rng);
    }
  }

  hidden_layers.resize(hidden_sizes.size());
  for (size_t l = 0; l < hidden_sizes.size(); l++) {
    hidden_layers[l].resize(hidden_sizes[l]);
  }

  output_layers.resize(output_size);
}

MultiLayerPerceptron::MultiLayerPerceptron(std::filesystem::path model_file)
{
  if(!std::filesystem::exists(model_file))
  {
    std::cerr << "That model file was invalid!!!!!!!!!!\n";
    throw std::exception();
  }
  std::ifstream saved_model_file(model_file);
  if(!saved_model_file)
  {
    std::cerr << "Cannot open file for reading\n";
    throw std::exception();
  }

  size_t dimention_1;
  saved_model_file.read(reinterpret_cast<char*>(&dimention_1), sizeof(size_t));

  std::vector<std::vector<std::vector<double>>> vec(dimention_1);

  for(size_t i = 0; i < dimention_1; i ++)
  {
    size_t dimention_2;
    saved_model_file.read(reinterpret_cast<char*>(&dimention_2), sizeof(size_t));
    vec[i].resize(dimention_2);

    for(size_t j = 0; j < dimention_2; j++)
    {
      size_t dimention_3;
	    saved_model_file.read(reinterpret_cast<char*>(&dimention_3), sizeof(size_t));
      vec[i][j].resize(dimention_3);


      saved_model_file.read(reinterpret_cast<char*>(vec[i][j].data()), dimention_3 * sizeof(double));
    }
  }

  saved_model_file.close();
  weights_input_hidden = std::move(vec);
}


std::vector<double> MultiLayerPerceptron::forward(const std::vector<double>& input)
{
  for (size_t i = 0; i < hidden_layers[0].size(); i++) {
    hidden_layers[0][i] = 0.0;
    for (size_t j = 0; j < input.size(); j++) {
      hidden_layers[0][i] += input[j] * weights_input_hidden[0][j][i];
    }
    hidden_layers[0][i] = sigmoid(hidden_layers[0][i]);
  }

  for (size_t l = 1; l < hidden_layers.size(); l++) {
    for (size_t i = 0; i < hidden_layers[l].size(); i++) {
      hidden_layers[l][i] = 0.0;
      for (size_t j = 0; j < hidden_layers[l-1].size(); j++) {
        hidden_layers[l][i] += hidden_layers[l-1][j] * weights_input_hidden[l][j][i];
      }
      hidden_layers[l][i] = sigmoid(hidden_layers[l][i]);
    }
  }

  for (size_t i = 0; i < output_layers.size(); i++) {
    output_layers[i] = 0.0;
    for (size_t j = 0; j < hidden_layers.back().size(); j++) {
      output_layers[i] += hidden_layers.back()[j] * weights_input_hidden.back()[j][i];
    }
    output_layers[i] = sigmoid(output_layers[i]);
  }

  return output_layers;
}

void MultiLayerPerceptron::backward(const std::vector<double>& input, const std::vector<double>& target, double learning_rate)
{
  std::vector<double> output_errors(output_layers.size());
  std::vector<std::vector<double>> hidden_errors(hidden_layers.size());
  for (size_t l = 0; l < hidden_layers.size(); l++) {
    hidden_errors[l].resize(hidden_layers[l].size());
  }
  for (size_t i = 0; i < output_layers.size(); i++) {
    output_errors[i] = (target[i] - output_layers[i]) * sigmoid_derivative(output_layers[i]);
  }
  for (int l = (int)hidden_layers.size() - 1; l >= 0; l--) {
    for (size_t i = 0; i < hidden_layers[l].size(); i++) {
      hidden_errors[l][i] = 0.0;
      if (l == hidden_layers.size() - 1) {
        for (size_t j = 0; j < output_layers.size(); j++) {
          hidden_errors[l][i] += output_errors[j] * weights_input_hidden[l + 1][i][j];
        }
      } else {
        for (size_t j = 0; j < hidden_layers[l + 1].size(); j++) {
          hidden_errors[l][i] += hidden_errors[l + 1][j] * weights_input_hidden[l + 1][i][j];
        }
      }
      hidden_errors[l][i] *= sigmoid_derivative(hidden_layers[l][i]);
    }
  }
  for (size_t i = 0; i < hidden_layers.back().size(); i++) {
    for (size_t j = 0; j < output_layers.size(); j++) {
      weights_input_hidden.back()[i][j] += learning_rate * output_errors[j] * hidden_layers.back()[i];
    }
  }
  for (int l = (int)hidden_layers.size() - 1; l > 0; l--) {
    for (size_t i = 0; i < hidden_layers[l].size(); i++) {
      for (size_t j = 0; j < hidden_layers[l-1].size(); j++) {
        weights_input_hidden[l][j][i] += learning_rate * hidden_errors[l][i] * hidden_layers[l-1][j];
      }
    }
  }
  for (size_t i = 0; i < input.size(); i++) {
    for (size_t j = 0; j < hidden_layers[0].size(); j++) {
      weights_input_hidden[0][i][j] += learning_rate * hidden_errors[0][j] * input[i];
    }
  }
}

void MultiLayerPerceptron::train(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, size_t epochs, double learning_rate)
{
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
  for (size_t epoch = 0; epoch < epochs; epoch++) {
    for (size_t i = 0; i < training_inputs.size(); i++) {
      forward(training_inputs[i]);
      backward(training_inputs[i], training_outputs[i], learning_rate);
    }
    if ((epoch % 100) == 0) {
      auto temp_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
      printf("Completed Epochs: %zu (%lld ms)\n", epoch, temp_time - time);
      time = temp_time;
    }
  }
  auto temp_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
  printf("Completed Epochs: %zu (%lld ms)\n", epochs, temp_time - time);
}

double MultiLayerPerceptron::sigmoid(double x)
{
  return 1.0 / (1.0 + exp(-x));
}

double MultiLayerPerceptron::sigmoid_derivative(double x)
{
  return x * (1.0 - x);
}
bool MultiLayerPerceptron::save_model()
{
  std::ofstream saved_model_file("model.tet-mod", std::ios::binary);

  if(!saved_model_file)
  {
    std::cerr << "Cannot open file for writing: model.tet-mod" << std::endl;
    return false;
  }

  size_t dimention_1 = weights_input_hidden.size();
  saved_model_file.write(reinterpret_cast<const char*>(&dimention_1), sizeof(size_t));

  for(const auto& vec2d : weights_input_hidden)
  {
    size_t dimention_2 = vec2d.size();
    saved_model_file.write(reinterpret_cast<const char*>(&dimention_2), sizeof(size_t));

    for(const auto& vec1d : vec2d)
    {
      size_t dimention_3 = vec1d.size();
      saved_model_file.write(reinterpret_cast<const char*>(&dimention_3), sizeof(size_t));

      saved_model_file.write(reinterpret_cast<const char*>(vec1d.data()), dimention_3 * sizeof(double));
    }
  }

  saved_model_file.close();
  return true;
}
