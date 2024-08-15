/*
 * multilayer_perceptron
 * Author: bonsall2004
 * Description: 
 */
#pragma once
#include <vector>
#include <random>

class MultiLayerPerceptron
{
  public:
    explicit MultiLayerPerceptron(size_t input_size, size_t hidden_size, int output_size);
    std::vector<double> forward(const std::vector<double>& input);
    void backward(const std::vector<double>& input, const std::vector<double>& target, double learning_rate);
    void train(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, size_t epochs, double learning_rate);

  private:
    std::vector<std::vector<double>> weights_input_hidden; // TODO Save to file
    std::vector<std::vector<double>> weights_hidden_output; // // TODO Save to file
    std::vector<double> hidden_layers;
    std::vector<double> output_layers;

    std::mt19937 rng;
    std::uniform_real_distribution<double> dist;

    double sigmoid(double x);
    double sigmoid_derivative(double x);
    double random_weight();
};
