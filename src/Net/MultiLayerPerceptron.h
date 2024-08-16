/*
 * multilayer_perceptron
 * Author: bonsall2004
 * Description:
 */
#pragma once
#include <vector>
#include <random>
#include <filesystem>

class MultiLayerPerceptron
{
  public:
    explicit MultiLayerPerceptron(size_t input_size, const std::vector<size_t>& hidden_sizes, int output_size);
    explicit MultiLayerPerceptron(std::filesystem::path model_file);
    std::vector<double> forward(const std::vector<double>& input);
    void backward(const std::vector<double>& input, const std::vector<double>& target, double learning_rate);
    void train(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, size_t epochs, double learning_rate);
    bool save_model();

  private:
    std::vector<std::vector<std::vector<double>>> weights_input_hidden; // TODO: save to a binary file
    std::vector<std::vector<double>> hidden_layers;
    std::vector<double> output_layers;


    std::mt19937 rng;
    std::uniform_real_distribution<double> dist;

    static double sigmoid(double x);
    static double sigmoid_derivative(double x);
};
