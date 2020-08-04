using System;
using System.Collections.Generic;
using System.Text;

namespace NeuNet
{
    class NeuralNetwork
    {
        int s_input;
        int s_output;
        int s_hidden_layers;
        double[] input;
        double[,] hidden_layers;
        double[,] delta_hidden_layers;

        double[] output;

        double[,] weight_input_layers;
        double[] weight_input_layers_bias;

        double[,,] weight_hidden_layers;
        double[,] weight_hidden_layers_bias;
        int[] s_each_hiddenlayer;

        double[,] weight_output_layers;
        double[] weight_output_layers_bias;

        double[] delta_output;

        Random r = new Random();
        public NeuralNetwork(int s_input, int[] s_each_hiddenlayer, int s_output)
        {
            this.s_input = s_input;
            this.s_output = s_output;
            this.s_hidden_layers = s_each_hiddenlayer.Length;

            this.input = new double[s_input];

            this.delta_hidden_layers = new double[s_hidden_layers, 30];
            this.output = new double[s_output];
            this.hidden_layers = new double[s_hidden_layers, 30];

            this.weight_input_layers = new double[s_each_hiddenlayer[0], s_input];
            this.weight_input_layers_bias = new double[s_each_hiddenlayer[0]];

            this.weight_hidden_layers = new double[s_hidden_layers, 30, 30];
            this.weight_hidden_layers_bias = new double[s_hidden_layers, 30];
            this.s_each_hiddenlayer = s_each_hiddenlayer;

            this.weight_output_layers = new double[s_output, s_each_hiddenlayer[s_hidden_layers - 1]];
            this.weight_output_layers_bias = new double[s_output];

            this.delta_output = new double[s_output];

            set_weights();

        }
        public double[] activate(double[] input)
        {
            this.input = input;
            for (int i = 0; i < s_each_hiddenlayer[0]; i++)
            {
                hidden_layers[0, i] = 0;
                for (int j = 0; j < s_input; j++)
                {
                    hidden_layers[0, i] += weight_input_layers[i, j] * input[j];
                }
                hidden_layers[0, i] += weight_input_layers_bias[i];
                hidden_layers[0, i] = function(hidden_layers[0, i]);
            }

            for (int i = 1; i < s_hidden_layers; i++)
            {
                for (int j = 0; j < s_each_hiddenlayer[i]; j++)
                {
                    hidden_layers[i, j] = 0;
                    for (int k = 0; k < s_each_hiddenlayer[i - 1]; k++)
                    {
                        hidden_layers[i, j] += weight_hidden_layers[i - 1, j, k] * hidden_layers[i - 1, k];

                    }
                    hidden_layers[i, j] += weight_hidden_layers_bias[i, j];
                    hidden_layers[i, j] = function(hidden_layers[i, j]);
                }
            }

            for (int i = 0; i < s_output; i++)
            {
                output[i] = 0;
                for (int j = 0; j < s_each_hiddenlayer[s_hidden_layers - 1]; j++)
                {
                    output[i] += weight_output_layers[i, j] * hidden_layers[s_hidden_layers - 1, j];
                }
                output[i] += weight_output_layers_bias[i];
                output[i] = function(output[i]);

            }
            return output;
        }
        void set_weights()
        {
            for (int i = 0; i < s_each_hiddenlayer[0]; i++)
            {
                weight_input_layers_bias[i] = getRandom();
                for (int j = 0; j < s_input; j++)
                {
                    weight_input_layers[i, j] = getRandom();
                }
            }
            for (int i = 0; i < s_hidden_layers - 1; i++)
            {
                for (int j = 0; j < s_each_hiddenlayer[i + 1]; j++)
                {
                    weight_hidden_layers_bias[i + 1, j] = getRandom();
                    for (int k = 0; k < s_each_hiddenlayer[i]; k++)
                    {
                        weight_hidden_layers[i, j, k] = getRandom();
                    }
                }
            }
            for (int i = 0; i < s_output; i++)
            {
                weight_output_layers_bias[i] = getRandom();
                for (int j = 0; j < s_each_hiddenlayer[s_hidden_layers - 1]; j++)
                {
                    weight_output_layers[i, j] = getRandom();
                }
            }
        }
        public void train(double learning_rate, double[] target)
        {
            for (int i = 0; i < s_output; i++)
            {
                delta_output[i] = (target[i] - output[i]) * dfunction(output[i]);
            }

            for (int i = 0; i < s_each_hiddenlayer[s_hidden_layers - 1]; i++)
            {
                delta_hidden_layers[s_hidden_layers - 1, i]=0;
                for (int j = 0; j < s_output; j++)
                {
                    delta_hidden_layers[s_hidden_layers - 1, i] += delta_output[j] * weight_output_layers[j, i] * dfunction(hidden_layers[s_hidden_layers - 1, i]);
                }
            }

            for (int i = s_hidden_layers - 2; i >= 0; i--)
            {
                for (int j = 0; j < s_each_hiddenlayer[i]; j++)
                {
                    delta_hidden_layers[i, j] = 0;
                    for (int k = 0; k < s_each_hiddenlayer[i + 1]; k++)
                    {
                        delta_hidden_layers[i, j] += delta_hidden_layers[i + 1, k] * weight_hidden_layers[i, k, j] * dfunction(hidden_layers[i, j]);

                    }
                }
            }

            for (int i = 0; i < s_hidden_layers - 1; i++)
            {
                for (int j = 0; j < s_each_hiddenlayer[i + 1]; j++)
                {
                    for (int k = 0; k < s_each_hiddenlayer[i]; k++)
                    {
                        weight_hidden_layers[i, j, k] += hidden_layers[i, k] * delta_hidden_layers[i + 1, j] * learning_rate;
                       
                    }
                    weight_hidden_layers_bias[i, j] += delta_hidden_layers[i + 1, j] * learning_rate;
                }
            }
            for (int i = 0; i < s_each_hiddenlayer[0]; i++)
            {
                for (int j = 0; j < s_input; j++)
                {
                    weight_input_layers[i, j] += input[j] * delta_hidden_layers[0, i] * learning_rate;
                }
                weight_input_layers_bias[i] += delta_hidden_layers[0, i] * learning_rate;
            }
            for (int i = 0; i < s_output; i++)
            {
                for (int j = 0; j < s_each_hiddenlayer[s_hidden_layers - 1]; j++)
                {
                    weight_output_layers[i, j] += hidden_layers[s_hidden_layers - 1, j] * delta_output[i] * learning_rate;
                }
                weight_output_layers_bias[i] += delta_output[i] * learning_rate;
            }
        }
        double getRandom()
        {
            return (r.NextDouble() - 0.5) * 2;
        }
        double function(double value)
        {
            return 1 / (1 + Math.Exp(-value));
        }
        double dfunction(double value)
        {
            return value * (1 - value);
        }
    }
}
