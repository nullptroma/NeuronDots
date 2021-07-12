using System;
using System.Collections.Generic;
using System.Text;

namespace NeuronDotsOnigiri
{
    class NeuralNetwork
    {
        private double learningRate;
        private Layer[] layers;
        private Func<double, double> activation;
        private Func<double, double> derivative;

        public NeuralNetwork(double learningRate, Func<double, double> activation, Func<double, double> derivative, params int[] sizes)
        {
            this.learningRate = learningRate;
            this.activation = activation;
            this.derivative = derivative;
            layers = new Layer[sizes.Length];
            Random rnd = new Random();
            for (int i = 0; i < sizes.Length; i++)
            {
                int nextSize = 0;
                if (i < sizes.Length - 1) nextSize = sizes[i + 1];
                layers[i] = new Layer(sizes[i], nextSize);
                for (int j = 0; j < sizes[i]; j++)
                {
                    layers[i].biases[j] = 1;
                    for (int k = 0; k < nextSize; k++)
                    {
                        layers[i].weights[j][k] = rnd.NextDouble() * 2.0 - 1.0;
                    }
                }
            }
        }

        public double[] feedForward(double[] inputs)
        {
            Array.Copy(inputs, layers[0].neurons, inputs.Length);
            for (int i = 1; i < layers.Length; i++)
            {
                Layer l = layers[i - 1];
                Layer l1 = layers[i];
                for (int j = 0; j < l1.size; j++)
                {
                    l1.neurons[j] = 0;
                    for (int k = 0; k < l.size; k++)
                    {
                        l1.neurons[j] += l.neurons[k] * l.weights[k][j];
                    }
                    l1.neurons[j] += l1.biases[j];
                    l1.neurons[j] = activation(l1.neurons[j]);
                }
            }
            return layers[layers.Length - 1].neurons;
        }

        public void backpropagation(double[] targets)
        {
            double[] errors = new double[layers[layers.Length - 1].size];
            for (int i = 0; i < layers[layers.Length - 1].size; i++)
            {
                errors[i] = targets[i] - layers[layers.Length - 1].neurons[i];
            }
            for (int k = layers.Length - 2; k >= 0; k--)
            {
                Layer l = layers[k];
                Layer l1 = layers[k + 1];
                double[] errorsNext = new double[l.size];
                double[] gradients = new double[l1.size];
                for (int i = 0; i < l1.size; i++)
                {
                    gradients[i] = errors[i] * derivative(layers[k + 1].neurons[i]);
                    gradients[i] *= learningRate;
                }
                double[][] deltas = new double[l1.size][];
                for (int i = 0; i < l1.size; i++)
                    deltas[i] = new double[l.size]; 
                for (int i = 0; i < l1.size; i++)
                {
                    for (int j = 0; j < l.size; j++)
                    {
                        deltas[i][j] = gradients[i] * l.neurons[j];
                    }
                }
                for (int i = 0; i < l.size; i++)
                {
                    errorsNext[i] = 0;
                    for (int j = 0; j < l1.size; j++)
                    {
                        errorsNext[i] += l.weights[i][j] * errors[j];
                    }
                }
                errors = new double[l.size];
                Array.Copy(errorsNext, errors, l.size);
                double[][] weightsNew = new double[l.weights.Length][];
                for (int i = 0; i < l.weights.Length; i++)
                    weightsNew[i] = new double[l.weights[0].Length];
                for (int i = 0; i < l1.size; i++)
                {
                    for (int j = 0; j < l.size; j++)
                    {
                        weightsNew[j][i] = l.weights[j][i] + deltas[i][j];
                    }
                }
                l.weights = weightsNew;
                for (int i = 0; i < l1.size; i++)
                {
                    l1.biases[i] += gradients[i];
                }
            }
        }

        public class Layer
        {

            public int size;
            public double[] neurons;
            public double[] biases;
            public double[][] weights;

            public Layer(int size, int nextSize)
            {
                this.size = size;
                neurons = new double[size];
                biases = new double[size];
                weights = new double[size][];
                for (int i = 0; i < size; i++)
                    weights[i] = new double[nextSize];
            }

        }
    }
}
