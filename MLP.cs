using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP_2025
{
    internal class MLP
    {
        // Liczba warstw w sieci (np. wejściowa, ukryte, wyjściowa)
        public int LayerCount { get; private set; }

        // Tablica warstw, każda warstwa to tablica perceptronów
        public Perceptron[][] Layers { get; private set; }

        // Liczba perceptronów w każdej warstwie
        public int[] NeuronsPerLayer { get; private set; }

        // Współczynnik uczenia
        public double LearningRate { get; set; }

        // Konstruktor do inicjalizacji sieci
        public MLP(int[] neuronsPerLayer, double learningRate = 0.1)
        {
            LayerCount = neuronsPerLayer.Length;
            NeuronsPerLayer = neuronsPerLayer;
            LearningRate = learningRate;
            Layers = new Perceptron[LayerCount][];

            for (int l = 0; l < LayerCount; l++)
            {
                int inputCount = l == 0 ? neuronsPerLayer[0] : neuronsPerLayer[l - 1];
                Layers[l] = new Perceptron[neuronsPerLayer[l]];
                for (int n = 0; n < neuronsPerLayer[l]; n++)
                {
                    Layers[l][n] = new Perceptron(inputCount);
                }
            }
        }
        public double[] Compute(double[] input)
        {
            double[] currentInput = input;

            for (int l = 0; l < LayerCount; l++)
            {
                double[] currentOutput = new double[Layers[l].Length];
                for (int n = 0; n < Layers[l].Length; n++)
                {
                    // Ustaw wejścia perceptrona
                    Layers[l][n].Inputs = currentInput;
                    // Oblicz wyjście perceptrona
                    currentOutput[n] = Layers[l][n].CalculateOutput();
                }
                currentInput = currentOutput; // Wyjście tej warstwy jest wejściem do następnej
            }

            return currentInput; // Wyjście ostatniej warstwy
        }
        public void Train(double[] input, double[] target)
        {
            // Forward pass: zapisz wyjścia wszystkich warstw
            double[][] layerOutputs = new double[LayerCount][];
            double[] currentInput = input;

            for (int l = 0; l < LayerCount; l++)
            {
                layerOutputs[l] = new double[Layers[l].Length];
                for (int n = 0; n < Layers[l].Length; n++)
                {
                    Layers[l][n].Inputs = currentInput;
                    layerOutputs[l][n] = Layers[l][n].CalculateOutput();
                }
                currentInput = layerOutputs[l];
            }

            // Backward pass: oblicz delty
            double[][] deltas = new double[LayerCount][];
            for (int l = LayerCount - 1; l >= 0; l--)
            {
                deltas[l] = new double[Layers[l].Length];
                for (int n = 0; n < Layers[l].Length; n++)
                {
                    double output = layerOutputs[l][n];
                    double error;
                    if (l == LayerCount - 1)
                    {
                        // Warstwa wyjściowa
                        error = target[n] - output;
                    }
                    else
                    {
                        // Warstwa ukryta
                        error = 0.0;
                        for (int k = 0; k < Layers[l + 1].Length; k++)
                        {
                            error += deltas[l + 1][k] * Layers[l + 1][k].Weights[n];
                        }
                    }
                    deltas[l][n] = error * output * (1 - output); // pochodna sigmoidy
                }
            }

            // Aktualizacja wag i biasów
            for (int l = 0; l < LayerCount; l++)
            {
                double[] prevOutput = l == 0 ? input : layerOutputs[l - 1];
                for (int n = 0; n < Layers[l].Length; n++)
                {
                    for (int w = 0; w < Layers[l][n].Weights.Length; w++)
                    {
                        Layers[l][n].Weights[w] += LearningRate * deltas[l][n] * prevOutput[w];
                    }
                    // Aktualizacja biasu
                    Layers[l][n].Bias += LearningRate * deltas[l][n];
                }
            }
        }
    }
}
