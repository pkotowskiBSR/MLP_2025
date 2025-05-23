using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP_2025
{
    internal class Perceptron
    {
        public int InputCount { get; private set; }
        public double[] Weights { get; set; }
        public double[] Inputs { get; set; }
        public double Bias { get; set; }
        double Output { get; set; }

        public Perceptron(int inputCount)
        {
            InputCount = inputCount;
            Weights = new double[inputCount];
            Inputs = new double[inputCount];
            Random rand = new Random();
            for (int i = 0; i < inputCount; i++)
            {
                Weights[i] = 2 * rand.NextDouble() - 1;
            }
            Bias = 2 * rand.NextDouble() - 1; // losowa inicjalizacja biasu
        }

        public double CalculateOutput()
        {
            double sum = 0.0;
            for (int i = 0; i < InputCount; i++)
            {
                sum += Weights[i] * Inputs[i];
            }
            sum += Bias; // dodanie biasu
            Output = Sigmoid(sum);
            return Output;
        }

        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
    }
}
