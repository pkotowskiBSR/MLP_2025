namespace MLP_2025
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Start MLP 2025");

            // MLP: 10 wejść, 20 neuronów ukrytych, 10 wyjść
            var mlp = new MLP(new int[] { 10, 20, 10 });

            Random rand = new Random();

            // --- CZĘŚĆ 1: TRENING ---
            for (int i = 0; i < 100000; i++)
            {
                int n = rand.Next(0, 10);
                int[] vector = GetRandomOnesArray(n);
                double[] input = Array.ConvertAll(vector, x => (double)x);
                double[] target = new double[10];
                target[n] = 1.0;
                mlp.Train(input, target);
            }

            // --- CZĘŚĆ 2: TESTOWANIE I WYPISYWANIE WYNIKÓW ---
            for (int i = 0; i < 20; i++)
            {
                int n = rand.Next(0, 10);
                int[] vector = GetRandomOnesArray(n);
                double[] input = Array.ConvertAll(vector, x => (double)x);
                double[] output = mlp.Compute(input);
                int maxIndex = Array.IndexOf(output, output.Max());
                Console.WriteLine(
                    $"Test: n = {n}, vector = [{string.Join(", ", vector)}], max index = {maxIndex}"
                );
            }
        }
        public static int[] GetRandomOnesArray(int n)
        {
            if (n < 0 || n > 10)
                throw new ArgumentOutOfRangeException(nameof(n), "n must be between 0 and 10.");

            int[] result = new int[10];
            List<int> indices = Enumerable.Range(0, 10).ToList();
            Random rand = new Random();

            for (int i = 0; i < n; i++)
            {
                int idx = rand.Next(indices.Count);
                result[indices[idx]] = 1;
                indices.RemoveAt(idx);
            }

            return result;
        }
    }
}
