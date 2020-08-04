using System;

namespace NeuNet
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            int[] size = {6,5};
            Random r = new Random();
            int sizee = 10000;
            double[,] value = new double[sizee, 3];
            double[,] target = new double[sizee, 2];
            for (int i = 0; i < sizee; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    value[i, j] = r.NextDouble() * 5;
                }
            }
            for (int i = 0; i < sizee; i++)
            {
                
                if (value[i, 0] * value[i, 1] * value[i, 2] < 15)
                {
                    target[i, 0] = 1;
                }
                else
                {
                    target[i, 1] = 1;
                }

            }

            NeuralNetwork n = new NeuralNetwork(3, size, 2);
            double[] input = new double[3];
            double[] targets = new double[2];
            double[] deneme = { 1, 1,5 };
            for (int i = 0; i < sizee; i++)
            {
                input[0] = value[i, 0];
                input[1] = value[i, 1];
                input[2] = value[i, 2];
                targets[0] = target[i, 0];
                targets[1] = target[i, 1];
                n.activate(input);
                n.train(0.1, targets);
                Console.WriteLine(n.activate(deneme)[0]);
                Console.WriteLine(n.activate(deneme)[1]);

            }
           
          
           

        }
    }
}
