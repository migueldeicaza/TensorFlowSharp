using Learn.Mnist;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorFlow;
using Xunit;
using Xunit.Abstractions;

using static System.FormattableString;

namespace TensorFlowSharp.Tests.CSharp
{
    public class OptimizerTests
    {
        private readonly string _testDataPath = "TestData";

        private readonly ITestOutputHelper outputHelper;

        public OptimizerTests(ITestOutputHelper output)
        {
            outputHelper = output;
        }


        [Fact]
        public void LinearRegresionTrainingTest()
        {
            Console.WriteLine("Linear regression");
            // Parameters
            var learning_rate = 0.01f;
            var training_epochs = 5;

            // Training data
            var train_x = new float[] {
                3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f
            };
            var train_y = new float[] {
                1.7f, 2.76f,2.09f,3.19f,1.694f,1.573f,3.366f,2.596f,2.53f,1.221f,
                 2.827f,3.465f,1.65f,2.904f,2.42f,2.94f,1.3f
            };
            var n_samples = train_x.Length;
            using (var graph = new TFGraph())
            { 
                var rng = new Random(0);
                // tf Graph Input

                var X = graph.Placeholder(TFDataType.Float, TFShape.Scalar);
                var Y = graph.Placeholder(TFDataType.Float, TFShape.Scalar);

                var W = graph.Variable(graph.Const(0.1f), operName: "weight");
                var b = graph.Variable(graph.Const(0.1f), operName: "bias");
                var pred = graph.Add(graph.Mul(X, W.Read, "x_w"), b.Read);

                var cost = graph.Div(graph.ReduceSum(graph.Pow(graph.Sub(pred, Y), graph.Const(2f))), graph.Mul(graph.Const(2f), graph.Const((float)n_samples), "2_n_samples"));

                var sgd = new SGD(graph, learning_rate);
                var updateOps = sgd.Minimize(cost);

                var readW = W.ReadAfter(updateOps);
                var readb = b.ReadAfter(updateOps);

                using (var sesssion = new TFSession(graph))
                {
                    sesssion.GetRunner().AddTarget(graph.GetGlobalVariablesInitializer()).Run();

                    var expectedLines = File.ReadAllLines(Path.Combine(_testDataPath, "SGD", "expected.txt"));
                    for (int i = 0; i < training_epochs; i++)
                    {
                        for (int j = 0; j < n_samples; j++)
                        {
                            var tensors = sesssion.GetRunner()
                                .AddInput(X, new TFTensor(train_x[j]))
                                .AddInput(Y, new TFTensor(train_y[j]))
                                .AddTarget(updateOps).Fetch(cost, readW, readb, pred).Run();
                            var output = Invariant($"loss: {tensors[0].GetValue():F4}, W: {tensors[1].GetValue():F4}, b: {tensors[2].GetValue():F4}");
                            Assert.Equal(expectedLines[i * n_samples + j], output);
                        }
                    }
                }
            }
        }

        [Fact]
        public void LinearRegresionTrainingSGDWithDecayTest()
        {
            Console.WriteLine("Linear regression");
            // Parameters
            var learning_rate = 0.01f;
            var training_epochs = 5;

            // Training data
            var train_x = new float[] {
                3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f
            };
            var train_y = new float[] {
                1.7f, 2.76f,2.09f,3.19f,1.694f,1.573f,3.366f,2.596f,2.53f,1.221f,
                 2.827f,3.465f,1.65f,2.904f,2.42f,2.94f,1.3f
            };
            var n_samples = train_x.Length;
            using (var graph = new TFGraph())
            {
                var rng = new Random(0);
                // tf Graph Input

                var X = graph.Placeholder(TFDataType.Float, TFShape.Scalar);
                var Y = graph.Placeholder(TFDataType.Float, TFShape.Scalar);

                var W = graph.Variable(graph.Const(0.1f), operName: "weight");
                var b = graph.Variable(graph.Const(0.1f), operName: "bias");
                var pred = graph.Add(graph.Mul(X, W.Read, "x_w"), b.Read);

                var cost = graph.Div(graph.ReduceSum(graph.Pow(graph.Sub(pred, Y), graph.Const(2f))), graph.Mul(graph.Const(2f), graph.Const((float)n_samples), "2_n_samples"));

                var sgd = new SGD(graph, learning_rate, decay:0.5f);
                var updateOps = sgd.Minimize(cost);

                var iter = sgd.Iterations.ReadAfter(updateOps);
                var readW = W.ReadAfter(updateOps);
                var readb = b.ReadAfter(updateOps);

                using (var sesssion = new TFSession(graph))
                {
                    sesssion.GetRunner().AddTarget(graph.GetGlobalVariablesInitializer()).Run();

                    var expectedLines = File.ReadAllLines(Path.Combine(_testDataPath, "SGDTimeDecay", "expected.txt"));
                    for (int i = 0; i < training_epochs; i++)
                    {
                        for (int j = 0; j < n_samples; j++)
                        {
                            var tensors = sesssion.GetRunner()
                               .AddInput(X, new TFTensor(train_x[j]))
                               .AddInput(Y, new TFTensor(train_y[j]))
                               .AddTarget(updateOps).Fetch(iter, cost, readW, readb, sgd.LearningRate).Run();
                            var output = Invariant($"step: {tensors[0].GetValue():D}, loss: {tensors[1].GetValue():F4}, W: {tensors[2].GetValue():F4}, b: {tensors[3].GetValue():F4}, lr: {tensors[4].GetValue():F8}");
                            Assert.Equal(expectedLines[i * n_samples + j], output);
                        }
                    }
                }
            }
        }

        [Fact]
        public void LinearRegresionTrainingMomentumTest()
        {
            Console.WriteLine("Linear regression");
            // Parameters
            var learning_rate = 0.01f;
            var training_epochs = 2;

            // Training data
            var train_x = new float[] {
                3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f
            };
            var train_y = new float[] {
                1.7f, 2.76f,2.09f,3.19f,1.694f,1.573f,3.366f,2.596f,2.53f,1.221f,
                 2.827f,3.465f,1.65f,2.904f,2.42f,2.94f,1.3f
            };
            var n_samples = train_x.Length;
            using (var graph = new TFGraph())
            {
                var rng = new Random(0);
                // tf Graph Input

                var X = graph.Placeholder(TFDataType.Float, TFShape.Scalar);
                var Y = graph.Placeholder(TFDataType.Float, TFShape.Scalar);

                var W = graph.Variable(graph.Const(0.1f), operName: "weight");
                var b = graph.Variable(graph.Const(0.1f), operName: "bias");
                var pred = graph.Add(graph.Mul(X, W.Read, "x_w"), b.Read);

                var cost = graph.Div(graph.ReduceSum(graph.Pow(graph.Sub(pred, Y), graph.Const(2f))), graph.Mul(graph.Const(2f), graph.Const((float)n_samples), "2_n_samples"));

                var sgd = new SGD(graph, learning_rate, 0.9f);
                var updateOps = sgd.Minimize(cost);

                var readW = W.ReadAfter(updateOps);
                var readb = b.ReadAfter(updateOps);

                using (var sesssion = new TFSession(graph))
                {
                    sesssion.GetRunner().AddTarget(graph.GetGlobalVariablesInitializer()).Run();

                    var expectedLines = File.ReadAllLines(Path.Combine(_testDataPath, "Momentum", "expected.txt"));
                    for (int i = 0; i < training_epochs; i++)
                    {
                        for (int j = 0; j < n_samples; j++)
                        {
                            var tensors = sesssion.GetRunner()
                                .AddInput(X, new TFTensor(train_x[j]))
                                .AddInput(Y, new TFTensor(train_y[j]))
                                .AddTarget(updateOps).Fetch(cost, readW, readb, pred).Run();
                            var output = Invariant($"loss: {tensors[0].GetValue():F4}, W: {tensors[1].GetValue():F4}, b: {tensors[2].GetValue():F4}");
                            Assert.Equal(expectedLines[i * n_samples + j], output);
                        }
                    }
                }
            }
        }

        [Fact]
        public void LinearRegresionTrainingMomentumWithDecayTest()
        {
            Console.WriteLine("Linear regression");
            // Parameters
            var learning_rate = 0.01f;
            var training_epochs = 2;

            // Training data
            var train_x = new float[] {
                3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f
            };
            var train_y = new float[] {
                1.7f, 2.76f,2.09f,3.19f,1.694f,1.573f,3.366f,2.596f,2.53f,1.221f,
                 2.827f,3.465f,1.65f,2.904f,2.42f,2.94f,1.3f
            };
            var n_samples = train_x.Length;
            using (var graph = new TFGraph())
            {
                var rng = new Random(0);
                // tf Graph Input

                var X = graph.Placeholder(TFDataType.Float, TFShape.Scalar);
                var Y = graph.Placeholder(TFDataType.Float, TFShape.Scalar);

                var W = graph.Variable(graph.Const(0.1f), operName: "weight");
                var b = graph.Variable(graph.Const(0.1f), operName: "bias");
                var pred = graph.Add(graph.Mul(X, W.Read, "x_w"), b.Read);

                var cost = graph.Div(graph.ReduceSum(graph.Pow(graph.Sub(pred, Y), graph.Const(2f))), graph.Mul(graph.Const(2f), graph.Const((float)n_samples), "2_n_samples"));

                var sgd = new SGD(graph, learning_rate, 0.9f, 0.5f);
                var updateOps = sgd.Minimize(cost);

                var iter = sgd.Iterations.ReadAfter(updateOps);
                var readW = W.ReadAfter(updateOps);
                var readb = b.ReadAfter(updateOps);

                using (var sesssion = new TFSession(graph))
                {
                    sesssion.GetRunner().AddTarget(graph.GetGlobalVariablesInitializer()).Run();

                    var expectedLines = File.ReadAllLines(Path.Combine(_testDataPath, "MomentumTimeDecay", "expected.txt"));
                    for (int i = 0; i < training_epochs; i++)
                    {
                        for (int j = 0; j < n_samples; j++)
                        {
                            var tensors = sesssion.GetRunner()
                               .AddInput(X, new TFTensor(train_x[j]))
                               .AddInput(Y, new TFTensor(train_y[j]))
                               .AddTarget(updateOps).Fetch(iter, cost, readW, readb, sgd.LearningRate).Run();
                            var output = Invariant($"step: {tensors[0].GetValue():D}, loss: {tensors[1].GetValue():F4}, W: {tensors[2].GetValue():F4}, b: {tensors[3].GetValue():F4}, lr: {tensors[4].GetValue():F8}");
                            Assert.Equal(expectedLines[i * n_samples + j], output);
                        }
                    }
                }
            }
        }

        [Fact]
        public void LinearRegresionTrainingMomentumNesterovTest()
        {
            Console.WriteLine("Linear regression");
            // Parameters
            var learning_rate = 0.01f;
            var training_epochs = 2;

            // Training data
            var train_x = new float[] {
                3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f
            };
            var train_y = new float[] {
                1.7f, 2.76f,2.09f,3.19f,1.694f,1.573f,3.366f,2.596f,2.53f,1.221f,
                 2.827f,3.465f,1.65f,2.904f,2.42f,2.94f,1.3f
            };
            var n_samples = train_x.Length;
            using (var graph = new TFGraph())
            {
                var rng = new Random(0);
                // tf Graph Input

                var X = graph.Placeholder(TFDataType.Float, TFShape.Scalar);
                var Y = graph.Placeholder(TFDataType.Float, TFShape.Scalar);

                var W = graph.Variable(graph.Const(0.1f), operName: "weight");
                var b = graph.Variable(graph.Const(0.1f), operName: "bias");
                var pred = graph.Add(graph.Mul(X, W.Read, "x_w"), b.Read);

                var cost = graph.Div(graph.ReduceSum(graph.Pow(graph.Sub(pred, Y), graph.Const(2f))), graph.Mul(graph.Const(2f), graph.Const((float)n_samples), "2_n_samples"));

                var sgd = new SGD(graph, learning_rate, 0.9f, nesterov: true);
                var updateOps = sgd.Minimize(cost);

                var readW = W.ReadAfter(updateOps);
                var readb = b.ReadAfter(updateOps);

                using (var sesssion = new TFSession(graph))
                {
                    sesssion.GetRunner().AddTarget(graph.GetGlobalVariablesInitializer()).Run();

                    var expectedLines = File.ReadAllLines(Path.Combine(_testDataPath, "MomentumNesterov", "expected.txt"));
                    for (int i = 0; i < training_epochs; i++)
                    {
                        for (int j = 0; j < n_samples; j++)
                        {
                            var tensors = sesssion.GetRunner()
                                .AddInput(X, new TFTensor(train_x[j]))
                                .AddInput(Y, new TFTensor(train_y[j]))
                                .AddTarget(updateOps).Fetch(cost, readW, readb, pred).Run();
                            var output = Invariant($"loss: {tensors[0].GetValue():F4}, W: {tensors[1].GetValue():F4}, b: {tensors[2].GetValue():F4}");
                            Assert.Equal(expectedLines[i * n_samples + j], output);
                        }
                    }
                }
            }
        }

        [Fact]
        public void LinearRegresionTrainingMomentumNesterovWithTimeDecayTest()
        {
            Console.WriteLine("Linear regression");
            // Parameters
            var learning_rate = 0.01f;
            var training_epochs = 2;

            // Training data
            var train_x = new float[] {
                3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f
            };
            var train_y = new float[] {
                1.7f, 2.76f,2.09f,3.19f,1.694f,1.573f,3.366f,2.596f,2.53f,1.221f,
                 2.827f,3.465f,1.65f,2.904f,2.42f,2.94f,1.3f
            };
            var n_samples = train_x.Length;
            using (var graph = new TFGraph())
            {
                var rng = new Random(0);
                // tf Graph Input

                var X = graph.Placeholder(TFDataType.Float, TFShape.Scalar);
                var Y = graph.Placeholder(TFDataType.Float, TFShape.Scalar);

                var W = graph.Variable(graph.Const(0.1f), operName: "weight");
                var b = graph.Variable(graph.Const(0.1f), operName: "bias");
                var pred = graph.Add(graph.Mul(X, W.Read, "x_w"), b.Read);

                var cost = graph.Div(graph.ReduceSum(graph.Pow(graph.Sub(pred, Y), graph.Const(2f))), graph.Mul(graph.Const(2f), graph.Const((float)n_samples), "2_n_samples"));

                var sgd = new SGD(graph, learning_rate, 0.9f, 0.5f, nesterov: true);
                var updateOps = sgd.Minimize(cost);

                var readIter = sgd.Iterations.ReadAfter(updateOps);
                var readW = W.ReadAfter(updateOps);
                var readb = b.ReadAfter(updateOps);

                using (var sesssion = new TFSession(graph))
                {
                    sesssion.GetRunner().AddTarget(graph.GetGlobalVariablesInitializer()).Run();

                    var expectedLines = File.ReadAllLines(Path.Combine(_testDataPath, "MomentumNesterovTimeDecay", "expected.txt"));
                    for (int i = 0; i < training_epochs; i++)
                    {
                        for (int j = 0; j < n_samples; j++)
                        {
                            var tensors = sesssion.GetRunner()
                               .AddInput(X, new TFTensor(train_x[j]))
                               .AddInput(Y, new TFTensor(train_y[j]))
                               .AddTarget(updateOps)
                               .Fetch(readIter, cost, readW, readb, sgd.LearningRate).Run();
                            var output = Invariant($"step: {tensors[0].GetValue():D}, loss: {tensors[1].GetValue():F4}, W: {tensors[2].GetValue():F4}, b: {tensors[3].GetValue():F4}, lr: {tensors[4].GetValue():F8}");
                            Assert.Equal(expectedLines[i * n_samples + j], output);
                        }
                    }
                }
            }
        }

        [Fact]
        public void MNISTTwoHiddenLayerNetworkTest()
        {
            // Parameters
            var learningRate = 0.1f;
            var epochs = 5;


            var mnist = new Mnist();
            mnist.ReadDataSets("/tmp");
            int batchSize = 100;
            int numBatches = mnist.TrainImages.Length / batchSize;

            using (var graph = new TFGraph())
            {
                var X = graph.Placeholder(TFDataType.Float, new TFShape(-1, 784));
                var Y = graph.Placeholder(TFDataType.Float, new TFShape(-1, 10));

                graph.Seed = 1;
                var initB = (float)(4 * Math.Sqrt(6) / Math.Sqrt(784 + 500));
                var W1 = graph.Variable(graph.RandomUniform(new TFShape(784,500), minval: -initB, maxval: initB), operName: "W1");
                var b1 = graph.Variable(graph.Constant(0f, new TFShape(500), TFDataType.Float), operName: "b1");
                var layer1 = graph.Sigmoid(graph.Add(graph.MatMul(X, W1.Read), b1.Read, operName: "layer1"));

                initB = (float)(4 * Math.Sqrt(6) / Math.Sqrt(500 + 100));
                var W2 = graph.Variable(graph.RandomUniform(new TFShape(500, 100), minval: -initB, maxval: initB), operName: "W2");
                var b2 = graph.Variable(graph.Constant(0f, new TFShape(100), TFDataType.Float), operName: "b2");
                var layer2 = graph.Sigmoid(graph.Add(graph.MatMul(layer1, W2.Read), b2.Read, operName: "layer2"));

                initB = (float)(4 * Math.Sqrt(6) / Math.Sqrt(100 + 10));
                var W3 = graph.Variable(graph.RandomUniform(new TFShape(100, 10), minval: -initB, maxval: initB), operName: "W3");
                var b3 = graph.Variable(graph.Constant(0f, new TFShape(10), TFDataType.Float), operName: "b3");
                var layer3 = graph.Add(graph.MatMul(layer2, W3.Read), b3.Read, operName: "layer3");

                // No support for computing gradient for the SparseSoftmaxCrossEntropyWithLogits function
                // instead using SoftmaxCrossEntropyWithLogits
                var cost = graph.ReduceMean(graph.SoftmaxCrossEntropyWithLogits(layer3, Y, "cost").loss);

                var prediction = graph.ArgMax(graph.Softmax(layer3), graph.Const(1));
                var labels = graph.ArgMax(Y, graph.Const(1));
                var areCorrect = graph.Equal(prediction, labels);
                var accuracy = graph.ReduceMean(graph.Cast(areCorrect,TFDataType.Float));

                var sgd = new SGD(graph, learningRate, 0.9f);
                var updateOps = sgd.Minimize(cost);

                using (var sesssion = new TFSession(graph))
                {
                    sesssion.GetRunner().AddTarget(graph.GetGlobalVariablesInitializer()).Run();

                    var expectedLines = File.ReadAllLines(Path.Combine(_testDataPath, "SGDMnist", "expected.txt"));
                    
                    for (int i = 0; i < epochs; i++)
                    {
                        var reader = mnist.GetTrainReader();
                        float avgLoss = 0;
                        float avgAccuracy = 0;
                        for (int j = 0; j < numBatches; j++)
                        {
                            var batch = reader.NextBatch(batchSize);
                            var tensors = sesssion.GetRunner()
                                .AddInput(X, batch.Item1)
                                .AddInput(Y, batch.Item2)
                                .AddTarget(updateOps).Fetch(cost, accuracy, prediction, labels).Run();

                            avgLoss += (float)tensors[0].GetValue();
                            avgAccuracy += (float)tensors[1].GetValue();
                        }
                        var output = Invariant($"Epoch: {i}, loss(Cross-Entropy): {avgLoss / numBatches:F4}, Accuracy:{avgAccuracy / numBatches:F4}");
                        Assert.Equal(expectedLines[i], output);
                    }
                }
            }
        }


        /// <summary>
        /// Get the variable if it has already been created in the graph otherwise creates the new one.
        /// </summary>
        public Variable GetVariable(TFGraph graph, Dictionary<Variable, List<TFOutput>> variables, TFOutput initialValue, bool trainable = true, string operName = null)
        {
            var variable = variables.Where(a => a.Key.VariableOp.Operation.Name.StartsWith(operName));
            if (variable != null && variable.Count() > 0)
                return variable.First().Key;

            return graph.Variable(initialValue, trainable, operName);
        }

        private (TFOutput cost, TFOutput model, TFOutput accracy) CreateNetwork(TFGraph graph, TFOutput X, TFOutput Y, Dictionary<Variable, List<TFOutput>> variables)
        {
            graph.Seed = 1;
            var initB = (float)(4 * Math.Sqrt(6) / Math.Sqrt(784 + 500));
            var W1 = GetVariable(graph, variables, graph.RandomUniform(new TFShape(784, 500), minval: -initB, maxval: initB), operName: "W1");
            var b1 = GetVariable(graph, variables, graph.Constant(0f, new TFShape(500), TFDataType.Float), operName: "b1");
            var layer1 = graph.Sigmoid(graph.Add(graph.MatMul(X, W1.Read), b1.Read));

            initB = (float)(4 * Math.Sqrt(6) / Math.Sqrt(500 + 100));
            var W2 = GetVariable(graph, variables, graph.RandomUniform(new TFShape(500, 100), minval: -initB, maxval: initB), operName: "W2");
            var b2 = GetVariable(graph, variables, graph.Constant(0f, new TFShape(100), TFDataType.Float), operName: "b2");
            var layer2 = graph.Sigmoid(graph.Add(graph.MatMul(layer1, W2.Read), b2.Read));

            initB = (float)(4 * Math.Sqrt(6) / Math.Sqrt(100 + 10));
            var W3 = GetVariable(graph, variables, graph.RandomUniform(new TFShape(100, 10), minval: -initB, maxval: initB), operName: "W3");
            var b3 = GetVariable(graph, variables, graph.Constant(0f, new TFShape(10), TFDataType.Float), operName: "b3");
            var model = graph.Add(graph.MatMul(layer2, W3.Read), b3.Read);

            // No support for computing gradient for the SparseSoftmaxCrossEntropyWithLogits function
            // instead using SoftmaxCrossEntropyWithLogits
            var cost = graph.ReduceMean(graph.SoftmaxCrossEntropyWithLogits(model, Y).loss);

            var prediction = graph.ArgMax(graph.Softmax(model), graph.Const(1));
            var labels = graph.ArgMax(Y, graph.Const(1));
            var areCorrect = graph.Equal(prediction, labels);
            var accuracy = graph.ReduceMean(graph.Cast(areCorrect, TFDataType.Float));

            return (cost, model, accuracy);
        }

        [Fact(Skip = "Disabled because it requires GPUs and need to set numGPUs to available GPUs on system." +
            " It has been tested on GPU machine with 4 GPUs and it passed there.")]
        public void MNISTTwoHiddenLayerNetworkGPUTest()
        {
            // Parameters
            var learningRate = 0.1f;
            var epochs = 5;
            var numGPUs = 4;

            var mnist = new Mnist();
            mnist.ReadDataSets("/tmp");
            int batchSize = 400;
            int numBatches = mnist.TrainImages.Length / batchSize;

            using (var graph = new TFGraph())
            {
                var X = graph.Placeholder(TFDataType.Float, new TFShape(-1, 784));
                var Y = graph.Placeholder(TFDataType.Float, new TFShape(-1, 10));

                var Xs = graph.Split(graph.Const(0), X, numGPUs);
                var Ys = graph.Split(graph.Const(0), Y, numGPUs);

                var sgd = new SGD(graph, learningRate, 0.9f);
                TFOutput[] costs = new TFOutput[numGPUs];
                TFOutput[] accuracys = new TFOutput[numGPUs];
                var variablesAndGradients = new Dictionary<Variable, List<TFOutput>>();
                for (int i = 0; i < numGPUs; i++)
                {
                    using (var device = graph.WithDevice("/GPU:" + i))
                    {
                        (costs[i], _, accuracys[i]) = CreateNetwork(graph, Xs[i], Ys[i], variablesAndGradients);
                        foreach (var gv in sgd.ComputeGradient(costs[i], colocateGradientsWithOps: true))
                        {
                            if (!variablesAndGradients.ContainsKey(gv.variable))
                                variablesAndGradients[gv.variable] = new List<TFOutput>();
                            variablesAndGradients[gv.variable].Add(gv.gradient);
                        }
                    }
                }
                var cost = graph.ReduceMean(graph.Stack(costs));
                var accuracy = graph.ReduceMean(graph.Stack(accuracys));

                var gradientsAndVariables = new (TFOutput gradient, Variable variable)[variablesAndGradients.Count];
                int index = 0;
                foreach (var key in variablesAndGradients.Keys)
                {
                    gradientsAndVariables[index].variable = key;
                    gradientsAndVariables[index++].gradient = graph.ReduceMean(graph.Stack(variablesAndGradients[key].ToArray()), graph.Const(0));
                }

                var updateOps = sgd.ApplyGradient(gradientsAndVariables);

                using (var sesssion = new TFSession(graph))
                {
                    sesssion.GetRunner().AddTarget(graph.GetGlobalVariablesInitializer()).Run();

                    var expectedLines = File.ReadAllLines(Path.Combine(_testDataPath, "SGDMnistGPU", "expected.txt"));

                    for (int i = 0; i < epochs; i++)
                    {
                        var reader = mnist.GetTrainReader();
                        float avgLoss = 0;
                        float avgAccuracy = 0;
                        for (int j = 0; j < numBatches; j++)
                        {
                            var batch = reader.NextBatch(batchSize);
                            var tensors = sesssion.GetRunner()
                                .AddInput(X, batch.Item1)
                                .AddInput(Y, batch.Item2)
                                .AddTarget(updateOps).Fetch(cost, accuracy).Run();

                            avgLoss += (float)tensors[0].GetValue();
                            avgAccuracy += (float)tensors[1].GetValue();
                        }
                        var output = Invariant($"Epoch: {i}, loss(Cross-Entropy): {avgLoss / numBatches:F4}, Accuracy:{avgAccuracy / numBatches:F4}");
                        Assert.Equal(expectedLines[i], output);
                    }
                }
            }
        }


        [Fact]
        public void LinearRegresionTrainingWithAdagradTest()
        {
            Console.WriteLine("Linear regression");
            // Parameters
            var learning_rate = 0.01f;
            var training_epochs = 5;

            // Training data
            var train_x = new float[] {
                3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f
            };
            var train_y = new float[] {
                1.7f, 2.76f,2.09f,3.19f,1.694f,1.573f,3.366f,2.596f,2.53f,1.221f,
                 2.827f,3.465f,1.65f,2.904f,2.42f,2.94f,1.3f
            };
            var n_samples = train_x.Length;
            using (var graph = new TFGraph())
            {
                var rng = new Random(0);
                // tf Graph Input

                var X = graph.Placeholder(TFDataType.Float, TFShape.Scalar);
                var Y = graph.Placeholder(TFDataType.Float, TFShape.Scalar);

                var W = graph.Variable(graph.Const(0.1f), operName: "weight");
                var b = graph.Variable(graph.Const(0.1f), operName: "bias");
                var pred = graph.Add(graph.Mul(X, W.Read, "x_w"), b.Read);

                var cost = graph.Div(graph.ReduceSum(graph.Pow(graph.Sub(pred, Y), graph.Const(2f))), graph.Mul(graph.Const(2f), graph.Const((float)n_samples), "2_n_samples"));

                var sgd = new AdaGradOptimizer(graph, learning_rate);
                var updateOps = sgd.Minimize(cost);

                var readW = W.ReadAfter(updateOps);
                var readb = b.ReadAfter(updateOps);

                using (var sesssion = new TFSession(graph))
                {
                    sesssion.GetRunner().AddTarget(graph.GetGlobalVariablesInitializer()).Run();

                    var expectedLines = File.ReadAllLines(Path.Combine(_testDataPath, "Adagrad", "expected.txt"));
                    for (int i = 0; i < training_epochs; i++)
                    {
                        for (int j = 0; j < n_samples; j++)
                        {
                            var tensors = sesssion.GetRunner()
                                .AddInput(X, new TFTensor(train_x[j]))
                                .AddInput(Y, new TFTensor(train_y[j]))
                                .AddTarget(updateOps).Fetch(cost, readW, readb, pred).Run();
                            var output = Invariant($"loss: {tensors[0].GetValue():F4}, W: {tensors[1].GetValue():F4}, b: {tensors[2].GetValue():F4}");
                            Assert.Equal(expectedLines[i * n_samples + j], output);
                        }
                    }
                }
            }
        }

        [Fact]
        public void LinearRegresionTrainingWithAdagradDecayTest()
        {
            Console.WriteLine("Linear regression");
            // Parameters
            var learning_rate = 0.01f;
            var training_epochs = 5;

            // Training data
            var train_x = new float[] {
                3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f
            };
            var train_y = new float[] {
                1.7f, 2.76f,2.09f,3.19f,1.694f,1.573f,3.366f,2.596f,2.53f,1.221f,
                 2.827f,3.465f,1.65f,2.904f,2.42f,2.94f,1.3f
            };
            var n_samples = train_x.Length;
            using (var graph = new TFGraph())
            {
                var rng = new Random(0);
                // tf Graph Input

                var X = graph.Placeholder(TFDataType.Float, TFShape.Scalar);
                var Y = graph.Placeholder(TFDataType.Float, TFShape.Scalar);

                var W = graph.Variable(graph.Const(0.1f), operName: "weight");
                var b = graph.Variable(graph.Const(0.1f), operName: "bias");
                var pred = graph.Add(graph.Mul(X, W.Read, "x_w"), b.Read);

                var cost = graph.Div(graph.ReduceSum(graph.Pow(graph.Sub(pred, Y), graph.Const(2f))), graph.Mul(graph.Const(2f), graph.Const((float)n_samples), "2_n_samples"));

                var sgd = new AdaGradOptimizer(graph, learning_rate, decay: 0.5f);
                var updateOps = sgd.Minimize(cost);

                var iter = sgd.Iterations.ReadAfter(updateOps);
                var readW = W.ReadAfter(updateOps);
                var readb = b.ReadAfter(updateOps);

                using (var sesssion = new TFSession(graph))
                {
                    sesssion.GetRunner().AddTarget(graph.GetGlobalVariablesInitializer()).Run();

                    var expectedLines = File.ReadAllLines(Path.Combine(_testDataPath, "AdagradTimeDecay", "expected.txt"));
                    for (int i = 0; i < training_epochs; i++)
                    {
                        for (int j = 0; j < n_samples; j++)
                        {
                            var tensors = sesssion.GetRunner()
                               .AddInput(X, new TFTensor(train_x[j]))
                               .AddInput(Y, new TFTensor(train_y[j]))
                               .AddTarget(updateOps).Fetch(iter, cost, readW, readb, sgd.LearningRate).Run();
                            var output = Invariant($"step: {tensors[0].GetValue():D}, loss: {tensors[1].GetValue():F4}, W: {tensors[2].GetValue():F4}, b: {tensors[3].GetValue():F4}, lr: {tensors[4].GetValue():F8}");
                            Assert.Equal(expectedLines[i * n_samples + j], output);
                        }
                    }
                }
            }
        }

        [Fact]
        public void LinearRegresionTrainingWithRMSPropTest()
        {
            Console.WriteLine("Linear regression");
            // Parameters
            var learning_rate = 0.01f;
            var training_epochs = 5;

            // Training data
            var train_x = new float[] {
                3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f
            };
            var train_y = new float[] {
                1.7f, 2.76f,2.09f,3.19f,1.694f,1.573f,3.366f,2.596f,2.53f,1.221f,
                 2.827f,3.465f,1.65f,2.904f,2.42f,2.94f,1.3f
            };
            var n_samples = train_x.Length;
            using (var graph = new TFGraph())
            {
                var rng = new Random(0);
                // tf Graph Input

                var X = graph.Placeholder(TFDataType.Float, TFShape.Scalar);
                var Y = graph.Placeholder(TFDataType.Float, TFShape.Scalar);

                var W = graph.Variable(graph.Const(0.1f), operName: "weight");
                var b = graph.Variable(graph.Const(0.1f), operName: "bias");
                var pred = graph.Add(graph.Mul(X, W.Read, "x_w"), b.Read);

                var cost = graph.Div(graph.ReduceSum(graph.Pow(graph.Sub(pred, Y), graph.Const(2f))), graph.Mul(graph.Const(2f), graph.Const((float)n_samples), "2_n_samples"));

                var sgd = new RMSPropOptimizer(graph, learning_rate, initialAccumulatorValue: 1.0f);
                var updateOps = sgd.Minimize(cost);

                var iter = sgd.Iterations.ReadAfter(updateOps);
                var readW = W.ReadAfter(updateOps);
                var readb = b.ReadAfter(updateOps);

                using (var sesssion = new TFSession(graph))
                {
                    sesssion.GetRunner().AddTarget(graph.GetGlobalVariablesInitializer()).Run();

                    var expectedLines = File.ReadAllLines(Path.Combine(_testDataPath, "RMSProp", "expected.txt"));
                    for (int i = 0; i < training_epochs; i++)
                    {
                        for (int j = 0; j < n_samples; j++)
                        {
                            var tensors = sesssion.GetRunner()
                                .AddInput(X, new TFTensor(train_x[j]))
                                .AddInput(Y, new TFTensor(train_y[j]))
                                .AddTarget(updateOps).Fetch(cost, readW, readb, pred).Run();
                            var output = Invariant($"loss: {tensors[0].GetValue():F4}, W: {tensors[1].GetValue():F4}, b: {tensors[2].GetValue():F4}");
                            Assert.Equal(expectedLines[i * n_samples + j], output);
                        }
                    }
                }
            }
        }

        [Fact]
        public void LinearRegresionTrainingWithRMSPropDecayTest()
        {
            Console.WriteLine("Linear regression");
            // Parameters
            var learning_rate = 0.01f;
            var training_epochs = 5;

            // Training data
            var train_x = new float[] {
                3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f
            };
            var train_y = new float[] {
                1.7f, 2.76f,2.09f,3.19f,1.694f,1.573f,3.366f,2.596f,2.53f,1.221f,
                 2.827f,3.465f,1.65f,2.904f,2.42f,2.94f,1.3f
            };
            var n_samples = train_x.Length;
            using (var graph = new TFGraph())
            {
                var rng = new Random(0);
                // tf Graph Input

                var X = graph.Placeholder(TFDataType.Float, TFShape.Scalar);
                var Y = graph.Placeholder(TFDataType.Float, TFShape.Scalar);

                var W = graph.Variable(graph.Const(0.1f), operName: "weight");
                var b = graph.Variable(graph.Const(0.1f), operName: "bias");
                var pred = graph.Add(graph.Mul(X, W.Read, "x_w"), b.Read);

                var cost = graph.Div(graph.ReduceSum(graph.Pow(graph.Sub(pred, Y), graph.Const(2f))), graph.Mul(graph.Const(2f), graph.Const((float)n_samples), "2_n_samples"));

                var sgd = new RMSPropOptimizer(graph, learning_rate, decay: 0.5f, initialAccumulatorValue: 1.0f);
                var updateOps = sgd.Minimize(cost);

                var iter = sgd.Iterations.ReadAfter(updateOps);
                var readW = W.ReadAfter(updateOps);
                var readb = b.ReadAfter(updateOps);

                using (var sesssion = new TFSession(graph))
                {
                    sesssion.GetRunner().AddTarget(graph.GetGlobalVariablesInitializer()).Run();

                    var expectedLines = File.ReadAllLines(Path.Combine(_testDataPath, "RMSPropTimeDecay", "expected.txt"));
                    for (int i = 0; i < training_epochs; i++)
                    {
                        for (int j = 0; j < n_samples; j++)
                        {
                            var tensors = sesssion.GetRunner()
                               .AddInput(X, new TFTensor(train_x[j]))
                               .AddInput(Y, new TFTensor(train_y[j]))
                               .AddTarget(updateOps).Fetch(iter, cost, readW, readb, sgd.LearningRate).Run();
                            var output = Invariant($"step: {tensors[0].GetValue():D}, loss: {tensors[1].GetValue():F4}, W: {tensors[2].GetValue():F4}, b: {tensors[3].GetValue():F4}, lr: {tensors[4].GetValue():F8}");
                            Assert.Equal(expectedLines[i * n_samples + j], output);
                        }
                    }
                }
            }
        }

        [Fact]
        public void LinearRegresionTrainingWithAdamOptimizerTest()
        {
            Console.WriteLine("Linear regression");
            // Parameters
            var learning_rate = 0.01f;
            var training_epochs = 5;

            // Training data
            var train_x = new float[] {
                3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f
            };
            var train_y = new float[] {
                1.7f, 2.76f,2.09f,3.19f,1.694f,1.573f,3.366f,2.596f,2.53f,1.221f,
                 2.827f,3.465f,1.65f,2.904f,2.42f,2.94f,1.3f
            };
            var n_samples = train_x.Length;
            using (var graph = new TFGraph())
            {
                var rng = new Random(0);
                // tf Graph Input

                var X = graph.Placeholder(TFDataType.Float, TFShape.Scalar);
                var Y = graph.Placeholder(TFDataType.Float, TFShape.Scalar);

                var W = graph.Variable(graph.Const(0.1f), operName: "weight");
                var b = graph.Variable(graph.Const(0.1f), operName: "bias");
                var pred = graph.Add(graph.Mul(X, W.Read, "x_w"), b.Read);

                var cost = graph.Div(graph.ReduceSum(graph.Pow(graph.Sub(pred, Y), graph.Const(2f))), graph.Mul(graph.Const(2f), graph.Const((float)n_samples), "2_n_samples"));

                var sgd = new AdamOptimizer(graph, learning_rate);
                var updateOps = sgd.Minimize(cost);

                using (var sesssion = new TFSession(graph))
                {
                    sesssion.GetRunner().AddTarget(graph.GetGlobalVariablesInitializer()).Run();

                    var expectedLines = File.ReadAllLines(Path.Combine(_testDataPath, "Adam", "expected.txt"));
                    for (int i = 0; i < training_epochs; i++)
                    {
                        for (int j = 0; j < n_samples; j++)
                        {
                            var tensors = sesssion.GetRunner()
                                .AddInput(X, new TFTensor(train_x[j]))
                                .AddInput(Y, new TFTensor(train_y[j]))
                                .AddTarget(updateOps).Fetch(cost, W.Read, b.Read, pred).Run();
                            var output = Invariant($"loss: {tensors[0].GetValue():F4}, W: {tensors[1].GetValue():F4}, b: {tensors[2].GetValue():F4}");
                            Assert.Equal(expectedLines[i * n_samples + j], output);
                        }
                    }
                }
            }
        }

        [Fact]
        public void LinearRegresionTrainingWithAdamOptimizerDecayTest()
        {
            Console.WriteLine("Linear regression");
            // Parameters
            var learning_rate = 0.01f;
            var training_epochs = 5;

            // Training data
            var train_x = new float[] {
                3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f
            };
            var train_y = new float[] {
                1.7f, 2.76f,2.09f,3.19f,1.694f,1.573f,3.366f,2.596f,2.53f,1.221f,
                 2.827f,3.465f,1.65f,2.904f,2.42f,2.94f,1.3f
            };
            var n_samples = train_x.Length;
            using (var graph = new TFGraph())
            {
                var rng = new Random(0);
                // tf Graph Input

                var X = graph.Placeholder(TFDataType.Float, TFShape.Scalar);
                var Y = graph.Placeholder(TFDataType.Float, TFShape.Scalar);

                var W = graph.Variable(graph.Const(0.1f), operName: "weight");
                var b = graph.Variable(graph.Const(0.1f), operName: "bias");
                var pred = graph.Add(graph.Mul(X, W.Read, "x_w"), b.Read);

                var cost = graph.Div(graph.ReduceSum(graph.Pow(graph.Sub(pred, Y), graph.Const(2f))), graph.Mul(graph.Const(2f), graph.Const((float)n_samples), "2_n_samples"));

                var sgd = new AdamOptimizer(graph, learning_rate, decay: 0.5f);
                var updateOps = sgd.Minimize(cost);

                using (var sesssion = new TFSession(graph))
                {
                    sesssion.GetRunner().AddTarget(graph.GetGlobalVariablesInitializer()).Run();

                    var expectedLines = File.ReadAllLines(Path.Combine(_testDataPath, "AdamTimeDecay", "expected.txt"));
                    for (int i = 0; i < training_epochs; i++)
                    {
                        for (int j = 0; j < n_samples; j++)
                        {
                            var tensors = sesssion.GetRunner()
                                .AddInput(X, new TFTensor(train_x[j]))
                                .AddInput(Y, new TFTensor(train_y[j]))
                                .AddTarget(updateOps).Fetch(cost, W.Read, b.Read, pred).Run();
                            var output = Invariant($"loss: {tensors[0].GetValue():F4}, W: {tensors[1].GetValue():F4}, b: {tensors[2].GetValue():F4}");
                            Assert.Equal(expectedLines[i * n_samples + j], output);
                        }
                    }
                }
            }
        }
    }
}
