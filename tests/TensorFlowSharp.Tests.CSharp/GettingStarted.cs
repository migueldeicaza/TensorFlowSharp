using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using System.Linq;
using TensorFlow;

namespace GettingStarted.Tests
{
    /// <summary>
    /// These are basic tests based on the Tensorflow gettings started page
    /// https://www.tensorflow.org/get_started/get_started
    /// </summary>

    public class GettingStartedTests 
	{
        TFGraph g = null;
        TFSession s = null;
        [TestInitialize]
        public void TestInit()
        {
            g = new TFGraph();
            Assert.IsNotNull(g, "Graph could not be initialized");

            s = new TFSession(g);
            Assert.IsNotNull(s, "Session could not be initialized");
        }

        [TestCleanup]
        public void TestCleanup()
        {
            if (s != null)
            {
                s.CloseSession();
                s.Dispose();
                s = null;
            }
            if (g != null)
            {
                g.Dispose();
                g = null;
            }
        }

        [TestMethod]
        public void Version()
        {
            Assert.IsNotNull(TFCore.Version);
            Assert.IsInstanceOfType(TFCore.Version, typeof(string));
        }

        [TestMethod]
        public void Basic()
        {
            //this could be that the TF did nit initialize
            //specifically on testing machines that may not have GPU or TF .dlls
            if (s == null) return;

            var runner = s.GetRunner();
            
            //3 # a rank 0 tensor; this is a scalar with shape []
            //[1. , 2., 3.] # a rank 1 tensor; this is a vector with shape [3]
            //[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
            //[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3] 
            //
            TFTensor rank0 = 3;//shape[]
            Assert.AreEqual(rank0.NumDims, 0);
            Assert.AreEqual(rank0.ToString(), "3");


            TFTensor rank1 = new double[] { 1.0, 2.0, 3.0 };//shape [3]                
            Assert.AreEqual(rank1.NumDims, 1);
            Assert.AreEqual(rank1.ToString(), "[3]");

            TFTensor rank2 = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };//shape[2,3]
            Assert.AreEqual(rank2.NumDims, 2);
            Assert.AreEqual(rank2.ToString(), "[2x3]");

            TFTensor rank3 = new double[,,]
            {
                    { { 1, 2, 3 } },
                    { { 4, 5, 6 } }
            };//shape [2, 1, 3] 
            Assert.AreEqual(rank3.NumDims, 3);
            Assert.AreEqual(rank3.ToString(), "[2x1x3]");
            
            //node1 = tf.constant(3.0, tf.float32)
            TFOutput node1 = g.Const(3.0F, TFDataType.Float);
            Assert.IsNotNull(node1);
            Assert.AreEqual(node1.OutputType, TFDataType.Float);
            //node2 = tf.constant(4.0) # also tf.float32 implicitly
            TFOutput node2 = g.Const(4.0F);
            Assert.IsNotNull(node2);
            Assert.AreEqual(node1.OutputType, TFDataType.Float);

            TFTensor[] results = runner.Run(new TFOutput[] { node1, node2 });
            Assert.IsNotNull(results);
            Assert.AreEqual(results.Length, 2);
            Assert.AreEqual(results[0].GetValue(), 3.0F);
            Assert.AreEqual(results[1].GetValue(), 4.0F);
            
            TFOutput node3 = g.Add(node1, node2);
            Assert.IsNotNull(node3);
            TFTensor result = runner.Run(node3);
            Assert.IsNotNull(result);
            Assert.AreEqual(result.GetValue(), 7.0F);

            TFOutput a = g.Placeholder(TFDataType.Float);
            Assert.IsNotNull(a);

            TFOutput b = g.Placeholder(TFDataType.Float);
            Assert.IsNotNull(b);

            TFOutput adder_node = g.Add(a, b);
            Assert.IsNotNull(adder_node);

            result = runner.AddInput(a, 3.0F).AddInput(b, 4.5F).Run(adder_node);
            Assert.IsNotNull(result);
            Assert.AreEqual(result.GetValue(), 7.5F);

            runner = s.GetRunner();
            result = runner.AddInput(a, new float[] { 1F, 3F }).AddInput(b, new float[] { 2F, 4F }).Run(adder_node);
            Assert.IsNotNull(result);
            Assert.AreEqual(result.NumDims, 1);
            Assert.AreEqual(((float[])result.GetValue())[0], 3.0F);
            Assert.AreEqual(((float[])result.GetValue())[1], 7.0F);
            
            var add_and_triple = g.Mul(g.Const(3.0F, TFDataType.Float), adder_node);
            Assert.IsNotNull(add_and_triple);

            runner = s.GetRunner();
            result = runner.AddInput(a, 3.0F).AddInput(b, 4.5F).Run(add_and_triple);
            Assert.IsNotNull(result);
            Assert.AreEqual(result.NumDims, 0);
            Assert.AreEqual(result.GetValue(), 22.5F);

            runner = s.GetRunner();
            result = runner.AddInput(a, new float[] { 1F, 3F }).AddInput(b, new float[] { 2F, 4F }).Run(add_and_triple);
            Assert.IsNotNull(result);
            Assert.AreEqual(result.NumDims, 1);
            Assert.AreEqual(((float[])result.GetValue())[0], 9.0F);
            Assert.AreEqual(((float[])result.GetValue())[1], 21.0F);
        }

        [TestMethod]
        public void Variables()
        {
            TFStatus status = new TFStatus();
            var runner = s.GetRunner();

            TFOutput vW, vb, vlinmodel;
            var hW = g.Variable(g.Const(0.3F, TFDataType.Float), out vW);
            var hb = g.Variable(g.Const(-0.3F, TFDataType.Float), out vb);
            var hlinearmodel = g.Variable(g.Const(0.0F, TFDataType.Float), out vlinmodel);
            var x = g.Placeholder(TFDataType.Float);

            var hoplm = g.AssignVariableOp(hlinearmodel, g.Add(g.Mul(vW, x), vb));

            //init all variable
            runner
                .AddTarget(g.GetGlobalVariablesInitializer())
                .AddTarget(hoplm)
                .AddInput(x, new float[] { 1F, 2F, 3F, 4F })
                .Run(status);

            //now get actual value
            var result = s.GetRunner()
                .Fetch(vlinmodel)
                .Run();

            Assert.IsNotNull(result);
            Assert.AreEqual(result.Length, 1);
            Assert.IsInstanceOfType(result[0].GetValue(), typeof(float[]));

            float[] values = (float[])result[0].GetValue();
            Assert.IsNotNull(values);
            Assert.AreEqual(values.Length, 4);
            Assert.AreEqual(values[0], 0.0F, 0.0000001F);
            Assert.AreEqual(values[1], 0.3F, 0.0000001F);
            Assert.AreEqual(values[2], 0.6F, 0.0000001F);
            Assert.AreEqual(values[3], 0.9F, 0.0000001F);
            
        }

        [TestMethod]
        public void BasicConstantOps()
        {
            var a = g.Const(2);
            Assert.IsNotNull(a);
            Assert.AreEqual(a.OutputType, TFDataType.Int32);

            var b = g.Const(3);
            Assert.IsNotNull(b);
            Assert.AreEqual(b.OutputType, TFDataType.Int32);

            // Add two constants
            var results = s.GetRunner().Run(g.Add(a, b));

            Assert.IsNotNull(results);
            Assert.IsInstanceOfType(results, typeof(TFTensor));
            Assert.AreEqual(results.TensorType, TFDataType.Int32);
            Assert.AreEqual(results.NumDims, 0);
            Assert.AreEqual(results.Shape.Length, 0);

            var val = results.GetValue();
            Assert.IsNotNull(val);
            Assert.AreEqual(val, 5);

            // Multiply two constants
            results = s.GetRunner().Run(g.Mul(a, b));
            Assert.IsNotNull(results);
            Assert.IsInstanceOfType(results, typeof(TFTensor));
            Assert.AreEqual(results.TensorType, TFDataType.Int32);
            Assert.AreEqual(results.NumDims, 0);
            Assert.AreEqual(results.Shape.Length, 0);
            Assert.IsNotNull(results.GetValue());
            Assert.AreEqual(results.GetValue(), 6);
        }

        [TestMethod]
        public void BasicConstantZerosAndOnes()
        {
            // Test Zeros, Ones for n x n shape
            var o = g.Ones(new TFShape(4, 4));
            Assert.IsNotNull(o);
            Assert.AreEqual(o.OutputType, TFDataType.Double);

            var z = g.Zeros(new TFShape(4, 4));
            Assert.IsNotNull(z);
            Assert.AreEqual(z.OutputType, TFDataType.Double);

            var r = g.RandomNormal(new TFShape(4, 4));
            Assert.IsNotNull(r);
            Assert.AreEqual(r.OutputType, TFDataType.Double);

            var res1 = s.GetRunner().Run(g.Mul(o, r));
            Assert.IsNotNull(res1);
            Assert.AreEqual(res1.TensorType, TFDataType.Double);
            Assert.AreEqual(res1.NumDims, 2);
            Assert.AreEqual(res1.Shape[0], 4);
            Assert.AreEqual(res1.Shape[1], 4);
            Assert.AreEqual(res1.ToString(), "[4x4]");

            var matval1 = res1.GetValue();
            Assert.IsNotNull(matval1);
            Assert.IsInstanceOfType(matval1, typeof(double[,]));
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    Assert.IsNotNull(((double[,])matval1)[i, j]);
                }
            }

            var res2 = s.GetRunner().Run(g.Mul(g.Mul(o, r), z));
            Assert.IsNotNull(res2);
            Assert.AreEqual(res2.TensorType, TFDataType.Double);
            Assert.AreEqual(res2.NumDims, 2);
            Assert.AreEqual(res2.Shape[0], 4);
            Assert.AreEqual(res2.Shape[1], 4);
            Assert.AreEqual(res2.ToString(), "[4x4]");

            var matval2 = res2.GetValue();
            Assert.IsNotNull(matval2);
            Assert.IsInstanceOfType(matval2, typeof(double[,]));
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    Assert.IsNotNull(((double[,])matval2)[i, j]);
                    Assert.AreEqual(((double[,])matval2)[i, j], 0.0);
                }
            }

        }

        [TestMethod]
        public void BasicConstantsOnSymmetricalShapes()
        {
            //build some test vectors
            var o = g.Ones(new TFShape(4, 4));
            var z = g.Zeros(new TFShape(4, 4));
            var r = g.RandomNormal(new TFShape(4, 4));
            var matval = s.GetRunner().Run(g.Mul(o, r)).GetValue();
            var matvalzero = s.GetRunner().Run(g.Mul(g.Mul(o, r), z)).GetValue();

            var co = g.Constant(1.0, new TFShape(4, 4), TFDataType.Double);
            var cz = g.Constant(0.0, new TFShape(4, 4), TFDataType.Double);
            var res1 = s.GetRunner().Run(g.Mul(co, r));

            Assert.IsNotNull(res1);
            Assert.AreEqual(res1.TensorType, TFDataType.Double);
            Assert.AreEqual(res1.NumDims, 2);
            Assert.AreEqual(res1.Shape[0], 4);
            Assert.AreEqual(res1.Shape[1], 4);
            Assert.AreEqual(res1.ToString(), "[4x4]");

            var cmatval1 = res1.GetValue();
            Assert.IsNotNull(cmatval1);
            Assert.IsInstanceOfType(cmatval1, typeof(double[,]));
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    Assert.IsNotNull(((double[,])cmatval1)[i, j]);                    
                }
            }

            var cres2 = s.GetRunner().Run(g.Mul(g.Mul(co, r), cz));

            Assert.IsNotNull(cres2);
            Assert.AreEqual(cres2.TensorType, TFDataType.Double);
            Assert.AreEqual(cres2.NumDims, 2);
            Assert.AreEqual(cres2.Shape[0], 4);
            Assert.AreEqual(cres2.Shape[1], 4);
            Assert.AreEqual(cres2.ToString(), "[4x4]");

            var cmatval2 = cres2.GetValue();
            Assert.IsNotNull(cmatval2);
            Assert.IsInstanceOfType(cmatval2, typeof(double[,]));
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    Assert.IsNotNull(((double[,])cmatval2)[i, j]);
                    Assert.AreEqual(((double[,])matvalzero)[i, j], ((double[,])cmatval2)[i, j]);
                }
            }

        }

        [TestMethod]
        public void BasicConstantsUnSymmetrical()
        {
            var o = g.Ones(new TFShape(4, 3));
            Assert.IsNotNull(o);
            Assert.AreEqual(o.OutputType, TFDataType.Double);

            var r = g.RandomNormal(new TFShape(3, 5));
            Assert.IsNotNull(o);
            Assert.AreEqual(o.OutputType, TFDataType.Double);
            
            //expect incompatible shapes
            Assert.ThrowsException<TFException>(() => s.GetRunner().Run(g.Mul(o, r)));
            
            var res = s.GetRunner().Run(g.MatMul(o, r));
            Assert.IsNotNull(res);
            Assert.AreEqual(res.TensorType, TFDataType.Double);
            Assert.AreEqual(res.NumDims, 2);
            Assert.AreEqual(res.Shape[0], 4);
            Assert.AreEqual(res.Shape[1], 5);

            double[,] val = (double[,])res.GetValue();
            Assert.IsNotNull(val);
            Assert.IsInstanceOfType(val, typeof(double[,]));
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    Assert.IsNotNull(((double[,])val)[i, j]);
                }
            }
            
        }

        [TestMethod]
        public void BasicMultidimensionalArray()
        {
            var var_a = g.Placeholder(TFDataType.Int32);
            var mul = g.Mul(var_a, g.Const(2));

            var a = new int[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 5 }, { 6, 7 } } };
            var res = s.GetRunner().AddInput(var_a, a).Fetch(mul).Run()[0];

            Assert.IsNotNull(res);
            Assert.AreEqual(res.TensorType, TFDataType.Int32);
            Assert.AreEqual(res.NumDims, 3);
            Assert.AreEqual(res.Shape[0], 2);
            Assert.AreEqual(res.Shape[1], 2);
            Assert.AreEqual(res.Shape[2], 2);

            var actual = (int[,,])res.GetValue();
            var expected = new int[,,] { { { 0, 2 }, { 4, 6 } }, { { 8, 10 }, { 12, 14 } } };

            Assert.IsTrue(expected.Cast<int>().SequenceEqual(actual.Cast<int>()));
        }

        [TestMethod]
        public void BasicMatrix()
        {
            // 1x2 matrix
            var matrix1 = g.Const(new double[,] { { 3, 3 } });
            // 2x1 matrix
            var matrix2 = g.Const(new double[,] { { 2 }, { 2 } });

            var expected = new double[,] { { 12 } };

            var res = s.GetRunner().Run(g.MatMul(matrix1, matrix2));

            Assert.IsNotNull(res);
            Assert.AreEqual(res.TensorType, TFDataType.Double);
            Assert.AreEqual(res.NumDims, 2);
            Assert.AreEqual(res.Shape[0], 1);
            Assert.AreEqual(res.Shape[1], 1);

            double[,] val = (double[,])res.GetValue();
            Assert.IsNotNull(val);
            Assert.IsInstanceOfType(val, typeof(double[,]));

            Assert.AreEqual(val[0, 0], expected[0,0]);

        }

    }
}