using System;
using System.Collections.Generic;
using System.Linq;
using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests
{
    /// <summary>
    /// These are basic tests based on the Tensorflow gettings started page
    /// https://www.tensorflow.org/get_started/get_started
    /// </summary>

    public class GettingStartedTests 
	{
        [Fact]
        public void Version()
        {
            Assert.NotNull(TFCore.Version);
            Assert.IsType<string>(TFCore.Version);
        }

        [Fact]
        public void Basic()
        {
            using (TFGraph g = new TFGraph ())
			using (TFSession s = new TFSession (g)) 
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
                Assert.Equal(0, rank0.NumDims);
                Assert.Equal("3", rank0.ToString());


                TFTensor rank1 = new double[] { 1.0, 2.0, 3.0 };//shape [3]                
                Assert.Equal(1, rank1.NumDims);
                Assert.Equal("[3]", rank1.ToString());

                TFTensor rank2 = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };//shape[2,3]
                Assert.Equal(2,rank2.NumDims);
                Assert.Equal("[2x3]", rank2.ToString());

                TFTensor rank3 = new double[,,]
                {
                        { { 1, 2, 3 } },
                        { { 4, 5, 6 } }
                };//shape [2, 1, 3] 
                Assert.Equal(3, rank3.NumDims);
                Assert.Equal("[2x1x3]", rank3.ToString());
                
                //node1 = tf.constant(3.0, tf.float32)
                TFOutput node1 = g.Const(3.0F, TFDataType.Float);
                Assert.NotNull(node1);
                Assert.Equal(TFDataType.Float, node1.OutputType);
                //node2 = tf.constant(4.0) # also tf.float32 implicitly
                TFOutput node2 = g.Const(4.0F);
                Assert.NotNull(node2);
                Assert.Equal(TFDataType.Float, node1.OutputType);

                TFTensor[] results = runner.Fetch(new TFOutput[] { node1, node2 }).Run();
                Assert.NotNull(results);
                Assert.Equal(2, results.Length);
                Assert.Equal(3.0F, results[0].GetValue());
                Assert.Equal(4.0F, results[1].GetValue());
                
                TFOutput node3 = g.Add(node1, node2);
                Assert.NotNull(node3);
                TFTensor result = runner.Run(node3);
                Assert.NotNull(result);
                Assert.Equal(7.0F, result.GetValue());

                TFOutput a = g.Placeholder(TFDataType.Float);
                Assert.NotNull(a);

                TFOutput b = g.Placeholder(TFDataType.Float);
                Assert.NotNull(b);

                TFOutput adder_node = g.Add(a, b);
                Assert.NotNull(adder_node);

                result = runner.AddInput(a, 3.0F).AddInput(b, 4.5F).Run(adder_node);
                Assert.NotNull(result);
                Assert.Equal(7.5F, result.GetValue());

                runner = s.GetRunner();
                result = runner.AddInput(a, new float[] { 1F, 3F }).AddInput(b, new float[] { 2F, 4F }).Run(adder_node);
                Assert.NotNull(result);
                Assert.Equal(1, result.NumDims);
                Assert.Equal(3.0F, ((float[])result.GetValue())[0]);
                Assert.Equal(7.0F, ((float[])result.GetValue())[1]);
                
                var add_and_triple = g.Mul(g.Const(3.0F, TFDataType.Float), adder_node);
                Assert.NotNull(add_and_triple);

                runner = s.GetRunner();
                result = runner.AddInput(a, 3.0F).AddInput(b, 4.5F).Run(add_and_triple);
                Assert.NotNull(result);
                Assert.Equal(0, result.NumDims);
                Assert.Equal(22.5F, result.GetValue());

                runner = s.GetRunner();
                result = runner.AddInput(a, new float[] { 1F, 3F }).AddInput(b, new float[] { 2F, 4F }).Run(add_and_triple);
                Assert.NotNull(result);
                Assert.Equal(1, result.NumDims);
                Assert.Equal(9.0F, ((float[])result.GetValue())[0]);
                Assert.Equal(21.0F, ((float[])result.GetValue())[1]);
            }
        }

        [Fact]
        public void Variables()
        {
            using (TFGraph g = new TFGraph ())
			using (TFSession s = new TFSession (g)) 
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

                Assert.NotNull(result);
                Assert.Equal(1, result.Length);
                Assert.IsType<float[]>(result[0].GetValue());

                float[] values = (float[])result[0].GetValue();
                Assert.NotNull(values);
                Assert.Equal(4, values.Length);
                Assert.Equal(0.0F, values[0], 7);
                Assert.Equal(0.3F, values[1], 7);
                Assert.Equal(0.6F, values[2], 7);
                Assert.Equal(0.9F, values[3], 7);
            }
        }

        [Fact]
        public void BasicConstantOps()
        {
            using (TFGraph g = new TFGraph ())
			using (TFSession s = new TFSession (g)) 
			{
                var a = g.Const(2);
                Assert.NotNull(a);
                Assert.Equal(TFDataType.Int32, a.OutputType);

                var b = g.Const(3);
                Assert.NotNull(b);
                Assert.Equal(TFDataType.Int32, b.OutputType);

                // Add two constants
                var results = s.GetRunner().Run(g.Add(a, b));

                Assert.NotNull(results);
                Assert.IsType<TFTensor>(results);
                Assert.Equal(TFDataType.Int32, results.TensorType);
                Assert.Equal(0, results.NumDims);
                Assert.Equal(0, results.Shape.Length);

                var val = results.GetValue();
                Assert.NotNull(val);
                Assert.Equal(5, val);

                // Multiply two constants
                results = s.GetRunner().Run(g.Mul(a, b));
                Assert.NotNull(results);
                Assert.IsType<TFTensor>(results);
                Assert.Equal(TFDataType.Int32, results.TensorType);
                Assert.Equal(0, results.NumDims);
                Assert.Equal(0, results.Shape.Length);
                Assert.NotNull(results.GetValue());
                Assert.Equal(6, results.GetValue());
            }
        }

        [Fact]
        public void BasicConstantZerosAndOnes()
        {
            using (TFGraph g = new TFGraph ())
			using (TFSession s = new TFSession (g)) 
			{
                // Test Zeros, Ones for n x n shape
                var o = g.Ones(new TFShape(4, 4));
                Assert.NotNull(o);
                Assert.Equal(TFDataType.Double, o.OutputType);

                var z = g.Zeros(new TFShape(4, 4));
                Assert.NotNull(z);
                Assert.Equal(TFDataType.Double, z.OutputType);

                var r = g.RandomNormal(new TFShape(4, 4));
                Assert.NotNull(r);
                Assert.Equal(TFDataType.Double, r.OutputType);

                var res1 = s.GetRunner().Run(g.Mul(o, r));
                Assert.NotNull(res1);
                Assert.Equal(TFDataType.Double, res1.TensorType);
                Assert.Equal(2, res1.NumDims);
                Assert.Equal(4, res1.Shape[0]);
                Assert.Equal(4, res1.Shape[1]);
                Assert.Equal("[4x4]", res1.ToString());

                var matval1 = res1.GetValue();
                Assert.NotNull(matval1);
                Assert.IsType<double[,]>(matval1);
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        Assert.NotNull(((double[,])matval1)[i, j]);
                    }
                }

                var res2 = s.GetRunner().Run(g.Mul(g.Mul(o, r), z));
                Assert.NotNull(res2);
                Assert.Equal(TFDataType.Double, res2.TensorType);
                Assert.Equal(2, res2.NumDims);
                Assert.Equal(4, res2.Shape[0]);
                Assert.Equal(4, res2.Shape[1]);
                Assert.Equal("[4x4]", res2.ToString());

                var matval2 = res2.GetValue();
                Assert.NotNull(matval2);
                Assert.IsType<double[,]>(matval2);
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        Assert.NotNull(((double[,])matval2)[i, j]);
                        Assert.Equal(0.0, ((double[,])matval2)[i, j]);
                    }
                }
            }
        }

        [Fact]
        public void BasicConstantsOnSymmetricalShapes()
        {
            using (TFGraph g = new TFGraph ())
			using (TFSession s = new TFSession (g)) 
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

                Assert.NotNull(res1);
                Assert.Equal(TFDataType.Double, res1.TensorType);
                Assert.Equal(2, res1.NumDims);
                Assert.Equal(4, res1.Shape[0]);
                Assert.Equal(4, res1.Shape[1]);
                Assert.Equal("[4x4]", res1.ToString());

                var cmatval1 = res1.GetValue();
                Assert.NotNull(cmatval1);
                Assert.IsType<double[,]>(cmatval1);
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        Assert.NotNull(((double[,])cmatval1)[i, j]);                    
                    }
                }

                var cres2 = s.GetRunner().Run(g.Mul(g.Mul(co, r), cz));

                Assert.NotNull(cres2);
                Assert.Equal(TFDataType.Double, cres2.TensorType);
                Assert.Equal(2, cres2.NumDims);
                Assert.Equal(4, cres2.Shape[0]);
                Assert.Equal(4, cres2.Shape[1]);
                Assert.Equal("[4x4]", cres2.ToString());

                var cmatval2 = cres2.GetValue();
                Assert.NotNull(cmatval2);
                Assert.IsType<double[,]>(cmatval2);
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        Assert.NotNull(((double[,])cmatval2)[i, j]);
                        Assert.Equal(((double[,])matvalzero)[i, j], ((double[,])cmatval2)[i, j]);
                    }
                }
            }
        }

        [Fact]
        public void BasicConstantsUnSymmetrical()
        {
            using (TFGraph g = new TFGraph ())
			using (TFSession s = new TFSession (g)) 
			{
                var o = g.Ones(new TFShape(4, 3));
                Assert.NotNull(o);
                Assert.Equal(TFDataType.Double, o.OutputType);

                var r = g.RandomNormal(new TFShape(3, 5));
                Assert.NotNull(o);
                Assert.Equal(TFDataType.Double, o.OutputType);
                
                //expect incompatible shapes
                Assert.Throws<TFException>(() => s.GetRunner().Run(g.Mul(o, r)));
                
                var res = s.GetRunner().Run(g.MatMul(o, r));
                Assert.NotNull(res);
                Assert.Equal(TFDataType.Double, res.TensorType);
                Assert.Equal(2, res.NumDims);
                Assert.Equal(4, res.Shape[0]);
                Assert.Equal(5, res.Shape[1]);

                double[,] val = (double[,])res.GetValue();
                Assert.NotNull(val);
                Assert.IsType<double[,]>(val);
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 5; j++)
                    {
                        Assert.NotNull(((double[,])val)[i, j]);
                    }
                }
            }
        }

        [Fact]
        public void BasicMultidimensionalArray()
        {
            using (TFGraph g = new TFGraph ())
			using (TFSession s = new TFSession (g)) 
			{
                var var_a = g.Placeholder(TFDataType.Int32);
                var mul = g.Mul(var_a, g.Const(2));

                var a = new int[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 5 }, { 6, 7 } } };
                var res = s.GetRunner().AddInput(var_a, a).Fetch(mul).Run()[0];

                Assert.NotNull(res);
                Assert.Equal(TFDataType.Int32, res.TensorType);
                Assert.Equal(3, res.NumDims);
                Assert.Equal(2, res.Shape[0]);
                Assert.Equal(2, res.Shape[1]);
                Assert.Equal(2, res.Shape[2]);

                var actual = (int[,,])res.GetValue();
                var expected = new int[,,] { { { 0, 2 }, { 4, 6 } }, { { 8, 10 }, { 12, 14 } } };

                Assert.Equal(expected.Cast<int>(), actual.Cast<int>());
            }
        }

        [Fact]
        public void BasicMatrix()
        {
            using (TFGraph g = new TFGraph ())
			using (TFSession s = new TFSession (g)) 
			{
                // 1x2 matrix
                var matrix1 = g.Const(new double[,] { { 3, 3 } });
                // 2x1 matrix
                var matrix2 = g.Const(new double[,] { { 2 }, { 2 } });

                var expected = new double[,] { { 12 } };

                var res = s.GetRunner().Run(g.MatMul(matrix1, matrix2));

                Assert.NotNull(res);
                Assert.Equal(TFDataType.Double, res.TensorType);
                Assert.Equal(2, res.NumDims);
                Assert.Equal(1, res.Shape[0]);
                Assert.Equal(1, res.Shape[1]);

                double[,] val = (double[,])res.GetValue();
                Assert.NotNull(val);
                Assert.IsType<double[,]>(val);

                Assert.Equal(expected[0,0], val[0, 0]);
            }
        }

    }
}