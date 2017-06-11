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

    [TestClass]
    public class GettingStarted
    {
        [ClassInitialize]
        public static void ClassInit(TestContext context)
        {
            //need TF# on Windows with GPU support?
            //find _pywrap_tensorflow_internal.pyd in the tensorflow_gpu-1.1.0-cp36-cp36m-win_amd64.whl
            //the location in the whl is here: \tensorflow_gpu-1.1.0.data\purelib\tensorflow\python\
            //extract, copy and rename _pywrap_tensorflow_internal.pyd to libtensorflow.dll
            //copy to you TF# folder
            //if running with libtensorflow.dll where Python 3.5 environment is needed - append the py35 environment to the path            
            //see https://github.com/larcai/cognibooks/blob/master/tensorflow/1_tf_setup.workbook 
            if (Environment.OSVersion.Platform == PlatformID.Unix) return;
            if (Environment.OSVersion.Platform == PlatformID.MacOSX) return;
            
            var envpaths = new List<string> { @"C:\ProgramData\Anaconda3\envs\py35" }
                .Union(Environment.GetEnvironmentVariable("PATH").Split(new char[] { ';' }, StringSplitOptions.RemoveEmptyEntries));
            Environment.SetEnvironmentVariable("PATH", string.Join(";", envpaths));
        }

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
            Assert.IsNotNull(node1.OutputType == TFDataType.Float);
            //node2 = tf.constant(4.0) # also tf.float32 implicitly
            TFOutput node2 = g.Const(4.0F);
            Assert.IsNotNull(node2);
            Assert.IsNotNull(node2.OutputType == TFDataType.Float);

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
            var b = g.Const(3);
            //Console.WriteLine("a=2 b=3");

            // Add two constants
            var results = s.GetRunner().Run(g.Add(a, b));
            var val = results.GetValue();
            //Console.WriteLine("a+b={0}", val);

            // Multiply two constants
            results = s.GetRunner().Run(g.Mul(a, b));
            //Console.WriteLine("a*b={0}", results.GetValue());

            // Test Zeros, Ones
            var o = g.Ones(new TFShape(4, 4));
            var r = g.RandomNormal(new TFShape(4, 4));
            var z = g.Zeros(new TFShape(4, 4));
            var m = g.Mul(o, r);
            var res1 = s.GetRunner().Run(m);
            var res2 = s.GetRunner().Run(g.Mul(g.Mul(o, r), z));

            //Test Constants
            var co = g.Constant(1.0, new TFShape(4, 4), TFDataType.Double);
            var cz = g.Constant(0.0, new TFShape(4, 4), TFDataType.Double);
            var cr = g.RandomNormal(new TFShape(4, 4));
            var cm = g.Mul(co, cr);
            var cres1 = s.GetRunner().Run(cm);
            var cres2 = s.GetRunner().Run(g.Mul(g.Mul(co, cr), cz));

            var so = g.Ones(new TFShape(4, 3));
            var sr = g.RandomNormal(new TFShape(3, 5));
            var sz = g.Zeros(new TFShape(5, 6));
            var sm = g.MatMul(so, sr);
            var sres1 = s.GetRunner().Run(sm);
            var sres2 = s.GetRunner().Run(g.MatMul(g.MatMul(so, sr), sz));

            // TODO: API-wise, perhaps session.Run () can have a simple
            // overload where we only care about the fetched values, 
            // making the above:
            // s.Run (g.Mul (a, b));

        }

    }
}
