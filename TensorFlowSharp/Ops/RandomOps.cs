//
// RandomOps: operations for generating random numbers
//
// Authors:
//   Miguel de Icaza
//
// This is a port of the Python code in tensorflow
//
// TODO: need an overload that allows minval/maxval to be float16, float32, float64, bfloat16, int32, int64
// Or perhaps TFTensor overload.
//
// Some of our operations only take doubles, they should either take tensors, or objects that can be 
// converted to it.

using System;
namespace TensorFlow
{
	public partial class TFGraph
	{

		/// <summary>
		/// Outputs random values from a normal distribution
		/// </summary>
		/// <returns>A tensor of the specified shape filled with random normal values.</returns>
		/// <param name="shape">Shape of the output tensor.</param>
		/// <param name="mean">The mean of the standard distribution.</param>
		/// <param name="stddev">The standard deviation of the normal distribution.</param>
		/// <param name="seed">Integer seed used for the random distribution, using the TensorFlow SetRandomSeed .</param>
		/// <param name="operName">Operation name, optional.</param>
		public TFOutput RandomNormal (TFShape shape, double mean = 0, double stddev = 1, int? seed = null, string operName = null)
		{
			var scopeName = MakeName ("RandomNormal", operName);

			using (var newScope = WithScope (scopeName)) {
				var shapeTensor = ShapeTensorOutput (shape);

				var tmean = Const (mean, "mean");
				var tstddev = Const (stddev, "stddev");

				int graph, local;
				GetRandomSeeds (seed, out graph, out local);

				var rnd = RandomStandardNormal (shapeTensor, TFDataType.Double, graph, local);
				var mul = Mul (rnd, tstddev);
				return Add (mul, tmean);
			}
		}

		/// <summary>
		/// Randoms the uniform.
		/// </summary>
		/// <returns>The uniform.</returns>
		/// <param name="shape">Shape.</param>
		/// <param name="minval">Minval.</param>
		/// <param name="maxval">Maxval.</param>
		/// <param name="seed">Seed.</param>
		/// <param name="operName">Oper name.</param>
		public TFOutput RandomUniform (TFShape shape, double minval = 0, double maxval = 1, int? seed = null, string operName = null)
		{
			using (var scope = WithScope (MakeName ("random_uniform", operName))) {
				var shapeTensor = ShapeTensorOutput (shape);
				var minvalTensor = Const (minval, "minval");
				var maxvalTensor = Const (maxval, "maxval");

				int graph, local;
				GetRandomSeeds (seed, out graph, out local);

				var rnd = RandomUniform (shapeTensor, TFDataType.Double, graph, local);
				var mul = Mul (rnd, Sub (maxvalTensor, minvalTensor));
				return Add (mul, minvalTensor);
			}
		}

	}
}
