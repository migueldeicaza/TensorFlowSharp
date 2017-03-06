using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorFlow
{
	public partial class TFGraph
	{
		/// <summary>
		/// Creates a constant operation from a TFTensor or constant
		/// </summary>
		/// <param name="value">Value.</param>
		/// <param name="operName">Oper name.</param>
		/// <remarks>
		/// Since TFTensor have implicit conversion operators, you can call this method with
		/// a constant like this: graph.Const (23)
		/// </remarks>
		public TFOutput Const (TFTensor value, string operName = null)
		{
			return Const (value, value.TensorType, operName);
		}

		// Returns range(0, rank(x)) if reduction_indices is null
		TFOutput ReduceDims (TFOutput input, TFOutput? axis = null)
		{
			if (axis.HasValue)
				return axis.Value;

			// Fast path: avoid creating Rank and Range ops if ndims is known.
			var shape = GetTensorShape (input);
			if (shape.Length >= 0) {
				// The python code distinguishes between tensor and sparsetensor

				var array = new int [shape.Length];
				for (int i = 0; i < array.Length; i++)
					array [i] = i;

				return this.Const (array, TFDataType.Int32);                   
			}
			return Range (Const (0), Const (shape.Length), Const (1));
		}

		/// <summary>
		/// Computes the sum of elements across dimensions of a tensor.
		/// </summary>
		/// <returns>The reduced tensor.</returns>
		/// <param name="input">The tensor to reduce. Should have numeric type.</param>
		/// <param name="axis">The dimensions to reduce. If not se (the default), reduces all dimensions.</param>
		/// <param name="keep_dims">If set to <c>true</c> retains reduced dimensions with length 1.</param>
		/// <param name="operName">A name for the operation, optional.</param>
		/// <remarks>
		///   Reduces input_tensor along the dimensions given in axis.
		/// Unless keep_dims is true, the rank of the tensor is reduced by 1 for each
		/// entry in axis. If keep_dims is true, the reduced dimensions
		/// are retained with length 1.
		/// 
		/// If axis has no entries, all dimensions are reduced, and a
		/// tensor with a single element is returned.
		/// </remarks>
		public TFOutput ReduceSum (TFOutput input, TFOutput? axis = null, bool? keep_dims = false, string operName = null)
		{
			return Sum (input, this.ReduceDims (input, axis), keep_dims, operName);
		}

		/// <summary>
		/// Variable node, with a starting initial value.
		/// </summary>
		/// <param name="initialValue">Initial value.</param>
		/// <param name="init">Returns the operation that initializes the value of the variable.</param>
		/// <param name="value">Returns the value of the variable.</param>
		/// <param name="operName">Operation name, optional.</param>
		/// <returns>The returning TFOutput returns the handle to the variable.</returns>
		/// <remarks>
		/// Variables need to be initialized before the main execution so you will typically want to
		/// run the session on the variable
		/// </remarks>
		public TFOutput Variable (TFOutput initialValue, out TFOperation init, out TFOutput value, string operName = null)
		{
			var scopeName = MakeName ("Variable", operName);

			using (var newScope = WithScope (scopeName)) {
				var type = initialValue.OutputType;
				var handle = VarHandleOp (type, new TFShape (GetShape (initialValue)));
				using (var aScope = WithScope ("Assign")) {
					init = AssignVariableOp (handle, initialValue);
					using (var rScope = WithScope ("Read")) {
						value = ReadVariableOp (handle, type);
						return handle;
					}
				}
			}
		}

		List<TFOperation> pending_init_variables;
		public void AddInitVariable (TFOperation variable)
		{
			if (pending_init_variables == null)
				pending_init_variables = new List<TFOperation> ();
			pending_init_variables.Add (variable);
		}

		public TFOperation [] GetGlobalVariablesInitializer ()
		{
			var res = pending_init_variables.ToArray ();
			pending_init_variables.Clear ();
			return res;
		}

		/// <summary>
		/// Variable node, with a starting initial value.  Convenience that registers the init variable to a global queue.
		/// </summary>
		/// <param name="initialValue">Initial value.</param>
		/// <param name="value">Returns the value of the variable.</param>
		/// <param name="operName">Operation name, optional.</param>
		/// <returns>The returning TFOutput returns the handle to the variable.</returns>
		/// <remarks>
		/// Variables need to be initialized before the main execution so you will typically want to
		/// run the session on the variable.
		/// 
		/// The init sequence for the variable is stored in the graph, you must manually initialize 
		/// those by running the session on the global variables.
		/// </remarks>
		public TFOutput Variable (TFOutput initialValue, out TFOutput value, string operName = null)
		{
			var scopeName = MakeName ("Variable", operName);

			using (var newScope = WithScope (scopeName)) {
				var type = initialValue.OutputType;
				var handle = VarHandleOp (type, new TFShape (GetShape (initialValue)));
				using (var aScope = WithScope ("Assign")) {
					var init = AssignVariableOp (handle, initialValue);
					AddInitVariable (init);
					using (var rScope = WithScope ("Read")) {
						value = ReadVariableOp (handle, type);
						return handle;
					}
				}
			}
		}

		/// <summary>
		/// Variable node, with a starting initial value.  Convenience that registers the init variable to a global queue.
		/// </summary>
		/// <param name="initialValue">Initial value.</param>
		/// <param name="operName">Operation name, optional.</param>
		/// <returns>The returning TFOutput returns the handle to the variable.</returns>
		/// <remarks>
		/// Variables need to be initialized before the main execution so you will typically want to
		/// run the session on the variable.
		/// 
		/// The init sequence for the variable is stored in the graph, you must manually initialize 
		/// those by running the session on the global variables.
		/// </remarks>
		public TFOutput Variable (TFOutput initialValue, string operName = null)
		{
			var scopeName = MakeName ("Variable", operName);

			using (var newScope = WithScope (scopeName)) {
				var type = initialValue.OutputType;
				var handle = VarHandleOp (type, new TFShape (GetShape (initialValue)));
				using (var aScope = WithScope ("Assign")) {
					var init = AssignVariableOp (handle, initialValue);
					AddInitVariable (init);
					return handle;
				}
			}
		}

		//
		// Converts a shape to a tensor
		TFTensor ShapeTensor (TFShape shape)
		{
			Array a;

			if (shape.IsLongArray)
				a = shape.ToArray ();
			else 
				a = shape.ToIntArray ();

			return (TFTensor)a;
		}

		/// <summary>
		/// Outputs random values from a normal distribution
		/// </summary>
		/// <returns>A tensor of the specified shape filled with random normal values.</returns>
		/// <param name="shape">Shape of the output tensor.</param>
		/// <param name="mean">The mean of the standard distribution.</param>
		/// <param name="stddev">The standard deviation of the normal distribution.</param>
		/// <param name="seed">Integer seed used for the random distribution, using the TensorFlow SetRandomSeed .</param>
		/// <param name="operName">>Operation name, optional.</param>
		public TFOutput RandomNormal (TFShape shape, double mean = 0, double stddev = 1, int? seed = null, string operName = null)
		{
			var scopeName = MakeName ("RandomNormal", operName);

			using (var newScope = WithScope (scopeName)) {
				var st = ShapeTensor (shape);
				var tmean = Const (mean, "mean");
				var tstddev = Const (stddev, "stddev");

				//seed1, seed2 = random_seed.get_seed (seed)
    				//rnd = gen_random_ops._random_standard_normal (
				//shape_tensor, dtype, seed = seed1, seed2 = seed2)
    				//mul = rnd * stddev_tensor
    				//value = math_ops.add (mul, mean_tensor, name = name)
    				// return value
			}
			throw new NotImplementedException ();
		}
	}
}
