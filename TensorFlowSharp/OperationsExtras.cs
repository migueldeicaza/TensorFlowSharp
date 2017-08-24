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

		/// <summary>
		/// Registers a specified variable as an initialization variable.
		/// </summary>
		/// <param name="variable">Variable to register.</param>
		/// <remarks>
		/// <para>
		/// This is a convenience method to track the variables that need to be initialized in the graph,
		/// you can retrieve the list of all those variables by calling the <see cref="M:TensorFlow.TFGraph.GetGlobalVariablesInitializer"/>
		/// which will return this list and clear the state at that point.
		/// </para>
		/// <para>
		/// You typically use this method from helper methods to register all the variables that you want
		/// initialized, and a higher level method will retrieve all these variables and initialize them
		/// at their convenience.
		/// </para>
		/// </remarks>
		public void AddInitVariable (TFOperation variable)
		{
			if (pending_init_variables == null)
				pending_init_variables = new List<TFOperation> ();
			pending_init_variables.Add (variable);
		}

		/// <summary>
		/// Gets the list of all registered global variables.
		/// </summary>
		/// <returns>The array of variables that should be initialized.</returns>
		/// <remarks>
		/// After this method is invoked the list of pending initialization variables
		/// is cleared.
		/// </remarks>
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
		// Converts a shape to a tensor, to a TFOutput
		//
		TFOutput ShapeTensorOutput (TFShape shape)
		{
			if (shape.IsLongArray)
				return Const (shape.ToArray (), TFDataType.Int64);
			else
				return Const (shape.ToIntArray (), TFDataType.Int32);
		}

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
		/// Gets or sets the graph random seed, see remarks for details.
		/// </summary>
		/// <value>The seed.</value>
		/// <remarks>
		///  Operations that rely on a random seed actually derive it from two seeds:
		///  the graph-level and operation-level seeds.This sets the graph-level seed.
		///
		/// Its interactions with operation-level seeds is as follows:
		/// 1. If neither the graph-level nor the operation seed is set:
		///    A random seed is used for this op.
		/// 2. If the graph-level seed is set, but the operation seed is not:
		///    The system deterministically picks an operation seed in conjunction
		///    with the graph-level seed so that it gets a unique random sequence.
		/// 3. If the graph-level seed is not set, but the operation seed is set:
		///    A default graph-level seed and the specified operation seed are used to
		///    determine the random sequence.
		/// 4. If both the graph-level and the operation seed are set:
		///    Both seeds are used in conjunction to determine the random sequence.
		/// </remarks>
		public int? Seed { get; set; }

		/// <summary>
		/// Returns the graph and local seeds based on an optionally set incoming seed value.
		/// </summary>
		/// <param name="operationSeed">The seed value that might be set.</param>
		/// <param name="graphSeed">Returned graph seed.</param>
		/// <param name="localSeed">Returned local seed.</param>
		/// <remarks>
		/// This helper function returns two seeds derived from graph-level and op-level seeds.
		/// Many random operations internally use the two seeds to allow user to change 
		/// the seed globally for a graph, or for only specific operations.
		/// </remarks>
		public void GetRandomSeeds (int? operationSeed, out int graphSeed, out int localSeed)
		{
			if (Seed.HasValue) {
				graphSeed = Seed.Value;
				if (operationSeed.HasValue)
					localSeed = operationSeed.Value;
				else
					localSeed = LastId;
			} else {
				graphSeed = 87654321;
				if (operationSeed.HasValue) {
					localSeed = operationSeed.Value;
				} else {
					localSeed = 0;
				}					
			}
		}

        /// <summary>
        /// Computes dropout. 
        /// </summary>
        /// <param name="x">A tensor.</param>
        /// <param name="keep_prob">A scalar Tensor with the same type as x. The probability that each element is kept.</param>
        /// <param name="noise_shape">A 1-D Tensor of type int32, representing the shape for randomly generated keep/drop flags.</param>
        /// <param name="seed">Integer seed used for the random distribution, using the TensorFlow SetRandomSeed .</param>
        /// <param name="operName">Operation name, optional.</param>
        /// <remarks>
        /// With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, 
        /// otherwise outputs 0. The scaling is so that the expected sum is unchanged.
        /// </remarks>
        public TFOutput Dropout (TFOutput x, TFOutput keep_prob, TFShape noise_shape = null, int? seed= null, string operName= null)
        {
            var scopeName = MakeName("dropout", operName);

            using (var newScope = WithScope(scopeName)) {
                if (noise_shape == null)
                    noise_shape = new TFShape(GetShape(x));

                TFOutput shapeTensor = ShapeTensorOutput(noise_shape);

                // uniform [keep_prob, 1.0 + keep_prob)
                TFOutput random_tensor = keep_prob;
                random_tensor = Add(random_tensor, RandomUniform(shapeTensor, seed: seed, dtype: x.OutputType));

                // 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
                TFOutput binary_tensor = Floor(random_tensor);
                TFOutput ret = Mul(Div(x, keep_prob) , binary_tensor);
                SetTensorShape(ret, GetShape(x));
                return ret;
            }
        }

        /// <summary>
        /// Computes dropout. 
        /// </summary>
        /// <param name="x">A tensor.</param>
        /// <param name="keep_prob">A scalar Tensor with the same type as x. The probability that each element is kept.</param>
        /// <param name="noise_shape">A 1-D Tensor of type int32, representing the shape for randomly generated keep/drop flags.</param>
        /// <param name="seed">Integer seed used for the random distribution, using the TensorFlow SetRandomSeed .</param>
        /// <param name="operName">Operation name, optional.</param>
        /// <remarks>
        /// With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, 
        /// otherwise outputs 0. The scaling is so that the expected sum is unchanged.
        /// </remarks>
        public TFOutput Dropout (TFOutput x, double keep_prob, TFShape noise_shape = null, int? seed = null, string operName = null)
        {
            if (keep_prob < 0 || keep_prob >= 1)
                throw new ArgumentOutOfRangeException("keep_prob must be a scalar tensor or a float in the range (0, 1], got " + keep_prob);

            if (keep_prob == 1)
                return x;

            var scopeName = MakeName("dropout", operName);
            using (var newScope = WithScope(scopeName)) {
                var tkeep_prob = Const(keep_prob);
                return Dropout(x, tkeep_prob, noise_shape, seed, operName);
            }
        }
    }

}
