﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace TensorFlow
{
    /// <summary>
    /// Base class for all the optimizers.
    /// </summary>
    public abstract class Optimizer
    {

        /// <summary>
        /// Varaible to keep track of number of iterations (mini-batch processed)
        /// </summary>
        public Variable Iterations { get; }

        private readonly string _lrName = "LearningRate";
        /// <summary>
        /// Variable to keep track of the learning rate.
        /// </summary>
        public Variable LearningRate { get; }

        /// <summary>
        /// The graph object. It is used for creating Ops through the construction of optimizer.
        /// </summary>
        protected readonly TFGraph _graph;

        /// <summary>
        /// Name the optimization operation in the graph.
        /// All the operation will be created under this name scope.
        /// </summary>
        protected readonly string _optimizerName;

        /// <summary>
        /// List to hold all the operations which are updated on each iteration of optimizer.
        /// </summary>
        protected readonly IList<TFOperation> _updateOps = new List<TFOperation>();

        private float _initialAccumulatorValue;
        /// <summary>
        /// Construct optimizer.
        /// </summary>
        /// <param name="graph">The graph object.</param>
        /// <param name="operName">Name of the operation.</param>
        /// <param name="learningRate">The learning rate for the SGD update.</param>
        /// <param name="decay">Learning rate decay over each update.</param>
        /// /// <param name="initialAccumulatorValue">A floating point value. Starting value for the accumulators, must be >=0.</param>
        public Optimizer(TFGraph graph, string operName, float learningRate, float decay, float initialAccumulatorValue)
        {
            if (initialAccumulatorValue < 0)
                throw new ArgumentException($"Value must be positive. initialAccumulatorValue = {initialAccumulatorValue}");

            _graph = graph;
            _optimizerName = operName;
            _initialAccumulatorValue = initialAccumulatorValue;
            using (var scope = _graph.WithScope(_optimizerName))
            {
                Iterations = _graph.Variable(_graph.Const(new TFTensor(0L)), trainable: false, operName: "iterations");
                _updateOps.Add(_graph.AssignAddVariableOp(Iterations, _graph.Const(1L)));
                var initialLearningRate = _graph.Const(learningRate);
                LearningRate = _graph.Variable(initialLearningRate, trainable: false, operName: _lrName);
                CreateDecayOps(decay, initialLearningRate);
            }
        }

        /// <summary>
        /// Create learning rate time decay operation.
        /// </summary>
        protected void CreateDecayOps(float decay, TFOutput initialLearningRate)
        {
            if (decay > 0)
            {
                var _decay = _graph.Const(decay, "Decay");
                var one = _graph.Const(1f);
                _updateOps.Add(_graph.AssignVariableOp(LearningRate,
                    _graph.Mul(initialLearningRate,
                                _graph.Div(one,
                                            _graph.Add(one,
                                                        _graph.Mul(_decay,
                                                                    _graph.Cast(Iterations.Read, _decay.OutputType)
                                                                  )
                                                       )
                                           )
                               )));
            }
        }

        /// <summary>
        /// Initialize the accumulators
        /// </summary>
        protected TFOutput[] InitMoments((TFOutput gradient, Variable variable)[] gradientsAndVariables)
        {
            var accumulators = new TFOutput[gradientsAndVariables.Length];
            for (int i = 0; i < gradientsAndVariables.Length; i++)
            {
                var gv = gradientsAndVariables[i];
                var varType = gv.variable.Read.OutputType;
                var varShape = _graph.GetTensorShape(gv.variable.Read);
                accumulators[i] = _graph.VariableV2(varShape, varType);
                _graph.AddInitVariable(_graph.Assign(accumulators[i], _graph.Constant(_initialAccumulatorValue, varShape, varType)).Operation);
            }
            return accumulators;
        }

        /// <summary>
        /// Computes the gradient of the trainable variables in the graph.
        /// </summary>
        /// <param name="loss">The loss operation to compute the gradient on.</param>
        /// <param name="varList">list of variable to compute the gradients for.
        /// If null the gradient is computed for all the trainable variables in the graph.</param>
        /// <param name="colocateGradientsWithOps">Place the gradient op on the same device as variable.</param>
        /// <returns>A list of (gradient, variable) pairs. Variable is always present, but
        /// gradient can be `None`.</returns>
        public virtual (TFOutput gradient, Variable variable)[] ComputeGradient(TFOutput loss, Variable[] varList = null, bool colocateGradientsWithOps = false)
        {
            varList = varList ?? _graph.GetTrainableVariables();
            var gradientsAndVariables = new (TFOutput gradient, Variable variable)[varList.Length];
            for (int i = 0; i < varList.Length; i++)
            {
                gradientsAndVariables[i].variable = varList[i];
                gradientsAndVariables[i].gradient = _graph.AddGradients(new TFOutput[] { loss }, new TFOutput[] { varList[i].Read })[0];
                if (colocateGradientsWithOps)
                {
                    var desc = new TFOperationDesc(_graph, gradientsAndVariables[i].gradient.Operation.OpType, gradientsAndVariables[i].gradient.Operation.Name);
                    desc.ColocateWith(gradientsAndVariables[i].variable.VariableOp.Operation);
                }
            }
            return gradientsAndVariables;
        }

        /// <summary>
        /// Returns the ops to update the variables in the graph.
        /// </summary>
        /// <param name="gradientsAndVariables">Gradient and Variable tuple.</param>
        public abstract TFOperation[] ApplyGradient((TFOutput gradient, Variable variable)[] gradientsAndVariables);

        /// <summary>
        /// Add operations to minimize `loss` by updating `var_list`.
        /// 
        /// This method simply combines calls `compute_gradients()` and 
        /// `apply_gradients()`. If you want to process the gradient before applying
        /// them call `compute_gradients()` and `apply_gradients()` explicitly instead
        /// of using this function.
        /// </summary>
        /// <param name="loss">A `Tensor` containing the value to minimize.</param>
        /// <param name="varList">list of variable to compute the gradients for.
        /// If null the gradient is computed for all the trainable variables in the graph.</param>
        /// <returns>An Operation that updates the variables.</returns>
        public virtual TFOperation[] Minimize(TFOutput loss, Variable[] varList = null)
        {
            return ApplyGradient(ComputeGradient(loss, varList));
        }
    }

    /// <summary>
    /// Stochastic gradient descent optimizer.
    /// Includes support for momentum, learning rate decay, and Nesterov momentum
    /// </summary>
    public sealed class SGD : Optimizer
    {
        private readonly TFOutput _momentum;
        private readonly string _momentumName = "Momentum";

        private readonly bool _nesterov;

        /// <summary>
        /// Construct SGD optimizer.
        /// </summary>
        /// <param name="graph">The graph object.</param>
        /// <param name="learningRate">The learning rate for the SGD update.</param>
        /// <param name="momentum">Parameter that accelerates SGD in the relevant direction and dampens oscillations.</param>
        /// <param name="decay">Learning rate decay over each update.</param>
        /// <param name="nesterov"> Whether to apply Nesterov momentum.</param>
        /// <param name="operName">Name the optimizer. All the variable that are created in this class will be created under this scope.</param>
        public SGD(TFGraph graph, float learningRate, float momentum = 0, float decay = 0, bool nesterov = false, string operName = "SGDOptimizer") 
            : base(graph, operName, learningRate, decay, 0f)
        {
            using (var scope = _graph.WithScope(_optimizerName))
            {
                _momentum = _graph.Const(momentum, _momentumName);
            }
            _nesterov = nesterov;
        }

        /// <inheritdoc />
        public override TFOperation[] ApplyGradient((TFOutput gradient, Variable variable)[] gradientsAndVariables)
        {
            var moments = InitMoments(gradientsAndVariables);
            for (int i = 0; i < gradientsAndVariables.Length; i++)
            {
                var gv = gradientsAndVariables[i];
                var lr = _graph.Cast(LearningRate.Read, gv.gradient.OutputType);
                var m = _graph.Cast(_momentum, gv.gradient.OutputType);
                // v = m * moment - lr * g
                var velocity = _graph.Sub(_graph.Mul(m, moments[i]), _graph.Mul(lr, gv.gradient));
                // moment = v
                _updateOps.Add(_graph.Assign(moments[i], velocity).Operation);

                if (_nesterov)
                {
                    // w = w + m * v - lr * g
                    var op = _graph.AssignAddVariableOp(gv.variable, _graph.Sub(_graph.Mul(m, velocity), _graph.Mul(lr, gv.gradient)));
                    _updateOps.Add(op);
                }
                else
                {
                    // w = w + v
                    _updateOps.Add(_graph.AssignAddVariableOp(gv.variable, velocity));
                }
            }
            return _updateOps.ToArray();
        }
    }

    /// <summary>
    /// The base class for all the adaptive optimizers.
    /// </summary>
    public abstract class AdaptiveOptimizer : Optimizer
    {
        /// <summary>
        /// Constant value used for avoiding division overflow.
        /// </summary>
        protected readonly TFOutput _epsilon;

        /// <summary>
        /// Construct Adagrad optimizer.
        /// </summary>
        /// <param name="graph">The graph object.</param>
        /// <param name="learningRate">The learning rate for the SGD update.</param>
        /// <param name="decay">Learning rate decay over each update.</param>
        /// <param name="initialAccumulatorValue">A floating point value. Starting value for the accumulators, must be positive.</param>
        /// <param name="operName">Name the optimizer. All the variable that are created in this class will be created under this scope.</param>
        public AdaptiveOptimizer(TFGraph graph, float learningRate, float decay = 0, float initialAccumulatorValue = 0.1f, string operName = "AdagradOptimizer") 
            : base(graph, operName, learningRate, decay, initialAccumulatorValue)
        {
            _epsilon = _graph.Const(1e-7f);
        }
    }

    /// <summary>
    /// Adaptive stochastic gradient descent optimizer.
    /// </summary>
    public sealed class Adagrad : AdaptiveOptimizer
    {
        /// <summary>
        /// Construct Adagrad optimizer.
        /// </summary>
        /// <param name="graph">The graph object.</param>
        /// <param name="learningRate">The learning rate for the SGD update.</param>
        /// <param name="decay">Learning rate decay over each update.</param>
        /// <param name="initialAccumulatorValue">A floating point value. Starting value for the accumulators, must be positive.</param>
        /// <param name="operName">Name the optimizer. All the variable that are created in this class will be created under this scope.</param>
        public Adagrad(TFGraph graph, float learningRate, float decay = 0, float initialAccumulatorValue = 0.1f, string operName = "AdagradOptimizer")
            : base(graph, learningRate, decay, initialAccumulatorValue, operName)
        {
        }

        /// <inheritdoc />
        public override TFOperation[] ApplyGradient((TFOutput gradient, Variable variable)[] gradientsAndVariables)
        {
            var accumulators = InitMoments(gradientsAndVariables);
            for (int i = 0; i < gradientsAndVariables.Length; i++)
            {
                var gv = gradientsAndVariables[i];
                var lr = _graph.Cast(LearningRate.Read, gv.gradient.OutputType);
                
                // accum = g ** 2;
                var accum = _graph.Add(accumulators[i], _graph.Square(gv.gradient));
                
                // accumulators[i] = accum
                _updateOps.Add(_graph.Assign(accumulators[i], accum).Operation);
               
                // w = w - lr * g / sqrt(accum + 1e-7)
                var denom = _graph.Div(_graph.Mul(lr, gv.gradient), _graph.Sqrt(_graph.Add(accum, _epsilon)));
                _updateOps.Add(_graph.AssignSubVariableOp(gv.variable, denom));
            }
            return _updateOps.ToArray();
        }
    }

    /// <summary>
    /// RMSProp: Adaptive stochastic gradient descent optimizer.
    /// </summary>
    public sealed class RMSProp : AdaptiveOptimizer
    {
        private readonly TFOutput _beta;

        /// <summary>
        /// Construct Adagrad optimizer.
        /// </summary>
        /// <param name="graph">The graph object.</param>
        /// <param name="learningRate">The learning rate for the SGD update.</param>
        /// <param name="beta">Factor to compute the moving average over square of gradients.</param>
        /// <param name="decay">Learning rate decay over each update.</param>
        /// <param name="initialAccumulatorValue">A floating point value. Starting value for the accumulators, must be positive.</param>
        /// <param name="operName">Name the optimizer. All the variable that are created in this class will be created under this scope.</param>
        public RMSProp(TFGraph graph, float learningRate, float beta = 0.9f, float decay = 0, float initialAccumulatorValue = 0.1f, string operName = "AdagradOptimizer") 
            : base(graph, learningRate, decay, initialAccumulatorValue, operName)
        {
            _beta = _graph.Const(beta);
        }

        /// <inheritdoc />
        public override TFOperation[] ApplyGradient((TFOutput gradient, Variable variable)[] gradientsAndVariables)
        {
            var accumulators = InitMoments(gradientsAndVariables);
            for (int i = 0; i < gradientsAndVariables.Length; i++)
            {
                var gv = gradientsAndVariables[i];
                var lr = _graph.Cast(LearningRate.Read, gv.gradient.OutputType);
                
                // accum = beta * accum + (1 - beta) * g ** 2;
                var first = _graph.Mul(_beta, accumulators[i]);
                var second = _graph.Mul(_graph.Sub(_graph.Const(1.0f), _beta), _graph.Square(gv.gradient));
                var accum = _graph.Add(first, second);
                
                // accumulators[i] = accum
                _updateOps.Add(_graph.Assign(accumulators[i], accum).Operation);
                
                // w = w - lr * g / sqrt(accum + eps)
                var denom = _graph.Div(_graph.Mul(lr, gv.gradient), _graph.Sqrt(_graph.Add(accum, _epsilon)));
                _updateOps.Add(_graph.AssignSubVariableOp(gv.variable, denom));
            }
            return _updateOps.ToArray();
        }
    }
}
