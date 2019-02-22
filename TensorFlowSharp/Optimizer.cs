using System;
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
        /// The graph object. It is used for creating Ops through the construction of optimizer.
        /// </summary>
        protected readonly TFGraph _graph;

        /// <summary>
        /// Construct optimizer.
        /// </summary>
        /// <param name="graph">The graph object.</param>
        public Optimizer(TFGraph graph)
        {
            _graph = graph;
        }
        /// <summary>
        /// Computes the gradient of the trainable variables in the graph.
        /// </summary>
        /// <param name="loss">The loss operation to compute the gradient on.</param>
        /// <param name="varList">list of variable to compute the gradients for.
        /// If null the gradient is computed for all the trainable variables in the graph.</param>
        /// <returns>A list of (gradient, variable) pairs. Variable is always present, but
        /// gradient can be `None`.</returns>
        public abstract (TFOutput gradient, Variable variable)[] ComputeGradient(TFOutput loss, Variable[] varList = null);

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
        /// If null the gradient is computed for all the trainable variables in the graph./param>
        /// <returns>An Operation that updates the variables.</returns>
        public abstract TFOperation[] Minimize(TFOutput loss, Variable[] varList = null);
    }

    /// <summary>
    /// Stochastic gradient descent optimizer.
    /// Includes support for momentum, learning rate decay, and Nesterov momentum
    /// </summary>
    public sealed class SGD : Optimizer
    {
        /// <summary>
        /// Varaible to keep track of number of iterations (mini-batch processed)
        /// </summary>
        public Variable Iterations { get; }

        /// <summary>
        /// Variable to keep track of the learning rate.
        /// </summary>
        public Variable LearningRate { get; }

        private readonly string _lrName = "LearningRate";

        private readonly TFOutput _momentum;
        private readonly string _momentumName = "Momentum";

        private readonly bool _nesterov;

        private readonly IList<TFOperation> _updateOps = new List<TFOperation>();

        /// <summary>
        /// Construct SGD optimizer.
        /// </summary>
        /// <param name="graph">The graph object.</param>
        /// <param name="learningRate">The learning rate for the SGD update.</param>
        /// <param name="momentum">Parameter that accelerates SGD in the relevant direction and dampens oscillations.</param>
        /// <param name="decay">Learning rate decay over each update.</param>
        /// <param name="nesterov"> Whether to apply Nesterov momentum.</param>
        public SGD(TFGraph graph, float learningRate, float momentum = 0, float decay = 0, bool nesterov = false) : base(graph)
        {
            Iterations = _graph.Variable(_graph.Const(new TFTensor(0L)), trainable: false, operName: "iterations");
            _updateOps.Add(_graph.AssignAddVariableOp(Iterations, _graph.Const(1L)));
            var initialLearningRate = _graph.Const(learningRate);
            LearningRate = _graph.Variable(initialLearningRate, trainable: false, operName: _lrName);
            _nesterov = nesterov;
            _momentum = _graph.Const(momentum, _momentumName);
            CreateDecayOps(decay, initialLearningRate);
        }

        private void CreateDecayOps(float decay, TFOutput initialLearningRate)
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

        private TFOutput[] InitMoments((TFOutput gradient, Variable variable)[] gradientsAndVariables)
        {
            var moments = new TFOutput[gradientsAndVariables.Length];
            for (int i = 0; i < gradientsAndVariables.Length; i++)
            {
                var gv = gradientsAndVariables[i];
                var varType = gv.variable.Read.OutputType;
                var varShape = _graph.GetTensorShape(gv.variable.Read);
                moments[i] = _graph.VariableV2(varShape, varType, operName: "moments_" + i);
                _graph.AddInitVariable(_graph.Assign(moments[i], _graph.Zeros(varShape, varType)).Operation);
            }
            return moments;
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
                var velocity = _graph.Sub(_graph.Mul(m, moments[i]), _graph.Mul(lr, gv.gradient), "velocity_" + i);
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

        /// <inheritdoc />
        public override (TFOutput gradient, Variable variable)[] ComputeGradient(TFOutput loss, Variable[] varList = null)
        {
            varList = varList ?? _graph.GetTrainableVariables();
            var gradientsAndVariables = new (TFOutput gradient, Variable variable)[varList.Length];
            for (int i = 0; i < varList.Length; i++)
            {
                gradientsAndVariables[i].variable = varList[i];
                gradientsAndVariables[i].gradient = _graph.AddGradients(new TFOutput[] { loss }, new TFOutput[] { varList[i].Read })[0];
            }
            return gradientsAndVariables;
        }

        /// <inheritdoc />
        public override TFOperation[] Minimize(TFOutput loss, Variable[] varList = null)
        {
            return ApplyGradient(ComputeGradient(loss, varList));
        }
    }
}
