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
        /// The graph object.
        /// </summary>
        protected readonly TFGraph _graph;

        /// <summary>
        /// Construct optimizer.
        /// </summary>
        /// <param name="graph">The graph to construct optimizer for.</param>
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
        /// <param name="varList">A `Tensor` containing the value to minimize.</param>
        /// <returns>An Operation that updates the variables in `var_list`.</returns>
        public abstract TFOperation[] Minimize(TFOutput loss, Variable[] varList = null);
    }

    /// <summary>
    /// 
    /// </summary>
    public sealed class SGD : Optimizer
    {
        private readonly Variable _iterations;
        private readonly TFOutput _learningRate;
        private readonly string _lrName = "LearningRate";

        private readonly TFOutput _momentum;
        private readonly string _momentumName = "Momentum";

        private readonly TFOutput _decay;
        private readonly string _decayName = "Decay";

        private readonly float initial_decay;

        private readonly bool _nesterov;

        private readonly IList<TFOperation> _updateOps = new List<TFOperation>();

        /// <summary>
        /// 
        /// </summary>
        /// <param name="graph"></param>
        /// <param name="learningRate"></param>
        /// <param name="momentum"></param>
        /// <param name="decay"></param>
        /// <param name="nesterov"></param>
        public SGD(TFGraph graph, float learningRate, float momentum = 0, float decay=0, bool nesterov=false) : base(graph)
        {
            _iterations = _graph.Variable(_graph.Const(new TFTensor(0L)), trainable: false, operName: "iterations");
            _learningRate = _graph.Const(learningRate); // _graph.Variable(_graph.Const(learningRate), trainable: false, operName: _lrName);            
            _nesterov = nesterov;
            _momentum = _graph.Const(momentum, _momentumName);
            if (decay > 0)
            {
                _decay = _graph.Const(decay, _decayName);
                _learningRate = _graph.Mul(_learningRate, _graph.Div(_graph.Const(1f),
                    _graph.Add(_graph.Const(1f), _graph.Mul(_decay, _iterations))));
                _updateOps.Add(_learningRate.Operation);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="gradientsAndVariables"></param>
        public override TFOperation[] ApplyGradient((TFOutput gradient, Variable variable)[] gradientsAndVariables)
        {
            for (int i = 0; i < gradientsAndVariables.Length; i++)
            {
                var gv = gradientsAndVariables[i];
                var varType = gv.variable.Read.OutputType;
                var dims = _graph.GetShape(gv.variable);
                var varShape = dims == null ? TFShape.Scalar : new TFShape(dims);
                var moment = _graph.VariableV2(varShape, varType, operName: "moments_" + i);

                _graph.AddInitVariable(_graph.Assign(moment, _graph.Zeros(varShape, varType)).Operation);
                var velocity = _graph.Sub(_graph.Mul(_momentum, moment), _graph.Mul(_learningRate, gv.gradient));
                _updateOps.Add(_graph.Assign(moment, velocity).Operation);

                if (_nesterov)
                {
                    var op = _graph.AssignAddVariableOp(gv.variable, _graph.Sub(_graph.Mul(_momentum, velocity), _graph.Mul(_learningRate, gv.gradient)));
                    _updateOps.Add(op);
                }
                else
                {
                    _updateOps.Add(_graph.AssignAddVariableOp(gv.variable, velocity));
                }
            }
            return _updateOps.ToArray();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="loss"></param>
        /// <param name="varList"></param>
        /// <returns></returns>
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

        /// <summary>
        /// 
        /// </summary>
        /// <param name="loss"></param>
        /// /// <param name="varList"></param>
        /// <returns></returns>
        public override TFOperation[] Minimize(TFOutput loss, Variable[] varList = null)
        {
            return ApplyGradient(ComputeGradient(loss, varList));
        }
    }
}
