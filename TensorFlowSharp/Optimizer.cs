using System;
using System.Collections.Generic;
using System.Linq;
using TensorFlow;

namespace TensorFlowSharp
{
  //  public class Optimizer
  //  {
  //      private bool _useLocking;
  //      private string _name;
  //      private Dictionary<string, string> _slots;
  //      private static List<TFDataType> _validDataTypes = new List<TFDataType> { TFDataType.BFloat16, TFDataType.Float };

  //      public Optimizer(bool useLocking, string name)
  //      {
  //          _useLocking = useLocking;
  //          _name = name;
  //          _slots = new Dictionary<string, string>();
  //      }

  //      public void Minimize(TFTensor loss,
  //          GateGradientsValues globalStep = GateGradientsValues.None,
  //          GateGradientsValues variableList = GateGradientsValues.None,
  //            GateGradientsValues gateGradients = GateGradientsValues.Op,
  //            GateGradientsValues aggregationMethod = GateGradientsValues.None,
  //            bool colocateGradientsWithOps = false,
  //            GateGradientsValues name = GateGradientsValues.None,
  //             TFTensor gradLoss = null)
  //      {
  //          var grads_and_vars = ComputeGradients(loss, variableList, gateGradients, aggregationMethod, colocateGradientsWithOps, gradLoss);

  //      }

  //      private void ComputeGradients(TFTensor loss,
  //          GateGradientsValues variableList = GateGradientsValues.None,
  //            GateGradientsValues gateGradients = GateGradientsValues.Op,
  //            GateGradientsValues aggregationMethod = GateGradientsValues.None,
  //            bool colocateGradientsWithOps = false,
  //             TFTensor gradLoss = null)
  //      {
  //          if (gateGradients != GateGradientsValues.Graph && gateGradients != GateGradientsValues.None && gateGradients != GateGradientsValues.Op)
  //          {
  //              throw new ArgumentOutOfRangeException();
  //          }

  //          AssertValidDataTypes(new[] { loss });

  //          if (gradLoss != null)
  //          {
  //              AssertValidDataTypes(new[] { gradLoss });
  //          }

  //          if (variableList == GateGradientsValues.None)
  //          {

  //          }
  //      }

  //      private void AssertValidDataTypes(TFTensor[] tensors)
  //      {
  //          foreach (var t in tensors)
  //          {
  //              if (!_validDataTypes.Contains(t.TensorType))
  //              {
  //                  throw new ArgumentException();// TODO: message
  //              }
  //          }
  //      }

  //      //      class _DenseResourceVariableProcessor(_OptimizableVariable):
  //      //"""Processor for dense ResourceVariables."""

  //      //def __init__(self, v):
  //      //  self._v = v

  //      //def target(self):
  //      //  return self._v

  //      //def update_op(self, optimizer, g):
  //      //  # pylint: disable=protected-access
  //      //  if isinstance(g, ops.IndexedSlices):
  //      //    if self._v.constraint is not None:
  //      //      raise RuntimeError(
  //      //          "Cannot use a constraint function on a sparse variable.")
  //      //    return optimizer._resource_apply_sparse_duplicate_indices(
  //      //        g.values, self._v, g.indices)
  //      //  update_op = optimizer._resource_apply_dense(g, self._v)
  //      //  if self._v.constraint is not None:
  //      //    with ops.control_dependencies([update_op]):
  //      //      return self._v.assign(self._v.constraint(self._v))
  //      //  else:
  //      //    return update_op
  //      private class DenseResourceVariableProcessor
  //      {
  //          public DenseResourceVariableProcessor(v)
  //          {

  //          }
  //          public void Update(Optimizer optimizer, g)
  //          {
  //              update_op = optimizer._resource_apply_dense(g, self._v)
  //          }
  //      }
  //      public enum AggregationMethod
  //      {
  //          ADD_N = 0,
  //DEFAULT = ADD_N,
  //EXPERIMENTAL_TREE = 1,
  //EXPERIMENTAL_ACCUMULATE_N = 2
  //      }
  //      private class Gradients
  //      {
  //          public void Gradients(
  //              IEnumerable<TFOutput> ys,
  //           IEnumerable<TFOutput> xs,
  //          IEnumerable<TFTensor> grad_ys = null,
  //          string  name= "gradients",
  //            bool colocate_gradients_with_ops= false,
  //           bool gate_gradients= false,
  //           AggregationMethod? aggregation_method = null)
  //          {
  //              ys = ConvertNToTensorOrIndexedSlices(ys, name: "y");
  //              xs = ConvertNToTensorOrIndexedSlices(xs, name: "x", as_ref: true);

  //              grad_ys = _DefaultGradYs(grad_ys, ys, colocate_gradients_with_ops);
  //              if (ys.Count() > 1)
  //              {

  //              }

  //              var toOps = ys.Select(x => x.Operation);
  //              var fromOps = xs.Select(x => x.Operation);
  //          }

  //          private void _PendingCount(TFGraph graph, IEnumerable<TFOperation> to_ops, IEnumerable<TFOperation> from_ops, bool colocate_gradients_with_ops)
  //          {
  //              var reached_ops = new bool [graph.LastId + 1];
  //              foreach(var op in to_ops)
  //              {
  //                  reached_ops[op.Id] = true;
  //              }

  //          }

  //          private void _MarkReachedOps(IEnumerable<TFOperation> from_ops, IEnumerable<TFOperation> reached_ops)
  //          {

  //          }

  //          private IEnumerable<object> _DefaultGradYs(IEnumerable<TFTensor> grad_ys, IEnumerable<TFTensor> ys, bool colocate_gradients_with_op)
  //          {
  //              if (grad_ys.Count() != ys.Count())
  //              {
  //                  throw new ArgumentException($"Lengthes of '{nameof(grad_ys)}' and '{nameof(ys)}' are mismatching");
  //              }

  //              // TODO: Implement https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/gradients_impl.py
  //              // line 206

  //              return grad_ys;
  //          }

  //          private IEnumerable<TFTensor> ConvertNToTensorOrIndexedSlices(IEnumerable<object> values,
  //                                               TFDataType dtype = TFDataType.None,
  //                                                 string name = null,
  //                                                 bool as_ref = false)
  //          {
  //              return InternalConvertNToTensorOrIndexedSlices(values, dtype, name, as_ref);
  //          }

  //          private IEnumerable<TFTensor> InternalConvertNToTensorOrIndexedSlices(IEnumerable<object> values,
  //                                               TFDataType dtype = TFDataType.None,
  //                                                 string name = null,
  //                                                 bool as_ref = false)
  //          {
  //              return values.Select(x => InternalConvertToTensorOrIndexedSlices(x, dtype, name, as_ref));
  //          }

  //          private TFTensor InternalConvertToTensorOrIndexedSlices(object value,
  //                                               TFDataType dtype = TFDataType.None,
  //                                                 string name = null,
  //                                                 bool as_ref = false)
  //          {
  //              return InternalConvertToTensor(value, dtype, name, as_ref);
  //          }

  //          private TFTensor InternalConvertToTensor(object value,
  //              TFDataType dtype = TFDataType.None,
  //                                                 string name = null,
  //                                                 bool as_ref = false)
  //          {
  //              var functionsAtPriority = _tensorConversionFuncRegistry.OrderBy(x => x.Key).Select(x => x.Value);
  //              TFTensor ret = null;
  //              foreach (var funcsAtPriority in functionsAtPriority)
  //              {
  //                  foreach (var func in funcsAtPriority)
  //                  {
  //                      ret = func(value);
  //                  }
  //              }

  //              return ret;
  //          }

  //          private static Dictionary<int, List<Func<object, TFTensor>>> _tensorConversionFuncRegistry = new Dictionary<int, List<Func<object, TFTensor>>>();
  //      }
  //  }

  //  public enum GateGradientsValues
  //  {
  //      None = 0,
  //      Op = 1,
  //      Graph = 2
  //  }
}