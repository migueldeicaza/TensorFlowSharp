using System.Collections.Generic;

namespace TensorFlowSharp
{
    public class Optimizer
    {
        private bool _useLocking;
        private string _name;
        private Dictionary<string, string> _slots;

        public Optimizer(bool useLocking, string name)
        {
            _useLocking = useLocking;
            _name = name;
            _slots = new Dictionary<string, string>();
        }

        public void Minimize()
        {

        }
    }

    public enum GateGradientsValues
    {
        None = 0,
        Op = 1,
        Graph = 2
    }
}