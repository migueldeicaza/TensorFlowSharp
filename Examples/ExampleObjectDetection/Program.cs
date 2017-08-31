using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace ExampleObjectDetection
{
    class Program
    {
        TFGraph _g = new TFGraph();
        static void Main(string[] args)
        {
        }

        private void Train(Dictionary<string, double> H, test_images)
        {
            H["grid_width"] = H["image_width"] / H["region_size"];
            H["grid_height"] = H["image_height"] / H["region_size"];

            var x_in = _g.Placeholder(TFDataType.Float);
        }
    }
}
