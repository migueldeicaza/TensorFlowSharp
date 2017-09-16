using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorFlow;
using ExampleCommon;
using Mono.Options;

namespace ExampleObjectDetection
{
	class Program
	{
		private static IEnumerable<CatalogItem> _catalog;
		private static string _input;
		private static string _output;
		private static string _catalogPath;
		private static string _modelPath;

		private static double MIN_SCORE_FOR_OBJECT_HIGHLIGHTING = 0.5;

		static OptionSet options = new OptionSet ()
		{
			{ "input_image=",  "Specifies the path to an image ", v => _input = v },
			{ "output_image=",  "Specifies the path to the output image with detected objects", v => _output = v },
			{ "catalog=", "Specifies the path to the .pbtxt objects catalog", v=> _catalogPath = v},
			{ "model=", "Specifies the path to the trained model", v=> _modelPath = v},
			{ "h|help", v => Help () }
		};

		/// <summary>
		/// The utility processes the image and produces output image highlighting detected objects on it.
		/// You need to proceed following steps to get the example working:
		/// 1. Download and unzip one of trained models from 
		/// https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md
		/// 
		/// for instance 'faster_rcnn_inception_resnet_v2_atrous_coco'
		/// 2. Download mscoco_label_map.pbtxt from
		/// https://github.com/tensorflow/models/blob/master/object_detection/data/mscoco_label_map.pbtxt
		/// 
		/// 3. Run the ExampleObjectDetection util from command line specifying input_image, output_image, catalog and model options
		/// where input_image - the path to the image for processing
		/// output_image - the path where the image with detected objects will be saved
		/// catalog - the path to the 'mscoco_label_map.pbtxt' file (see 2)
		/// model - the path to the 'frozen_inference_graph.pb' file (see 1)
		/// 
		/// for instance, 
		/// ExampleObjectDetection --input_image="/demo/input.jpg" --output_image="/demo/output.jpg" --catalog="/demo/mscoco_label_map.pbtxt" --model="/demo/frozen_inference_graph.pb"
		/// </summary>
		/// <param name="args"></param>
		static void Main (string [] args)
		{
			options.Parse (args);

			if(_input == null) {
				throw new ArgumentException ("Missing required option --input_image=");
			}

			if (_output == null) {
				throw new ArgumentException ("Missing required option --output_image=");
			}

			if (_catalogPath == null) {
				throw new ArgumentException ("Missing required option --catalog=");
			}

			if (_modelPath == null) {
				throw new ArgumentException ("Missing required option --model=");
			}

			_catalog = CatalogUtil.ReadCatalogItems (_catalogPath);
			var fileTuples = new List<(string input, string output)> () { (_input, _output) };
			string modelFile = _modelPath;

			using (var graph = new TFGraph ()) {
				var model = File.ReadAllBytes (modelFile);
				graph.Import (new TFBuffer (model));

				using (var session = new TFSession (graph)) {
					foreach (var tuple in fileTuples) {
						var tensor = ImageUtil.CreateTensorFromImageFile (tuple.input, TFDataType.UInt8);
						var runner = session.GetRunner ();


						runner
							.AddInput (graph ["image_tensor"] [0], tensor)
							.Fetch (
							graph ["detection_boxes"] [0],
							graph ["detection_scores"] [0],
							graph ["detection_classes"] [0],
							graph ["num_detections"] [0]);
						var output = runner.Run ();

						var boxes = (float [,,])output [0].GetValue (jagged: false);
						var scores = (float [,])output [1].GetValue (jagged: false);
						var classes = (float [,])output [2].GetValue (jagged: false);
						var num = (float [])output [3].GetValue (jagged: false);

						DrawBoxes (boxes, scores, classes, tuple.input, tuple.output, MIN_SCORE_FOR_OBJECT_HIGHLIGHTING);
					}
				}
			}
		}
		
		private static void DrawBoxes (float [,,] boxes, float [,] scores, float [,] classes, string inputFile, string outputFile, double minScore)
		{
			var x = boxes.GetLength (0);
			var y = boxes.GetLength (1);
			var z = boxes.GetLength (2);

			float ymin = 0, xmin = 0, ymax = 0, xmax = 0;

			using (var editor = new ImageEditor (inputFile, outputFile)) {
				for (int i = 0; i < x; i++) {
					for (int j = 0; j < y; j++) {
						if (scores [i, j] < minScore) continue;

						for (int k = 0; k < z; k++) {
							var box = boxes [i, j, k];
							switch (k) {
								case 0:
									ymin = box;
									break;
								case 1:
									xmin = box;
									break;
								case 2:
									ymax = box;
									break;
								case 3:
									xmax = box;
									break;
							}

						}

						int value = Convert.ToInt32 (classes [i, j]);
						CatalogItem catalogItem = _catalog.FirstOrDefault (item => item.Id == value);
						editor.AddBox (xmin, xmax, ymin, ymax, $"{catalogItem.DisplayName} : {(scores [i, j] * 100).ToString ("0")}%");
					}
				}
			}
		}

		private static void Help ()
		{
			options.WriteOptionDescriptions (Console.Out);
		}
	}
}
