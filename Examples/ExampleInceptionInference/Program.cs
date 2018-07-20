﻿// An example for using the TensorFlow C# API for image recognition
// using a pre-trained inception model (http://arxiv.org/abs/1512.00567).
// 
// Sample usage: <program> -dir=/tmp/modeldir imagefile
// 
// The pre-trained model takes input in the form of a 4-dimensional
// tensor with shape [ BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3 ],
// where:
// - BATCH_SIZE allows for inference of multiple images in one pass through the graph
// - IMAGE_HEIGHT is the height of the images on which the model was trained
// - IMAGE_WIDTH is the width of the images on which the model was trained
// - 3 is the (R, G, B) values of the pixel colors represented as a float.
// 
// And produces as output a vector with shape [ NUM_LABELS ].
// output[i] is the probability that the input image was recognized as
// having the i-th label.
// 
// A separate file contains a list of string labels corresponding to the
// integer indices of the output.
// 
// This example:
// - Loads the serialized representation of the pre-trained model into a Graph
// - Creates a Session to execute operations on the Graph
// - Converts an image file to a Tensor to provide as input to a Session run
// - Executes the Session and prints out the label with the highest probability
// 
// To convert an image file to a Tensor suitable for input to the Inception model,
// this example:
// - Constructs another TensorFlow graph to normalize the image into a
//   form suitable for the model (for example, resizing the image)
// - Creates an executes a Session to obtain a Tensor in this normalized form.
using System;
using TensorFlow;
using Mono.Options;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Collections.Generic;
using ExampleCommon;

namespace ExampleInceptionInference
{
	class MainClass
	{
		static void Error (string msg)
		{
			Console.WriteLine ("Error: {0}", msg);
			Environment.Exit (1);
		}

		static void Help ()
		{
			options.WriteOptionDescriptions (Console.Out);
		}

		static bool jagged = true;

		static OptionSet options = new OptionSet ()
		{
			{ "m|dir=",  "Specifies the directory where the model and labels are stored", v => dir = v },
			{ "h|help", v => Help () },

			{ "amulti", "Use multi-dimensional arrays instead of jagged arrays", v => jagged = false }
		};


		static string dir, modelFile, labelsFile;
		public static void Main (string [] args)
		{
			var files = options.Parse (args);
			if (dir == null) {
				dir = "/tmp";
				//Error ("Must specify a directory with -m to store the training data");
			}

			//if (files == null || files.Count == 0)
			//	Error ("No files were specified");

			if (files.Count == 0)
				files = new List<string> () { "demo-picture.jpg" };
			
			ModelFiles (dir);

			// Construct an in-memory graph from the serialized form.
			var graph = new TFGraph ();
			// Load the serialized GraphDef from a file.
			var model = File.ReadAllBytes (modelFile);

			graph.Import (model, "");
			using (var session = new TFSession (graph)) {
				var labels = File.ReadAllLines (labelsFile);

				foreach (var file in files) {
					// Run inference on the image files
					// For multiple images, session.Run() can be called in a loop (and
					// concurrently). Alternatively, images can be batched since the model
					// accepts batches of image data as input.
					var tensor = ImageUtil.CreateTensorFromImageFile (file);

					var runner = session.GetRunner ();
					runner.AddInput (graph ["input"] [0], tensor).Fetch (graph ["output"] [0]);
					var output = runner.Run ();
					// output[0].Value() is a vector containing probabilities of
					// labels for each image in the "batch". The batch size was 1.
					// Find the most probably label index.

					var result = output [0];
					var rshape = result.Shape;
					if (result.NumDims != 2 || rshape [0] != 1) {
						var shape = "";
						foreach (var d in rshape) {
							shape += $"{d} ";
						}
						shape = shape.Trim ();
						Console.WriteLine ($"Error: expected to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape [{shape}]");
						Environment.Exit (1);
					}

					// You can get the data in two ways, as a multi-dimensional array, or arrays of arrays, 
					// code can be nicer to read with one or the other, pick it based on how you want to process
					// it
					bool jagged = true;

					var bestIdx = 0;
					float p = 0, best = 0;

					if (jagged) {
						var probabilities = ((float [] [])result.GetValue (jagged: true)) [0];
						for (int i = 0; i < probabilities.Length; i++) {
							if (probabilities [i] > best) {
								bestIdx = i;
								best = probabilities [i];
							}
						}

					} else {
						var val = (float [,])result.GetValue (jagged: false);	

						// Result is [1,N], flatten array
						for (int i = 0; i < val.GetLength (1); i++) {
							if (val [0, i] > best) {
								bestIdx = i;
								best = val [0, i];
							}
						}
					}

					Console.WriteLine ($"{file} best match: [{bestIdx}] {best * 100.0}% {labels [bestIdx]}");
				}
			}
		}
		
		//
		// Downloads the inception graph and labels
		//
		static void ModelFiles (string dir)
		{
			string url = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip";

			modelFile = Path.Combine (dir, "tensorflow_inception_graph.pb");
			labelsFile = Path.Combine (dir, "imagenet_comp_graph_label_strings.txt");
			var zipfile = Path.Combine (dir, "inception5h.zip");

			if (File.Exists (modelFile) && File.Exists (labelsFile))
				return;

			Directory.CreateDirectory (dir);
			var wc = new WebClient ();
			wc.DownloadFile (url, zipfile);
			ZipFile.ExtractToDirectory (zipfile, dir);
			File.Delete (zipfile);
		}
	}
}
