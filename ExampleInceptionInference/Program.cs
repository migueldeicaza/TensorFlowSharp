using System;
using TensorFlow;
using Mono.Options;
using System.IO;
using System.IO.Compression;
using System.Net;

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

		static OptionSet options = new OptionSet ()
		{
			{ "m|dir=",  v => dir = v },
			{ "h|help", v => Help () }
		};

		static string dir, modelFile, labelsFile;
		public static void Main (string [] args)
		{
			var files = options.Parse (args);
			if (dir == null)
				Error ("Must specify a directory with -m to store the training data");
			if (files == null)
				Error ("No files were specified");

			ModelFiles (dir);

			var model = File.ReadAllBytes (modelFile);

			var g = new TFGraph ();
			g.Import (model, "");
			using (var s = new TFSession (g)) {
			}
		}

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
