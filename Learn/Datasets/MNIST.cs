//
// Code to download and load the MNIST data.
//

using System;
using System.IO;
using System.IO.Compression;
using Mono;
using TensorFlow;

namespace Learn
{
	public class DataSet
	{
		
	}

	public class Mnist 
	{
		public DataSet Train { get; private set; }
		public DataSet Validation { get; private set; }
		public DataSet Test { get; private set; }

		const string SourceUrl = "http://yann.lecun.com/exdb/mnist/";
		const string TrainImages = "train-images-idx3-ubyte.gz";
		const string TrainLabels = "train-labels-idx1-ubyte.gz";
		const string TestImages = "t10k-images-idx3-ubyte.gz";
		const string TestLabels = "t10k-labels-idx1-ubyte.gz";


		int Read32 (Stream s)
		{
			var x = new byte [4];
			s.Read (x, 0, 4);
			return DataConverter.BigEndian.GetInt32 (x);
		}

		void ExtractImages (Stream input, string file)
		{
			var gz = new GZipStream (input, CompressionMode.Decompress);
			if (Read32 (gz) != 2051) 
				throw new Exception ("Invalid magic number found on the MNIST " + file);
			var count = Read32 (gz);
			var rows = Read32 (gz);
			var cols = Read32 (gz);
			var buffer = new byte [rows * cols * count];


		}

		public void ReadDataSets (string trainDir, bool fakeData = false, bool oneHot = false, TFDataType dtype = TFDataType.Float, bool reshape = true, int validationSize = 5000)
		{
			if (fakeData) {
				return;
			}

			ExtractImages (Helper.MaybeDownload (SourceUrl, trainDir, TrainImages), TrainImages);

		}
	}
}
