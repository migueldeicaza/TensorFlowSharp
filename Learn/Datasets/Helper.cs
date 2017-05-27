using System;
using System.IO;
using System.Net;

namespace Learn
{
	public class Helper
	{
		public static Stream MaybeDownload (string urlBase, string trainDir, string file)
		{
			if (!Directory.Exists (trainDir))
				Directory.CreateDirectory (trainDir);
			var target = Path.Combine (trainDir, file);
			if (!File.Exists (target)) {
				var wc = new WebClient ();
				wc.DownloadFile (urlBase + file, target);
			}
			return File.OpenRead (target);
		}
	}
}
