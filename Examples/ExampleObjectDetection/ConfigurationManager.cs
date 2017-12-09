using Microsoft.Extensions.Configuration;
using System;
using System.IO;

namespace ExampleCommon
{
    public class ConfigurationManager
    {
		public static Lazy<IConfigurationRoot> AppSettings => new Lazy<IConfigurationRoot> (() => {

			var builder = new ConfigurationBuilder ()
			.SetBasePath (Directory.GetCurrentDirectory ())
			.AddJsonFile ("appsettings.json");

			return builder.Build ();
		});
	}
}
