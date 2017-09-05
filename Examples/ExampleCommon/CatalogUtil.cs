using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;

namespace ExampleCommon
{
	public static class CatalogUtil
	{
		private static string CATALOG_ITEM_PATTERN = @"item {\r\n  name: ""(?<name>.*)""\r\n  id: (?<id>\d+)\r\n  display_name: ""(?<displayName>.*)""\r\n}";

		/// <summary>
		/// Reads catalog of well-known objects from text file.
		/// </summary>
		/// <param name="file">path to the text file</param>
		/// <returns>collection of items</returns>
		public static IEnumerable<CatalogItem> ReadCatalogItems (string file)
		{
			using (FileStream stream = File.OpenRead (file))
			using (StreamReader reader = new StreamReader (stream)) {
				string text = reader.ReadToEnd ();
				if (string.IsNullOrWhiteSpace (text)) {
					yield break;
				}

				Regex regex = new Regex (CATALOG_ITEM_PATTERN);
				var matches = regex.Matches (text);
				foreach (Match match in matches) {
					var name = match.Groups [1].Value;
					var id = int.Parse (match.Groups [2].Value);
					var displayName = match.Groups [3].Value;

					yield return new CatalogItem () {
						Id = id,
						Name = name,
						DisplayName = displayName
					};
				}
			}
		}
	}

	public class CatalogItem
	{
		public int Id { get; set; }
		public string Name { get; set; }
		public string DisplayName { get; set; }
	}
}
