using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{
    public static class TestUtils
    {
        public static void MatrixEqual(double[,] expected, double[,] actual, int precision)
        {
            for (int i = 0; i < expected.GetLength(0); i++)
                for (int j = 0; j < expected.GetLength(1); j++)
                    Assert.Equal(expected[i, j], actual[i, j], precision: precision);
        }

        public static void MatrixEqual(Array expected, Array actual, int precision)
        {
            Assert.Equal(expected.Length, actual.Length);
            Assert.Equal(expected.Rank, actual.Rank);
            Assert.Equal(expected.GetType(), actual.GetType());

            var ei = expected.GetEnumerator();
            var ai = actual.GetEnumerator();

            var expectedType = expected.GetType().GetElementType();

            if (expectedType == typeof(double))
            {
                while (ei.MoveNext() && ai.MoveNext())
                    Assert.Equal((double)ei.Current, (double)ai.Current, precision: 8);
            }
            else if (expectedType == typeof(float))
            {
                while (ei.MoveNext() && ai.MoveNext())
                    Assert.Equal((float)ei.Current, (float)ai.Current, precision: 8);
            }
            else
            {
                while (ei.MoveNext() && ai.MoveNext())
                    Assert.True(Object.Equals(ei.Current, ai.Current));
            }
        }

		public static void MatrixEqual (object expected, object actual, int precision)
		{
			if (expected is Array) {
				MatrixEqual (expected as Array, actual as Array, precision);
				return;
			}
			var expectedType = expected.GetType ();

			if (expectedType == typeof (double)) {
				Assert.Equal ((double)expected, (double)actual, precision: precision);
			} else if (expectedType == typeof (float)) {
				Assert.Equal ((float)expected, (float)actual, precision: precision);
			} else if (expectedType == typeof (int)) {
				Assert.Equal ((int)expected, (int)actual);
			} else {
				Assert.True (Object.Equals (expected, actual));
			}
		}
	}
}
