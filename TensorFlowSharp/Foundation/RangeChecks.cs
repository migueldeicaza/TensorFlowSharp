using System;
using System.Runtime.CompilerServices;

namespace TensorFlowSharp.Foundation
{
    internal static class RangeChecks
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsWithinRange(int value, int minInclusive, int maxExclusive) => minInclusive <= value && value < maxExclusive;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsWithinRange<T>(T obj, T minInclusive, T maxExcluded)
            where T : IComparable<T>
        {
            return minInclusive.CompareTo(obj) <= 0 && 0 < maxExcluded.CompareTo(obj);
        }

        public static int Assert(int value, string name, int minInclusive, int maxExclusive)
        {
            return IsWithinRange(value, minInclusive, maxExclusive)
                ? value
                : throw new ArgumentOutOfRangeException(FormattableString.Invariant($"Expected '{name}' to be within [{minInclusive}, {maxExclusive}), but was {value}"));
        }
    }
}
