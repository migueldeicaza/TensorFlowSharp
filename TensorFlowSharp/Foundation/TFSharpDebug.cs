using System;
using System.Runtime.CompilerServices;

namespace TensorFlowSharp.Foundation
{
    internal delegate void DebugAction();
    internal delegate T DebugFunc<T>();

    internal static class TFSharpDebug
    {
        private const bool IsDebug = 
#if DEBUG
            true;
#else
            false;
#endif

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T AssertNotNull<T>(T obj, string name)
            where T : class
        {
            return IsDebug
                ? obj ?? throw new ArgumentNullException(name)
                : obj;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T AssertWithinLimits<T>(T obj, string name, T minInclusive, T maxExcluded)
            where T : IComparable<T>
        {

            if (!IsDebug || RangeChecks.IsWithinRange(obj, minInclusive, maxExcluded))
                return obj;
            else
                throw new ArgumentOutOfRangeException(FormattableString.Invariant($"Expected '{name}' to be between [{minInclusive}, {maxExcluded})"));
        }
    }
}
