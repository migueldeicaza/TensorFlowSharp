//
// TensorFlow API for .NET languages
// https://github.com/migueldeicaza/TensorFlowSharp
//
// Authors:
//    Miguel de Icaza (miguel@microsoft.com)
//    Gustavo J Knuppe (https://github.com/knuppe/)
//

namespace TensorFlow
{
	/// <summary>
	/// Status code for invoking a tensorflow operation.
	/// </summary>
	public enum TFCode : uint
	{
		Ok = 0,
		Cancelled = 1,
		Unknown = 2,
		InvalidArgument = 3,
		DeadlineExceeded = 4,
		NotFound = 5,
		AlreadyExists = 6,
		PermissionDenied = 7,
		Unauthenticated = 16,
		ResourceExhausted = 8,
		FailedPrecondition = 9,
		Aborted = 10,
		OutOfRange = 11,
		Unimplemented = 12,
		Internal = 13,
		Unavailable = 14,
		DataLoss = 15
	}
}
