//
// TensorFlow API for .NET languages
// https://github.com/migueldeicaza/TensorFlowSharp
//
// Authors:
//    Miguel de Icaza (miguel@microsoft.com)
//    Gustavo J Knuppe (https://github.com/knuppe/)
//

using System;

namespace TensorFlow
{
	/// <summary>
	/// Represents an TensorFlow exception.
	/// </summary>
	public sealed class TFException : Exception
	{
		/// <summary>
		/// Initializes a new instance of the <see cref="TFException"/> class with a specified error message.
		/// </summary>
		/// <param name="message">The message that describes the error.</param>
		public TFException (string message) : base (message) 
		{ 
			
		}

		/// <summary>
		/// Initializes a new instance of the <see cref="TFException"/> class with a specified 
		/// error message and a reference to the inner exception that is the cause of this exception.
		/// </summary>
		/// <param name="message">The error message that explains the reason for the exception.</param>
		/// <param name="innerException">The exception that is the cause of the current exception, or a null reference if no inner exception is specified.</param>
		public TFException (string message, Exception innerException) : base(message, innerException) 
		{

		}
	}
}
