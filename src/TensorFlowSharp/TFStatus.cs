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

	using static NativeMethods;

	/// <summary>
	/// Used to track the result of TensorFlow operations.
	/// </summary>
	/// <remarks>
	/// TFStatus is used to track the status of a call to some TensorFlow
	/// operations. Instances of this object are passed to various
	/// TensorFlow operations and you can use the <see cref="P:IsOk"/>
	/// to quickly check if the operation succeeded, or get more detail from the
	/// <see cref="P:Code"/> and a human-readable text
	/// using the <see cref="P:Message"/> property.
	/// 
	/// The convenience <see cref="M:TensorFlow.TFStatus.Raise"/> can be used
	/// to raise a <see cref="P:TensorFlow.TFException"/> if the status of the
	/// operation did not succeed.
	/// </remarks>
	public class TFStatus : TFDisposable
	{

		/// <summary>
		/// The default status.
		/// </summary>
		[ThreadStatic]
		public static TFStatus Default = new TFStatus ();

		/// <summary>
		/// Initializes a new instance of the <see cref="T:TensorFlow.TFStatus"/> class.
		/// </summary>
		public TFStatus () : base (TF_NewStatus ())
		{
			
		}

		protected override void NativeDispose (IntPtr handle)
		{
			TF_DeleteStatus (handle);
		}

		/// <summary>
		/// Sets the TensorFlow status code and status message.
		/// </summary>
		/// <param name="statusMessage">The status code.</param>
		/// <param name="statusMessage">The status message.</param>
		public void SetStatus (TFStatusCode statusCode, string statusMessage)
		{
			TF_SetStatus (Handle, statusCode, statusMessage);
		}

		/// <summary>
		/// Gets the status code for the status code.
		/// </summary>
		/// <value>The status code as an enumeration.</value>
		public TFStatusCode Code {
			get {
				return TF_GetCode (Handle);
			}
			set {
				CheckDisposed ();
				TF_SetStatus (Handle, value, Message);
			}
		}

		/// <summary>
		/// Gets or sets a human-readable status message.
		/// </summary>
		/// <value>The status message.</value>
		public string Message {
			get {
				return TF_Message (Handle).GetStr ();
			}
			set {
				CheckDisposed ();
				TF_SetStatus (Handle, Code, value);
			}
		}

		/// <summary>
		/// Returns a <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFStatus"/>.
		/// </summary>
		/// <returns>A <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFStatus"/>.</returns>
		public override string ToString ()
		{
			return string.Format ("[TFStatus: StatusCode={0}, StatusMessage={1}]", Code, Message);
		}


		/// <summary>
		/// Gets a value indicating whether this <see cref="TFStatus"/> state has been set to ok.
		/// </summary>
		/// <value><c>true</c> if ok; otherwise, <c>false</c>.</value>
		public bool IsOk => Code == TFStatusCode.Ok;

		/// <summary>
		/// Gets a value indicating whether this <see cref="TFStatus"/> state has been set to an error.
		/// </summary>
		/// <value><c>true</c> if the <see cref="Code"/> is different than <see cref="TFStatusCode.Ok"/>; otherwise, <c>false</c>.</value>
		public bool IsError => Code != TFStatusCode.Ok;

		/// <summary>
		/// Convenience method that raises an exception if the current status is an error.
		/// </summary>
		/// <remarks>
		/// You can use this method as a convenience to raise an exception after you
		/// invoke an operation if the operation did not succeed.
		/// </remarks>
		public void Raise ()
		{
			if (TF_GetCode (Handle) != TFStatusCode.Ok)
				throw new TFException (Message);
		}

		// 
		// Utility function used to simplify implementing the idiom
		// where the user optionally provides a TFStatus, if it is provided,
		// the error is returned there;   If it is not provided, then an
		// exception is raised.
		//
		internal bool CheckMaybeRaise (TFStatus incomingStatus, bool last = true)
		{
			if (incomingStatus == null) {
				CheckDisposed ();
				
				if (Code != TFStatusCode.Ok) {
					var e = new TFException (Message);
					Dispose ();
					throw e;
				}

				if (last)
					Dispose ();
				
				return true;
			}
			return Code == TFStatusCode.Ok;
		}

		internal static TFStatus Setup (TFStatus incoming)
		{
			return incoming ?? new TFStatus ();
		}
	}
}
