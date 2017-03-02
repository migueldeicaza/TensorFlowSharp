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
	/// Represents a disposable TensorFlow object.
	/// </summary>
	public abstract class TFDisposable : IDisposable
	{

        /// <summary>
        /// Occurs when the object is disposed.
        /// </summary>
        public event EventHandler<EventArgs> Disposed;

		/// <summary>
		/// Gets the handle to the unmanaged TensorFlow object.
		/// </summary>
		/// <value>The handle to the unmanaged TensorFlow object.</value>
		public IntPtr Handle { get; protected set; }

		protected TFDisposable() 
		{ 
			
		}

		/// <summary>
		/// Initializes a new instance of the <see cref="T:TensorFlowSharp.TFDisposable"/> class with the specified handle.
		/// </summary>
		/// <param name="handle">The TensorFlow handle.</param>
		protected TFDisposable(IntPtr handle)
		{
			Handle = handle;
		}

		/// <summary>
		/// Gets a value indicating whether this instance is disposed.
		/// </summary>
		/// <value><c>true</c> if this instance is disposed; otherwise, <c>false</c>.</value>
		protected bool IsDisposed { get; private set; }

		/// <summary>
		/// Releases all resource used by the <see cref="T:TensorFlowSharp.TFDisposable"/> object.
		/// </summary>
		/// <remarks>Call <see cref="M:Dispose"/> when you are finished using the <see cref="T:TensorFlowSharp.TFDisposable"/>. The
		/// <see cref="M:Dispose"/> method leaves the <see cref="T:TensorFlowSharp.TFDisposable"/> in an unusable state. After
		/// calling <see cref="M:Dispose"/>, you must release all references to the <see cref="T:TensorFlowSharp.TFDisposable"/>
		/// so the garbage collector can reclaim the memory that the <see cref="T:TensorFlowSharp.TFDisposable"/> was occupying.</remarks>
		public void Dispose()
		{
			Dispose(true);
			GC.SuppressFinalize(this);
		}

		/// <summary>
		/// Releases unmanaged resources and performs other cleanup operations 
		/// before the <see cref="TFDisposable"/> is reclaimed by garbage collection.
		/// </summary>
		~TFDisposable()
		{
			Dispose(false);
		}
	
		/// <summary>
		/// Disposes the native TensorFlow handler.
		/// </summary>
		/// <param name="handle">The handle to dispose.</param>
		/// <remarks>
		/// Must be implemented in subclasses to dispose the unmanaged object, it does
		/// not need to take care of zeroing out the handle, that is done by the Dispose
		/// method inherited from TFDisposable
		/// </remarks>
		protected abstract void NativeDispose(IntPtr handle);

		/// <summary>
		/// Releases unmanaged and - optionally - managed resources.
		/// </summary>
		/// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
		public virtual void Dispose(bool disposing)
		{
			if (IsDisposed)
				return;
				
			try {
				if (disposing) // dispose managed resources
					try {
						if (Handle != IntPtr.Zero)
							NativeDispose (Handle);
					
					} finally {
						Handle = IntPtr.Zero;
					}
				
			} finally {
				IsDisposed = true;

				Disposed?.Invoke(this, EventArgs.Empty);
			}
		}

		/// <summary>
		/// Checks the if the object is disposed.
		/// </summary>
		/// <exception cref="ObjectDisposedException">The object is disposed.</exception>
		protected void CheckDisposed ()
		{
			if (IsDisposed)
				throw new ObjectDisposedException (GetType().FullName);
		}
	}
}