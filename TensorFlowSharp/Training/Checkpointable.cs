//
// Port of the checkpointable code from Python.
//
// Authors:
//   Miguel de Icaza
//
using System;
using System.Collections.Generic;

namespace TensorFlow
{
	/// <summary>
	/// Checkpointable reference.
	/// </summary>
	public class CheckpointableReference
	{
		/// <summary>
		/// Local name for the dependency
		/// </summary>
		/// <value>The name.</value>
		public string Name { get; private set; }

		/// <summary>
		/// The Checkpointable object being referenced.
		/// </summary>
		/// <value>The reference.</value>
		public CheckpointableBase Reference { get; private set; }

		public CheckpointableReference (string name, CheckpointableBase reference)
		{
			Name = name;
			Reference = reference;
		}
	}

	/// <summary>
	/// Indicates a position within a Checkpoint
	/// </summary>
	public class CheckpointPosition
	{
	}

	/// <summary>
	/// Base class for `Checkpointable` objects without automatic dependencies.
	/// </summary>
	/// <remarks>
	/// Dependencies must be added explicitly, unless attribute assignment 
	/// is performance-critical use <see cref="T:TensorFlow.Checkpointable"/>
	/// </remarks>
	public class CheckpointableBase
	{
		List<CheckpointableReference> _unconditional_checkpoint_dependencies;
		Dictionary<string, CheckpointableReference> _unconditional_dependency_names;
		Dictionary<string, CheckpointPosition> _deferred_dependencies;
		int update_uid;

		public void MaybeInitializeCheckpointTable ()
		{
			// If we have already been initialized
			if (_unconditional_checkpoint_dependencies != null)
				return;
			
			_unconditional_checkpoint_dependencies = new List<CheckpointableReference> ();
			_unconditional_dependency_names = new Dictionary<string, CheckpointableReference> ();
			_deferred_dependencies = new Dictionary<string, CheckpointPosition> ();
			update_uid = -1;
		}

		/// <summary>
		/// All dependencies for this object
		/// </summary>
		/// <value>A list of CheckpointableReference objects indicating named Checkpointable dependencies which should be saved along with this object.</value>
		/// <remarks>
		/// <para>
		/// May be overridden to include conditional dependencies.  
		/// </para>
		/// <para>
		/// </para>
		/// </remarks>
		public virtual IList<CheckpointableReference> CheckPointDependencies => _unconditional_checkpoint_dependencies;

		/// <summary>
		/// Look up a dependency by name, may be overridden to include conditional dependencies.
		/// </summary>
		/// <param name="name">Name.</param>
		public virtual CheckpointableReference LookupDependency (string name) => _unconditional_dependency_names.TryGetValue (name, out var res) ? res : null;


	}
}
