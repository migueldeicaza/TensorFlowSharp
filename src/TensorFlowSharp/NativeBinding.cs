//
// TensorFlow API for .NET languages
// https://github.com/migueldeicaza/TensorFlowSharp
// 
// Authors:
//   Miguel de Icaza (miguel@microsoft.com)
//   Gustavo J Knuppe (https://github.com/knuppe/)
//

using System;
using System.Runtime.InteropServices;
using size_t = System.UIntPtr;

namespace TensorFlow
{
	internal static class NativeMethods
	{
		internal const string TensorFlowLibrary = "libtensorflow";

		// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h

		internal static string GetStr (this IntPtr x) => Marshal.PtrToStringAnsi (x);

		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_Version ();

		[StructLayout (LayoutKind.Sequential)]
		internal struct LLBuffer
		{
			internal IntPtr data;
			internal size_t length;
			internal IntPtr data_deallocator;
		}

		// extern TF_Buffer * TF_NewBufferFromString (const void *proto, size_t proto_len);
		[DllImport (TensorFlowLibrary)]
		internal static extern unsafe LLBuffer* TF_NewBufferFromString (IntPtr proto, IntPtr proto_len);

		// extern TF_Buffer * TF_NewBuffer ();
		[DllImport (TensorFlowLibrary)]
		internal static extern unsafe LLBuffer* TF_NewBuffer ();

		// extern void TF_DeleteBuffer (TF_Buffer *);
		[DllImport (TensorFlowLibrary)]
		internal static extern unsafe void TF_DeleteBuffer (LLBuffer* buffer);
		// extern TF_Buffer TF_GetBuffer (TF_Buffer *buffer);

		[DllImport (TensorFlowLibrary)]
		internal static extern unsafe LLBuffer TF_GetBuffer (LLBuffer* buffer);



		// extern size_t TF_DataTypeSize (TF_DataType dt);
		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_DataTypeSize (TFDataType dt);

		// extern TF_Buffer * TF_GetAllOpList ();
		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_GetAllOpList ();

		// extern TF_Status * TF_NewStatus ();
		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_NewStatus ();

		// extern void TF_DeleteStatus (TF_Status *);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_DeleteStatus (IntPtr status);

		// extern void TF_SetStatus (TF_Status *s, TF_Code code, const char *msg);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern void TF_SetStatus (IntPtr s, TFCode code, string msg);

		// extern TF_Code TF_GetCode (const TF_Status *s);
		[DllImport (TensorFlowLibrary)]
		internal static extern TFCode TF_GetCode (IntPtr s);

		// extern const char * TF_Message (const TF_Status *s);
		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_Message (IntPtr s);

		// extern size_t TF_StringEncode (const char *src, size_t src_len, char *dst, size_t dst_len, TF_Status *status);
		[DllImport (TensorFlowLibrary)]
		internal static extern unsafe size_t TF_StringEncode (byte* src, size_t src_len, sbyte* dst, size_t dst_len, IntPtr status);

		// extern size_t TF_StringDecode (const char *src, size_t src_len, const char **dst, size_t *dst_len, TF_Status *status);
		[DllImport (TensorFlowLibrary)]
		internal static extern unsafe size_t TF_StringDecode (sbyte* src, size_t src_len, sbyte** dst, size_t* dst_len, IntPtr status);

		// extern size_t TF_StringEncodedSize (size_t len);
		[DllImport (TensorFlowLibrary)]
		internal static extern size_t TF_StringEncodedSize (size_t len);

		// extern TF_SessionOptions * TF_NewSessionOptions ();
		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_NewSessionOptions ();

		// extern void TF_DeleteSessionOptions (TF_SessionOptions *);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_DeleteSessionOptions (IntPtr options);

		// extern void TF_SetTarget (TF_SessionOptions *options, const char *target);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern void TF_SetTarget (IntPtr options, string target);

		// extern void TF_SetConfig (TF_SessionOptions *options, const void *proto, size_t proto_len, TF_Status *status);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_SetConfig (IntPtr options, IntPtr proto, size_t proto_len, IntPtr status);

		// extern TF_Graph * TF_NewGraph ();
		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_NewGraph ();

		// extern void TF_DeleteGraph (TF_Graph *);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_DeleteGraph (IntPtr graph);

		// extern void TF_GraphSetTensorShape (TF_Graph *graph, TF_Output output, const int64_t *dims, const int num_dims, TF_Status *status);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_GraphSetTensorShape (IntPtr graph, TFOutput output, ref long [] dims, int num_dims, IntPtr status);

		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_GraphSetTensorShape (IntPtr graph, TFOutput output, IntPtr dims, int num_dims, IntPtr status);

		// extern int TF_GraphGetTensorNumDims (TF_Graph *graph, TF_Output output, TF_Status *status);
		[DllImport (TensorFlowLibrary)]
		internal static extern int TF_GraphGetTensorNumDims (IntPtr graph, TFOutput output, IntPtr status);

		// extern void TF_GraphGetTensorShape (TF_Graph *graph, TF_Output output, int64_t *dims, int num_dims, TF_Status *status);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_GraphGetTensorShape (IntPtr graph, TFOutput output, ref long [] dims, int num_dims, IntPtr status);

		// extern void TF_GraphToGraphDef (TF_Graph *graph, TF_Buffer *output_graph_def, TF_Status *status);
		[DllImport (TensorFlowLibrary)]
		internal static extern unsafe void TF_GraphToGraphDef (IntPtr graph, LLBuffer* output_graph_def, IntPtr status);

		// extern void TF_GraphImportGraphDef (TF_Graph *graph, const TF_Buffer *graph_def, const TF_ImportGraphDefOptions *options, TF_Status *status);
		[DllImport (TensorFlowLibrary)]
		internal static extern unsafe void TF_GraphImportGraphDef (IntPtr graph, LLBuffer* graph_def, IntPtr options, IntPtr status);

		// extern TF_Operation * TF_GraphOperationByName (TF_Graph *graph, const char *oper_name);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern IntPtr TF_GraphOperationByName (IntPtr graph, string oper_name);


		// extern TF_Operation * TF_GraphNextOperation (TF_Graph *graph, size_t *pos);
		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_GraphNextOperation (IntPtr graph, ref IntPtr token);

		// Import the graph serialized in `graph_def` into `graph`.
		//
		// `num_return_outputs` must be the number of return outputs added (i.e. the
		// result of TF_ImportGraphDefOptionsNumReturnOutputs()).  If
		// `num_return_outputs` is non-zero, `return_outputs` must be of length
		// `num_return_outputs`. Otherwise it can be null.
		// extern void TF_GraphImportGraphDefWithReturnOutputs ( TF_Graph* graph, const TF_Buffer* graph_def, const TF_ImportGraphDefOptions* options, TF_Output* return_outputs,     int num_return_outputs, TF_Status* status);
		[DllImport (TensorFlowLibrary)]
		internal static extern unsafe void TF_GraphImportGraphDefWithReturnOutputs (IntPtr graph, LLBuffer* graph_def, IntPtr options, TFOutput* return_outputs, int num_return_outputs, IntPtr status);

		[StructLayout (LayoutKind.Sequential)]
		internal unsafe struct TFWhileParams
		{
			public int ninputs;
			public IntPtr cond_graph;
			public TFOutput* cond_inputs;
			public TFOutput cond_output;
			public IntPtr body_graph;
			public TFOutput* body_inputs;
			public TFOutput* body_outputs;
			public IntPtr charPtrName;
		}

		[DllImport (TensorFlowLibrary)]
		internal static extern TFWhileParams TF_NewWhile (IntPtr g, TFOutput [] inputs, int ninputs, IntPtr status);

		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_AbortWhile (ref TFWhileParams pars);

		[DllImport (TensorFlowLibrary)]
		internal static extern unsafe void TF_FinishWhile (ref TFWhileParams pars, IntPtr status, TFOutput* outputs);




		// extern TF_OperationDescription * TF_NewOperation (TF_Graph *graph, const char *op_type, const char *oper_name);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern IntPtr TF_NewOperation (IntPtr graph, string opType, string oper_name);

		// extern void TF_SetDevice (TF_OperationDescription *desc, const char *device);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern void TF_SetDevice (IntPtr desc, string device);

		// extern void TF_AddInput (TF_OperationDescription *desc, TF_Output input);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_AddInput (IntPtr desc, TFOutput input);

		// extern void TF_AddInputList (TF_OperationDescription *desc, const TF_Output *inputs, int num_inputs);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_AddInputList (IntPtr desc, TFOutput [] inputs, int num_inputs);


		// extern void TF_AddControlInput (TF_OperationDescription *desc, TF_Operation *input);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_AddControlInput (IntPtr desc, IntPtr input);

		// extern void TF_ColocateWith (TF_OperationDescription *desc, TF_Operation *op);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_ColocateWith (IntPtr desc, IntPtr op);

		// extern void TF_SetAttrString (TF_OperationDescription *desc, const char *attr_name, const void *value, size_t length);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_SetAttrString (IntPtr desc, string attr_name, IntPtr value, size_t length);

		// extern void TF_SetAttrStringList (TF_OperationDescription *desc, const char *attr_name, const void *const *values, const size_t *lengths, int num_values);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_SetAttrStringList (IntPtr desc, string attr_name, IntPtr [] values, size_t [] lengths, int num_values);


		// extern void TF_SetAttrInt (TF_OperationDescription *desc, const char *attr_name, int64_t value);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_SetAttrInt (IntPtr desc, string attr_name, long value);

		// extern void TF_SetAttrIntList (TF_OperationDescription *desc, const char *attr_name, const int64_t *values, int num_values);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_SetAttrIntList (IntPtr desc, string attr_name, long [] values, int num_values);

		// extern void TF_SetAttrFloat (TF_OperationDescription *desc, const char *attr_name, float value);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_SetAttrFloat (IntPtr desc, string attr_name, float value);

		// extern void TF_SetAttrFloatList (TF_OperationDescription *desc, const char *attr_name, const float *values, int num_values);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_SetAttrFloatList (IntPtr desc, string attr_name, float [] values, int num_values);

		// extern void TF_SetAttrBool (TF_OperationDescription *desc, const char *attr_name, unsigned char value);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern void TF_SetAttrBool (IntPtr desc, string attr_name, byte value);

		// extern void TF_SetAttrBoolList (TF_OperationDescription *desc, const char *attr_name, const unsigned char *values, int num_values);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern void TF_SetAttrBoolList (IntPtr desc, string attr_name, bool [] values, int num_values);

		// extern void TF_SetAttrType (TF_OperationDescription *desc, const char *attr_name, TF_DataType value);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern void TF_SetAttrType (IntPtr desc, string attr_name, TFDataType value);

		// extern void TF_SetAttrTypeList (TF_OperationDescription *desc, const char *attr_name, const TF_DataType *values, int num_values);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern void TF_SetAttrTypeList (IntPtr desc, string attr_name, TFDataType [] values, int num_values);

		// extern void TF_SetAttrShape (TF_OperationDescription *desc, const char *attr_name, const int64_t *dims, int num_dims);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern void TF_SetAttrShape (IntPtr desc, string attr_name, long [] dims, int num_dims);

		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern void TF_SetAttrShape (IntPtr desc, string attr_name, IntPtr dims, int num_dims);

		// extern void TF_SetAttrShapeList (TF_OperationDescription *desc, const char *attr_name, const int64_t *const *dims, const int *num_dims, int num_shapes);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern void TF_SetAttrShapeList (IntPtr desc, string attr_name, IntPtr dims, int [] num_dims, int num_shapes);

		// extern void TF_SetAttrTensorShapeProto (TF_OperationDescription *desc, const char *attr_name, const void *proto, size_t proto_len, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern void TF_SetAttrTensorShapeProto (IntPtr desc, string attr_name, IntPtr proto, size_t proto_len, IntPtr status);

		// extern void TF_SetAttrTensorShapeProtoList (TF_OperationDescription *desc, const char *attr_name, const void *const *protos, const size_t *proto_lens, int num_shapes, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe void TF_SetAttrTensorShapeProtoList (IntPtr desc, string attr_name, void** protos, size_t* proto_lens, int num_shapes, IntPtr status);

		// extern void TF_SetAttrTensor (TF_OperationDescription *desc, const char *attr_name, TF_Tensor *value, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern void TF_SetAttrTensor (IntPtr desc, string attr_name, IntPtr value, IntPtr status);

		// extern void TF_SetAttrTensorList (TF_OperationDescription *desc, const char *attr_name, TF_Tensor *const *values, int num_values, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern void TF_SetAttrTensorList (IntPtr desc, string attr_name, IntPtr [] values, int num_values, IntPtr status);

		// extern void TF_SetAttrValueProto (TF_OperationDescription *desc, const char *attr_name, const void *proto, size_t proto_len, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe void TF_SetAttrValueProto (IntPtr desc, string attr_name, void* proto, size_t proto_len, IntPtr status);

		// extern TF_Operation * TF_FinishOperation (TF_OperationDescription *desc, TF_Status *status);
		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_FinishOperation (IntPtr desc, IntPtr status);


		// extern const char * TF_OperationName (TF_Operation *oper);
		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_OperationName (IntPtr oper);

		// extern const char * TF_OperationOpType (TF_Operation *oper);
		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_OperationOpType (IntPtr oper);
		// extern const char * TF_OperationDevice (TF_Operation *oper);
		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_OperationDevice (IntPtr oper);

		// extern int TF_OperationNumOutputs (TF_Operation *oper);
		[DllImport (TensorFlowLibrary)]
		internal static extern int TF_OperationNumOutputs (IntPtr oper);

		// extern int TF_OperationOutputListLength (TF_Operation *oper, const char *arg_name, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern int TF_OperationOutputListLength (IntPtr oper, string arg_name, IntPtr status);

		// extern int TF_OperationNumInputs (TF_Operation *oper);
		[DllImport (TensorFlowLibrary)]
		internal static extern int TF_OperationNumInputs (IntPtr oper);
		// extern int TF_OperationInputListLength (TF_Operation *oper, const char *arg_name, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern int TF_OperationInputListLength (IntPtr oper, string arg_name, IntPtr status);

		// extern int TF_OperationNumControlInputs (TF_Operation *oper);
		[DllImport (TensorFlowLibrary)]
		internal static extern int TF_OperationNumControlInputs (IntPtr oper);

		// extern int TF_OperationGetControlInputs (TF_Operation *oper, TF_Operation **control_inputs, int max_control_inputs);
		[DllImport (TensorFlowLibrary)]
		internal static extern int TF_OperationGetControlInputs (IntPtr oper, IntPtr control_inputs, int max_control_inputs);

		// extern int TF_OperationNumControlOutputs (TF_Operation *oper);
		[DllImport (TensorFlowLibrary)]
		internal static extern int TF_OperationNumControlOutputs (IntPtr oper);

		// extern int TF_OperationGetControlOutputs (TF_Operation *oper, TF_Operation **control_outputs, int max_control_outputs);
		[DllImport (TensorFlowLibrary)]
		internal static extern int TF_OperationGetControlOutputs (IntPtr oper, [Out] [MarshalAs (UnmanagedType.LPArray, SizeParamIndex = 2)] IntPtr [] control_outputs, int max_control_outputs);

		// extern TF_AttrMetadata TF_OperationGetAttrMetadata (TF_Operation *oper, const char *attr_name, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern TFAttributeMetadata TF_OperationGetAttrMetadata (IntPtr oper, string attr_name, IntPtr status);

		// extern void TF_OperationGetAttrString (TF_Operation *oper, const char *attr_name, void *value, size_t max_length, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe void TF_OperationGetAttrString (IntPtr oper, string attr_name, void* value, size_t max_length, IntPtr status);

		// extern void TF_OperationGetAttrStringList (TF_Operation *oper, const char *attr_name, void **values, size_t *lengths, int max_values, void *storage, size_t storage_size, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe void TF_OperationGetAttrStringList (IntPtr oper, string attr_name, void** values, size_t* lengths, int max_values, void* storage, size_t storage_size, IntPtr status);

		// extern void TF_OperationGetAttrInt (TF_Operation *oper, const char *attr_name, int64_t *value, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe void TF_OperationGetAttrInt (IntPtr oper, string attr_name, long* value, IntPtr status);

		// extern void TF_OperationGetAttrIntList (TF_Operation *oper, const char *attr_name, int64_t *values, int max_values, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe void TF_OperationGetAttrIntList (IntPtr oper, string attr_name, long* values, int max_values, IntPtr status);

		// extern void TF_OperationGetAttrFloat (TF_Operation *oper, const char *attr_name, float *value, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe void TF_OperationGetAttrFloat (IntPtr oper, string attr_name, float* value, IntPtr status);

		// extern void TF_OperationGetAttrFloatList (TF_Operation *oper, const char *attr_name, float *values, int max_values, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe void TF_OperationGetAttrFloatList (IntPtr oper, string attr_name, float* values, int max_values, IntPtr status);

		// extern void TF_OperationGetAttrBool (TF_Operation *oper, const char *attr_name, unsigned char *value, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe void TF_OperationGetAttrBool (IntPtr oper, string attr_name, byte* value, IntPtr status);

		// extern void TF_OperationGetAttrBoolList (TF_Operation *oper, const char *attr_name, unsigned char *values, int max_values, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe void TF_OperationGetAttrBoolList (IntPtr oper, string attr_name, byte* values, int max_values, IntPtr status);

		// extern void TF_OperationGetAttrType (TF_Operation *oper, const char *attr_name, TF_DataType *value, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe void TF_OperationGetAttrType (IntPtr oper, string attr_name, TFDataType* value, IntPtr status);

		// extern void TF_OperationGetAttrTypeList (TF_Operation *oper, const char *attr_name, TF_DataType *values, int max_values, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe void TF_OperationGetAttrTypeList (IntPtr oper, string attr_name, TFDataType* values, int max_values, IntPtr status);

		// extern void TF_OperationGetAttrShape (TF_Operation *oper, const char *attr_name, int64_t *value, int num_dims, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe void TF_OperationGetAttrShape (IntPtr oper, string attr_name, long* value, int num_dims, IntPtr status);

		// extern void TF_OperationGetAttrShapeList (TF_Operation *oper, const char *attr_name, int64_t **dims, int *num_dims, int num_shapes, int64_t *storage, int storage_size, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe void TF_OperationGetAttrShapeList (IntPtr oper, string attr_name, long** dims, int* num_dims, int num_shapes, long* storage, int storage_size, IntPtr status);

		// extern void TF_OperationGetAttrTensorShapeProto (TF_Operation *oper, const char *attr_name, TF_Buffer *value, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe void TF_OperationGetAttrTensorShapeProto (IntPtr oper, string attr_name, LLBuffer* value, IntPtr status);

		// extern void TF_OperationGetAttrTensorShapeProtoList (TF_Operation *oper, const char *attr_name, TF_Buffer **values, int max_values, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe void TF_OperationGetAttrTensorShapeProtoList (IntPtr oper, string attr_name, LLBuffer** values, int max_values, IntPtr status);

		// extern void TF_OperationGetAttrTensor (TF_Operation *oper, const char *attr_name, TF_Tensor **value, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe void TF_OperationGetAttrTensor (IntPtr oper, string attr_name, IntPtr* value, IntPtr status);

		// extern void TF_OperationGetAttrTensorList (TF_Operation *oper, const char *attr_name, TF_Tensor **values, int max_values, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe void TF_OperationGetAttrTensorList (IntPtr oper, string attr_name, IntPtr* values, int max_values, IntPtr status);

		// extern void TF_OperationGetAttrValueProto (TF_Operation *oper, const char *attr_name, TF_Buffer *output_attr_value, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe void TF_OperationGetAttrValueProto (IntPtr oper, string attr_name, LLBuffer* output_attr_value, IntPtr status);

		// extern void TF_OperationToNodeDef (TF_Operation *oper, TF_Buffer *output_node_def, TF_Status *status);
		[DllImport (TensorFlowLibrary)]
		internal static extern unsafe void TF_OperationToNodeDef (IntPtr oper, LLBuffer* output_node_def, IntPtr status);

		// extern TF_ImportGraphDefOptions * TF_NewImportGraphDefOptions ();
		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_NewImportGraphDefOptions ();

		// extern void TF_DeleteImportGraphDefOptions (TF_ImportGraphDefOptions *opts);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_DeleteImportGraphDefOptions (IntPtr opts);

		// extern void TF_ImportGraphDefOptionsSetPrefix (TF_ImportGraphDefOptions *opts, const char *prefix);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern void TF_ImportGraphDefOptionsSetPrefix (IntPtr opts, string prefix);

		// extern void TF_ImportGraphDefOptionsAddInputMapping (TF_ImportGraphDefOptions *opts, const char* src_name, int src_index, TF_Output dst);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern void TF_ImportGraphDefOptionsAddInputMapping (IntPtr opts, string src_name, int src_index, TFOutput dst);

		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_ImportGraphDefOptionsAddControlDependency (IntPtr opts, IntPtr oper);

		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern void TF_ImportGraphDefOptionsAddReturnOutput (IntPtr opts, string oper_name, int index);

		[DllImport (TensorFlowLibrary)]
		internal static extern int TF_ImportGraphDefOptionsNumReturnOutputs (IntPtr opts);


		// extern TF_Output TF_OperationInput (TF_Input oper_in);
		[DllImport (TensorFlowLibrary)]
		internal static extern TFOutput TF_OperationInput (TFInput oper_in);

		// extern TF_DataType TF_OperationInputType (TF_Input oper_in);
		[DllImport (TensorFlowLibrary)]
		internal static extern TFDataType TF_OperationInputType (TFInput oper_in);



		// extern int TF_OperationOutputNumConsumers (TF_Output oper_out);
		[DllImport (TensorFlowLibrary)]
		internal static extern int TF_OperationOutputNumConsumers (TFOutput oper_out);

		// extern TF_DataType TF_OperationOutputType (TF_Output oper_out);
		[DllImport (TensorFlowLibrary)]
		internal static extern TFDataType TF_OperationOutputType (TFOutput oper_out);

		// extern int TF_OperationOutputConsumers (TF_Output oper_out, TF_Input *consumers, int max_consumers);
		[DllImport (TensorFlowLibrary)]
		internal static extern unsafe int TF_OperationOutputConsumers (TFOutput oper_out, TFInput* consumers, int max_consumers);



		// extern TF_Session * TF_NewSession (TF_Graph *graph, const TF_SessionOptions *opts, TF_Status *status);
		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_NewSession (IntPtr graph, IntPtr opts, IntPtr status);

		// extern TF_Session * TF_LoadSessionFromSavedModel (const TF_SessionOptions *session_options, const TF_Buffer *run_options, const char *export_dir, const char *const *tags, int tags_len, TF_Graph *graph, TF_Buffer *meta_graph_def, TF_Status *status);
		[DllImport (TensorFlowLibrary, CharSet = CharSet.Ansi)]
		internal static extern unsafe IntPtr TF_LoadSessionFromSavedModel (IntPtr session_options, LLBuffer* run_options, string export_dir, string [] tags, int tags_len, IntPtr graph, LLBuffer* meta_graph_def, IntPtr status);

		// extern void TF_CloseSession (TF_Session *, TF_Status *status);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_CloseSession (IntPtr session, IntPtr status);

		// extern void TF_DeleteSession (TF_Session *, TF_Status *status);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_DeleteSession (IntPtr session, IntPtr status);

		// extern void TF_SessionRun (TF_Session *session, const TF_Buffer *run_options, const TF_Output *inputs, TF_Tensor *const *input_values, int ninputs, const TF_Output *outputs, TF_Tensor **output_values, int noutputs, const TF_Operation *const *target_opers, int ntargets, TF_Buffer *run_metadata, TF_Status *);
		[DllImport (TensorFlowLibrary)]
		internal static extern unsafe void TF_SessionRun (IntPtr session, LLBuffer* run_options, TFOutput [] inputs, IntPtr [] input_values, int ninputs, TFOutput [] outputs, IntPtr [] output_values, int noutputs, IntPtr [] target_opers, int ntargets, LLBuffer* run_metadata, IntPtr status);

		// extern void TF_SessionPRunSetup (TF_Session, const TF_Output *inputs, int ninputs, const TF_Output *outputs, int noutputs, const TF_Operation *const *target_opers, int ntargets, const char **handle, TF_Status *);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_SessionPRunSetup (IntPtr session, TFOutput [] inputs, int ninputs, TFOutput [] outputs, int noutputs, IntPtr [] target_opers, int ntargets, out IntPtr returnHandle, IntPtr status);

		[DllImport (TensorFlowLibrary)]
		// extern void TF_SessionPRun (TF_Session *, const char *handle, const TF_Output *inputs, TF_Tensor *const *input_values, int ninputs, const TF_Output *outputs, TF_Tensor **output_values, int noutputs, const TF_Operation *const *target_opers, int ntargets, TF_Status *);
		internal static extern void TF_SessionPRun (IntPtr session, IntPtr partialHandle, TFOutput [] inputs, IntPtr [] input_values, int ninputs, TFOutput [] outputs, IntPtr [] output_values, int noutputs, IntPtr [] target_opers, int ntargets, IntPtr status);

		// extern void TF_DeletePRunHandle(const char* handle);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_DeletePRunHandle (IntPtr partialRunHandle);


		// extern TF_Library * TF_LoadLibrary (const char *library_filename, TF_Status *status);
		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_LoadLibrary (string library_filename, IntPtr status);

		// extern TF_Buffer TF_GetOpList (TF_Library *lib_handle);
		[DllImport (TensorFlowLibrary)]
		internal static extern TFBuffer TF_GetOpList (IntPtr lib_handle);

		// extern void TF_DeleteLibraryHandle (TF_Library *lib_handle);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_DeleteLibraryHandle (IntPtr lib_handle);




		// extern TF_Tensor * TF_NewTensor (TF_DataType, const int64_t *dims, int num_dims, void *data, size_t len, void (* deallocator)(void *, size_t, void *), void *deallocator_arg);
		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_NewTensor (TFDataType dataType, long [] dims, int num_dims, IntPtr data, size_t len, TFTensor.Deallocator deallocator, IntPtr deallocator_arg);

		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_NewTensor (TFDataType dataType, IntPtr zeroDims, int num_dims, IntPtr data, size_t len, TFTensor.Deallocator deallocator, IntPtr deallocator_arg);


		// extern TF_Tensor * TF_AllocateTensor (TF_DataType, const int64_t *dims, int num_dims, size_t len);
		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_AllocateTensor (TFDataType dataType, long [] dims, int num_dims, size_t len);
		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_AllocateTensor (TFDataType dataType, IntPtr zeroDim, int num_dims, size_t len);


		// extern void TF_DeleteTensor (TF_Tensor *);
		[DllImport (TensorFlowLibrary)]
		internal static extern void TF_DeleteTensor (IntPtr tensor);

		// extern TF_DataType TF_TensorType (const TF_Tensor *);
		[DllImport (TensorFlowLibrary)]
		internal static extern TFDataType TF_TensorType (IntPtr tensor);

		// extern int TF_NumDims (const TF_Tensor *);
		[DllImport (TensorFlowLibrary)]
		internal static extern int TF_NumDims (IntPtr tensor);

		// extern int64_t TF_Dim (const TF_Tensor *tensor, int dim_index);
		[DllImport (TensorFlowLibrary)]
		internal static extern long TF_Dim (IntPtr tensor, int dim_index);

		// extern size_t TF_TensorByteSize (const TF_Tensor *);
		[DllImport (TensorFlowLibrary)]
		internal static extern size_t TF_TensorByteSize (IntPtr tensor);

		// extern void * TF_TensorData (const TF_Tensor *);
		[DllImport (TensorFlowLibrary)]
		internal static extern IntPtr TF_TensorData (IntPtr tensor);



	}
}
