??
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12unknown8??
?
feature_embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_namefeature_embeddings
{
&feature_embeddings/Read/ReadVariableOpReadVariableOpfeature_embeddings* 
_output_shapes
:
??*
dtype0
v
feature_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namefeature_bias
o
 feature_bias/Read/ReadVariableOpReadVariableOpfeature_bias* 
_output_shapes
:
??*
dtype0
q
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:2?*
shared_name
Variable
j
Variable/Read/ReadVariableOpReadVariableOpVariable*#
_output_shapes
:2?*
dtype0
l

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_name
Variable_1
e
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:2*
dtype0
t

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_name
Variable_2
m
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*"
_output_shapes
:2*
dtype0
p

Variable_3VarHandleOp*
_output_shapes
: *
dtype0*
shape
:2 *
shared_name
Variable_3
i
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes

:2 *
dtype0
p

Variable_4VarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_name
Variable_4
i
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes

: *
dtype0
p

Variable_5VarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_name
Variable_5
i
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes

:  *
dtype0
p

Variable_6VarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_name
Variable_6
i
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes

: *
dtype0
l

Variable_7VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Variable_7
e
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*
_output_shapes
:*
dtype0
l

Variable_8VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Variable_8
e
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes
:*
dtype0
p

Variable_9VarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_name
Variable_9
i
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*
_output_shapes

: *
dtype0
j
Variable_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_10
c
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10*
_output_shapes
: *
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
deep_layers
dropout_deep
_weights
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
 
?
	feature_embeddings

feature_bias
product-linear
product-bias
product-quadratic-outer
layer_0

bias_0
layer_1

bias_1
batch_norm_scale_0
batch_norm_offset_0

output
output_bias
 
^
	0

1
2
3
4
5
6
7
8
9
10
11
12
^
	0

1
2
3
4
5
6
7
8
9
10
11
12
?
regularization_losses
trainable_variables

layers
layer_metrics
non_trainable_variables
layer_regularization_losses
	variables
metrics
 
^\
VARIABLE_VALUEfeature_embeddings6_weights/feature_embeddings/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEfeature_bias0_weights/feature_bias/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEVariable2_weights/product-linear/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUE
Variable_10_weights/product-bias/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUE
Variable_2;_weights/product-quadratic-outer/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUE
Variable_3+_weights/layer_0/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Variable_4*_weights/bias_0/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUE
Variable_5+_weights/layer_1/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Variable_6*_weights/bias_1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUE
Variable_76_weights/batch_norm_scale_0/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUE
Variable_87_weights/batch_norm_offset_0/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Variable_9*_weights/output/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEVariable_10/_weights/output_bias/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
|
serving_default_input_2Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2feature_embeddingsVariable
Variable_2
Variable_1
Variable_3
Variable_4
Variable_8
Variable_7
Variable_5
Variable_6
Variable_9Variable_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_16075377
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&feature_embeddings/Read/ReadVariableOp feature_bias/Read/ReadVariableOpVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable_2/Read/ReadVariableOpVariable_3/Read/ReadVariableOpVariable_4/Read/ReadVariableOpVariable_5/Read/ReadVariableOpVariable_6/Read/ReadVariableOpVariable_7/Read/ReadVariableOpVariable_8/Read/ReadVariableOpVariable_9/Read/ReadVariableOpVariable_10/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_save_16075440
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamefeature_embeddingsfeature_biasVariable
Variable_1
Variable_2
Variable_3
Variable_4
Variable_5
Variable_6
Variable_7
Variable_8
Variable_9Variable_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__traced_restore_16075489??
?%
?
!__inference__traced_save_16075440
file_prefix1
-savev2_feature_embeddings_read_readvariableop+
'savev2_feature_bias_read_readvariableop'
#savev2_variable_read_readvariableop)
%savev2_variable_1_read_readvariableop)
%savev2_variable_2_read_readvariableop)
%savev2_variable_3_read_readvariableop)
%savev2_variable_4_read_readvariableop)
%savev2_variable_5_read_readvariableop)
%savev2_variable_6_read_readvariableop)
%savev2_variable_7_read_readvariableop)
%savev2_variable_8_read_readvariableop)
%savev2_variable_9_read_readvariableop*
&savev2_variable_10_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6_weights/feature_embeddings/.ATTRIBUTES/VARIABLE_VALUEB0_weights/feature_bias/.ATTRIBUTES/VARIABLE_VALUEB2_weights/product-linear/.ATTRIBUTES/VARIABLE_VALUEB0_weights/product-bias/.ATTRIBUTES/VARIABLE_VALUEB;_weights/product-quadratic-outer/.ATTRIBUTES/VARIABLE_VALUEB+_weights/layer_0/.ATTRIBUTES/VARIABLE_VALUEB*_weights/bias_0/.ATTRIBUTES/VARIABLE_VALUEB+_weights/layer_1/.ATTRIBUTES/VARIABLE_VALUEB*_weights/bias_1/.ATTRIBUTES/VARIABLE_VALUEB6_weights/batch_norm_scale_0/.ATTRIBUTES/VARIABLE_VALUEB7_weights/batch_norm_offset_0/.ATTRIBUTES/VARIABLE_VALUEB*_weights/output/.ATTRIBUTES/VARIABLE_VALUEB/_weights/output_bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_feature_embeddings_read_readvariableop'savev2_feature_bias_read_readvariableop#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableop%savev2_variable_3_read_readvariableop%savev2_variable_4_read_readvariableop%savev2_variable_5_read_readvariableop%savev2_variable_6_read_readvariableop%savev2_variable_7_read_readvariableop%savev2_variable_8_read_readvariableop%savev2_variable_9_read_readvariableop&savev2_variable_10_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
: :
??:
??:2?:2:2:2 : :  : ::: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:)%
#
_output_shapes
:2?: 

_output_shapes
:2:($
"
_output_shapes
:2:$ 

_output_shapes

:2 :$ 

_output_shapes

: :$ 

_output_shapes

:  :$	 

_output_shapes

: : 


_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

: :

_output_shapes
: :

_output_shapes
: 
??

?
A__inference_pnn_layer_call_and_return_conditional_losses_16075314
input_1
input_2
embedding_lookup_16074104
readvariableop_resource
readvariableop_50_resource!
add_1_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_2_readvariableop_resource)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource$
 matmul_2_readvariableop_resource!
add_3_readvariableop_resource$
 matmul_3_readvariableop_resource!
add_4_readvariableop_resource
identity??MatMul_1/ReadVariableOp?MatMul_2/ReadVariableOp?MatMul_3/ReadVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_10?ReadVariableOp_11?ReadVariableOp_12?ReadVariableOp_13?ReadVariableOp_14?ReadVariableOp_15?ReadVariableOp_16?ReadVariableOp_17?ReadVariableOp_18?ReadVariableOp_19?ReadVariableOp_2?ReadVariableOp_20?ReadVariableOp_21?ReadVariableOp_22?ReadVariableOp_23?ReadVariableOp_24?ReadVariableOp_25?ReadVariableOp_26?ReadVariableOp_27?ReadVariableOp_28?ReadVariableOp_29?ReadVariableOp_3?ReadVariableOp_30?ReadVariableOp_31?ReadVariableOp_32?ReadVariableOp_33?ReadVariableOp_34?ReadVariableOp_35?ReadVariableOp_36?ReadVariableOp_37?ReadVariableOp_38?ReadVariableOp_39?ReadVariableOp_4?ReadVariableOp_40?ReadVariableOp_41?ReadVariableOp_42?ReadVariableOp_43?ReadVariableOp_44?ReadVariableOp_45?ReadVariableOp_46?ReadVariableOp_47?ReadVariableOp_48?ReadVariableOp_49?ReadVariableOp_5?ReadVariableOp_50?ReadVariableOp_51?ReadVariableOp_52?ReadVariableOp_53?ReadVariableOp_54?ReadVariableOp_55?ReadVariableOp_56?ReadVariableOp_57?ReadVariableOp_58?ReadVariableOp_59?ReadVariableOp_6?ReadVariableOp_60?ReadVariableOp_61?ReadVariableOp_62?ReadVariableOp_63?ReadVariableOp_64?ReadVariableOp_65?ReadVariableOp_66?ReadVariableOp_67?ReadVariableOp_68?ReadVariableOp_69?ReadVariableOp_7?ReadVariableOp_70?ReadVariableOp_71?ReadVariableOp_72?ReadVariableOp_73?ReadVariableOp_74?ReadVariableOp_75?ReadVariableOp_76?ReadVariableOp_77?ReadVariableOp_78?ReadVariableOp_79?ReadVariableOp_8?ReadVariableOp_80?ReadVariableOp_81?ReadVariableOp_82?ReadVariableOp_83?ReadVariableOp_84?ReadVariableOp_85?ReadVariableOp_86?ReadVariableOp_87?ReadVariableOp_88?ReadVariableOp_89?ReadVariableOp_9?ReadVariableOp_90?ReadVariableOp_91?ReadVariableOp_92?ReadVariableOp_93?ReadVariableOp_94?ReadVariableOp_95?ReadVariableOp_96?ReadVariableOp_97?ReadVariableOp_98?ReadVariableOp_99?add_1/ReadVariableOp?add_2/ReadVariableOp?add_3/ReadVariableOp?add_4/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?batchnorm_1/ReadVariableOp?batchnorm_1/mul/ReadVariableOp?embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_16074104input_1",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/16074104*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/16074104*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1s
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"?????     2
Reshape/shapeu
ReshapeReshapeinput_2Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2	
Reshape?
MulMul$embedding_lookup/Identity_1:output:0Reshape:output:0*
T0*,
_output_shapes
:??????????2
Mul}
ReadVariableOpReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOpt
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slicem
Mul_1MulMul:z:0strided_slice:output:0*
T0*,
_output_shapes
:??????????2
Mul_1
Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum/reduction_indicesj
SumSum	Mul_1:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sums
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_1/shape{
	Reshape_1ReshapeSum:output:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_1?
ReadVariableOp_1ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_1o
Mul_2MulMul:z:0strided_slice_1:output:0*
T0*,
_output_shapes
:??????????2
Mul_2?
Sum_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_1/reduction_indicesp
Sum_1Sum	Mul_2:z:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_1s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_2/shape}
	Reshape_2ReshapeSum_1:output:0Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_2x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_2o
Mul_3MulMul:z:0strided_slice_2:output:0*
T0*,
_output_shapes
:??????????2
Mul_3?
Sum_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_2/reduction_indicesp
Sum_2Sum	Mul_3:z:0 Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_2s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_3/shape}
	Reshape_3ReshapeSum_2:output:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_3?
ReadVariableOp_3ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_3x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_3o
Mul_4MulMul:z:0strided_slice_3:output:0*
T0*,
_output_shapes
:??????????2
Mul_4?
Sum_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_3/reduction_indicesp
Sum_3Sum	Mul_4:z:0 Sum_3/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_3s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_4/shape}
	Reshape_4ReshapeSum_3:output:0Reshape_4/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_4?
ReadVariableOp_4ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_4x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_4o
Mul_5MulMul:z:0strided_slice_4:output:0*
T0*,
_output_shapes
:??????????2
Mul_5?
Sum_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_4/reduction_indicesp
Sum_4Sum	Mul_5:z:0 Sum_4/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_4s
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_5/shape}
	Reshape_5ReshapeSum_4:output:0Reshape_5/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_5?
ReadVariableOp_5ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_5x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_5o
Mul_6MulMul:z:0strided_slice_5:output:0*
T0*,
_output_shapes
:??????????2
Mul_6?
Sum_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_5/reduction_indicesp
Sum_5Sum	Mul_6:z:0 Sum_5/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_5s
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_6/shape}
	Reshape_6ReshapeSum_5:output:0Reshape_6/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_6?
ReadVariableOp_6ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_6x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_6o
Mul_7MulMul:z:0strided_slice_6:output:0*
T0*,
_output_shapes
:??????????2
Mul_7?
Sum_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_6/reduction_indicesp
Sum_6Sum	Mul_7:z:0 Sum_6/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_6s
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_7/shape}
	Reshape_7ReshapeSum_6:output:0Reshape_7/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_7?
ReadVariableOp_7ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_7x
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack|
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_1|
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_7o
Mul_8MulMul:z:0strided_slice_7:output:0*
T0*,
_output_shapes
:??????????2
Mul_8?
Sum_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_7/reduction_indicesp
Sum_7Sum	Mul_8:z:0 Sum_7/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_7s
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_8/shape}
	Reshape_8ReshapeSum_7:output:0Reshape_8/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_8?
ReadVariableOp_8ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_8x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:	2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_8o
Mul_9MulMul:z:0strided_slice_8:output:0*
T0*,
_output_shapes
:??????????2
Mul_9?
Sum_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_8/reduction_indicesp
Sum_8Sum	Mul_9:z:0 Sum_8/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_8s
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_9/shape}
	Reshape_9ReshapeSum_8:output:0Reshape_9/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_9?
ReadVariableOp_9ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_9x
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:	2
strided_slice_9/stack|
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
2
strided_slice_9/stack_1|
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack_2?
strided_slice_9StridedSliceReadVariableOp_9:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_9q
Mul_10MulMul:z:0strided_slice_9:output:0*
T0*,
_output_shapes
:??????????2
Mul_10?
Sum_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_9/reduction_indicesq
Sum_9Sum
Mul_10:z:0 Sum_9/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_9u
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_10/shape?

Reshape_10ReshapeSum_9:output:0Reshape_10/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_10?
ReadVariableOp_10ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_10z
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB:
2
strided_slice_10/stack~
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_10/stack_1~
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_10/stack_2?
strided_slice_10StridedSliceReadVariableOp_10:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_10r
Mul_11MulMul:z:0strided_slice_10:output:0*
T0*,
_output_shapes
:??????????2
Mul_11?
Sum_10/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_10/reduction_indicest
Sum_10Sum
Mul_11:z:0!Sum_10/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_10u
Reshape_11/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_11/shape?

Reshape_11ReshapeSum_10:output:0Reshape_11/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_11?
ReadVariableOp_11ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_11z
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack~
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_1~
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_2?
strided_slice_11StridedSliceReadVariableOp_11:value:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_11r
Mul_12MulMul:z:0strided_slice_11:output:0*
T0*,
_output_shapes
:??????????2
Mul_12?
Sum_11/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_11/reduction_indicest
Sum_11Sum
Mul_12:z:0!Sum_11/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_11u
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_12/shape?

Reshape_12ReshapeSum_11:output:0Reshape_12/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_12?
ReadVariableOp_12ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_12z
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack~
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack_1~
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack_2?
strided_slice_12StridedSliceReadVariableOp_12:value:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_12r
Mul_13MulMul:z:0strided_slice_12:output:0*
T0*,
_output_shapes
:??????????2
Mul_13?
Sum_12/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_12/reduction_indicest
Sum_12Sum
Mul_13:z:0!Sum_12/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_12u
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_13/shape?

Reshape_13ReshapeSum_12:output:0Reshape_13/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_13?
ReadVariableOp_13ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_13z
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_13/stack~
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_13/stack_1~
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_13/stack_2?
strided_slice_13StridedSliceReadVariableOp_13:value:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_13r
Mul_14MulMul:z:0strided_slice_13:output:0*
T0*,
_output_shapes
:??????????2
Mul_14?
Sum_13/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_13/reduction_indicest
Sum_13Sum
Mul_14:z:0!Sum_13/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_13u
Reshape_14/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_14/shape?

Reshape_14ReshapeSum_13:output:0Reshape_14/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_14?
ReadVariableOp_14ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_14z
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack~
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack_1~
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack_2?
strided_slice_14StridedSliceReadVariableOp_14:value:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_14r
Mul_15MulMul:z:0strided_slice_14:output:0*
T0*,
_output_shapes
:??????????2
Mul_15?
Sum_14/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_14/reduction_indicest
Sum_14Sum
Mul_15:z:0!Sum_14/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_14u
Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_15/shape?

Reshape_15ReshapeSum_14:output:0Reshape_15/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_15?
ReadVariableOp_15ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_15z
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_15/stack~
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_15/stack_1~
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_15/stack_2?
strided_slice_15StridedSliceReadVariableOp_15:value:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_15r
Mul_16MulMul:z:0strided_slice_15:output:0*
T0*,
_output_shapes
:??????????2
Mul_16?
Sum_15/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_15/reduction_indicest
Sum_15Sum
Mul_16:z:0!Sum_15/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_15u
Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_16/shape?

Reshape_16ReshapeSum_15:output:0Reshape_16/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_16?
ReadVariableOp_16ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_16z
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_16/stack~
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_16/stack_1~
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_16/stack_2?
strided_slice_16StridedSliceReadVariableOp_16:value:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_16r
Mul_17MulMul:z:0strided_slice_16:output:0*
T0*,
_output_shapes
:??????????2
Mul_17?
Sum_16/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_16/reduction_indicest
Sum_16Sum
Mul_17:z:0!Sum_16/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_16u
Reshape_17/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_17/shape?

Reshape_17ReshapeSum_16:output:0Reshape_17/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_17?
ReadVariableOp_17ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_17z
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_17/stack~
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_17/stack_1~
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_17/stack_2?
strided_slice_17StridedSliceReadVariableOp_17:value:0strided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_17r
Mul_18MulMul:z:0strided_slice_17:output:0*
T0*,
_output_shapes
:??????????2
Mul_18?
Sum_17/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_17/reduction_indicest
Sum_17Sum
Mul_18:z:0!Sum_17/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_17u
Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_18/shape?

Reshape_18ReshapeSum_17:output:0Reshape_18/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_18?
ReadVariableOp_18ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_18z
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_18/stack~
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_18/stack_1~
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_18/stack_2?
strided_slice_18StridedSliceReadVariableOp_18:value:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_18r
Mul_19MulMul:z:0strided_slice_18:output:0*
T0*,
_output_shapes
:??????????2
Mul_19?
Sum_18/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_18/reduction_indicest
Sum_18Sum
Mul_19:z:0!Sum_18/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_18u
Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_19/shape?

Reshape_19ReshapeSum_18:output:0Reshape_19/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_19?
ReadVariableOp_19ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_19z
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_19/stack~
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_19/stack_1~
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_19/stack_2?
strided_slice_19StridedSliceReadVariableOp_19:value:0strided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_19r
Mul_20MulMul:z:0strided_slice_19:output:0*
T0*,
_output_shapes
:??????????2
Mul_20?
Sum_19/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_19/reduction_indicest
Sum_19Sum
Mul_20:z:0!Sum_19/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_19u
Reshape_20/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_20/shape?

Reshape_20ReshapeSum_19:output:0Reshape_20/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_20?
ReadVariableOp_20ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_20z
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_20/stack~
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_20/stack_1~
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_20/stack_2?
strided_slice_20StridedSliceReadVariableOp_20:value:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_20r
Mul_21MulMul:z:0strided_slice_20:output:0*
T0*,
_output_shapes
:??????????2
Mul_21?
Sum_20/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_20/reduction_indicest
Sum_20Sum
Mul_21:z:0!Sum_20/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_20u
Reshape_21/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_21/shape?

Reshape_21ReshapeSum_20:output:0Reshape_21/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_21?
ReadVariableOp_21ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_21z
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_21/stack~
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_21/stack_1~
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_21/stack_2?
strided_slice_21StridedSliceReadVariableOp_21:value:0strided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_21r
Mul_22MulMul:z:0strided_slice_21:output:0*
T0*,
_output_shapes
:??????????2
Mul_22?
Sum_21/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_21/reduction_indicest
Sum_21Sum
Mul_22:z:0!Sum_21/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_21u
Reshape_22/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_22/shape?

Reshape_22ReshapeSum_21:output:0Reshape_22/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_22?
ReadVariableOp_22ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_22z
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_22/stack~
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_22/stack_1~
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_22/stack_2?
strided_slice_22StridedSliceReadVariableOp_22:value:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_22r
Mul_23MulMul:z:0strided_slice_22:output:0*
T0*,
_output_shapes
:??????????2
Mul_23?
Sum_22/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_22/reduction_indicest
Sum_22Sum
Mul_23:z:0!Sum_22/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_22u
Reshape_23/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_23/shape?

Reshape_23ReshapeSum_22:output:0Reshape_23/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_23?
ReadVariableOp_23ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_23z
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_23/stack~
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_23/stack_1~
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_23/stack_2?
strided_slice_23StridedSliceReadVariableOp_23:value:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_23r
Mul_24MulMul:z:0strided_slice_23:output:0*
T0*,
_output_shapes
:??????????2
Mul_24?
Sum_23/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_23/reduction_indicest
Sum_23Sum
Mul_24:z:0!Sum_23/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_23u
Reshape_24/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_24/shape?

Reshape_24ReshapeSum_23:output:0Reshape_24/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_24?
ReadVariableOp_24ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_24z
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_24/stack~
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_24/stack_1~
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_24/stack_2?
strided_slice_24StridedSliceReadVariableOp_24:value:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_24r
Mul_25MulMul:z:0strided_slice_24:output:0*
T0*,
_output_shapes
:??????????2
Mul_25?
Sum_24/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_24/reduction_indicest
Sum_24Sum
Mul_25:z:0!Sum_24/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_24u
Reshape_25/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_25/shape?

Reshape_25ReshapeSum_24:output:0Reshape_25/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_25?
ReadVariableOp_25ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_25z
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_25/stack~
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_25/stack_1~
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_25/stack_2?
strided_slice_25StridedSliceReadVariableOp_25:value:0strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_25r
Mul_26MulMul:z:0strided_slice_25:output:0*
T0*,
_output_shapes
:??????????2
Mul_26?
Sum_25/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_25/reduction_indicest
Sum_25Sum
Mul_26:z:0!Sum_25/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_25u
Reshape_26/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_26/shape?

Reshape_26ReshapeSum_25:output:0Reshape_26/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_26?
ReadVariableOp_26ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_26z
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_26/stack~
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_26/stack_1~
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_26/stack_2?
strided_slice_26StridedSliceReadVariableOp_26:value:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_26r
Mul_27MulMul:z:0strided_slice_26:output:0*
T0*,
_output_shapes
:??????????2
Mul_27?
Sum_26/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_26/reduction_indicest
Sum_26Sum
Mul_27:z:0!Sum_26/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_26u
Reshape_27/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_27/shape?

Reshape_27ReshapeSum_26:output:0Reshape_27/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_27?
ReadVariableOp_27ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_27z
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_27/stack~
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_27/stack_1~
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_27/stack_2?
strided_slice_27StridedSliceReadVariableOp_27:value:0strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_27r
Mul_28MulMul:z:0strided_slice_27:output:0*
T0*,
_output_shapes
:??????????2
Mul_28?
Sum_27/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_27/reduction_indicest
Sum_27Sum
Mul_28:z:0!Sum_27/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_27u
Reshape_28/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_28/shape?

Reshape_28ReshapeSum_27:output:0Reshape_28/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_28?
ReadVariableOp_28ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_28z
strided_slice_28/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_28/stack~
strided_slice_28/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_28/stack_1~
strided_slice_28/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_28/stack_2?
strided_slice_28StridedSliceReadVariableOp_28:value:0strided_slice_28/stack:output:0!strided_slice_28/stack_1:output:0!strided_slice_28/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_28r
Mul_29MulMul:z:0strided_slice_28:output:0*
T0*,
_output_shapes
:??????????2
Mul_29?
Sum_28/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_28/reduction_indicest
Sum_28Sum
Mul_29:z:0!Sum_28/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_28u
Reshape_29/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_29/shape?

Reshape_29ReshapeSum_28:output:0Reshape_29/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_29?
ReadVariableOp_29ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_29z
strided_slice_29/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_29/stack~
strided_slice_29/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_29/stack_1~
strided_slice_29/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_29/stack_2?
strided_slice_29StridedSliceReadVariableOp_29:value:0strided_slice_29/stack:output:0!strided_slice_29/stack_1:output:0!strided_slice_29/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_29r
Mul_30MulMul:z:0strided_slice_29:output:0*
T0*,
_output_shapes
:??????????2
Mul_30?
Sum_29/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_29/reduction_indicest
Sum_29Sum
Mul_30:z:0!Sum_29/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_29u
Reshape_30/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_30/shape?

Reshape_30ReshapeSum_29:output:0Reshape_30/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_30?
ReadVariableOp_30ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_30z
strided_slice_30/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_30/stack~
strided_slice_30/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_30/stack_1~
strided_slice_30/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_30/stack_2?
strided_slice_30StridedSliceReadVariableOp_30:value:0strided_slice_30/stack:output:0!strided_slice_30/stack_1:output:0!strided_slice_30/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_30r
Mul_31MulMul:z:0strided_slice_30:output:0*
T0*,
_output_shapes
:??????????2
Mul_31?
Sum_30/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_30/reduction_indicest
Sum_30Sum
Mul_31:z:0!Sum_30/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_30u
Reshape_31/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_31/shape?

Reshape_31ReshapeSum_30:output:0Reshape_31/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_31?
ReadVariableOp_31ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_31z
strided_slice_31/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_31/stack~
strided_slice_31/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_31/stack_1~
strided_slice_31/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_31/stack_2?
strided_slice_31StridedSliceReadVariableOp_31:value:0strided_slice_31/stack:output:0!strided_slice_31/stack_1:output:0!strided_slice_31/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_31r
Mul_32MulMul:z:0strided_slice_31:output:0*
T0*,
_output_shapes
:??????????2
Mul_32?
Sum_31/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_31/reduction_indicest
Sum_31Sum
Mul_32:z:0!Sum_31/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_31u
Reshape_32/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_32/shape?

Reshape_32ReshapeSum_31:output:0Reshape_32/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_32?
ReadVariableOp_32ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_32z
strided_slice_32/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_32/stack~
strided_slice_32/stack_1Const*
_output_shapes
:*
dtype0*
valueB:!2
strided_slice_32/stack_1~
strided_slice_32/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_32/stack_2?
strided_slice_32StridedSliceReadVariableOp_32:value:0strided_slice_32/stack:output:0!strided_slice_32/stack_1:output:0!strided_slice_32/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_32r
Mul_33MulMul:z:0strided_slice_32:output:0*
T0*,
_output_shapes
:??????????2
Mul_33?
Sum_32/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_32/reduction_indicest
Sum_32Sum
Mul_33:z:0!Sum_32/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_32u
Reshape_33/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_33/shape?

Reshape_33ReshapeSum_32:output:0Reshape_33/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_33?
ReadVariableOp_33ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_33z
strided_slice_33/stackConst*
_output_shapes
:*
dtype0*
valueB:!2
strided_slice_33/stack~
strided_slice_33/stack_1Const*
_output_shapes
:*
dtype0*
valueB:"2
strided_slice_33/stack_1~
strided_slice_33/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_33/stack_2?
strided_slice_33StridedSliceReadVariableOp_33:value:0strided_slice_33/stack:output:0!strided_slice_33/stack_1:output:0!strided_slice_33/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_33r
Mul_34MulMul:z:0strided_slice_33:output:0*
T0*,
_output_shapes
:??????????2
Mul_34?
Sum_33/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_33/reduction_indicest
Sum_33Sum
Mul_34:z:0!Sum_33/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_33u
Reshape_34/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_34/shape?

Reshape_34ReshapeSum_33:output:0Reshape_34/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_34?
ReadVariableOp_34ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_34z
strided_slice_34/stackConst*
_output_shapes
:*
dtype0*
valueB:"2
strided_slice_34/stack~
strided_slice_34/stack_1Const*
_output_shapes
:*
dtype0*
valueB:#2
strided_slice_34/stack_1~
strided_slice_34/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_34/stack_2?
strided_slice_34StridedSliceReadVariableOp_34:value:0strided_slice_34/stack:output:0!strided_slice_34/stack_1:output:0!strided_slice_34/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_34r
Mul_35MulMul:z:0strided_slice_34:output:0*
T0*,
_output_shapes
:??????????2
Mul_35?
Sum_34/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_34/reduction_indicest
Sum_34Sum
Mul_35:z:0!Sum_34/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_34u
Reshape_35/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_35/shape?

Reshape_35ReshapeSum_34:output:0Reshape_35/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_35?
ReadVariableOp_35ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_35z
strided_slice_35/stackConst*
_output_shapes
:*
dtype0*
valueB:#2
strided_slice_35/stack~
strided_slice_35/stack_1Const*
_output_shapes
:*
dtype0*
valueB:$2
strided_slice_35/stack_1~
strided_slice_35/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_35/stack_2?
strided_slice_35StridedSliceReadVariableOp_35:value:0strided_slice_35/stack:output:0!strided_slice_35/stack_1:output:0!strided_slice_35/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_35r
Mul_36MulMul:z:0strided_slice_35:output:0*
T0*,
_output_shapes
:??????????2
Mul_36?
Sum_35/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_35/reduction_indicest
Sum_35Sum
Mul_36:z:0!Sum_35/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_35u
Reshape_36/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_36/shape?

Reshape_36ReshapeSum_35:output:0Reshape_36/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_36?
ReadVariableOp_36ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_36z
strided_slice_36/stackConst*
_output_shapes
:*
dtype0*
valueB:$2
strided_slice_36/stack~
strided_slice_36/stack_1Const*
_output_shapes
:*
dtype0*
valueB:%2
strided_slice_36/stack_1~
strided_slice_36/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_36/stack_2?
strided_slice_36StridedSliceReadVariableOp_36:value:0strided_slice_36/stack:output:0!strided_slice_36/stack_1:output:0!strided_slice_36/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_36r
Mul_37MulMul:z:0strided_slice_36:output:0*
T0*,
_output_shapes
:??????????2
Mul_37?
Sum_36/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_36/reduction_indicest
Sum_36Sum
Mul_37:z:0!Sum_36/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_36u
Reshape_37/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_37/shape?

Reshape_37ReshapeSum_36:output:0Reshape_37/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_37?
ReadVariableOp_37ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_37z
strided_slice_37/stackConst*
_output_shapes
:*
dtype0*
valueB:%2
strided_slice_37/stack~
strided_slice_37/stack_1Const*
_output_shapes
:*
dtype0*
valueB:&2
strided_slice_37/stack_1~
strided_slice_37/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_37/stack_2?
strided_slice_37StridedSliceReadVariableOp_37:value:0strided_slice_37/stack:output:0!strided_slice_37/stack_1:output:0!strided_slice_37/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_37r
Mul_38MulMul:z:0strided_slice_37:output:0*
T0*,
_output_shapes
:??????????2
Mul_38?
Sum_37/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_37/reduction_indicest
Sum_37Sum
Mul_38:z:0!Sum_37/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_37u
Reshape_38/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_38/shape?

Reshape_38ReshapeSum_37:output:0Reshape_38/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_38?
ReadVariableOp_38ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_38z
strided_slice_38/stackConst*
_output_shapes
:*
dtype0*
valueB:&2
strided_slice_38/stack~
strided_slice_38/stack_1Const*
_output_shapes
:*
dtype0*
valueB:'2
strided_slice_38/stack_1~
strided_slice_38/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_38/stack_2?
strided_slice_38StridedSliceReadVariableOp_38:value:0strided_slice_38/stack:output:0!strided_slice_38/stack_1:output:0!strided_slice_38/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_38r
Mul_39MulMul:z:0strided_slice_38:output:0*
T0*,
_output_shapes
:??????????2
Mul_39?
Sum_38/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_38/reduction_indicest
Sum_38Sum
Mul_39:z:0!Sum_38/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_38u
Reshape_39/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_39/shape?

Reshape_39ReshapeSum_38:output:0Reshape_39/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_39?
ReadVariableOp_39ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_39z
strided_slice_39/stackConst*
_output_shapes
:*
dtype0*
valueB:'2
strided_slice_39/stack~
strided_slice_39/stack_1Const*
_output_shapes
:*
dtype0*
valueB:(2
strided_slice_39/stack_1~
strided_slice_39/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_39/stack_2?
strided_slice_39StridedSliceReadVariableOp_39:value:0strided_slice_39/stack:output:0!strided_slice_39/stack_1:output:0!strided_slice_39/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_39r
Mul_40MulMul:z:0strided_slice_39:output:0*
T0*,
_output_shapes
:??????????2
Mul_40?
Sum_39/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_39/reduction_indicest
Sum_39Sum
Mul_40:z:0!Sum_39/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_39u
Reshape_40/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_40/shape?

Reshape_40ReshapeSum_39:output:0Reshape_40/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_40?
ReadVariableOp_40ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_40z
strided_slice_40/stackConst*
_output_shapes
:*
dtype0*
valueB:(2
strided_slice_40/stack~
strided_slice_40/stack_1Const*
_output_shapes
:*
dtype0*
valueB:)2
strided_slice_40/stack_1~
strided_slice_40/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_40/stack_2?
strided_slice_40StridedSliceReadVariableOp_40:value:0strided_slice_40/stack:output:0!strided_slice_40/stack_1:output:0!strided_slice_40/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_40r
Mul_41MulMul:z:0strided_slice_40:output:0*
T0*,
_output_shapes
:??????????2
Mul_41?
Sum_40/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_40/reduction_indicest
Sum_40Sum
Mul_41:z:0!Sum_40/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_40u
Reshape_41/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_41/shape?

Reshape_41ReshapeSum_40:output:0Reshape_41/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_41?
ReadVariableOp_41ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_41z
strided_slice_41/stackConst*
_output_shapes
:*
dtype0*
valueB:)2
strided_slice_41/stack~
strided_slice_41/stack_1Const*
_output_shapes
:*
dtype0*
valueB:*2
strided_slice_41/stack_1~
strided_slice_41/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_41/stack_2?
strided_slice_41StridedSliceReadVariableOp_41:value:0strided_slice_41/stack:output:0!strided_slice_41/stack_1:output:0!strided_slice_41/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_41r
Mul_42MulMul:z:0strided_slice_41:output:0*
T0*,
_output_shapes
:??????????2
Mul_42?
Sum_41/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_41/reduction_indicest
Sum_41Sum
Mul_42:z:0!Sum_41/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_41u
Reshape_42/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_42/shape?

Reshape_42ReshapeSum_41:output:0Reshape_42/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_42?
ReadVariableOp_42ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_42z
strided_slice_42/stackConst*
_output_shapes
:*
dtype0*
valueB:*2
strided_slice_42/stack~
strided_slice_42/stack_1Const*
_output_shapes
:*
dtype0*
valueB:+2
strided_slice_42/stack_1~
strided_slice_42/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_42/stack_2?
strided_slice_42StridedSliceReadVariableOp_42:value:0strided_slice_42/stack:output:0!strided_slice_42/stack_1:output:0!strided_slice_42/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_42r
Mul_43MulMul:z:0strided_slice_42:output:0*
T0*,
_output_shapes
:??????????2
Mul_43?
Sum_42/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_42/reduction_indicest
Sum_42Sum
Mul_43:z:0!Sum_42/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_42u
Reshape_43/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_43/shape?

Reshape_43ReshapeSum_42:output:0Reshape_43/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_43?
ReadVariableOp_43ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_43z
strided_slice_43/stackConst*
_output_shapes
:*
dtype0*
valueB:+2
strided_slice_43/stack~
strided_slice_43/stack_1Const*
_output_shapes
:*
dtype0*
valueB:,2
strided_slice_43/stack_1~
strided_slice_43/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_43/stack_2?
strided_slice_43StridedSliceReadVariableOp_43:value:0strided_slice_43/stack:output:0!strided_slice_43/stack_1:output:0!strided_slice_43/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_43r
Mul_44MulMul:z:0strided_slice_43:output:0*
T0*,
_output_shapes
:??????????2
Mul_44?
Sum_43/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_43/reduction_indicest
Sum_43Sum
Mul_44:z:0!Sum_43/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_43u
Reshape_44/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_44/shape?

Reshape_44ReshapeSum_43:output:0Reshape_44/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_44?
ReadVariableOp_44ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_44z
strided_slice_44/stackConst*
_output_shapes
:*
dtype0*
valueB:,2
strided_slice_44/stack~
strided_slice_44/stack_1Const*
_output_shapes
:*
dtype0*
valueB:-2
strided_slice_44/stack_1~
strided_slice_44/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_44/stack_2?
strided_slice_44StridedSliceReadVariableOp_44:value:0strided_slice_44/stack:output:0!strided_slice_44/stack_1:output:0!strided_slice_44/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_44r
Mul_45MulMul:z:0strided_slice_44:output:0*
T0*,
_output_shapes
:??????????2
Mul_45?
Sum_44/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_44/reduction_indicest
Sum_44Sum
Mul_45:z:0!Sum_44/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_44u
Reshape_45/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_45/shape?

Reshape_45ReshapeSum_44:output:0Reshape_45/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_45?
ReadVariableOp_45ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_45z
strided_slice_45/stackConst*
_output_shapes
:*
dtype0*
valueB:-2
strided_slice_45/stack~
strided_slice_45/stack_1Const*
_output_shapes
:*
dtype0*
valueB:.2
strided_slice_45/stack_1~
strided_slice_45/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_45/stack_2?
strided_slice_45StridedSliceReadVariableOp_45:value:0strided_slice_45/stack:output:0!strided_slice_45/stack_1:output:0!strided_slice_45/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_45r
Mul_46MulMul:z:0strided_slice_45:output:0*
T0*,
_output_shapes
:??????????2
Mul_46?
Sum_45/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_45/reduction_indicest
Sum_45Sum
Mul_46:z:0!Sum_45/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_45u
Reshape_46/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_46/shape?

Reshape_46ReshapeSum_45:output:0Reshape_46/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_46?
ReadVariableOp_46ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_46z
strided_slice_46/stackConst*
_output_shapes
:*
dtype0*
valueB:.2
strided_slice_46/stack~
strided_slice_46/stack_1Const*
_output_shapes
:*
dtype0*
valueB:/2
strided_slice_46/stack_1~
strided_slice_46/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_46/stack_2?
strided_slice_46StridedSliceReadVariableOp_46:value:0strided_slice_46/stack:output:0!strided_slice_46/stack_1:output:0!strided_slice_46/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_46r
Mul_47MulMul:z:0strided_slice_46:output:0*
T0*,
_output_shapes
:??????????2
Mul_47?
Sum_46/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_46/reduction_indicest
Sum_46Sum
Mul_47:z:0!Sum_46/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_46u
Reshape_47/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_47/shape?

Reshape_47ReshapeSum_46:output:0Reshape_47/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_47?
ReadVariableOp_47ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_47z
strided_slice_47/stackConst*
_output_shapes
:*
dtype0*
valueB:/2
strided_slice_47/stack~
strided_slice_47/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02
strided_slice_47/stack_1~
strided_slice_47/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_47/stack_2?
strided_slice_47StridedSliceReadVariableOp_47:value:0strided_slice_47/stack:output:0!strided_slice_47/stack_1:output:0!strided_slice_47/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_47r
Mul_48MulMul:z:0strided_slice_47:output:0*
T0*,
_output_shapes
:??????????2
Mul_48?
Sum_47/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_47/reduction_indicest
Sum_47Sum
Mul_48:z:0!Sum_47/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_47u
Reshape_48/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_48/shape?

Reshape_48ReshapeSum_47:output:0Reshape_48/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_48?
ReadVariableOp_48ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_48z
strided_slice_48/stackConst*
_output_shapes
:*
dtype0*
valueB:02
strided_slice_48/stack~
strided_slice_48/stack_1Const*
_output_shapes
:*
dtype0*
valueB:12
strided_slice_48/stack_1~
strided_slice_48/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_48/stack_2?
strided_slice_48StridedSliceReadVariableOp_48:value:0strided_slice_48/stack:output:0!strided_slice_48/stack_1:output:0!strided_slice_48/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_48r
Mul_49MulMul:z:0strided_slice_48:output:0*
T0*,
_output_shapes
:??????????2
Mul_49?
Sum_48/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_48/reduction_indicest
Sum_48Sum
Mul_49:z:0!Sum_48/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_48u
Reshape_49/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_49/shape?

Reshape_49ReshapeSum_48:output:0Reshape_49/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_49?
ReadVariableOp_49ReadVariableOpreadvariableop_resource*#
_output_shapes
:2?*
dtype02
ReadVariableOp_49z
strided_slice_49/stackConst*
_output_shapes
:*
dtype0*
valueB:12
strided_slice_49/stack~
strided_slice_49/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
strided_slice_49/stack_1~
strided_slice_49/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_49/stack_2?
strided_slice_49StridedSliceReadVariableOp_49:value:0strided_slice_49/stack:output:0!strided_slice_49/stack_1:output:0!strided_slice_49/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
strided_slice_49r
Mul_50MulMul:z:0strided_slice_49:output:0*
T0*,
_output_shapes
:??????????2
Mul_50?
Sum_49/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_49/reduction_indicest
Sum_49Sum
Mul_50:z:0!Sum_49/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_49u
Reshape_50/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_50/shape?

Reshape_50ReshapeSum_49:output:0Reshape_50/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_50\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0Reshape_16:output:0Reshape_17:output:0Reshape_18:output:0Reshape_19:output:0Reshape_20:output:0Reshape_21:output:0Reshape_22:output:0Reshape_23:output:0Reshape_24:output:0Reshape_25:output:0Reshape_26:output:0Reshape_27:output:0Reshape_28:output:0Reshape_29:output:0Reshape_30:output:0Reshape_31:output:0Reshape_32:output:0Reshape_33:output:0Reshape_34:output:0Reshape_35:output:0Reshape_36:output:0Reshape_37:output:0Reshape_38:output:0Reshape_39:output:0Reshape_40:output:0Reshape_41:output:0Reshape_42:output:0Reshape_43:output:0Reshape_44:output:0Reshape_45:output:0Reshape_46:output:0Reshape_47:output:0Reshape_48:output:0Reshape_49:output:0Reshape_50:output:0concat/axis:output:0*
N2*
T0*'
_output_shapes
:?????????22
concatv
Sum_50/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_50/reduction_indicesu
Sum_50SumMul:z:0!Sum_50/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2
Sum_50b
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsSum_50:output:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsSum_50:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1?
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:?????????2
MatMul?
ReadVariableOp_50ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_50z
strided_slice_50/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_50/stack~
strided_slice_50/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_50/stack_1~
strided_slice_50/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_50/stack_2?
strided_slice_50StridedSliceReadVariableOp_50:value:0strided_slice_50/stack:output:0!strided_slice_50/stack_1:output:0!strided_slice_50/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_50f
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_2/dim?
ExpandDims_2
ExpandDimsstrided_slice_50:output:0ExpandDims_2/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_2u
Mul_51MulMatMul:output:0ExpandDims_2:output:0*
T0*+
_output_shapes
:?????????2
Mul_51?
Sum_51/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_51/reduction_indicest
Sum_51Sum
Mul_51:z:0!Sum_51/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_51u
Reshape_51/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_51/shape?

Reshape_51ReshapeSum_51:output:0Reshape_51/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_51?
ReadVariableOp_51ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_51z
strided_slice_51/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_51/stack~
strided_slice_51/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_51/stack_1~
strided_slice_51/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_51/stack_2?
strided_slice_51StridedSliceReadVariableOp_51:value:0strided_slice_51/stack:output:0!strided_slice_51/stack_1:output:0!strided_slice_51/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_51f
ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_3/dim?
ExpandDims_3
ExpandDimsstrided_slice_51:output:0ExpandDims_3/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_3u
Mul_52MulMatMul:output:0ExpandDims_3:output:0*
T0*+
_output_shapes
:?????????2
Mul_52?
Sum_52/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_52/reduction_indicest
Sum_52Sum
Mul_52:z:0!Sum_52/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_52u
Reshape_52/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_52/shape?

Reshape_52ReshapeSum_52:output:0Reshape_52/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_52?
ReadVariableOp_52ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_52z
strided_slice_52/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_52/stack~
strided_slice_52/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_52/stack_1~
strided_slice_52/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_52/stack_2?
strided_slice_52StridedSliceReadVariableOp_52:value:0strided_slice_52/stack:output:0!strided_slice_52/stack_1:output:0!strided_slice_52/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_52f
ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_4/dim?
ExpandDims_4
ExpandDimsstrided_slice_52:output:0ExpandDims_4/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_4u
Mul_53MulMatMul:output:0ExpandDims_4:output:0*
T0*+
_output_shapes
:?????????2
Mul_53?
Sum_53/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_53/reduction_indicest
Sum_53Sum
Mul_53:z:0!Sum_53/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_53u
Reshape_53/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_53/shape?

Reshape_53ReshapeSum_53:output:0Reshape_53/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_53?
ReadVariableOp_53ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_53z
strided_slice_53/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_53/stack~
strided_slice_53/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_53/stack_1~
strided_slice_53/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_53/stack_2?
strided_slice_53StridedSliceReadVariableOp_53:value:0strided_slice_53/stack:output:0!strided_slice_53/stack_1:output:0!strided_slice_53/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_53f
ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_5/dim?
ExpandDims_5
ExpandDimsstrided_slice_53:output:0ExpandDims_5/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_5u
Mul_54MulMatMul:output:0ExpandDims_5:output:0*
T0*+
_output_shapes
:?????????2
Mul_54?
Sum_54/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_54/reduction_indicest
Sum_54Sum
Mul_54:z:0!Sum_54/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_54u
Reshape_54/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_54/shape?

Reshape_54ReshapeSum_54:output:0Reshape_54/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_54?
ReadVariableOp_54ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_54z
strided_slice_54/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_54/stack~
strided_slice_54/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_54/stack_1~
strided_slice_54/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_54/stack_2?
strided_slice_54StridedSliceReadVariableOp_54:value:0strided_slice_54/stack:output:0!strided_slice_54/stack_1:output:0!strided_slice_54/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_54f
ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_6/dim?
ExpandDims_6
ExpandDimsstrided_slice_54:output:0ExpandDims_6/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_6u
Mul_55MulMatMul:output:0ExpandDims_6:output:0*
T0*+
_output_shapes
:?????????2
Mul_55?
Sum_55/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_55/reduction_indicest
Sum_55Sum
Mul_55:z:0!Sum_55/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_55u
Reshape_55/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_55/shape?

Reshape_55ReshapeSum_55:output:0Reshape_55/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_55?
ReadVariableOp_55ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_55z
strided_slice_55/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_55/stack~
strided_slice_55/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_55/stack_1~
strided_slice_55/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_55/stack_2?
strided_slice_55StridedSliceReadVariableOp_55:value:0strided_slice_55/stack:output:0!strided_slice_55/stack_1:output:0!strided_slice_55/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_55f
ExpandDims_7/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_7/dim?
ExpandDims_7
ExpandDimsstrided_slice_55:output:0ExpandDims_7/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_7u
Mul_56MulMatMul:output:0ExpandDims_7:output:0*
T0*+
_output_shapes
:?????????2
Mul_56?
Sum_56/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_56/reduction_indicest
Sum_56Sum
Mul_56:z:0!Sum_56/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_56u
Reshape_56/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_56/shape?

Reshape_56ReshapeSum_56:output:0Reshape_56/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_56?
ReadVariableOp_56ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_56z
strided_slice_56/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_56/stack~
strided_slice_56/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_56/stack_1~
strided_slice_56/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_56/stack_2?
strided_slice_56StridedSliceReadVariableOp_56:value:0strided_slice_56/stack:output:0!strided_slice_56/stack_1:output:0!strided_slice_56/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_56f
ExpandDims_8/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_8/dim?
ExpandDims_8
ExpandDimsstrided_slice_56:output:0ExpandDims_8/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_8u
Mul_57MulMatMul:output:0ExpandDims_8:output:0*
T0*+
_output_shapes
:?????????2
Mul_57?
Sum_57/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_57/reduction_indicest
Sum_57Sum
Mul_57:z:0!Sum_57/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_57u
Reshape_57/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_57/shape?

Reshape_57ReshapeSum_57:output:0Reshape_57/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_57?
ReadVariableOp_57ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_57z
strided_slice_57/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_57/stack~
strided_slice_57/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_57/stack_1~
strided_slice_57/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_57/stack_2?
strided_slice_57StridedSliceReadVariableOp_57:value:0strided_slice_57/stack:output:0!strided_slice_57/stack_1:output:0!strided_slice_57/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_57f
ExpandDims_9/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_9/dim?
ExpandDims_9
ExpandDimsstrided_slice_57:output:0ExpandDims_9/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_9u
Mul_58MulMatMul:output:0ExpandDims_9:output:0*
T0*+
_output_shapes
:?????????2
Mul_58?
Sum_58/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_58/reduction_indicest
Sum_58Sum
Mul_58:z:0!Sum_58/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_58u
Reshape_58/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_58/shape?

Reshape_58ReshapeSum_58:output:0Reshape_58/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_58?
ReadVariableOp_58ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_58z
strided_slice_58/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_58/stack~
strided_slice_58/stack_1Const*
_output_shapes
:*
dtype0*
valueB:	2
strided_slice_58/stack_1~
strided_slice_58/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_58/stack_2?
strided_slice_58StridedSliceReadVariableOp_58:value:0strided_slice_58/stack:output:0!strided_slice_58/stack_1:output:0!strided_slice_58/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_58h
ExpandDims_10/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_10/dim?
ExpandDims_10
ExpandDimsstrided_slice_58:output:0ExpandDims_10/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_10v
Mul_59MulMatMul:output:0ExpandDims_10:output:0*
T0*+
_output_shapes
:?????????2
Mul_59?
Sum_59/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_59/reduction_indicest
Sum_59Sum
Mul_59:z:0!Sum_59/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_59u
Reshape_59/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_59/shape?

Reshape_59ReshapeSum_59:output:0Reshape_59/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_59?
ReadVariableOp_59ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_59z
strided_slice_59/stackConst*
_output_shapes
:*
dtype0*
valueB:	2
strided_slice_59/stack~
strided_slice_59/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
2
strided_slice_59/stack_1~
strided_slice_59/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_59/stack_2?
strided_slice_59StridedSliceReadVariableOp_59:value:0strided_slice_59/stack:output:0!strided_slice_59/stack_1:output:0!strided_slice_59/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_59h
ExpandDims_11/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_11/dim?
ExpandDims_11
ExpandDimsstrided_slice_59:output:0ExpandDims_11/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_11v
Mul_60MulMatMul:output:0ExpandDims_11:output:0*
T0*+
_output_shapes
:?????????2
Mul_60?
Sum_60/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_60/reduction_indicest
Sum_60Sum
Mul_60:z:0!Sum_60/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_60u
Reshape_60/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_60/shape?

Reshape_60ReshapeSum_60:output:0Reshape_60/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_60?
ReadVariableOp_60ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_60z
strided_slice_60/stackConst*
_output_shapes
:*
dtype0*
valueB:
2
strided_slice_60/stack~
strided_slice_60/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_60/stack_1~
strided_slice_60/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_60/stack_2?
strided_slice_60StridedSliceReadVariableOp_60:value:0strided_slice_60/stack:output:0!strided_slice_60/stack_1:output:0!strided_slice_60/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_60h
ExpandDims_12/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_12/dim?
ExpandDims_12
ExpandDimsstrided_slice_60:output:0ExpandDims_12/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_12v
Mul_61MulMatMul:output:0ExpandDims_12:output:0*
T0*+
_output_shapes
:?????????2
Mul_61?
Sum_61/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_61/reduction_indicest
Sum_61Sum
Mul_61:z:0!Sum_61/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_61u
Reshape_61/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_61/shape?

Reshape_61ReshapeSum_61:output:0Reshape_61/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_61?
ReadVariableOp_61ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_61z
strided_slice_61/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_61/stack~
strided_slice_61/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_61/stack_1~
strided_slice_61/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_61/stack_2?
strided_slice_61StridedSliceReadVariableOp_61:value:0strided_slice_61/stack:output:0!strided_slice_61/stack_1:output:0!strided_slice_61/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_61h
ExpandDims_13/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_13/dim?
ExpandDims_13
ExpandDimsstrided_slice_61:output:0ExpandDims_13/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_13v
Mul_62MulMatMul:output:0ExpandDims_13:output:0*
T0*+
_output_shapes
:?????????2
Mul_62?
Sum_62/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_62/reduction_indicest
Sum_62Sum
Mul_62:z:0!Sum_62/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_62u
Reshape_62/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_62/shape?

Reshape_62ReshapeSum_62:output:0Reshape_62/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_62?
ReadVariableOp_62ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_62z
strided_slice_62/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_62/stack~
strided_slice_62/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_62/stack_1~
strided_slice_62/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_62/stack_2?
strided_slice_62StridedSliceReadVariableOp_62:value:0strided_slice_62/stack:output:0!strided_slice_62/stack_1:output:0!strided_slice_62/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_62h
ExpandDims_14/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_14/dim?
ExpandDims_14
ExpandDimsstrided_slice_62:output:0ExpandDims_14/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_14v
Mul_63MulMatMul:output:0ExpandDims_14:output:0*
T0*+
_output_shapes
:?????????2
Mul_63?
Sum_63/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_63/reduction_indicest
Sum_63Sum
Mul_63:z:0!Sum_63/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_63u
Reshape_63/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_63/shape?

Reshape_63ReshapeSum_63:output:0Reshape_63/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_63?
ReadVariableOp_63ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_63z
strided_slice_63/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_63/stack~
strided_slice_63/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_63/stack_1~
strided_slice_63/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_63/stack_2?
strided_slice_63StridedSliceReadVariableOp_63:value:0strided_slice_63/stack:output:0!strided_slice_63/stack_1:output:0!strided_slice_63/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_63h
ExpandDims_15/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_15/dim?
ExpandDims_15
ExpandDimsstrided_slice_63:output:0ExpandDims_15/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_15v
Mul_64MulMatMul:output:0ExpandDims_15:output:0*
T0*+
_output_shapes
:?????????2
Mul_64?
Sum_64/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_64/reduction_indicest
Sum_64Sum
Mul_64:z:0!Sum_64/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_64u
Reshape_64/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_64/shape?

Reshape_64ReshapeSum_64:output:0Reshape_64/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_64?
ReadVariableOp_64ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_64z
strided_slice_64/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_64/stack~
strided_slice_64/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_64/stack_1~
strided_slice_64/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_64/stack_2?
strided_slice_64StridedSliceReadVariableOp_64:value:0strided_slice_64/stack:output:0!strided_slice_64/stack_1:output:0!strided_slice_64/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_64h
ExpandDims_16/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_16/dim?
ExpandDims_16
ExpandDimsstrided_slice_64:output:0ExpandDims_16/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_16v
Mul_65MulMatMul:output:0ExpandDims_16:output:0*
T0*+
_output_shapes
:?????????2
Mul_65?
Sum_65/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_65/reduction_indicest
Sum_65Sum
Mul_65:z:0!Sum_65/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_65u
Reshape_65/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_65/shape?

Reshape_65ReshapeSum_65:output:0Reshape_65/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_65?
ReadVariableOp_65ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_65z
strided_slice_65/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_65/stack~
strided_slice_65/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_65/stack_1~
strided_slice_65/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_65/stack_2?
strided_slice_65StridedSliceReadVariableOp_65:value:0strided_slice_65/stack:output:0!strided_slice_65/stack_1:output:0!strided_slice_65/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_65h
ExpandDims_17/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_17/dim?
ExpandDims_17
ExpandDimsstrided_slice_65:output:0ExpandDims_17/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_17v
Mul_66MulMatMul:output:0ExpandDims_17:output:0*
T0*+
_output_shapes
:?????????2
Mul_66?
Sum_66/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_66/reduction_indicest
Sum_66Sum
Mul_66:z:0!Sum_66/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_66u
Reshape_66/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_66/shape?

Reshape_66ReshapeSum_66:output:0Reshape_66/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_66?
ReadVariableOp_66ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_66z
strided_slice_66/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_66/stack~
strided_slice_66/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_66/stack_1~
strided_slice_66/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_66/stack_2?
strided_slice_66StridedSliceReadVariableOp_66:value:0strided_slice_66/stack:output:0!strided_slice_66/stack_1:output:0!strided_slice_66/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_66h
ExpandDims_18/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_18/dim?
ExpandDims_18
ExpandDimsstrided_slice_66:output:0ExpandDims_18/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_18v
Mul_67MulMatMul:output:0ExpandDims_18:output:0*
T0*+
_output_shapes
:?????????2
Mul_67?
Sum_67/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_67/reduction_indicest
Sum_67Sum
Mul_67:z:0!Sum_67/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_67u
Reshape_67/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_67/shape?

Reshape_67ReshapeSum_67:output:0Reshape_67/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_67?
ReadVariableOp_67ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_67z
strided_slice_67/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_67/stack~
strided_slice_67/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_67/stack_1~
strided_slice_67/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_67/stack_2?
strided_slice_67StridedSliceReadVariableOp_67:value:0strided_slice_67/stack:output:0!strided_slice_67/stack_1:output:0!strided_slice_67/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_67h
ExpandDims_19/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_19/dim?
ExpandDims_19
ExpandDimsstrided_slice_67:output:0ExpandDims_19/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_19v
Mul_68MulMatMul:output:0ExpandDims_19:output:0*
T0*+
_output_shapes
:?????????2
Mul_68?
Sum_68/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_68/reduction_indicest
Sum_68Sum
Mul_68:z:0!Sum_68/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_68u
Reshape_68/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_68/shape?

Reshape_68ReshapeSum_68:output:0Reshape_68/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_68?
ReadVariableOp_68ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_68z
strided_slice_68/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_68/stack~
strided_slice_68/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_68/stack_1~
strided_slice_68/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_68/stack_2?
strided_slice_68StridedSliceReadVariableOp_68:value:0strided_slice_68/stack:output:0!strided_slice_68/stack_1:output:0!strided_slice_68/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_68h
ExpandDims_20/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_20/dim?
ExpandDims_20
ExpandDimsstrided_slice_68:output:0ExpandDims_20/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_20v
Mul_69MulMatMul:output:0ExpandDims_20:output:0*
T0*+
_output_shapes
:?????????2
Mul_69?
Sum_69/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_69/reduction_indicest
Sum_69Sum
Mul_69:z:0!Sum_69/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_69u
Reshape_69/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_69/shape?

Reshape_69ReshapeSum_69:output:0Reshape_69/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_69?
ReadVariableOp_69ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_69z
strided_slice_69/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_69/stack~
strided_slice_69/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_69/stack_1~
strided_slice_69/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_69/stack_2?
strided_slice_69StridedSliceReadVariableOp_69:value:0strided_slice_69/stack:output:0!strided_slice_69/stack_1:output:0!strided_slice_69/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_69h
ExpandDims_21/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_21/dim?
ExpandDims_21
ExpandDimsstrided_slice_69:output:0ExpandDims_21/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_21v
Mul_70MulMatMul:output:0ExpandDims_21:output:0*
T0*+
_output_shapes
:?????????2
Mul_70?
Sum_70/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_70/reduction_indicest
Sum_70Sum
Mul_70:z:0!Sum_70/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_70u
Reshape_70/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_70/shape?

Reshape_70ReshapeSum_70:output:0Reshape_70/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_70?
ReadVariableOp_70ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_70z
strided_slice_70/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_70/stack~
strided_slice_70/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_70/stack_1~
strided_slice_70/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_70/stack_2?
strided_slice_70StridedSliceReadVariableOp_70:value:0strided_slice_70/stack:output:0!strided_slice_70/stack_1:output:0!strided_slice_70/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_70h
ExpandDims_22/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_22/dim?
ExpandDims_22
ExpandDimsstrided_slice_70:output:0ExpandDims_22/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_22v
Mul_71MulMatMul:output:0ExpandDims_22:output:0*
T0*+
_output_shapes
:?????????2
Mul_71?
Sum_71/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_71/reduction_indicest
Sum_71Sum
Mul_71:z:0!Sum_71/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_71u
Reshape_71/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_71/shape?

Reshape_71ReshapeSum_71:output:0Reshape_71/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_71?
ReadVariableOp_71ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_71z
strided_slice_71/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_71/stack~
strided_slice_71/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_71/stack_1~
strided_slice_71/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_71/stack_2?
strided_slice_71StridedSliceReadVariableOp_71:value:0strided_slice_71/stack:output:0!strided_slice_71/stack_1:output:0!strided_slice_71/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_71h
ExpandDims_23/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_23/dim?
ExpandDims_23
ExpandDimsstrided_slice_71:output:0ExpandDims_23/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_23v
Mul_72MulMatMul:output:0ExpandDims_23:output:0*
T0*+
_output_shapes
:?????????2
Mul_72?
Sum_72/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_72/reduction_indicest
Sum_72Sum
Mul_72:z:0!Sum_72/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_72u
Reshape_72/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_72/shape?

Reshape_72ReshapeSum_72:output:0Reshape_72/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_72?
ReadVariableOp_72ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_72z
strided_slice_72/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_72/stack~
strided_slice_72/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_72/stack_1~
strided_slice_72/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_72/stack_2?
strided_slice_72StridedSliceReadVariableOp_72:value:0strided_slice_72/stack:output:0!strided_slice_72/stack_1:output:0!strided_slice_72/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_72h
ExpandDims_24/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_24/dim?
ExpandDims_24
ExpandDimsstrided_slice_72:output:0ExpandDims_24/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_24v
Mul_73MulMatMul:output:0ExpandDims_24:output:0*
T0*+
_output_shapes
:?????????2
Mul_73?
Sum_73/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_73/reduction_indicest
Sum_73Sum
Mul_73:z:0!Sum_73/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_73u
Reshape_73/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_73/shape?

Reshape_73ReshapeSum_73:output:0Reshape_73/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_73?
ReadVariableOp_73ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_73z
strided_slice_73/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_73/stack~
strided_slice_73/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_73/stack_1~
strided_slice_73/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_73/stack_2?
strided_slice_73StridedSliceReadVariableOp_73:value:0strided_slice_73/stack:output:0!strided_slice_73/stack_1:output:0!strided_slice_73/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_73h
ExpandDims_25/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_25/dim?
ExpandDims_25
ExpandDimsstrided_slice_73:output:0ExpandDims_25/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_25v
Mul_74MulMatMul:output:0ExpandDims_25:output:0*
T0*+
_output_shapes
:?????????2
Mul_74?
Sum_74/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_74/reduction_indicest
Sum_74Sum
Mul_74:z:0!Sum_74/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_74u
Reshape_74/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_74/shape?

Reshape_74ReshapeSum_74:output:0Reshape_74/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_74?
ReadVariableOp_74ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_74z
strided_slice_74/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_74/stack~
strided_slice_74/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_74/stack_1~
strided_slice_74/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_74/stack_2?
strided_slice_74StridedSliceReadVariableOp_74:value:0strided_slice_74/stack:output:0!strided_slice_74/stack_1:output:0!strided_slice_74/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_74h
ExpandDims_26/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_26/dim?
ExpandDims_26
ExpandDimsstrided_slice_74:output:0ExpandDims_26/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_26v
Mul_75MulMatMul:output:0ExpandDims_26:output:0*
T0*+
_output_shapes
:?????????2
Mul_75?
Sum_75/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_75/reduction_indicest
Sum_75Sum
Mul_75:z:0!Sum_75/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_75u
Reshape_75/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_75/shape?

Reshape_75ReshapeSum_75:output:0Reshape_75/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_75?
ReadVariableOp_75ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_75z
strided_slice_75/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_75/stack~
strided_slice_75/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_75/stack_1~
strided_slice_75/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_75/stack_2?
strided_slice_75StridedSliceReadVariableOp_75:value:0strided_slice_75/stack:output:0!strided_slice_75/stack_1:output:0!strided_slice_75/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_75h
ExpandDims_27/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_27/dim?
ExpandDims_27
ExpandDimsstrided_slice_75:output:0ExpandDims_27/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_27v
Mul_76MulMatMul:output:0ExpandDims_27:output:0*
T0*+
_output_shapes
:?????????2
Mul_76?
Sum_76/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_76/reduction_indicest
Sum_76Sum
Mul_76:z:0!Sum_76/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_76u
Reshape_76/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_76/shape?

Reshape_76ReshapeSum_76:output:0Reshape_76/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_76?
ReadVariableOp_76ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_76z
strided_slice_76/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_76/stack~
strided_slice_76/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_76/stack_1~
strided_slice_76/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_76/stack_2?
strided_slice_76StridedSliceReadVariableOp_76:value:0strided_slice_76/stack:output:0!strided_slice_76/stack_1:output:0!strided_slice_76/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_76h
ExpandDims_28/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_28/dim?
ExpandDims_28
ExpandDimsstrided_slice_76:output:0ExpandDims_28/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_28v
Mul_77MulMatMul:output:0ExpandDims_28:output:0*
T0*+
_output_shapes
:?????????2
Mul_77?
Sum_77/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_77/reduction_indicest
Sum_77Sum
Mul_77:z:0!Sum_77/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_77u
Reshape_77/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_77/shape?

Reshape_77ReshapeSum_77:output:0Reshape_77/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_77?
ReadVariableOp_77ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_77z
strided_slice_77/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_77/stack~
strided_slice_77/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_77/stack_1~
strided_slice_77/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_77/stack_2?
strided_slice_77StridedSliceReadVariableOp_77:value:0strided_slice_77/stack:output:0!strided_slice_77/stack_1:output:0!strided_slice_77/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_77h
ExpandDims_29/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_29/dim?
ExpandDims_29
ExpandDimsstrided_slice_77:output:0ExpandDims_29/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_29v
Mul_78MulMatMul:output:0ExpandDims_29:output:0*
T0*+
_output_shapes
:?????????2
Mul_78?
Sum_78/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_78/reduction_indicest
Sum_78Sum
Mul_78:z:0!Sum_78/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_78u
Reshape_78/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_78/shape?

Reshape_78ReshapeSum_78:output:0Reshape_78/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_78?
ReadVariableOp_78ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_78z
strided_slice_78/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_78/stack~
strided_slice_78/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_78/stack_1~
strided_slice_78/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_78/stack_2?
strided_slice_78StridedSliceReadVariableOp_78:value:0strided_slice_78/stack:output:0!strided_slice_78/stack_1:output:0!strided_slice_78/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_78h
ExpandDims_30/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_30/dim?
ExpandDims_30
ExpandDimsstrided_slice_78:output:0ExpandDims_30/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_30v
Mul_79MulMatMul:output:0ExpandDims_30:output:0*
T0*+
_output_shapes
:?????????2
Mul_79?
Sum_79/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_79/reduction_indicest
Sum_79Sum
Mul_79:z:0!Sum_79/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_79u
Reshape_79/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_79/shape?

Reshape_79ReshapeSum_79:output:0Reshape_79/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_79?
ReadVariableOp_79ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_79z
strided_slice_79/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_79/stack~
strided_slice_79/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_79/stack_1~
strided_slice_79/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_79/stack_2?
strided_slice_79StridedSliceReadVariableOp_79:value:0strided_slice_79/stack:output:0!strided_slice_79/stack_1:output:0!strided_slice_79/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_79h
ExpandDims_31/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_31/dim?
ExpandDims_31
ExpandDimsstrided_slice_79:output:0ExpandDims_31/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_31v
Mul_80MulMatMul:output:0ExpandDims_31:output:0*
T0*+
_output_shapes
:?????????2
Mul_80?
Sum_80/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_80/reduction_indicest
Sum_80Sum
Mul_80:z:0!Sum_80/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_80u
Reshape_80/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_80/shape?

Reshape_80ReshapeSum_80:output:0Reshape_80/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_80?
ReadVariableOp_80ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_80z
strided_slice_80/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_80/stack~
strided_slice_80/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_80/stack_1~
strided_slice_80/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_80/stack_2?
strided_slice_80StridedSliceReadVariableOp_80:value:0strided_slice_80/stack:output:0!strided_slice_80/stack_1:output:0!strided_slice_80/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_80h
ExpandDims_32/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_32/dim?
ExpandDims_32
ExpandDimsstrided_slice_80:output:0ExpandDims_32/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_32v
Mul_81MulMatMul:output:0ExpandDims_32:output:0*
T0*+
_output_shapes
:?????????2
Mul_81?
Sum_81/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_81/reduction_indicest
Sum_81Sum
Mul_81:z:0!Sum_81/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_81u
Reshape_81/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_81/shape?

Reshape_81ReshapeSum_81:output:0Reshape_81/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_81?
ReadVariableOp_81ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_81z
strided_slice_81/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_81/stack~
strided_slice_81/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_81/stack_1~
strided_slice_81/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_81/stack_2?
strided_slice_81StridedSliceReadVariableOp_81:value:0strided_slice_81/stack:output:0!strided_slice_81/stack_1:output:0!strided_slice_81/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_81h
ExpandDims_33/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_33/dim?
ExpandDims_33
ExpandDimsstrided_slice_81:output:0ExpandDims_33/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_33v
Mul_82MulMatMul:output:0ExpandDims_33:output:0*
T0*+
_output_shapes
:?????????2
Mul_82?
Sum_82/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_82/reduction_indicest
Sum_82Sum
Mul_82:z:0!Sum_82/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_82u
Reshape_82/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_82/shape?

Reshape_82ReshapeSum_82:output:0Reshape_82/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_82?
ReadVariableOp_82ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_82z
strided_slice_82/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_82/stack~
strided_slice_82/stack_1Const*
_output_shapes
:*
dtype0*
valueB:!2
strided_slice_82/stack_1~
strided_slice_82/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_82/stack_2?
strided_slice_82StridedSliceReadVariableOp_82:value:0strided_slice_82/stack:output:0!strided_slice_82/stack_1:output:0!strided_slice_82/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_82h
ExpandDims_34/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_34/dim?
ExpandDims_34
ExpandDimsstrided_slice_82:output:0ExpandDims_34/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_34v
Mul_83MulMatMul:output:0ExpandDims_34:output:0*
T0*+
_output_shapes
:?????????2
Mul_83?
Sum_83/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_83/reduction_indicest
Sum_83Sum
Mul_83:z:0!Sum_83/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_83u
Reshape_83/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_83/shape?

Reshape_83ReshapeSum_83:output:0Reshape_83/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_83?
ReadVariableOp_83ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_83z
strided_slice_83/stackConst*
_output_shapes
:*
dtype0*
valueB:!2
strided_slice_83/stack~
strided_slice_83/stack_1Const*
_output_shapes
:*
dtype0*
valueB:"2
strided_slice_83/stack_1~
strided_slice_83/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_83/stack_2?
strided_slice_83StridedSliceReadVariableOp_83:value:0strided_slice_83/stack:output:0!strided_slice_83/stack_1:output:0!strided_slice_83/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_83h
ExpandDims_35/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_35/dim?
ExpandDims_35
ExpandDimsstrided_slice_83:output:0ExpandDims_35/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_35v
Mul_84MulMatMul:output:0ExpandDims_35:output:0*
T0*+
_output_shapes
:?????????2
Mul_84?
Sum_84/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_84/reduction_indicest
Sum_84Sum
Mul_84:z:0!Sum_84/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_84u
Reshape_84/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_84/shape?

Reshape_84ReshapeSum_84:output:0Reshape_84/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_84?
ReadVariableOp_84ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_84z
strided_slice_84/stackConst*
_output_shapes
:*
dtype0*
valueB:"2
strided_slice_84/stack~
strided_slice_84/stack_1Const*
_output_shapes
:*
dtype0*
valueB:#2
strided_slice_84/stack_1~
strided_slice_84/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_84/stack_2?
strided_slice_84StridedSliceReadVariableOp_84:value:0strided_slice_84/stack:output:0!strided_slice_84/stack_1:output:0!strided_slice_84/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_84h
ExpandDims_36/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_36/dim?
ExpandDims_36
ExpandDimsstrided_slice_84:output:0ExpandDims_36/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_36v
Mul_85MulMatMul:output:0ExpandDims_36:output:0*
T0*+
_output_shapes
:?????????2
Mul_85?
Sum_85/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_85/reduction_indicest
Sum_85Sum
Mul_85:z:0!Sum_85/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_85u
Reshape_85/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_85/shape?

Reshape_85ReshapeSum_85:output:0Reshape_85/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_85?
ReadVariableOp_85ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_85z
strided_slice_85/stackConst*
_output_shapes
:*
dtype0*
valueB:#2
strided_slice_85/stack~
strided_slice_85/stack_1Const*
_output_shapes
:*
dtype0*
valueB:$2
strided_slice_85/stack_1~
strided_slice_85/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_85/stack_2?
strided_slice_85StridedSliceReadVariableOp_85:value:0strided_slice_85/stack:output:0!strided_slice_85/stack_1:output:0!strided_slice_85/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_85h
ExpandDims_37/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_37/dim?
ExpandDims_37
ExpandDimsstrided_slice_85:output:0ExpandDims_37/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_37v
Mul_86MulMatMul:output:0ExpandDims_37:output:0*
T0*+
_output_shapes
:?????????2
Mul_86?
Sum_86/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_86/reduction_indicest
Sum_86Sum
Mul_86:z:0!Sum_86/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_86u
Reshape_86/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_86/shape?

Reshape_86ReshapeSum_86:output:0Reshape_86/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_86?
ReadVariableOp_86ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_86z
strided_slice_86/stackConst*
_output_shapes
:*
dtype0*
valueB:$2
strided_slice_86/stack~
strided_slice_86/stack_1Const*
_output_shapes
:*
dtype0*
valueB:%2
strided_slice_86/stack_1~
strided_slice_86/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_86/stack_2?
strided_slice_86StridedSliceReadVariableOp_86:value:0strided_slice_86/stack:output:0!strided_slice_86/stack_1:output:0!strided_slice_86/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_86h
ExpandDims_38/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_38/dim?
ExpandDims_38
ExpandDimsstrided_slice_86:output:0ExpandDims_38/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_38v
Mul_87MulMatMul:output:0ExpandDims_38:output:0*
T0*+
_output_shapes
:?????????2
Mul_87?
Sum_87/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_87/reduction_indicest
Sum_87Sum
Mul_87:z:0!Sum_87/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_87u
Reshape_87/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_87/shape?

Reshape_87ReshapeSum_87:output:0Reshape_87/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_87?
ReadVariableOp_87ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_87z
strided_slice_87/stackConst*
_output_shapes
:*
dtype0*
valueB:%2
strided_slice_87/stack~
strided_slice_87/stack_1Const*
_output_shapes
:*
dtype0*
valueB:&2
strided_slice_87/stack_1~
strided_slice_87/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_87/stack_2?
strided_slice_87StridedSliceReadVariableOp_87:value:0strided_slice_87/stack:output:0!strided_slice_87/stack_1:output:0!strided_slice_87/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_87h
ExpandDims_39/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_39/dim?
ExpandDims_39
ExpandDimsstrided_slice_87:output:0ExpandDims_39/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_39v
Mul_88MulMatMul:output:0ExpandDims_39:output:0*
T0*+
_output_shapes
:?????????2
Mul_88?
Sum_88/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_88/reduction_indicest
Sum_88Sum
Mul_88:z:0!Sum_88/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_88u
Reshape_88/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_88/shape?

Reshape_88ReshapeSum_88:output:0Reshape_88/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_88?
ReadVariableOp_88ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_88z
strided_slice_88/stackConst*
_output_shapes
:*
dtype0*
valueB:&2
strided_slice_88/stack~
strided_slice_88/stack_1Const*
_output_shapes
:*
dtype0*
valueB:'2
strided_slice_88/stack_1~
strided_slice_88/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_88/stack_2?
strided_slice_88StridedSliceReadVariableOp_88:value:0strided_slice_88/stack:output:0!strided_slice_88/stack_1:output:0!strided_slice_88/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_88h
ExpandDims_40/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_40/dim?
ExpandDims_40
ExpandDimsstrided_slice_88:output:0ExpandDims_40/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_40v
Mul_89MulMatMul:output:0ExpandDims_40:output:0*
T0*+
_output_shapes
:?????????2
Mul_89?
Sum_89/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_89/reduction_indicest
Sum_89Sum
Mul_89:z:0!Sum_89/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_89u
Reshape_89/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_89/shape?

Reshape_89ReshapeSum_89:output:0Reshape_89/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_89?
ReadVariableOp_89ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_89z
strided_slice_89/stackConst*
_output_shapes
:*
dtype0*
valueB:'2
strided_slice_89/stack~
strided_slice_89/stack_1Const*
_output_shapes
:*
dtype0*
valueB:(2
strided_slice_89/stack_1~
strided_slice_89/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_89/stack_2?
strided_slice_89StridedSliceReadVariableOp_89:value:0strided_slice_89/stack:output:0!strided_slice_89/stack_1:output:0!strided_slice_89/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_89h
ExpandDims_41/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_41/dim?
ExpandDims_41
ExpandDimsstrided_slice_89:output:0ExpandDims_41/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_41v
Mul_90MulMatMul:output:0ExpandDims_41:output:0*
T0*+
_output_shapes
:?????????2
Mul_90?
Sum_90/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_90/reduction_indicest
Sum_90Sum
Mul_90:z:0!Sum_90/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_90u
Reshape_90/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_90/shape?

Reshape_90ReshapeSum_90:output:0Reshape_90/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_90?
ReadVariableOp_90ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_90z
strided_slice_90/stackConst*
_output_shapes
:*
dtype0*
valueB:(2
strided_slice_90/stack~
strided_slice_90/stack_1Const*
_output_shapes
:*
dtype0*
valueB:)2
strided_slice_90/stack_1~
strided_slice_90/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_90/stack_2?
strided_slice_90StridedSliceReadVariableOp_90:value:0strided_slice_90/stack:output:0!strided_slice_90/stack_1:output:0!strided_slice_90/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_90h
ExpandDims_42/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_42/dim?
ExpandDims_42
ExpandDimsstrided_slice_90:output:0ExpandDims_42/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_42v
Mul_91MulMatMul:output:0ExpandDims_42:output:0*
T0*+
_output_shapes
:?????????2
Mul_91?
Sum_91/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_91/reduction_indicest
Sum_91Sum
Mul_91:z:0!Sum_91/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_91u
Reshape_91/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_91/shape?

Reshape_91ReshapeSum_91:output:0Reshape_91/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_91?
ReadVariableOp_91ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_91z
strided_slice_91/stackConst*
_output_shapes
:*
dtype0*
valueB:)2
strided_slice_91/stack~
strided_slice_91/stack_1Const*
_output_shapes
:*
dtype0*
valueB:*2
strided_slice_91/stack_1~
strided_slice_91/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_91/stack_2?
strided_slice_91StridedSliceReadVariableOp_91:value:0strided_slice_91/stack:output:0!strided_slice_91/stack_1:output:0!strided_slice_91/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_91h
ExpandDims_43/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_43/dim?
ExpandDims_43
ExpandDimsstrided_slice_91:output:0ExpandDims_43/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_43v
Mul_92MulMatMul:output:0ExpandDims_43:output:0*
T0*+
_output_shapes
:?????????2
Mul_92?
Sum_92/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_92/reduction_indicest
Sum_92Sum
Mul_92:z:0!Sum_92/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_92u
Reshape_92/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_92/shape?

Reshape_92ReshapeSum_92:output:0Reshape_92/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_92?
ReadVariableOp_92ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_92z
strided_slice_92/stackConst*
_output_shapes
:*
dtype0*
valueB:*2
strided_slice_92/stack~
strided_slice_92/stack_1Const*
_output_shapes
:*
dtype0*
valueB:+2
strided_slice_92/stack_1~
strided_slice_92/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_92/stack_2?
strided_slice_92StridedSliceReadVariableOp_92:value:0strided_slice_92/stack:output:0!strided_slice_92/stack_1:output:0!strided_slice_92/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_92h
ExpandDims_44/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_44/dim?
ExpandDims_44
ExpandDimsstrided_slice_92:output:0ExpandDims_44/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_44v
Mul_93MulMatMul:output:0ExpandDims_44:output:0*
T0*+
_output_shapes
:?????????2
Mul_93?
Sum_93/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_93/reduction_indicest
Sum_93Sum
Mul_93:z:0!Sum_93/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_93u
Reshape_93/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_93/shape?

Reshape_93ReshapeSum_93:output:0Reshape_93/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_93?
ReadVariableOp_93ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_93z
strided_slice_93/stackConst*
_output_shapes
:*
dtype0*
valueB:+2
strided_slice_93/stack~
strided_slice_93/stack_1Const*
_output_shapes
:*
dtype0*
valueB:,2
strided_slice_93/stack_1~
strided_slice_93/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_93/stack_2?
strided_slice_93StridedSliceReadVariableOp_93:value:0strided_slice_93/stack:output:0!strided_slice_93/stack_1:output:0!strided_slice_93/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_93h
ExpandDims_45/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_45/dim?
ExpandDims_45
ExpandDimsstrided_slice_93:output:0ExpandDims_45/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_45v
Mul_94MulMatMul:output:0ExpandDims_45:output:0*
T0*+
_output_shapes
:?????????2
Mul_94?
Sum_94/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_94/reduction_indicest
Sum_94Sum
Mul_94:z:0!Sum_94/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_94u
Reshape_94/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_94/shape?

Reshape_94ReshapeSum_94:output:0Reshape_94/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_94?
ReadVariableOp_94ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_94z
strided_slice_94/stackConst*
_output_shapes
:*
dtype0*
valueB:,2
strided_slice_94/stack~
strided_slice_94/stack_1Const*
_output_shapes
:*
dtype0*
valueB:-2
strided_slice_94/stack_1~
strided_slice_94/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_94/stack_2?
strided_slice_94StridedSliceReadVariableOp_94:value:0strided_slice_94/stack:output:0!strided_slice_94/stack_1:output:0!strided_slice_94/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_94h
ExpandDims_46/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_46/dim?
ExpandDims_46
ExpandDimsstrided_slice_94:output:0ExpandDims_46/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_46v
Mul_95MulMatMul:output:0ExpandDims_46:output:0*
T0*+
_output_shapes
:?????????2
Mul_95?
Sum_95/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_95/reduction_indicest
Sum_95Sum
Mul_95:z:0!Sum_95/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_95u
Reshape_95/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_95/shape?

Reshape_95ReshapeSum_95:output:0Reshape_95/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_95?
ReadVariableOp_95ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_95z
strided_slice_95/stackConst*
_output_shapes
:*
dtype0*
valueB:-2
strided_slice_95/stack~
strided_slice_95/stack_1Const*
_output_shapes
:*
dtype0*
valueB:.2
strided_slice_95/stack_1~
strided_slice_95/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_95/stack_2?
strided_slice_95StridedSliceReadVariableOp_95:value:0strided_slice_95/stack:output:0!strided_slice_95/stack_1:output:0!strided_slice_95/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_95h
ExpandDims_47/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_47/dim?
ExpandDims_47
ExpandDimsstrided_slice_95:output:0ExpandDims_47/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_47v
Mul_96MulMatMul:output:0ExpandDims_47:output:0*
T0*+
_output_shapes
:?????????2
Mul_96?
Sum_96/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_96/reduction_indicest
Sum_96Sum
Mul_96:z:0!Sum_96/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_96u
Reshape_96/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_96/shape?

Reshape_96ReshapeSum_96:output:0Reshape_96/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_96?
ReadVariableOp_96ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_96z
strided_slice_96/stackConst*
_output_shapes
:*
dtype0*
valueB:.2
strided_slice_96/stack~
strided_slice_96/stack_1Const*
_output_shapes
:*
dtype0*
valueB:/2
strided_slice_96/stack_1~
strided_slice_96/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_96/stack_2?
strided_slice_96StridedSliceReadVariableOp_96:value:0strided_slice_96/stack:output:0!strided_slice_96/stack_1:output:0!strided_slice_96/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_96h
ExpandDims_48/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_48/dim?
ExpandDims_48
ExpandDimsstrided_slice_96:output:0ExpandDims_48/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_48v
Mul_97MulMatMul:output:0ExpandDims_48:output:0*
T0*+
_output_shapes
:?????????2
Mul_97?
Sum_97/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_97/reduction_indicest
Sum_97Sum
Mul_97:z:0!Sum_97/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_97u
Reshape_97/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_97/shape?

Reshape_97ReshapeSum_97:output:0Reshape_97/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_97?
ReadVariableOp_97ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_97z
strided_slice_97/stackConst*
_output_shapes
:*
dtype0*
valueB:/2
strided_slice_97/stack~
strided_slice_97/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02
strided_slice_97/stack_1~
strided_slice_97/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_97/stack_2?
strided_slice_97StridedSliceReadVariableOp_97:value:0strided_slice_97/stack:output:0!strided_slice_97/stack_1:output:0!strided_slice_97/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_97h
ExpandDims_49/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_49/dim?
ExpandDims_49
ExpandDimsstrided_slice_97:output:0ExpandDims_49/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_49v
Mul_98MulMatMul:output:0ExpandDims_49:output:0*
T0*+
_output_shapes
:?????????2
Mul_98?
Sum_98/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_98/reduction_indicest
Sum_98Sum
Mul_98:z:0!Sum_98/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_98u
Reshape_98/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_98/shape?

Reshape_98ReshapeSum_98:output:0Reshape_98/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_98?
ReadVariableOp_98ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_98z
strided_slice_98/stackConst*
_output_shapes
:*
dtype0*
valueB:02
strided_slice_98/stack~
strided_slice_98/stack_1Const*
_output_shapes
:*
dtype0*
valueB:12
strided_slice_98/stack_1~
strided_slice_98/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_98/stack_2?
strided_slice_98StridedSliceReadVariableOp_98:value:0strided_slice_98/stack:output:0!strided_slice_98/stack_1:output:0!strided_slice_98/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_98h
ExpandDims_50/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_50/dim?
ExpandDims_50
ExpandDimsstrided_slice_98:output:0ExpandDims_50/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_50v
Mul_99MulMatMul:output:0ExpandDims_50:output:0*
T0*+
_output_shapes
:?????????2
Mul_99?
Sum_99/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_99/reduction_indicest
Sum_99Sum
Mul_99:z:0!Sum_99/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_99u
Reshape_99/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_99/shape?

Reshape_99ReshapeSum_99:output:0Reshape_99/shape:output:0*
T0*'
_output_shapes
:?????????2

Reshape_99?
ReadVariableOp_99ReadVariableOpreadvariableop_50_resource*"
_output_shapes
:2*
dtype02
ReadVariableOp_99z
strided_slice_99/stackConst*
_output_shapes
:*
dtype0*
valueB:12
strided_slice_99/stack~
strided_slice_99/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
strided_slice_99/stack_1~
strided_slice_99/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_99/stack_2?
strided_slice_99StridedSliceReadVariableOp_99:value:0strided_slice_99/stack:output:0!strided_slice_99/stack_1:output:0!strided_slice_99/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_99h
ExpandDims_51/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_51/dim?
ExpandDims_51
ExpandDimsstrided_slice_99:output:0ExpandDims_51/dim:output:0*
T0*"
_output_shapes
:2
ExpandDims_51x
Mul_100MulMatMul:output:0ExpandDims_51:output:0*
T0*+
_output_shapes
:?????????2	
Mul_100?
Sum_100/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Sum_100/reduction_indicesx
Sum_100SumMul_100:z:0"Sum_100/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2	
Sum_100w
Reshape_100/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_100/shape?
Reshape_100ReshapeSum_100:output:0Reshape_100/shape:output:0*
T0*'
_output_shapes
:?????????2
Reshape_100`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?	
concat_1ConcatV2Reshape_51:output:0Reshape_52:output:0Reshape_53:output:0Reshape_54:output:0Reshape_55:output:0Reshape_56:output:0Reshape_57:output:0Reshape_58:output:0Reshape_59:output:0Reshape_60:output:0Reshape_61:output:0Reshape_62:output:0Reshape_63:output:0Reshape_64:output:0Reshape_65:output:0Reshape_66:output:0Reshape_67:output:0Reshape_68:output:0Reshape_69:output:0Reshape_70:output:0Reshape_71:output:0Reshape_72:output:0Reshape_73:output:0Reshape_74:output:0Reshape_75:output:0Reshape_76:output:0Reshape_77:output:0Reshape_78:output:0Reshape_79:output:0Reshape_80:output:0Reshape_81:output:0Reshape_82:output:0Reshape_83:output:0Reshape_84:output:0Reshape_85:output:0Reshape_86:output:0Reshape_87:output:0Reshape_88:output:0Reshape_89:output:0Reshape_90:output:0Reshape_91:output:0Reshape_92:output:0Reshape_93:output:0Reshape_94:output:0Reshape_95:output:0Reshape_96:output:0Reshape_97:output:0Reshape_98:output:0Reshape_99:output:0Reshape_100:output:0concat_1/axis:output:0*
N2*
T0*'
_output_shapes
:?????????22

concat_1i
addAddV2concat:output:0concat_1:output:0*
T0*'
_output_shapes
:?????????22
add?
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:2*
dtype02
add_1/ReadVariableOpp
add_1AddV2add:z:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
add_1Q
ReluRelu	add_1:z:0*
T0*'
_output_shapes
:?????????22
Reluc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const
dropout/MulMulRelu:activations:0dropout/Const:output:0*
T0*'
_output_shapes
:?????????22
dropout/Mul`
dropout/ShapeShapeRelu:activations:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed?2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????22
dropout/Mul_1?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:2 *
dtype02
MatMul_1/ReadVariableOp?
MatMul_1MatMuldropout/Mul_1:z:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2

MatMul_1?
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes

: *
dtype02
add_2/ReadVariableOp{
add_2AddV2MatMul_1:product:0add_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
add_2?
moments/mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2 
moments/mean/reduction_indices?
moments/meanMean	add_2:z:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifference	add_2:z:0moments/StopGradient:output:0*
T0*'
_output_shapes
:????????? 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/muly
batchnorm/mul_1Mul	add_2:z:0batchnorm/mul:z:0*
T0*'
_output_shapes
:????????? 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:????????? 2
batchnorm/add_1_
Relu_1Relubatchnorm/add_1:z:0*
T0*'
_output_shapes
:????????? 2
Relu_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/Const?
dropout_1/MulMulRelu_1:activations:0dropout_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_1/Mulf
dropout_1/ShapeShapeRelu_1:activations:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed?*
seed22(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_1/Mul_1?
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul_2/ReadVariableOp?
MatMul_2MatMuldropout_1/Mul_1:z:0MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2

MatMul_2?
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes

: *
dtype02
add_3/ReadVariableOp{
add_3AddV2MatMul_2:product:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
add_3?
 moments_1/mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2"
 moments_1/mean/reduction_indices?
moments_1/meanMean	add_3:z:0)moments_1/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
moments_1/mean?
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*
_output_shapes

: 2
moments_1/StopGradient?
moments_1/SquaredDifferenceSquaredDifference	add_3:z:0moments_1/StopGradient:output:0*
T0*'
_output_shapes
:????????? 2
moments_1/SquaredDifference?
$moments_1/variance/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2&
$moments_1/variance/reduction_indices?
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
moments_1/variance?
moments_1/SqueezeSqueezemoments_1/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments_1/Squeeze?
moments_1/Squeeze_1Squeezemoments_1/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments_1/Squeeze_1k
batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2
batchnorm_1/add/y?
batchnorm_1/addAddV2moments_1/Squeeze_1:output:0batchnorm_1/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm_1/addi
batchnorm_1/RsqrtRsqrtbatchnorm_1/add:z:0*
T0*
_output_shapes
: 2
batchnorm_1/Rsqrt?
batchnorm_1/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02 
batchnorm_1/mul/ReadVariableOp?
batchnorm_1/mulMulbatchnorm_1/Rsqrt:y:0&batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm_1/mul
batchnorm_1/mul_1Mul	add_3:z:0batchnorm_1/mul:z:0*
T0*'
_output_shapes
:????????? 2
batchnorm_1/mul_1?
batchnorm_1/mul_2Mulmoments_1/Squeeze:output:0batchnorm_1/mul:z:0*
T0*
_output_shapes
: 2
batchnorm_1/mul_2?
batchnorm_1/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm_1/ReadVariableOp?
batchnorm_1/subSub"batchnorm_1/ReadVariableOp:value:0batchnorm_1/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm_1/sub?
batchnorm_1/add_1AddV2batchnorm_1/mul_1:z:0batchnorm_1/sub:z:0*
T0*'
_output_shapes
:????????? 2
batchnorm_1/add_1a
Relu_2Relubatchnorm_1/add_1:z:0*
T0*'
_output_shapes
:????????? 2
Relu_2g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/Const?
dropout_2/MulMulRelu_2:activations:0dropout_2/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_2/Mulf
dropout_2/ShapeShapeRelu_2:activations:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed?*
seed22(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_2/Mul_1?
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul_3/ReadVariableOp?
MatMul_3MatMuldropout_2/Mul_1:z:0MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes
: *
dtype02
add_4/ReadVariableOp{
add_4AddV2MatMul_3:product:0add_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
add_4Z
SigmoidSigmoid	add_4:z:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_10^ReadVariableOp_11^ReadVariableOp_12^ReadVariableOp_13^ReadVariableOp_14^ReadVariableOp_15^ReadVariableOp_16^ReadVariableOp_17^ReadVariableOp_18^ReadVariableOp_19^ReadVariableOp_2^ReadVariableOp_20^ReadVariableOp_21^ReadVariableOp_22^ReadVariableOp_23^ReadVariableOp_24^ReadVariableOp_25^ReadVariableOp_26^ReadVariableOp_27^ReadVariableOp_28^ReadVariableOp_29^ReadVariableOp_3^ReadVariableOp_30^ReadVariableOp_31^ReadVariableOp_32^ReadVariableOp_33^ReadVariableOp_34^ReadVariableOp_35^ReadVariableOp_36^ReadVariableOp_37^ReadVariableOp_38^ReadVariableOp_39^ReadVariableOp_4^ReadVariableOp_40^ReadVariableOp_41^ReadVariableOp_42^ReadVariableOp_43^ReadVariableOp_44^ReadVariableOp_45^ReadVariableOp_46^ReadVariableOp_47^ReadVariableOp_48^ReadVariableOp_49^ReadVariableOp_5^ReadVariableOp_50^ReadVariableOp_51^ReadVariableOp_52^ReadVariableOp_53^ReadVariableOp_54^ReadVariableOp_55^ReadVariableOp_56^ReadVariableOp_57^ReadVariableOp_58^ReadVariableOp_59^ReadVariableOp_6^ReadVariableOp_60^ReadVariableOp_61^ReadVariableOp_62^ReadVariableOp_63^ReadVariableOp_64^ReadVariableOp_65^ReadVariableOp_66^ReadVariableOp_67^ReadVariableOp_68^ReadVariableOp_69^ReadVariableOp_7^ReadVariableOp_70^ReadVariableOp_71^ReadVariableOp_72^ReadVariableOp_73^ReadVariableOp_74^ReadVariableOp_75^ReadVariableOp_76^ReadVariableOp_77^ReadVariableOp_78^ReadVariableOp_79^ReadVariableOp_8^ReadVariableOp_80^ReadVariableOp_81^ReadVariableOp_82^ReadVariableOp_83^ReadVariableOp_84^ReadVariableOp_85^ReadVariableOp_86^ReadVariableOp_87^ReadVariableOp_88^ReadVariableOp_89^ReadVariableOp_9^ReadVariableOp_90^ReadVariableOp_91^ReadVariableOp_92^ReadVariableOp_93^ReadVariableOp_94^ReadVariableOp_95^ReadVariableOp_96^ReadVariableOp_97^ReadVariableOp_98^ReadVariableOp_99^add_1/ReadVariableOp^add_2/ReadVariableOp^add_3/ReadVariableOp^add_4/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp^batchnorm_1/ReadVariableOp^batchnorm_1/mul/ReadVariableOp^embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:??????????:??????????::::::::::::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12&
ReadVariableOp_10ReadVariableOp_102&
ReadVariableOp_11ReadVariableOp_112&
ReadVariableOp_12ReadVariableOp_122&
ReadVariableOp_13ReadVariableOp_132&
ReadVariableOp_14ReadVariableOp_142&
ReadVariableOp_15ReadVariableOp_152&
ReadVariableOp_16ReadVariableOp_162&
ReadVariableOp_17ReadVariableOp_172&
ReadVariableOp_18ReadVariableOp_182&
ReadVariableOp_19ReadVariableOp_192$
ReadVariableOp_2ReadVariableOp_22&
ReadVariableOp_20ReadVariableOp_202&
ReadVariableOp_21ReadVariableOp_212&
ReadVariableOp_22ReadVariableOp_222&
ReadVariableOp_23ReadVariableOp_232&
ReadVariableOp_24ReadVariableOp_242&
ReadVariableOp_25ReadVariableOp_252&
ReadVariableOp_26ReadVariableOp_262&
ReadVariableOp_27ReadVariableOp_272&
ReadVariableOp_28ReadVariableOp_282&
ReadVariableOp_29ReadVariableOp_292$
ReadVariableOp_3ReadVariableOp_32&
ReadVariableOp_30ReadVariableOp_302&
ReadVariableOp_31ReadVariableOp_312&
ReadVariableOp_32ReadVariableOp_322&
ReadVariableOp_33ReadVariableOp_332&
ReadVariableOp_34ReadVariableOp_342&
ReadVariableOp_35ReadVariableOp_352&
ReadVariableOp_36ReadVariableOp_362&
ReadVariableOp_37ReadVariableOp_372&
ReadVariableOp_38ReadVariableOp_382&
ReadVariableOp_39ReadVariableOp_392$
ReadVariableOp_4ReadVariableOp_42&
ReadVariableOp_40ReadVariableOp_402&
ReadVariableOp_41ReadVariableOp_412&
ReadVariableOp_42ReadVariableOp_422&
ReadVariableOp_43ReadVariableOp_432&
ReadVariableOp_44ReadVariableOp_442&
ReadVariableOp_45ReadVariableOp_452&
ReadVariableOp_46ReadVariableOp_462&
ReadVariableOp_47ReadVariableOp_472&
ReadVariableOp_48ReadVariableOp_482&
ReadVariableOp_49ReadVariableOp_492$
ReadVariableOp_5ReadVariableOp_52&
ReadVariableOp_50ReadVariableOp_502&
ReadVariableOp_51ReadVariableOp_512&
ReadVariableOp_52ReadVariableOp_522&
ReadVariableOp_53ReadVariableOp_532&
ReadVariableOp_54ReadVariableOp_542&
ReadVariableOp_55ReadVariableOp_552&
ReadVariableOp_56ReadVariableOp_562&
ReadVariableOp_57ReadVariableOp_572&
ReadVariableOp_58ReadVariableOp_582&
ReadVariableOp_59ReadVariableOp_592$
ReadVariableOp_6ReadVariableOp_62&
ReadVariableOp_60ReadVariableOp_602&
ReadVariableOp_61ReadVariableOp_612&
ReadVariableOp_62ReadVariableOp_622&
ReadVariableOp_63ReadVariableOp_632&
ReadVariableOp_64ReadVariableOp_642&
ReadVariableOp_65ReadVariableOp_652&
ReadVariableOp_66ReadVariableOp_662&
ReadVariableOp_67ReadVariableOp_672&
ReadVariableOp_68ReadVariableOp_682&
ReadVariableOp_69ReadVariableOp_692$
ReadVariableOp_7ReadVariableOp_72&
ReadVariableOp_70ReadVariableOp_702&
ReadVariableOp_71ReadVariableOp_712&
ReadVariableOp_72ReadVariableOp_722&
ReadVariableOp_73ReadVariableOp_732&
ReadVariableOp_74ReadVariableOp_742&
ReadVariableOp_75ReadVariableOp_752&
ReadVariableOp_76ReadVariableOp_762&
ReadVariableOp_77ReadVariableOp_772&
ReadVariableOp_78ReadVariableOp_782&
ReadVariableOp_79ReadVariableOp_792$
ReadVariableOp_8ReadVariableOp_82&
ReadVariableOp_80ReadVariableOp_802&
ReadVariableOp_81ReadVariableOp_812&
ReadVariableOp_82ReadVariableOp_822&
ReadVariableOp_83ReadVariableOp_832&
ReadVariableOp_84ReadVariableOp_842&
ReadVariableOp_85ReadVariableOp_852&
ReadVariableOp_86ReadVariableOp_862&
ReadVariableOp_87ReadVariableOp_872&
ReadVariableOp_88ReadVariableOp_882&
ReadVariableOp_89ReadVariableOp_892$
ReadVariableOp_9ReadVariableOp_92&
ReadVariableOp_90ReadVariableOp_902&
ReadVariableOp_91ReadVariableOp_912&
ReadVariableOp_92ReadVariableOp_922&
ReadVariableOp_93ReadVariableOp_932&
ReadVariableOp_94ReadVariableOp_942&
ReadVariableOp_95ReadVariableOp_952&
ReadVariableOp_96ReadVariableOp_962&
ReadVariableOp_97ReadVariableOp_972&
ReadVariableOp_98ReadVariableOp_982&
ReadVariableOp_99ReadVariableOp_992,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
add_4/ReadVariableOpadd_4/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp28
batchnorm_1/ReadVariableOpbatchnorm_1/ReadVariableOp2@
batchnorm_1/mul/ReadVariableOpbatchnorm_1/mul/ReadVariableOp2$
embedding_lookupembedding_lookup:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1:QM
(
_output_shapes
:??????????
!
_user_specified_name	input_2
?

?
&__inference_pnn_layer_call_fn_16075345
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_pnn_layer_call_and_return_conditional_losses_160753142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:??????????:??????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1:QM
(
_output_shapes
:??????????
!
_user_specified_name	input_2
?8
?
$__inference__traced_restore_16075489
file_prefix'
#assignvariableop_feature_embeddings#
assignvariableop_1_feature_bias
assignvariableop_2_variable!
assignvariableop_3_variable_1!
assignvariableop_4_variable_2!
assignvariableop_5_variable_3!
assignvariableop_6_variable_4!
assignvariableop_7_variable_5!
assignvariableop_8_variable_6!
assignvariableop_9_variable_7"
assignvariableop_10_variable_8"
assignvariableop_11_variable_9#
assignvariableop_12_variable_10
identity_14??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6_weights/feature_embeddings/.ATTRIBUTES/VARIABLE_VALUEB0_weights/feature_bias/.ATTRIBUTES/VARIABLE_VALUEB2_weights/product-linear/.ATTRIBUTES/VARIABLE_VALUEB0_weights/product-bias/.ATTRIBUTES/VARIABLE_VALUEB;_weights/product-quadratic-outer/.ATTRIBUTES/VARIABLE_VALUEB+_weights/layer_0/.ATTRIBUTES/VARIABLE_VALUEB*_weights/bias_0/.ATTRIBUTES/VARIABLE_VALUEB+_weights/layer_1/.ATTRIBUTES/VARIABLE_VALUEB*_weights/bias_1/.ATTRIBUTES/VARIABLE_VALUEB6_weights/batch_norm_scale_0/.ATTRIBUTES/VARIABLE_VALUEB7_weights/batch_norm_offset_0/.ATTRIBUTES/VARIABLE_VALUEB*_weights/output/.ATTRIBUTES/VARIABLE_VALUEB/_weights/output_bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp#assignvariableop_feature_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_feature_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_variableIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_3Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_4Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_5Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_6Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_7Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_variable_8Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_9Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_variable_10Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_13Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_13?
Identity_14IdentityIdentity_13:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_14"#
identity_14Identity_14:output:0*I
_input_shapes8
6: :::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
??
?
#__inference__wrapped_model_16074099
input_1
input_2!
pnn_embedding_lookup_16072889
pnn_readvariableop_resource"
pnn_readvariableop_50_resource%
!pnn_add_1_readvariableop_resource(
$pnn_matmul_1_readvariableop_resource%
!pnn_add_2_readvariableop_resource-
)pnn_batchnorm_mul_readvariableop_resource)
%pnn_batchnorm_readvariableop_resource(
$pnn_matmul_2_readvariableop_resource%
!pnn_add_3_readvariableop_resource(
$pnn_matmul_3_readvariableop_resource%
!pnn_add_4_readvariableop_resource
identity??pnn/MatMul_1/ReadVariableOp?pnn/MatMul_2/ReadVariableOp?pnn/MatMul_3/ReadVariableOp?pnn/ReadVariableOp?pnn/ReadVariableOp_1?pnn/ReadVariableOp_10?pnn/ReadVariableOp_11?pnn/ReadVariableOp_12?pnn/ReadVariableOp_13?pnn/ReadVariableOp_14?pnn/ReadVariableOp_15?pnn/ReadVariableOp_16?pnn/ReadVariableOp_17?pnn/ReadVariableOp_18?pnn/ReadVariableOp_19?pnn/ReadVariableOp_2?pnn/ReadVariableOp_20?pnn/ReadVariableOp_21?pnn/ReadVariableOp_22?pnn/ReadVariableOp_23?pnn/ReadVariableOp_24?pnn/ReadVariableOp_25?pnn/ReadVariableOp_26?pnn/ReadVariableOp_27?pnn/ReadVariableOp_28?pnn/ReadVariableOp_29?pnn/ReadVariableOp_3?pnn/ReadVariableOp_30?pnn/ReadVariableOp_31?pnn/ReadVariableOp_32?pnn/ReadVariableOp_33?pnn/ReadVariableOp_34?pnn/ReadVariableOp_35?pnn/ReadVariableOp_36?pnn/ReadVariableOp_37?pnn/ReadVariableOp_38?pnn/ReadVariableOp_39?pnn/ReadVariableOp_4?pnn/ReadVariableOp_40?pnn/ReadVariableOp_41?pnn/ReadVariableOp_42?pnn/ReadVariableOp_43?pnn/ReadVariableOp_44?pnn/ReadVariableOp_45?pnn/ReadVariableOp_46?pnn/ReadVariableOp_47?pnn/ReadVariableOp_48?pnn/ReadVariableOp_49?pnn/ReadVariableOp_5?pnn/ReadVariableOp_50?pnn/ReadVariableOp_51?pnn/ReadVariableOp_52?pnn/ReadVariableOp_53?pnn/ReadVariableOp_54?pnn/ReadVariableOp_55?pnn/ReadVariableOp_56?pnn/ReadVariableOp_57?pnn/ReadVariableOp_58?pnn/ReadVariableOp_59?pnn/ReadVariableOp_6?pnn/ReadVariableOp_60?pnn/ReadVariableOp_61?pnn/ReadVariableOp_62?pnn/ReadVariableOp_63?pnn/ReadVariableOp_64?pnn/ReadVariableOp_65?pnn/ReadVariableOp_66?pnn/ReadVariableOp_67?pnn/ReadVariableOp_68?pnn/ReadVariableOp_69?pnn/ReadVariableOp_7?pnn/ReadVariableOp_70?pnn/ReadVariableOp_71?pnn/ReadVariableOp_72?pnn/ReadVariableOp_73?pnn/ReadVariableOp_74?pnn/ReadVariableOp_75?pnn/ReadVariableOp_76?pnn/ReadVariableOp_77?pnn/ReadVariableOp_78?pnn/ReadVariableOp_79?pnn/ReadVariableOp_8?pnn/ReadVariableOp_80?pnn/ReadVariableOp_81?pnn/ReadVariableOp_82?pnn/ReadVariableOp_83?pnn/ReadVariableOp_84?pnn/ReadVariableOp_85?pnn/ReadVariableOp_86?pnn/ReadVariableOp_87?pnn/ReadVariableOp_88?pnn/ReadVariableOp_89?pnn/ReadVariableOp_9?pnn/ReadVariableOp_90?pnn/ReadVariableOp_91?pnn/ReadVariableOp_92?pnn/ReadVariableOp_93?pnn/ReadVariableOp_94?pnn/ReadVariableOp_95?pnn/ReadVariableOp_96?pnn/ReadVariableOp_97?pnn/ReadVariableOp_98?pnn/ReadVariableOp_99?pnn/add_1/ReadVariableOp?pnn/add_2/ReadVariableOp?pnn/add_3/ReadVariableOp?pnn/add_4/ReadVariableOp?pnn/batchnorm/ReadVariableOp? pnn/batchnorm/mul/ReadVariableOp?pnn/batchnorm_1/ReadVariableOp?"pnn/batchnorm_1/mul/ReadVariableOp?pnn/embedding_lookup?
pnn/embedding_lookupResourceGatherpnn_embedding_lookup_16072889input_1",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*0
_class&
$"loc:@pnn/embedding_lookup/16072889*,
_output_shapes
:??????????*
dtype02
pnn/embedding_lookup?
pnn/embedding_lookup/IdentityIdentitypnn/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*0
_class&
$"loc:@pnn/embedding_lookup/16072889*,
_output_shapes
:??????????2
pnn/embedding_lookup/Identity?
pnn/embedding_lookup/Identity_1Identity&pnn/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2!
pnn/embedding_lookup/Identity_1{
pnn/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"?????     2
pnn/Reshape/shape?
pnn/ReshapeReshapeinput_2pnn/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
pnn/Reshape?
pnn/MulMul(pnn/embedding_lookup/Identity_1:output:0pnn/Reshape:output:0*
T0*,
_output_shapes
:??????????2	
pnn/Mul?
pnn/ReadVariableOpReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp|
pnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
pnn/strided_slice/stack?
pnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice/stack_1?
pnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice/stack_2?
pnn/strided_sliceStridedSlicepnn/ReadVariableOp:value:0 pnn/strided_slice/stack:output:0"pnn/strided_slice/stack_1:output:0"pnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice}
	pnn/Mul_1Mulpnn/Mul:z:0pnn/strided_slice:output:0*
T0*,
_output_shapes
:??????????2
	pnn/Mul_1?
pnn/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum/reduction_indicesz
pnn/SumSumpnn/Mul_1:z:0"pnn/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2	
pnn/Sum{
pnn/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_1/shape?
pnn/Reshape_1Reshapepnn/Sum:output:0pnn/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_1?
pnn/ReadVariableOp_1ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_1?
pnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_1/stack?
pnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_1/stack_1?
pnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_1/stack_2?
pnn/strided_slice_1StridedSlicepnn/ReadVariableOp_1:value:0"pnn/strided_slice_1/stack:output:0$pnn/strided_slice_1/stack_1:output:0$pnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_1
	pnn/Mul_2Mulpnn/Mul:z:0pnn/strided_slice_1:output:0*
T0*,
_output_shapes
:??????????2
	pnn/Mul_2?
pnn/Sum_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_1/reduction_indices?
	pnn/Sum_1Sumpnn/Mul_2:z:0$pnn/Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
	pnn/Sum_1{
pnn/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_2/shape?
pnn/Reshape_2Reshapepnn/Sum_1:output:0pnn/Reshape_2/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_2?
pnn/ReadVariableOp_2ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_2?
pnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_2/stack?
pnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_2/stack_1?
pnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_2/stack_2?
pnn/strided_slice_2StridedSlicepnn/ReadVariableOp_2:value:0"pnn/strided_slice_2/stack:output:0$pnn/strided_slice_2/stack_1:output:0$pnn/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_2
	pnn/Mul_3Mulpnn/Mul:z:0pnn/strided_slice_2:output:0*
T0*,
_output_shapes
:??????????2
	pnn/Mul_3?
pnn/Sum_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_2/reduction_indices?
	pnn/Sum_2Sumpnn/Mul_3:z:0$pnn/Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
	pnn/Sum_2{
pnn/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_3/shape?
pnn/Reshape_3Reshapepnn/Sum_2:output:0pnn/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_3?
pnn/ReadVariableOp_3ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_3?
pnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_3/stack?
pnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_3/stack_1?
pnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_3/stack_2?
pnn/strided_slice_3StridedSlicepnn/ReadVariableOp_3:value:0"pnn/strided_slice_3/stack:output:0$pnn/strided_slice_3/stack_1:output:0$pnn/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_3
	pnn/Mul_4Mulpnn/Mul:z:0pnn/strided_slice_3:output:0*
T0*,
_output_shapes
:??????????2
	pnn/Mul_4?
pnn/Sum_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_3/reduction_indices?
	pnn/Sum_3Sumpnn/Mul_4:z:0$pnn/Sum_3/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
	pnn/Sum_3{
pnn/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_4/shape?
pnn/Reshape_4Reshapepnn/Sum_3:output:0pnn/Reshape_4/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_4?
pnn/ReadVariableOp_4ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_4?
pnn/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_4/stack?
pnn/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_4/stack_1?
pnn/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_4/stack_2?
pnn/strided_slice_4StridedSlicepnn/ReadVariableOp_4:value:0"pnn/strided_slice_4/stack:output:0$pnn/strided_slice_4/stack_1:output:0$pnn/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_4
	pnn/Mul_5Mulpnn/Mul:z:0pnn/strided_slice_4:output:0*
T0*,
_output_shapes
:??????????2
	pnn/Mul_5?
pnn/Sum_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_4/reduction_indices?
	pnn/Sum_4Sumpnn/Mul_5:z:0$pnn/Sum_4/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
	pnn/Sum_4{
pnn/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_5/shape?
pnn/Reshape_5Reshapepnn/Sum_4:output:0pnn/Reshape_5/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_5?
pnn/ReadVariableOp_5ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_5?
pnn/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_5/stack?
pnn/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_5/stack_1?
pnn/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_5/stack_2?
pnn/strided_slice_5StridedSlicepnn/ReadVariableOp_5:value:0"pnn/strided_slice_5/stack:output:0$pnn/strided_slice_5/stack_1:output:0$pnn/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_5
	pnn/Mul_6Mulpnn/Mul:z:0pnn/strided_slice_5:output:0*
T0*,
_output_shapes
:??????????2
	pnn/Mul_6?
pnn/Sum_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_5/reduction_indices?
	pnn/Sum_5Sumpnn/Mul_6:z:0$pnn/Sum_5/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
	pnn/Sum_5{
pnn/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_6/shape?
pnn/Reshape_6Reshapepnn/Sum_5:output:0pnn/Reshape_6/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_6?
pnn/ReadVariableOp_6ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_6?
pnn/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_6/stack?
pnn/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_6/stack_1?
pnn/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_6/stack_2?
pnn/strided_slice_6StridedSlicepnn/ReadVariableOp_6:value:0"pnn/strided_slice_6/stack:output:0$pnn/strided_slice_6/stack_1:output:0$pnn/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_6
	pnn/Mul_7Mulpnn/Mul:z:0pnn/strided_slice_6:output:0*
T0*,
_output_shapes
:??????????2
	pnn/Mul_7?
pnn/Sum_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_6/reduction_indices?
	pnn/Sum_6Sumpnn/Mul_7:z:0$pnn/Sum_6/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
	pnn/Sum_6{
pnn/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_7/shape?
pnn/Reshape_7Reshapepnn/Sum_6:output:0pnn/Reshape_7/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_7?
pnn/ReadVariableOp_7ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_7?
pnn/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_7/stack?
pnn/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_7/stack_1?
pnn/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_7/stack_2?
pnn/strided_slice_7StridedSlicepnn/ReadVariableOp_7:value:0"pnn/strided_slice_7/stack:output:0$pnn/strided_slice_7/stack_1:output:0$pnn/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_7
	pnn/Mul_8Mulpnn/Mul:z:0pnn/strided_slice_7:output:0*
T0*,
_output_shapes
:??????????2
	pnn/Mul_8?
pnn/Sum_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_7/reduction_indices?
	pnn/Sum_7Sumpnn/Mul_8:z:0$pnn/Sum_7/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
	pnn/Sum_7{
pnn/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_8/shape?
pnn/Reshape_8Reshapepnn/Sum_7:output:0pnn/Reshape_8/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_8?
pnn/ReadVariableOp_8ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_8?
pnn/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_8/stack?
pnn/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:	2
pnn/strided_slice_8/stack_1?
pnn/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_8/stack_2?
pnn/strided_slice_8StridedSlicepnn/ReadVariableOp_8:value:0"pnn/strided_slice_8/stack:output:0$pnn/strided_slice_8/stack_1:output:0$pnn/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_8
	pnn/Mul_9Mulpnn/Mul:z:0pnn/strided_slice_8:output:0*
T0*,
_output_shapes
:??????????2
	pnn/Mul_9?
pnn/Sum_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_8/reduction_indices?
	pnn/Sum_8Sumpnn/Mul_9:z:0$pnn/Sum_8/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
	pnn/Sum_8{
pnn/Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_9/shape?
pnn/Reshape_9Reshapepnn/Sum_8:output:0pnn/Reshape_9/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_9?
pnn/ReadVariableOp_9ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_9?
pnn/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:	2
pnn/strided_slice_9/stack?
pnn/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
2
pnn/strided_slice_9/stack_1?
pnn/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_9/stack_2?
pnn/strided_slice_9StridedSlicepnn/ReadVariableOp_9:value:0"pnn/strided_slice_9/stack:output:0$pnn/strided_slice_9/stack_1:output:0$pnn/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_9?

pnn/Mul_10Mulpnn/Mul:z:0pnn/strided_slice_9:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_10?
pnn/Sum_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_9/reduction_indices?
	pnn/Sum_9Sumpnn/Mul_10:z:0$pnn/Sum_9/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
	pnn/Sum_9}
pnn/Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_10/shape?
pnn/Reshape_10Reshapepnn/Sum_9:output:0pnn/Reshape_10/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_10?
pnn/ReadVariableOp_10ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_10?
pnn/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB:
2
pnn/strided_slice_10/stack?
pnn/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_10/stack_1?
pnn/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_10/stack_2?
pnn/strided_slice_10StridedSlicepnn/ReadVariableOp_10:value:0#pnn/strided_slice_10/stack:output:0%pnn/strided_slice_10/stack_1:output:0%pnn/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_10?

pnn/Mul_11Mulpnn/Mul:z:0pnn/strided_slice_10:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_11?
pnn/Sum_10/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_10/reduction_indices?

pnn/Sum_10Sumpnn/Mul_11:z:0%pnn/Sum_10/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_10}
pnn/Reshape_11/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_11/shape?
pnn/Reshape_11Reshapepnn/Sum_10:output:0pnn/Reshape_11/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_11?
pnn/ReadVariableOp_11ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_11?
pnn/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_11/stack?
pnn/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_11/stack_1?
pnn/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_11/stack_2?
pnn/strided_slice_11StridedSlicepnn/ReadVariableOp_11:value:0#pnn/strided_slice_11/stack:output:0%pnn/strided_slice_11/stack_1:output:0%pnn/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_11?

pnn/Mul_12Mulpnn/Mul:z:0pnn/strided_slice_11:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_12?
pnn/Sum_11/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_11/reduction_indices?

pnn/Sum_11Sumpnn/Mul_12:z:0%pnn/Sum_11/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_11}
pnn/Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_12/shape?
pnn/Reshape_12Reshapepnn/Sum_11:output:0pnn/Reshape_12/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_12?
pnn/ReadVariableOp_12ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_12?
pnn/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_12/stack?
pnn/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_12/stack_1?
pnn/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_12/stack_2?
pnn/strided_slice_12StridedSlicepnn/ReadVariableOp_12:value:0#pnn/strided_slice_12/stack:output:0%pnn/strided_slice_12/stack_1:output:0%pnn/strided_slice_12/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_12?

pnn/Mul_13Mulpnn/Mul:z:0pnn/strided_slice_12:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_13?
pnn/Sum_12/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_12/reduction_indices?

pnn/Sum_12Sumpnn/Mul_13:z:0%pnn/Sum_12/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_12}
pnn/Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_13/shape?
pnn/Reshape_13Reshapepnn/Sum_12:output:0pnn/Reshape_13/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_13?
pnn/ReadVariableOp_13ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_13?
pnn/strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_13/stack?
pnn/strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_13/stack_1?
pnn/strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_13/stack_2?
pnn/strided_slice_13StridedSlicepnn/ReadVariableOp_13:value:0#pnn/strided_slice_13/stack:output:0%pnn/strided_slice_13/stack_1:output:0%pnn/strided_slice_13/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_13?

pnn/Mul_14Mulpnn/Mul:z:0pnn/strided_slice_13:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_14?
pnn/Sum_13/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_13/reduction_indices?

pnn/Sum_13Sumpnn/Mul_14:z:0%pnn/Sum_13/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_13}
pnn/Reshape_14/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_14/shape?
pnn/Reshape_14Reshapepnn/Sum_13:output:0pnn/Reshape_14/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_14?
pnn/ReadVariableOp_14ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_14?
pnn/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_14/stack?
pnn/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_14/stack_1?
pnn/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_14/stack_2?
pnn/strided_slice_14StridedSlicepnn/ReadVariableOp_14:value:0#pnn/strided_slice_14/stack:output:0%pnn/strided_slice_14/stack_1:output:0%pnn/strided_slice_14/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_14?

pnn/Mul_15Mulpnn/Mul:z:0pnn/strided_slice_14:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_15?
pnn/Sum_14/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_14/reduction_indices?

pnn/Sum_14Sumpnn/Mul_15:z:0%pnn/Sum_14/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_14}
pnn/Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_15/shape?
pnn/Reshape_15Reshapepnn/Sum_14:output:0pnn/Reshape_15/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_15?
pnn/ReadVariableOp_15ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_15?
pnn/strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_15/stack?
pnn/strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_15/stack_1?
pnn/strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_15/stack_2?
pnn/strided_slice_15StridedSlicepnn/ReadVariableOp_15:value:0#pnn/strided_slice_15/stack:output:0%pnn/strided_slice_15/stack_1:output:0%pnn/strided_slice_15/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_15?

pnn/Mul_16Mulpnn/Mul:z:0pnn/strided_slice_15:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_16?
pnn/Sum_15/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_15/reduction_indices?

pnn/Sum_15Sumpnn/Mul_16:z:0%pnn/Sum_15/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_15}
pnn/Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_16/shape?
pnn/Reshape_16Reshapepnn/Sum_15:output:0pnn/Reshape_16/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_16?
pnn/ReadVariableOp_16ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_16?
pnn/strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_16/stack?
pnn/strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_16/stack_1?
pnn/strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_16/stack_2?
pnn/strided_slice_16StridedSlicepnn/ReadVariableOp_16:value:0#pnn/strided_slice_16/stack:output:0%pnn/strided_slice_16/stack_1:output:0%pnn/strided_slice_16/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_16?

pnn/Mul_17Mulpnn/Mul:z:0pnn/strided_slice_16:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_17?
pnn/Sum_16/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_16/reduction_indices?

pnn/Sum_16Sumpnn/Mul_17:z:0%pnn/Sum_16/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_16}
pnn/Reshape_17/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_17/shape?
pnn/Reshape_17Reshapepnn/Sum_16:output:0pnn/Reshape_17/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_17?
pnn/ReadVariableOp_17ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_17?
pnn/strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_17/stack?
pnn/strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_17/stack_1?
pnn/strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_17/stack_2?
pnn/strided_slice_17StridedSlicepnn/ReadVariableOp_17:value:0#pnn/strided_slice_17/stack:output:0%pnn/strided_slice_17/stack_1:output:0%pnn/strided_slice_17/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_17?

pnn/Mul_18Mulpnn/Mul:z:0pnn/strided_slice_17:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_18?
pnn/Sum_17/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_17/reduction_indices?

pnn/Sum_17Sumpnn/Mul_18:z:0%pnn/Sum_17/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_17}
pnn/Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_18/shape?
pnn/Reshape_18Reshapepnn/Sum_17:output:0pnn/Reshape_18/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_18?
pnn/ReadVariableOp_18ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_18?
pnn/strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_18/stack?
pnn/strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_18/stack_1?
pnn/strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_18/stack_2?
pnn/strided_slice_18StridedSlicepnn/ReadVariableOp_18:value:0#pnn/strided_slice_18/stack:output:0%pnn/strided_slice_18/stack_1:output:0%pnn/strided_slice_18/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_18?

pnn/Mul_19Mulpnn/Mul:z:0pnn/strided_slice_18:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_19?
pnn/Sum_18/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_18/reduction_indices?

pnn/Sum_18Sumpnn/Mul_19:z:0%pnn/Sum_18/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_18}
pnn/Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_19/shape?
pnn/Reshape_19Reshapepnn/Sum_18:output:0pnn/Reshape_19/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_19?
pnn/ReadVariableOp_19ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_19?
pnn/strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_19/stack?
pnn/strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_19/stack_1?
pnn/strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_19/stack_2?
pnn/strided_slice_19StridedSlicepnn/ReadVariableOp_19:value:0#pnn/strided_slice_19/stack:output:0%pnn/strided_slice_19/stack_1:output:0%pnn/strided_slice_19/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_19?

pnn/Mul_20Mulpnn/Mul:z:0pnn/strided_slice_19:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_20?
pnn/Sum_19/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_19/reduction_indices?

pnn/Sum_19Sumpnn/Mul_20:z:0%pnn/Sum_19/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_19}
pnn/Reshape_20/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_20/shape?
pnn/Reshape_20Reshapepnn/Sum_19:output:0pnn/Reshape_20/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_20?
pnn/ReadVariableOp_20ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_20?
pnn/strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_20/stack?
pnn/strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_20/stack_1?
pnn/strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_20/stack_2?
pnn/strided_slice_20StridedSlicepnn/ReadVariableOp_20:value:0#pnn/strided_slice_20/stack:output:0%pnn/strided_slice_20/stack_1:output:0%pnn/strided_slice_20/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_20?

pnn/Mul_21Mulpnn/Mul:z:0pnn/strided_slice_20:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_21?
pnn/Sum_20/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_20/reduction_indices?

pnn/Sum_20Sumpnn/Mul_21:z:0%pnn/Sum_20/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_20}
pnn/Reshape_21/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_21/shape?
pnn/Reshape_21Reshapepnn/Sum_20:output:0pnn/Reshape_21/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_21?
pnn/ReadVariableOp_21ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_21?
pnn/strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_21/stack?
pnn/strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_21/stack_1?
pnn/strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_21/stack_2?
pnn/strided_slice_21StridedSlicepnn/ReadVariableOp_21:value:0#pnn/strided_slice_21/stack:output:0%pnn/strided_slice_21/stack_1:output:0%pnn/strided_slice_21/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_21?

pnn/Mul_22Mulpnn/Mul:z:0pnn/strided_slice_21:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_22?
pnn/Sum_21/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_21/reduction_indices?

pnn/Sum_21Sumpnn/Mul_22:z:0%pnn/Sum_21/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_21}
pnn/Reshape_22/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_22/shape?
pnn/Reshape_22Reshapepnn/Sum_21:output:0pnn/Reshape_22/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_22?
pnn/ReadVariableOp_22ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_22?
pnn/strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_22/stack?
pnn/strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_22/stack_1?
pnn/strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_22/stack_2?
pnn/strided_slice_22StridedSlicepnn/ReadVariableOp_22:value:0#pnn/strided_slice_22/stack:output:0%pnn/strided_slice_22/stack_1:output:0%pnn/strided_slice_22/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_22?

pnn/Mul_23Mulpnn/Mul:z:0pnn/strided_slice_22:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_23?
pnn/Sum_22/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_22/reduction_indices?

pnn/Sum_22Sumpnn/Mul_23:z:0%pnn/Sum_22/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_22}
pnn/Reshape_23/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_23/shape?
pnn/Reshape_23Reshapepnn/Sum_22:output:0pnn/Reshape_23/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_23?
pnn/ReadVariableOp_23ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_23?
pnn/strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_23/stack?
pnn/strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_23/stack_1?
pnn/strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_23/stack_2?
pnn/strided_slice_23StridedSlicepnn/ReadVariableOp_23:value:0#pnn/strided_slice_23/stack:output:0%pnn/strided_slice_23/stack_1:output:0%pnn/strided_slice_23/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_23?

pnn/Mul_24Mulpnn/Mul:z:0pnn/strided_slice_23:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_24?
pnn/Sum_23/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_23/reduction_indices?

pnn/Sum_23Sumpnn/Mul_24:z:0%pnn/Sum_23/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_23}
pnn/Reshape_24/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_24/shape?
pnn/Reshape_24Reshapepnn/Sum_23:output:0pnn/Reshape_24/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_24?
pnn/ReadVariableOp_24ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_24?
pnn/strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_24/stack?
pnn/strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_24/stack_1?
pnn/strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_24/stack_2?
pnn/strided_slice_24StridedSlicepnn/ReadVariableOp_24:value:0#pnn/strided_slice_24/stack:output:0%pnn/strided_slice_24/stack_1:output:0%pnn/strided_slice_24/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_24?

pnn/Mul_25Mulpnn/Mul:z:0pnn/strided_slice_24:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_25?
pnn/Sum_24/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_24/reduction_indices?

pnn/Sum_24Sumpnn/Mul_25:z:0%pnn/Sum_24/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_24}
pnn/Reshape_25/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_25/shape?
pnn/Reshape_25Reshapepnn/Sum_24:output:0pnn/Reshape_25/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_25?
pnn/ReadVariableOp_25ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_25?
pnn/strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_25/stack?
pnn/strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_25/stack_1?
pnn/strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_25/stack_2?
pnn/strided_slice_25StridedSlicepnn/ReadVariableOp_25:value:0#pnn/strided_slice_25/stack:output:0%pnn/strided_slice_25/stack_1:output:0%pnn/strided_slice_25/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_25?

pnn/Mul_26Mulpnn/Mul:z:0pnn/strided_slice_25:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_26?
pnn/Sum_25/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_25/reduction_indices?

pnn/Sum_25Sumpnn/Mul_26:z:0%pnn/Sum_25/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_25}
pnn/Reshape_26/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_26/shape?
pnn/Reshape_26Reshapepnn/Sum_25:output:0pnn/Reshape_26/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_26?
pnn/ReadVariableOp_26ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_26?
pnn/strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_26/stack?
pnn/strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_26/stack_1?
pnn/strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_26/stack_2?
pnn/strided_slice_26StridedSlicepnn/ReadVariableOp_26:value:0#pnn/strided_slice_26/stack:output:0%pnn/strided_slice_26/stack_1:output:0%pnn/strided_slice_26/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_26?

pnn/Mul_27Mulpnn/Mul:z:0pnn/strided_slice_26:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_27?
pnn/Sum_26/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_26/reduction_indices?

pnn/Sum_26Sumpnn/Mul_27:z:0%pnn/Sum_26/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_26}
pnn/Reshape_27/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_27/shape?
pnn/Reshape_27Reshapepnn/Sum_26:output:0pnn/Reshape_27/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_27?
pnn/ReadVariableOp_27ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_27?
pnn/strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_27/stack?
pnn/strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_27/stack_1?
pnn/strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_27/stack_2?
pnn/strided_slice_27StridedSlicepnn/ReadVariableOp_27:value:0#pnn/strided_slice_27/stack:output:0%pnn/strided_slice_27/stack_1:output:0%pnn/strided_slice_27/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_27?

pnn/Mul_28Mulpnn/Mul:z:0pnn/strided_slice_27:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_28?
pnn/Sum_27/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_27/reduction_indices?

pnn/Sum_27Sumpnn/Mul_28:z:0%pnn/Sum_27/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_27}
pnn/Reshape_28/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_28/shape?
pnn/Reshape_28Reshapepnn/Sum_27:output:0pnn/Reshape_28/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_28?
pnn/ReadVariableOp_28ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_28?
pnn/strided_slice_28/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_28/stack?
pnn/strided_slice_28/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_28/stack_1?
pnn/strided_slice_28/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_28/stack_2?
pnn/strided_slice_28StridedSlicepnn/ReadVariableOp_28:value:0#pnn/strided_slice_28/stack:output:0%pnn/strided_slice_28/stack_1:output:0%pnn/strided_slice_28/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_28?

pnn/Mul_29Mulpnn/Mul:z:0pnn/strided_slice_28:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_29?
pnn/Sum_28/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_28/reduction_indices?

pnn/Sum_28Sumpnn/Mul_29:z:0%pnn/Sum_28/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_28}
pnn/Reshape_29/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_29/shape?
pnn/Reshape_29Reshapepnn/Sum_28:output:0pnn/Reshape_29/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_29?
pnn/ReadVariableOp_29ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_29?
pnn/strided_slice_29/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_29/stack?
pnn/strided_slice_29/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_29/stack_1?
pnn/strided_slice_29/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_29/stack_2?
pnn/strided_slice_29StridedSlicepnn/ReadVariableOp_29:value:0#pnn/strided_slice_29/stack:output:0%pnn/strided_slice_29/stack_1:output:0%pnn/strided_slice_29/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_29?

pnn/Mul_30Mulpnn/Mul:z:0pnn/strided_slice_29:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_30?
pnn/Sum_29/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_29/reduction_indices?

pnn/Sum_29Sumpnn/Mul_30:z:0%pnn/Sum_29/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_29}
pnn/Reshape_30/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_30/shape?
pnn/Reshape_30Reshapepnn/Sum_29:output:0pnn/Reshape_30/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_30?
pnn/ReadVariableOp_30ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_30?
pnn/strided_slice_30/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_30/stack?
pnn/strided_slice_30/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_30/stack_1?
pnn/strided_slice_30/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_30/stack_2?
pnn/strided_slice_30StridedSlicepnn/ReadVariableOp_30:value:0#pnn/strided_slice_30/stack:output:0%pnn/strided_slice_30/stack_1:output:0%pnn/strided_slice_30/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_30?

pnn/Mul_31Mulpnn/Mul:z:0pnn/strided_slice_30:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_31?
pnn/Sum_30/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_30/reduction_indices?

pnn/Sum_30Sumpnn/Mul_31:z:0%pnn/Sum_30/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_30}
pnn/Reshape_31/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_31/shape?
pnn/Reshape_31Reshapepnn/Sum_30:output:0pnn/Reshape_31/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_31?
pnn/ReadVariableOp_31ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_31?
pnn/strided_slice_31/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_31/stack?
pnn/strided_slice_31/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
pnn/strided_slice_31/stack_1?
pnn/strided_slice_31/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_31/stack_2?
pnn/strided_slice_31StridedSlicepnn/ReadVariableOp_31:value:0#pnn/strided_slice_31/stack:output:0%pnn/strided_slice_31/stack_1:output:0%pnn/strided_slice_31/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_31?

pnn/Mul_32Mulpnn/Mul:z:0pnn/strided_slice_31:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_32?
pnn/Sum_31/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_31/reduction_indices?

pnn/Sum_31Sumpnn/Mul_32:z:0%pnn/Sum_31/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_31}
pnn/Reshape_32/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_32/shape?
pnn/Reshape_32Reshapepnn/Sum_31:output:0pnn/Reshape_32/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_32?
pnn/ReadVariableOp_32ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_32?
pnn/strided_slice_32/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
pnn/strided_slice_32/stack?
pnn/strided_slice_32/stack_1Const*
_output_shapes
:*
dtype0*
valueB:!2
pnn/strided_slice_32/stack_1?
pnn/strided_slice_32/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_32/stack_2?
pnn/strided_slice_32StridedSlicepnn/ReadVariableOp_32:value:0#pnn/strided_slice_32/stack:output:0%pnn/strided_slice_32/stack_1:output:0%pnn/strided_slice_32/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_32?

pnn/Mul_33Mulpnn/Mul:z:0pnn/strided_slice_32:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_33?
pnn/Sum_32/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_32/reduction_indices?

pnn/Sum_32Sumpnn/Mul_33:z:0%pnn/Sum_32/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_32}
pnn/Reshape_33/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_33/shape?
pnn/Reshape_33Reshapepnn/Sum_32:output:0pnn/Reshape_33/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_33?
pnn/ReadVariableOp_33ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_33?
pnn/strided_slice_33/stackConst*
_output_shapes
:*
dtype0*
valueB:!2
pnn/strided_slice_33/stack?
pnn/strided_slice_33/stack_1Const*
_output_shapes
:*
dtype0*
valueB:"2
pnn/strided_slice_33/stack_1?
pnn/strided_slice_33/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_33/stack_2?
pnn/strided_slice_33StridedSlicepnn/ReadVariableOp_33:value:0#pnn/strided_slice_33/stack:output:0%pnn/strided_slice_33/stack_1:output:0%pnn/strided_slice_33/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_33?

pnn/Mul_34Mulpnn/Mul:z:0pnn/strided_slice_33:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_34?
pnn/Sum_33/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_33/reduction_indices?

pnn/Sum_33Sumpnn/Mul_34:z:0%pnn/Sum_33/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_33}
pnn/Reshape_34/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_34/shape?
pnn/Reshape_34Reshapepnn/Sum_33:output:0pnn/Reshape_34/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_34?
pnn/ReadVariableOp_34ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_34?
pnn/strided_slice_34/stackConst*
_output_shapes
:*
dtype0*
valueB:"2
pnn/strided_slice_34/stack?
pnn/strided_slice_34/stack_1Const*
_output_shapes
:*
dtype0*
valueB:#2
pnn/strided_slice_34/stack_1?
pnn/strided_slice_34/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_34/stack_2?
pnn/strided_slice_34StridedSlicepnn/ReadVariableOp_34:value:0#pnn/strided_slice_34/stack:output:0%pnn/strided_slice_34/stack_1:output:0%pnn/strided_slice_34/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_34?

pnn/Mul_35Mulpnn/Mul:z:0pnn/strided_slice_34:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_35?
pnn/Sum_34/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_34/reduction_indices?

pnn/Sum_34Sumpnn/Mul_35:z:0%pnn/Sum_34/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_34}
pnn/Reshape_35/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_35/shape?
pnn/Reshape_35Reshapepnn/Sum_34:output:0pnn/Reshape_35/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_35?
pnn/ReadVariableOp_35ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_35?
pnn/strided_slice_35/stackConst*
_output_shapes
:*
dtype0*
valueB:#2
pnn/strided_slice_35/stack?
pnn/strided_slice_35/stack_1Const*
_output_shapes
:*
dtype0*
valueB:$2
pnn/strided_slice_35/stack_1?
pnn/strided_slice_35/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_35/stack_2?
pnn/strided_slice_35StridedSlicepnn/ReadVariableOp_35:value:0#pnn/strided_slice_35/stack:output:0%pnn/strided_slice_35/stack_1:output:0%pnn/strided_slice_35/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_35?

pnn/Mul_36Mulpnn/Mul:z:0pnn/strided_slice_35:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_36?
pnn/Sum_35/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_35/reduction_indices?

pnn/Sum_35Sumpnn/Mul_36:z:0%pnn/Sum_35/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_35}
pnn/Reshape_36/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_36/shape?
pnn/Reshape_36Reshapepnn/Sum_35:output:0pnn/Reshape_36/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_36?
pnn/ReadVariableOp_36ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_36?
pnn/strided_slice_36/stackConst*
_output_shapes
:*
dtype0*
valueB:$2
pnn/strided_slice_36/stack?
pnn/strided_slice_36/stack_1Const*
_output_shapes
:*
dtype0*
valueB:%2
pnn/strided_slice_36/stack_1?
pnn/strided_slice_36/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_36/stack_2?
pnn/strided_slice_36StridedSlicepnn/ReadVariableOp_36:value:0#pnn/strided_slice_36/stack:output:0%pnn/strided_slice_36/stack_1:output:0%pnn/strided_slice_36/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_36?

pnn/Mul_37Mulpnn/Mul:z:0pnn/strided_slice_36:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_37?
pnn/Sum_36/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_36/reduction_indices?

pnn/Sum_36Sumpnn/Mul_37:z:0%pnn/Sum_36/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_36}
pnn/Reshape_37/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_37/shape?
pnn/Reshape_37Reshapepnn/Sum_36:output:0pnn/Reshape_37/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_37?
pnn/ReadVariableOp_37ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_37?
pnn/strided_slice_37/stackConst*
_output_shapes
:*
dtype0*
valueB:%2
pnn/strided_slice_37/stack?
pnn/strided_slice_37/stack_1Const*
_output_shapes
:*
dtype0*
valueB:&2
pnn/strided_slice_37/stack_1?
pnn/strided_slice_37/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_37/stack_2?
pnn/strided_slice_37StridedSlicepnn/ReadVariableOp_37:value:0#pnn/strided_slice_37/stack:output:0%pnn/strided_slice_37/stack_1:output:0%pnn/strided_slice_37/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_37?

pnn/Mul_38Mulpnn/Mul:z:0pnn/strided_slice_37:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_38?
pnn/Sum_37/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_37/reduction_indices?

pnn/Sum_37Sumpnn/Mul_38:z:0%pnn/Sum_37/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_37}
pnn/Reshape_38/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_38/shape?
pnn/Reshape_38Reshapepnn/Sum_37:output:0pnn/Reshape_38/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_38?
pnn/ReadVariableOp_38ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_38?
pnn/strided_slice_38/stackConst*
_output_shapes
:*
dtype0*
valueB:&2
pnn/strided_slice_38/stack?
pnn/strided_slice_38/stack_1Const*
_output_shapes
:*
dtype0*
valueB:'2
pnn/strided_slice_38/stack_1?
pnn/strided_slice_38/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_38/stack_2?
pnn/strided_slice_38StridedSlicepnn/ReadVariableOp_38:value:0#pnn/strided_slice_38/stack:output:0%pnn/strided_slice_38/stack_1:output:0%pnn/strided_slice_38/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_38?

pnn/Mul_39Mulpnn/Mul:z:0pnn/strided_slice_38:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_39?
pnn/Sum_38/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_38/reduction_indices?

pnn/Sum_38Sumpnn/Mul_39:z:0%pnn/Sum_38/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_38}
pnn/Reshape_39/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_39/shape?
pnn/Reshape_39Reshapepnn/Sum_38:output:0pnn/Reshape_39/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_39?
pnn/ReadVariableOp_39ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_39?
pnn/strided_slice_39/stackConst*
_output_shapes
:*
dtype0*
valueB:'2
pnn/strided_slice_39/stack?
pnn/strided_slice_39/stack_1Const*
_output_shapes
:*
dtype0*
valueB:(2
pnn/strided_slice_39/stack_1?
pnn/strided_slice_39/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_39/stack_2?
pnn/strided_slice_39StridedSlicepnn/ReadVariableOp_39:value:0#pnn/strided_slice_39/stack:output:0%pnn/strided_slice_39/stack_1:output:0%pnn/strided_slice_39/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_39?

pnn/Mul_40Mulpnn/Mul:z:0pnn/strided_slice_39:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_40?
pnn/Sum_39/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_39/reduction_indices?

pnn/Sum_39Sumpnn/Mul_40:z:0%pnn/Sum_39/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_39}
pnn/Reshape_40/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_40/shape?
pnn/Reshape_40Reshapepnn/Sum_39:output:0pnn/Reshape_40/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_40?
pnn/ReadVariableOp_40ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_40?
pnn/strided_slice_40/stackConst*
_output_shapes
:*
dtype0*
valueB:(2
pnn/strided_slice_40/stack?
pnn/strided_slice_40/stack_1Const*
_output_shapes
:*
dtype0*
valueB:)2
pnn/strided_slice_40/stack_1?
pnn/strided_slice_40/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_40/stack_2?
pnn/strided_slice_40StridedSlicepnn/ReadVariableOp_40:value:0#pnn/strided_slice_40/stack:output:0%pnn/strided_slice_40/stack_1:output:0%pnn/strided_slice_40/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_40?

pnn/Mul_41Mulpnn/Mul:z:0pnn/strided_slice_40:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_41?
pnn/Sum_40/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_40/reduction_indices?

pnn/Sum_40Sumpnn/Mul_41:z:0%pnn/Sum_40/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_40}
pnn/Reshape_41/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_41/shape?
pnn/Reshape_41Reshapepnn/Sum_40:output:0pnn/Reshape_41/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_41?
pnn/ReadVariableOp_41ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_41?
pnn/strided_slice_41/stackConst*
_output_shapes
:*
dtype0*
valueB:)2
pnn/strided_slice_41/stack?
pnn/strided_slice_41/stack_1Const*
_output_shapes
:*
dtype0*
valueB:*2
pnn/strided_slice_41/stack_1?
pnn/strided_slice_41/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_41/stack_2?
pnn/strided_slice_41StridedSlicepnn/ReadVariableOp_41:value:0#pnn/strided_slice_41/stack:output:0%pnn/strided_slice_41/stack_1:output:0%pnn/strided_slice_41/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_41?

pnn/Mul_42Mulpnn/Mul:z:0pnn/strided_slice_41:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_42?
pnn/Sum_41/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_41/reduction_indices?

pnn/Sum_41Sumpnn/Mul_42:z:0%pnn/Sum_41/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_41}
pnn/Reshape_42/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_42/shape?
pnn/Reshape_42Reshapepnn/Sum_41:output:0pnn/Reshape_42/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_42?
pnn/ReadVariableOp_42ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_42?
pnn/strided_slice_42/stackConst*
_output_shapes
:*
dtype0*
valueB:*2
pnn/strided_slice_42/stack?
pnn/strided_slice_42/stack_1Const*
_output_shapes
:*
dtype0*
valueB:+2
pnn/strided_slice_42/stack_1?
pnn/strided_slice_42/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_42/stack_2?
pnn/strided_slice_42StridedSlicepnn/ReadVariableOp_42:value:0#pnn/strided_slice_42/stack:output:0%pnn/strided_slice_42/stack_1:output:0%pnn/strided_slice_42/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_42?

pnn/Mul_43Mulpnn/Mul:z:0pnn/strided_slice_42:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_43?
pnn/Sum_42/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_42/reduction_indices?

pnn/Sum_42Sumpnn/Mul_43:z:0%pnn/Sum_42/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_42}
pnn/Reshape_43/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_43/shape?
pnn/Reshape_43Reshapepnn/Sum_42:output:0pnn/Reshape_43/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_43?
pnn/ReadVariableOp_43ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_43?
pnn/strided_slice_43/stackConst*
_output_shapes
:*
dtype0*
valueB:+2
pnn/strided_slice_43/stack?
pnn/strided_slice_43/stack_1Const*
_output_shapes
:*
dtype0*
valueB:,2
pnn/strided_slice_43/stack_1?
pnn/strided_slice_43/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_43/stack_2?
pnn/strided_slice_43StridedSlicepnn/ReadVariableOp_43:value:0#pnn/strided_slice_43/stack:output:0%pnn/strided_slice_43/stack_1:output:0%pnn/strided_slice_43/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_43?

pnn/Mul_44Mulpnn/Mul:z:0pnn/strided_slice_43:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_44?
pnn/Sum_43/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_43/reduction_indices?

pnn/Sum_43Sumpnn/Mul_44:z:0%pnn/Sum_43/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_43}
pnn/Reshape_44/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_44/shape?
pnn/Reshape_44Reshapepnn/Sum_43:output:0pnn/Reshape_44/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_44?
pnn/ReadVariableOp_44ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_44?
pnn/strided_slice_44/stackConst*
_output_shapes
:*
dtype0*
valueB:,2
pnn/strided_slice_44/stack?
pnn/strided_slice_44/stack_1Const*
_output_shapes
:*
dtype0*
valueB:-2
pnn/strided_slice_44/stack_1?
pnn/strided_slice_44/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_44/stack_2?
pnn/strided_slice_44StridedSlicepnn/ReadVariableOp_44:value:0#pnn/strided_slice_44/stack:output:0%pnn/strided_slice_44/stack_1:output:0%pnn/strided_slice_44/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_44?

pnn/Mul_45Mulpnn/Mul:z:0pnn/strided_slice_44:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_45?
pnn/Sum_44/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_44/reduction_indices?

pnn/Sum_44Sumpnn/Mul_45:z:0%pnn/Sum_44/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_44}
pnn/Reshape_45/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_45/shape?
pnn/Reshape_45Reshapepnn/Sum_44:output:0pnn/Reshape_45/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_45?
pnn/ReadVariableOp_45ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_45?
pnn/strided_slice_45/stackConst*
_output_shapes
:*
dtype0*
valueB:-2
pnn/strided_slice_45/stack?
pnn/strided_slice_45/stack_1Const*
_output_shapes
:*
dtype0*
valueB:.2
pnn/strided_slice_45/stack_1?
pnn/strided_slice_45/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_45/stack_2?
pnn/strided_slice_45StridedSlicepnn/ReadVariableOp_45:value:0#pnn/strided_slice_45/stack:output:0%pnn/strided_slice_45/stack_1:output:0%pnn/strided_slice_45/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_45?

pnn/Mul_46Mulpnn/Mul:z:0pnn/strided_slice_45:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_46?
pnn/Sum_45/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_45/reduction_indices?

pnn/Sum_45Sumpnn/Mul_46:z:0%pnn/Sum_45/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_45}
pnn/Reshape_46/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_46/shape?
pnn/Reshape_46Reshapepnn/Sum_45:output:0pnn/Reshape_46/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_46?
pnn/ReadVariableOp_46ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_46?
pnn/strided_slice_46/stackConst*
_output_shapes
:*
dtype0*
valueB:.2
pnn/strided_slice_46/stack?
pnn/strided_slice_46/stack_1Const*
_output_shapes
:*
dtype0*
valueB:/2
pnn/strided_slice_46/stack_1?
pnn/strided_slice_46/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_46/stack_2?
pnn/strided_slice_46StridedSlicepnn/ReadVariableOp_46:value:0#pnn/strided_slice_46/stack:output:0%pnn/strided_slice_46/stack_1:output:0%pnn/strided_slice_46/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_46?

pnn/Mul_47Mulpnn/Mul:z:0pnn/strided_slice_46:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_47?
pnn/Sum_46/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_46/reduction_indices?

pnn/Sum_46Sumpnn/Mul_47:z:0%pnn/Sum_46/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_46}
pnn/Reshape_47/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_47/shape?
pnn/Reshape_47Reshapepnn/Sum_46:output:0pnn/Reshape_47/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_47?
pnn/ReadVariableOp_47ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_47?
pnn/strided_slice_47/stackConst*
_output_shapes
:*
dtype0*
valueB:/2
pnn/strided_slice_47/stack?
pnn/strided_slice_47/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02
pnn/strided_slice_47/stack_1?
pnn/strided_slice_47/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_47/stack_2?
pnn/strided_slice_47StridedSlicepnn/ReadVariableOp_47:value:0#pnn/strided_slice_47/stack:output:0%pnn/strided_slice_47/stack_1:output:0%pnn/strided_slice_47/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_47?

pnn/Mul_48Mulpnn/Mul:z:0pnn/strided_slice_47:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_48?
pnn/Sum_47/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_47/reduction_indices?

pnn/Sum_47Sumpnn/Mul_48:z:0%pnn/Sum_47/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_47}
pnn/Reshape_48/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_48/shape?
pnn/Reshape_48Reshapepnn/Sum_47:output:0pnn/Reshape_48/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_48?
pnn/ReadVariableOp_48ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_48?
pnn/strided_slice_48/stackConst*
_output_shapes
:*
dtype0*
valueB:02
pnn/strided_slice_48/stack?
pnn/strided_slice_48/stack_1Const*
_output_shapes
:*
dtype0*
valueB:12
pnn/strided_slice_48/stack_1?
pnn/strided_slice_48/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_48/stack_2?
pnn/strided_slice_48StridedSlicepnn/ReadVariableOp_48:value:0#pnn/strided_slice_48/stack:output:0%pnn/strided_slice_48/stack_1:output:0%pnn/strided_slice_48/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_48?

pnn/Mul_49Mulpnn/Mul:z:0pnn/strided_slice_48:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_49?
pnn/Sum_48/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_48/reduction_indices?

pnn/Sum_48Sumpnn/Mul_49:z:0%pnn/Sum_48/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_48}
pnn/Reshape_49/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_49/shape?
pnn/Reshape_49Reshapepnn/Sum_48:output:0pnn/Reshape_49/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_49?
pnn/ReadVariableOp_49ReadVariableOppnn_readvariableop_resource*#
_output_shapes
:2?*
dtype02
pnn/ReadVariableOp_49?
pnn/strided_slice_49/stackConst*
_output_shapes
:*
dtype0*
valueB:12
pnn/strided_slice_49/stack?
pnn/strided_slice_49/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
pnn/strided_slice_49/stack_1?
pnn/strided_slice_49/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_49/stack_2?
pnn/strided_slice_49StridedSlicepnn/ReadVariableOp_49:value:0#pnn/strided_slice_49/stack:output:0%pnn/strided_slice_49/stack_1:output:0%pnn/strided_slice_49/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?*
shrink_axis_mask2
pnn/strided_slice_49?

pnn/Mul_50Mulpnn/Mul:z:0pnn/strided_slice_49:output:0*
T0*,
_output_shapes
:??????????2

pnn/Mul_50?
pnn/Sum_49/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_49/reduction_indices?

pnn/Sum_49Sumpnn/Mul_50:z:0%pnn/Sum_49/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_49}
pnn/Reshape_50/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_50/shape?
pnn/Reshape_50Reshapepnn/Sum_49:output:0pnn/Reshape_50/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_50d
pnn/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
pnn/concat/axis?


pnn/concatConcatV2pnn/Reshape_1:output:0pnn/Reshape_2:output:0pnn/Reshape_3:output:0pnn/Reshape_4:output:0pnn/Reshape_5:output:0pnn/Reshape_6:output:0pnn/Reshape_7:output:0pnn/Reshape_8:output:0pnn/Reshape_9:output:0pnn/Reshape_10:output:0pnn/Reshape_11:output:0pnn/Reshape_12:output:0pnn/Reshape_13:output:0pnn/Reshape_14:output:0pnn/Reshape_15:output:0pnn/Reshape_16:output:0pnn/Reshape_17:output:0pnn/Reshape_18:output:0pnn/Reshape_19:output:0pnn/Reshape_20:output:0pnn/Reshape_21:output:0pnn/Reshape_22:output:0pnn/Reshape_23:output:0pnn/Reshape_24:output:0pnn/Reshape_25:output:0pnn/Reshape_26:output:0pnn/Reshape_27:output:0pnn/Reshape_28:output:0pnn/Reshape_29:output:0pnn/Reshape_30:output:0pnn/Reshape_31:output:0pnn/Reshape_32:output:0pnn/Reshape_33:output:0pnn/Reshape_34:output:0pnn/Reshape_35:output:0pnn/Reshape_36:output:0pnn/Reshape_37:output:0pnn/Reshape_38:output:0pnn/Reshape_39:output:0pnn/Reshape_40:output:0pnn/Reshape_41:output:0pnn/Reshape_42:output:0pnn/Reshape_43:output:0pnn/Reshape_44:output:0pnn/Reshape_45:output:0pnn/Reshape_46:output:0pnn/Reshape_47:output:0pnn/Reshape_48:output:0pnn/Reshape_49:output:0pnn/Reshape_50:output:0pnn/concat/axis:output:0*
N2*
T0*'
_output_shapes
:?????????22

pnn/concat~
pnn/Sum_50/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
pnn/Sum_50/reduction_indices?

pnn/Sum_50Sumpnn/Mul:z:0%pnn/Sum_50/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2

pnn/Sum_50j
pnn/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
pnn/ExpandDims/dim?
pnn/ExpandDims
ExpandDimspnn/Sum_50:output:0pnn/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
pnn/ExpandDimsn
pnn/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
pnn/ExpandDims_1/dim?
pnn/ExpandDims_1
ExpandDimspnn/Sum_50:output:0pnn/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
pnn/ExpandDims_1?

pnn/MatMulBatchMatMulV2pnn/ExpandDims:output:0pnn/ExpandDims_1:output:0*
T0*+
_output_shapes
:?????????2

pnn/MatMul?
pnn/ReadVariableOp_50ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_50?
pnn/strided_slice_50/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
pnn/strided_slice_50/stack?
pnn/strided_slice_50/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_50/stack_1?
pnn/strided_slice_50/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_50/stack_2?
pnn/strided_slice_50StridedSlicepnn/ReadVariableOp_50:value:0#pnn/strided_slice_50/stack:output:0%pnn/strided_slice_50/stack_1:output:0%pnn/strided_slice_50/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_50n
pnn/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_2/dim?
pnn/ExpandDims_2
ExpandDimspnn/strided_slice_50:output:0pnn/ExpandDims_2/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_2?

pnn/Mul_51Mulpnn/MatMul:output:0pnn/ExpandDims_2:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_51?
pnn/Sum_51/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_51/reduction_indices?

pnn/Sum_51Sumpnn/Mul_51:z:0%pnn/Sum_51/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_51}
pnn/Reshape_51/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_51/shape?
pnn/Reshape_51Reshapepnn/Sum_51:output:0pnn/Reshape_51/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_51?
pnn/ReadVariableOp_51ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_51?
pnn/strided_slice_51/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_51/stack?
pnn/strided_slice_51/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_51/stack_1?
pnn/strided_slice_51/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_51/stack_2?
pnn/strided_slice_51StridedSlicepnn/ReadVariableOp_51:value:0#pnn/strided_slice_51/stack:output:0%pnn/strided_slice_51/stack_1:output:0%pnn/strided_slice_51/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_51n
pnn/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_3/dim?
pnn/ExpandDims_3
ExpandDimspnn/strided_slice_51:output:0pnn/ExpandDims_3/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_3?

pnn/Mul_52Mulpnn/MatMul:output:0pnn/ExpandDims_3:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_52?
pnn/Sum_52/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_52/reduction_indices?

pnn/Sum_52Sumpnn/Mul_52:z:0%pnn/Sum_52/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_52}
pnn/Reshape_52/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_52/shape?
pnn/Reshape_52Reshapepnn/Sum_52:output:0pnn/Reshape_52/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_52?
pnn/ReadVariableOp_52ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_52?
pnn/strided_slice_52/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_52/stack?
pnn/strided_slice_52/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_52/stack_1?
pnn/strided_slice_52/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_52/stack_2?
pnn/strided_slice_52StridedSlicepnn/ReadVariableOp_52:value:0#pnn/strided_slice_52/stack:output:0%pnn/strided_slice_52/stack_1:output:0%pnn/strided_slice_52/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_52n
pnn/ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_4/dim?
pnn/ExpandDims_4
ExpandDimspnn/strided_slice_52:output:0pnn/ExpandDims_4/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_4?

pnn/Mul_53Mulpnn/MatMul:output:0pnn/ExpandDims_4:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_53?
pnn/Sum_53/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_53/reduction_indices?

pnn/Sum_53Sumpnn/Mul_53:z:0%pnn/Sum_53/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_53}
pnn/Reshape_53/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_53/shape?
pnn/Reshape_53Reshapepnn/Sum_53:output:0pnn/Reshape_53/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_53?
pnn/ReadVariableOp_53ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_53?
pnn/strided_slice_53/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_53/stack?
pnn/strided_slice_53/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_53/stack_1?
pnn/strided_slice_53/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_53/stack_2?
pnn/strided_slice_53StridedSlicepnn/ReadVariableOp_53:value:0#pnn/strided_slice_53/stack:output:0%pnn/strided_slice_53/stack_1:output:0%pnn/strided_slice_53/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_53n
pnn/ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_5/dim?
pnn/ExpandDims_5
ExpandDimspnn/strided_slice_53:output:0pnn/ExpandDims_5/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_5?

pnn/Mul_54Mulpnn/MatMul:output:0pnn/ExpandDims_5:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_54?
pnn/Sum_54/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_54/reduction_indices?

pnn/Sum_54Sumpnn/Mul_54:z:0%pnn/Sum_54/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_54}
pnn/Reshape_54/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_54/shape?
pnn/Reshape_54Reshapepnn/Sum_54:output:0pnn/Reshape_54/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_54?
pnn/ReadVariableOp_54ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_54?
pnn/strided_slice_54/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_54/stack?
pnn/strided_slice_54/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_54/stack_1?
pnn/strided_slice_54/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_54/stack_2?
pnn/strided_slice_54StridedSlicepnn/ReadVariableOp_54:value:0#pnn/strided_slice_54/stack:output:0%pnn/strided_slice_54/stack_1:output:0%pnn/strided_slice_54/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_54n
pnn/ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_6/dim?
pnn/ExpandDims_6
ExpandDimspnn/strided_slice_54:output:0pnn/ExpandDims_6/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_6?

pnn/Mul_55Mulpnn/MatMul:output:0pnn/ExpandDims_6:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_55?
pnn/Sum_55/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_55/reduction_indices?

pnn/Sum_55Sumpnn/Mul_55:z:0%pnn/Sum_55/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_55}
pnn/Reshape_55/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_55/shape?
pnn/Reshape_55Reshapepnn/Sum_55:output:0pnn/Reshape_55/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_55?
pnn/ReadVariableOp_55ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_55?
pnn/strided_slice_55/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_55/stack?
pnn/strided_slice_55/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_55/stack_1?
pnn/strided_slice_55/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_55/stack_2?
pnn/strided_slice_55StridedSlicepnn/ReadVariableOp_55:value:0#pnn/strided_slice_55/stack:output:0%pnn/strided_slice_55/stack_1:output:0%pnn/strided_slice_55/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_55n
pnn/ExpandDims_7/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_7/dim?
pnn/ExpandDims_7
ExpandDimspnn/strided_slice_55:output:0pnn/ExpandDims_7/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_7?

pnn/Mul_56Mulpnn/MatMul:output:0pnn/ExpandDims_7:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_56?
pnn/Sum_56/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_56/reduction_indices?

pnn/Sum_56Sumpnn/Mul_56:z:0%pnn/Sum_56/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_56}
pnn/Reshape_56/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_56/shape?
pnn/Reshape_56Reshapepnn/Sum_56:output:0pnn/Reshape_56/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_56?
pnn/ReadVariableOp_56ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_56?
pnn/strided_slice_56/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_56/stack?
pnn/strided_slice_56/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_56/stack_1?
pnn/strided_slice_56/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_56/stack_2?
pnn/strided_slice_56StridedSlicepnn/ReadVariableOp_56:value:0#pnn/strided_slice_56/stack:output:0%pnn/strided_slice_56/stack_1:output:0%pnn/strided_slice_56/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_56n
pnn/ExpandDims_8/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_8/dim?
pnn/ExpandDims_8
ExpandDimspnn/strided_slice_56:output:0pnn/ExpandDims_8/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_8?

pnn/Mul_57Mulpnn/MatMul:output:0pnn/ExpandDims_8:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_57?
pnn/Sum_57/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_57/reduction_indices?

pnn/Sum_57Sumpnn/Mul_57:z:0%pnn/Sum_57/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_57}
pnn/Reshape_57/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_57/shape?
pnn/Reshape_57Reshapepnn/Sum_57:output:0pnn/Reshape_57/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_57?
pnn/ReadVariableOp_57ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_57?
pnn/strided_slice_57/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_57/stack?
pnn/strided_slice_57/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_57/stack_1?
pnn/strided_slice_57/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_57/stack_2?
pnn/strided_slice_57StridedSlicepnn/ReadVariableOp_57:value:0#pnn/strided_slice_57/stack:output:0%pnn/strided_slice_57/stack_1:output:0%pnn/strided_slice_57/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_57n
pnn/ExpandDims_9/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_9/dim?
pnn/ExpandDims_9
ExpandDimspnn/strided_slice_57:output:0pnn/ExpandDims_9/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_9?

pnn/Mul_58Mulpnn/MatMul:output:0pnn/ExpandDims_9:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_58?
pnn/Sum_58/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_58/reduction_indices?

pnn/Sum_58Sumpnn/Mul_58:z:0%pnn/Sum_58/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_58}
pnn/Reshape_58/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_58/shape?
pnn/Reshape_58Reshapepnn/Sum_58:output:0pnn/Reshape_58/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_58?
pnn/ReadVariableOp_58ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_58?
pnn/strided_slice_58/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_58/stack?
pnn/strided_slice_58/stack_1Const*
_output_shapes
:*
dtype0*
valueB:	2
pnn/strided_slice_58/stack_1?
pnn/strided_slice_58/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_58/stack_2?
pnn/strided_slice_58StridedSlicepnn/ReadVariableOp_58:value:0#pnn/strided_slice_58/stack:output:0%pnn/strided_slice_58/stack_1:output:0%pnn/strided_slice_58/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_58p
pnn/ExpandDims_10/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_10/dim?
pnn/ExpandDims_10
ExpandDimspnn/strided_slice_58:output:0pnn/ExpandDims_10/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_10?

pnn/Mul_59Mulpnn/MatMul:output:0pnn/ExpandDims_10:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_59?
pnn/Sum_59/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_59/reduction_indices?

pnn/Sum_59Sumpnn/Mul_59:z:0%pnn/Sum_59/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_59}
pnn/Reshape_59/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_59/shape?
pnn/Reshape_59Reshapepnn/Sum_59:output:0pnn/Reshape_59/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_59?
pnn/ReadVariableOp_59ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_59?
pnn/strided_slice_59/stackConst*
_output_shapes
:*
dtype0*
valueB:	2
pnn/strided_slice_59/stack?
pnn/strided_slice_59/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
2
pnn/strided_slice_59/stack_1?
pnn/strided_slice_59/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_59/stack_2?
pnn/strided_slice_59StridedSlicepnn/ReadVariableOp_59:value:0#pnn/strided_slice_59/stack:output:0%pnn/strided_slice_59/stack_1:output:0%pnn/strided_slice_59/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_59p
pnn/ExpandDims_11/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_11/dim?
pnn/ExpandDims_11
ExpandDimspnn/strided_slice_59:output:0pnn/ExpandDims_11/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_11?

pnn/Mul_60Mulpnn/MatMul:output:0pnn/ExpandDims_11:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_60?
pnn/Sum_60/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_60/reduction_indices?

pnn/Sum_60Sumpnn/Mul_60:z:0%pnn/Sum_60/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_60}
pnn/Reshape_60/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_60/shape?
pnn/Reshape_60Reshapepnn/Sum_60:output:0pnn/Reshape_60/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_60?
pnn/ReadVariableOp_60ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_60?
pnn/strided_slice_60/stackConst*
_output_shapes
:*
dtype0*
valueB:
2
pnn/strided_slice_60/stack?
pnn/strided_slice_60/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_60/stack_1?
pnn/strided_slice_60/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_60/stack_2?
pnn/strided_slice_60StridedSlicepnn/ReadVariableOp_60:value:0#pnn/strided_slice_60/stack:output:0%pnn/strided_slice_60/stack_1:output:0%pnn/strided_slice_60/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_60p
pnn/ExpandDims_12/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_12/dim?
pnn/ExpandDims_12
ExpandDimspnn/strided_slice_60:output:0pnn/ExpandDims_12/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_12?

pnn/Mul_61Mulpnn/MatMul:output:0pnn/ExpandDims_12:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_61?
pnn/Sum_61/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_61/reduction_indices?

pnn/Sum_61Sumpnn/Mul_61:z:0%pnn/Sum_61/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_61}
pnn/Reshape_61/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_61/shape?
pnn/Reshape_61Reshapepnn/Sum_61:output:0pnn/Reshape_61/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_61?
pnn/ReadVariableOp_61ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_61?
pnn/strided_slice_61/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_61/stack?
pnn/strided_slice_61/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_61/stack_1?
pnn/strided_slice_61/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_61/stack_2?
pnn/strided_slice_61StridedSlicepnn/ReadVariableOp_61:value:0#pnn/strided_slice_61/stack:output:0%pnn/strided_slice_61/stack_1:output:0%pnn/strided_slice_61/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_61p
pnn/ExpandDims_13/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_13/dim?
pnn/ExpandDims_13
ExpandDimspnn/strided_slice_61:output:0pnn/ExpandDims_13/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_13?

pnn/Mul_62Mulpnn/MatMul:output:0pnn/ExpandDims_13:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_62?
pnn/Sum_62/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_62/reduction_indices?

pnn/Sum_62Sumpnn/Mul_62:z:0%pnn/Sum_62/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_62}
pnn/Reshape_62/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_62/shape?
pnn/Reshape_62Reshapepnn/Sum_62:output:0pnn/Reshape_62/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_62?
pnn/ReadVariableOp_62ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_62?
pnn/strided_slice_62/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_62/stack?
pnn/strided_slice_62/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_62/stack_1?
pnn/strided_slice_62/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_62/stack_2?
pnn/strided_slice_62StridedSlicepnn/ReadVariableOp_62:value:0#pnn/strided_slice_62/stack:output:0%pnn/strided_slice_62/stack_1:output:0%pnn/strided_slice_62/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_62p
pnn/ExpandDims_14/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_14/dim?
pnn/ExpandDims_14
ExpandDimspnn/strided_slice_62:output:0pnn/ExpandDims_14/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_14?

pnn/Mul_63Mulpnn/MatMul:output:0pnn/ExpandDims_14:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_63?
pnn/Sum_63/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_63/reduction_indices?

pnn/Sum_63Sumpnn/Mul_63:z:0%pnn/Sum_63/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_63}
pnn/Reshape_63/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_63/shape?
pnn/Reshape_63Reshapepnn/Sum_63:output:0pnn/Reshape_63/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_63?
pnn/ReadVariableOp_63ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_63?
pnn/strided_slice_63/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_63/stack?
pnn/strided_slice_63/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_63/stack_1?
pnn/strided_slice_63/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_63/stack_2?
pnn/strided_slice_63StridedSlicepnn/ReadVariableOp_63:value:0#pnn/strided_slice_63/stack:output:0%pnn/strided_slice_63/stack_1:output:0%pnn/strided_slice_63/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_63p
pnn/ExpandDims_15/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_15/dim?
pnn/ExpandDims_15
ExpandDimspnn/strided_slice_63:output:0pnn/ExpandDims_15/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_15?

pnn/Mul_64Mulpnn/MatMul:output:0pnn/ExpandDims_15:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_64?
pnn/Sum_64/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_64/reduction_indices?

pnn/Sum_64Sumpnn/Mul_64:z:0%pnn/Sum_64/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_64}
pnn/Reshape_64/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_64/shape?
pnn/Reshape_64Reshapepnn/Sum_64:output:0pnn/Reshape_64/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_64?
pnn/ReadVariableOp_64ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_64?
pnn/strided_slice_64/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_64/stack?
pnn/strided_slice_64/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_64/stack_1?
pnn/strided_slice_64/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_64/stack_2?
pnn/strided_slice_64StridedSlicepnn/ReadVariableOp_64:value:0#pnn/strided_slice_64/stack:output:0%pnn/strided_slice_64/stack_1:output:0%pnn/strided_slice_64/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_64p
pnn/ExpandDims_16/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_16/dim?
pnn/ExpandDims_16
ExpandDimspnn/strided_slice_64:output:0pnn/ExpandDims_16/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_16?

pnn/Mul_65Mulpnn/MatMul:output:0pnn/ExpandDims_16:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_65?
pnn/Sum_65/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_65/reduction_indices?

pnn/Sum_65Sumpnn/Mul_65:z:0%pnn/Sum_65/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_65}
pnn/Reshape_65/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_65/shape?
pnn/Reshape_65Reshapepnn/Sum_65:output:0pnn/Reshape_65/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_65?
pnn/ReadVariableOp_65ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_65?
pnn/strided_slice_65/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_65/stack?
pnn/strided_slice_65/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_65/stack_1?
pnn/strided_slice_65/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_65/stack_2?
pnn/strided_slice_65StridedSlicepnn/ReadVariableOp_65:value:0#pnn/strided_slice_65/stack:output:0%pnn/strided_slice_65/stack_1:output:0%pnn/strided_slice_65/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_65p
pnn/ExpandDims_17/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_17/dim?
pnn/ExpandDims_17
ExpandDimspnn/strided_slice_65:output:0pnn/ExpandDims_17/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_17?

pnn/Mul_66Mulpnn/MatMul:output:0pnn/ExpandDims_17:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_66?
pnn/Sum_66/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_66/reduction_indices?

pnn/Sum_66Sumpnn/Mul_66:z:0%pnn/Sum_66/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_66}
pnn/Reshape_66/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_66/shape?
pnn/Reshape_66Reshapepnn/Sum_66:output:0pnn/Reshape_66/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_66?
pnn/ReadVariableOp_66ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_66?
pnn/strided_slice_66/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_66/stack?
pnn/strided_slice_66/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_66/stack_1?
pnn/strided_slice_66/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_66/stack_2?
pnn/strided_slice_66StridedSlicepnn/ReadVariableOp_66:value:0#pnn/strided_slice_66/stack:output:0%pnn/strided_slice_66/stack_1:output:0%pnn/strided_slice_66/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_66p
pnn/ExpandDims_18/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_18/dim?
pnn/ExpandDims_18
ExpandDimspnn/strided_slice_66:output:0pnn/ExpandDims_18/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_18?

pnn/Mul_67Mulpnn/MatMul:output:0pnn/ExpandDims_18:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_67?
pnn/Sum_67/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_67/reduction_indices?

pnn/Sum_67Sumpnn/Mul_67:z:0%pnn/Sum_67/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_67}
pnn/Reshape_67/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_67/shape?
pnn/Reshape_67Reshapepnn/Sum_67:output:0pnn/Reshape_67/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_67?
pnn/ReadVariableOp_67ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_67?
pnn/strided_slice_67/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_67/stack?
pnn/strided_slice_67/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_67/stack_1?
pnn/strided_slice_67/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_67/stack_2?
pnn/strided_slice_67StridedSlicepnn/ReadVariableOp_67:value:0#pnn/strided_slice_67/stack:output:0%pnn/strided_slice_67/stack_1:output:0%pnn/strided_slice_67/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_67p
pnn/ExpandDims_19/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_19/dim?
pnn/ExpandDims_19
ExpandDimspnn/strided_slice_67:output:0pnn/ExpandDims_19/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_19?

pnn/Mul_68Mulpnn/MatMul:output:0pnn/ExpandDims_19:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_68?
pnn/Sum_68/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_68/reduction_indices?

pnn/Sum_68Sumpnn/Mul_68:z:0%pnn/Sum_68/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_68}
pnn/Reshape_68/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_68/shape?
pnn/Reshape_68Reshapepnn/Sum_68:output:0pnn/Reshape_68/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_68?
pnn/ReadVariableOp_68ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_68?
pnn/strided_slice_68/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_68/stack?
pnn/strided_slice_68/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_68/stack_1?
pnn/strided_slice_68/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_68/stack_2?
pnn/strided_slice_68StridedSlicepnn/ReadVariableOp_68:value:0#pnn/strided_slice_68/stack:output:0%pnn/strided_slice_68/stack_1:output:0%pnn/strided_slice_68/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_68p
pnn/ExpandDims_20/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_20/dim?
pnn/ExpandDims_20
ExpandDimspnn/strided_slice_68:output:0pnn/ExpandDims_20/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_20?

pnn/Mul_69Mulpnn/MatMul:output:0pnn/ExpandDims_20:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_69?
pnn/Sum_69/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_69/reduction_indices?

pnn/Sum_69Sumpnn/Mul_69:z:0%pnn/Sum_69/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_69}
pnn/Reshape_69/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_69/shape?
pnn/Reshape_69Reshapepnn/Sum_69:output:0pnn/Reshape_69/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_69?
pnn/ReadVariableOp_69ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_69?
pnn/strided_slice_69/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_69/stack?
pnn/strided_slice_69/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_69/stack_1?
pnn/strided_slice_69/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_69/stack_2?
pnn/strided_slice_69StridedSlicepnn/ReadVariableOp_69:value:0#pnn/strided_slice_69/stack:output:0%pnn/strided_slice_69/stack_1:output:0%pnn/strided_slice_69/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_69p
pnn/ExpandDims_21/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_21/dim?
pnn/ExpandDims_21
ExpandDimspnn/strided_slice_69:output:0pnn/ExpandDims_21/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_21?

pnn/Mul_70Mulpnn/MatMul:output:0pnn/ExpandDims_21:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_70?
pnn/Sum_70/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_70/reduction_indices?

pnn/Sum_70Sumpnn/Mul_70:z:0%pnn/Sum_70/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_70}
pnn/Reshape_70/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_70/shape?
pnn/Reshape_70Reshapepnn/Sum_70:output:0pnn/Reshape_70/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_70?
pnn/ReadVariableOp_70ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_70?
pnn/strided_slice_70/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_70/stack?
pnn/strided_slice_70/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_70/stack_1?
pnn/strided_slice_70/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_70/stack_2?
pnn/strided_slice_70StridedSlicepnn/ReadVariableOp_70:value:0#pnn/strided_slice_70/stack:output:0%pnn/strided_slice_70/stack_1:output:0%pnn/strided_slice_70/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_70p
pnn/ExpandDims_22/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_22/dim?
pnn/ExpandDims_22
ExpandDimspnn/strided_slice_70:output:0pnn/ExpandDims_22/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_22?

pnn/Mul_71Mulpnn/MatMul:output:0pnn/ExpandDims_22:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_71?
pnn/Sum_71/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_71/reduction_indices?

pnn/Sum_71Sumpnn/Mul_71:z:0%pnn/Sum_71/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_71}
pnn/Reshape_71/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_71/shape?
pnn/Reshape_71Reshapepnn/Sum_71:output:0pnn/Reshape_71/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_71?
pnn/ReadVariableOp_71ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_71?
pnn/strided_slice_71/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_71/stack?
pnn/strided_slice_71/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_71/stack_1?
pnn/strided_slice_71/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_71/stack_2?
pnn/strided_slice_71StridedSlicepnn/ReadVariableOp_71:value:0#pnn/strided_slice_71/stack:output:0%pnn/strided_slice_71/stack_1:output:0%pnn/strided_slice_71/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_71p
pnn/ExpandDims_23/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_23/dim?
pnn/ExpandDims_23
ExpandDimspnn/strided_slice_71:output:0pnn/ExpandDims_23/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_23?

pnn/Mul_72Mulpnn/MatMul:output:0pnn/ExpandDims_23:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_72?
pnn/Sum_72/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_72/reduction_indices?

pnn/Sum_72Sumpnn/Mul_72:z:0%pnn/Sum_72/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_72}
pnn/Reshape_72/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_72/shape?
pnn/Reshape_72Reshapepnn/Sum_72:output:0pnn/Reshape_72/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_72?
pnn/ReadVariableOp_72ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_72?
pnn/strided_slice_72/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_72/stack?
pnn/strided_slice_72/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_72/stack_1?
pnn/strided_slice_72/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_72/stack_2?
pnn/strided_slice_72StridedSlicepnn/ReadVariableOp_72:value:0#pnn/strided_slice_72/stack:output:0%pnn/strided_slice_72/stack_1:output:0%pnn/strided_slice_72/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_72p
pnn/ExpandDims_24/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_24/dim?
pnn/ExpandDims_24
ExpandDimspnn/strided_slice_72:output:0pnn/ExpandDims_24/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_24?

pnn/Mul_73Mulpnn/MatMul:output:0pnn/ExpandDims_24:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_73?
pnn/Sum_73/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_73/reduction_indices?

pnn/Sum_73Sumpnn/Mul_73:z:0%pnn/Sum_73/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_73}
pnn/Reshape_73/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_73/shape?
pnn/Reshape_73Reshapepnn/Sum_73:output:0pnn/Reshape_73/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_73?
pnn/ReadVariableOp_73ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_73?
pnn/strided_slice_73/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_73/stack?
pnn/strided_slice_73/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_73/stack_1?
pnn/strided_slice_73/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_73/stack_2?
pnn/strided_slice_73StridedSlicepnn/ReadVariableOp_73:value:0#pnn/strided_slice_73/stack:output:0%pnn/strided_slice_73/stack_1:output:0%pnn/strided_slice_73/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_73p
pnn/ExpandDims_25/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_25/dim?
pnn/ExpandDims_25
ExpandDimspnn/strided_slice_73:output:0pnn/ExpandDims_25/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_25?

pnn/Mul_74Mulpnn/MatMul:output:0pnn/ExpandDims_25:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_74?
pnn/Sum_74/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_74/reduction_indices?

pnn/Sum_74Sumpnn/Mul_74:z:0%pnn/Sum_74/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_74}
pnn/Reshape_74/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_74/shape?
pnn/Reshape_74Reshapepnn/Sum_74:output:0pnn/Reshape_74/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_74?
pnn/ReadVariableOp_74ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_74?
pnn/strided_slice_74/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_74/stack?
pnn/strided_slice_74/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_74/stack_1?
pnn/strided_slice_74/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_74/stack_2?
pnn/strided_slice_74StridedSlicepnn/ReadVariableOp_74:value:0#pnn/strided_slice_74/stack:output:0%pnn/strided_slice_74/stack_1:output:0%pnn/strided_slice_74/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_74p
pnn/ExpandDims_26/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_26/dim?
pnn/ExpandDims_26
ExpandDimspnn/strided_slice_74:output:0pnn/ExpandDims_26/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_26?

pnn/Mul_75Mulpnn/MatMul:output:0pnn/ExpandDims_26:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_75?
pnn/Sum_75/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_75/reduction_indices?

pnn/Sum_75Sumpnn/Mul_75:z:0%pnn/Sum_75/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_75}
pnn/Reshape_75/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_75/shape?
pnn/Reshape_75Reshapepnn/Sum_75:output:0pnn/Reshape_75/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_75?
pnn/ReadVariableOp_75ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_75?
pnn/strided_slice_75/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_75/stack?
pnn/strided_slice_75/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_75/stack_1?
pnn/strided_slice_75/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_75/stack_2?
pnn/strided_slice_75StridedSlicepnn/ReadVariableOp_75:value:0#pnn/strided_slice_75/stack:output:0%pnn/strided_slice_75/stack_1:output:0%pnn/strided_slice_75/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_75p
pnn/ExpandDims_27/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_27/dim?
pnn/ExpandDims_27
ExpandDimspnn/strided_slice_75:output:0pnn/ExpandDims_27/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_27?

pnn/Mul_76Mulpnn/MatMul:output:0pnn/ExpandDims_27:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_76?
pnn/Sum_76/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_76/reduction_indices?

pnn/Sum_76Sumpnn/Mul_76:z:0%pnn/Sum_76/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_76}
pnn/Reshape_76/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_76/shape?
pnn/Reshape_76Reshapepnn/Sum_76:output:0pnn/Reshape_76/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_76?
pnn/ReadVariableOp_76ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_76?
pnn/strided_slice_76/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_76/stack?
pnn/strided_slice_76/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_76/stack_1?
pnn/strided_slice_76/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_76/stack_2?
pnn/strided_slice_76StridedSlicepnn/ReadVariableOp_76:value:0#pnn/strided_slice_76/stack:output:0%pnn/strided_slice_76/stack_1:output:0%pnn/strided_slice_76/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_76p
pnn/ExpandDims_28/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_28/dim?
pnn/ExpandDims_28
ExpandDimspnn/strided_slice_76:output:0pnn/ExpandDims_28/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_28?

pnn/Mul_77Mulpnn/MatMul:output:0pnn/ExpandDims_28:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_77?
pnn/Sum_77/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_77/reduction_indices?

pnn/Sum_77Sumpnn/Mul_77:z:0%pnn/Sum_77/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_77}
pnn/Reshape_77/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_77/shape?
pnn/Reshape_77Reshapepnn/Sum_77:output:0pnn/Reshape_77/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_77?
pnn/ReadVariableOp_77ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_77?
pnn/strided_slice_77/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_77/stack?
pnn/strided_slice_77/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_77/stack_1?
pnn/strided_slice_77/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_77/stack_2?
pnn/strided_slice_77StridedSlicepnn/ReadVariableOp_77:value:0#pnn/strided_slice_77/stack:output:0%pnn/strided_slice_77/stack_1:output:0%pnn/strided_slice_77/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_77p
pnn/ExpandDims_29/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_29/dim?
pnn/ExpandDims_29
ExpandDimspnn/strided_slice_77:output:0pnn/ExpandDims_29/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_29?

pnn/Mul_78Mulpnn/MatMul:output:0pnn/ExpandDims_29:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_78?
pnn/Sum_78/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_78/reduction_indices?

pnn/Sum_78Sumpnn/Mul_78:z:0%pnn/Sum_78/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_78}
pnn/Reshape_78/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_78/shape?
pnn/Reshape_78Reshapepnn/Sum_78:output:0pnn/Reshape_78/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_78?
pnn/ReadVariableOp_78ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_78?
pnn/strided_slice_78/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_78/stack?
pnn/strided_slice_78/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_78/stack_1?
pnn/strided_slice_78/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_78/stack_2?
pnn/strided_slice_78StridedSlicepnn/ReadVariableOp_78:value:0#pnn/strided_slice_78/stack:output:0%pnn/strided_slice_78/stack_1:output:0%pnn/strided_slice_78/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_78p
pnn/ExpandDims_30/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_30/dim?
pnn/ExpandDims_30
ExpandDimspnn/strided_slice_78:output:0pnn/ExpandDims_30/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_30?

pnn/Mul_79Mulpnn/MatMul:output:0pnn/ExpandDims_30:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_79?
pnn/Sum_79/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_79/reduction_indices?

pnn/Sum_79Sumpnn/Mul_79:z:0%pnn/Sum_79/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_79}
pnn/Reshape_79/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_79/shape?
pnn/Reshape_79Reshapepnn/Sum_79:output:0pnn/Reshape_79/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_79?
pnn/ReadVariableOp_79ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_79?
pnn/strided_slice_79/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_79/stack?
pnn/strided_slice_79/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_79/stack_1?
pnn/strided_slice_79/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_79/stack_2?
pnn/strided_slice_79StridedSlicepnn/ReadVariableOp_79:value:0#pnn/strided_slice_79/stack:output:0%pnn/strided_slice_79/stack_1:output:0%pnn/strided_slice_79/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_79p
pnn/ExpandDims_31/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_31/dim?
pnn/ExpandDims_31
ExpandDimspnn/strided_slice_79:output:0pnn/ExpandDims_31/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_31?

pnn/Mul_80Mulpnn/MatMul:output:0pnn/ExpandDims_31:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_80?
pnn/Sum_80/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_80/reduction_indices?

pnn/Sum_80Sumpnn/Mul_80:z:0%pnn/Sum_80/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_80}
pnn/Reshape_80/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_80/shape?
pnn/Reshape_80Reshapepnn/Sum_80:output:0pnn/Reshape_80/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_80?
pnn/ReadVariableOp_80ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_80?
pnn/strided_slice_80/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_80/stack?
pnn/strided_slice_80/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_80/stack_1?
pnn/strided_slice_80/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_80/stack_2?
pnn/strided_slice_80StridedSlicepnn/ReadVariableOp_80:value:0#pnn/strided_slice_80/stack:output:0%pnn/strided_slice_80/stack_1:output:0%pnn/strided_slice_80/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_80p
pnn/ExpandDims_32/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_32/dim?
pnn/ExpandDims_32
ExpandDimspnn/strided_slice_80:output:0pnn/ExpandDims_32/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_32?

pnn/Mul_81Mulpnn/MatMul:output:0pnn/ExpandDims_32:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_81?
pnn/Sum_81/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_81/reduction_indices?

pnn/Sum_81Sumpnn/Mul_81:z:0%pnn/Sum_81/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_81}
pnn/Reshape_81/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_81/shape?
pnn/Reshape_81Reshapepnn/Sum_81:output:0pnn/Reshape_81/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_81?
pnn/ReadVariableOp_81ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_81?
pnn/strided_slice_81/stackConst*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_81/stack?
pnn/strided_slice_81/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
pnn/strided_slice_81/stack_1?
pnn/strided_slice_81/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_81/stack_2?
pnn/strided_slice_81StridedSlicepnn/ReadVariableOp_81:value:0#pnn/strided_slice_81/stack:output:0%pnn/strided_slice_81/stack_1:output:0%pnn/strided_slice_81/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_81p
pnn/ExpandDims_33/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_33/dim?
pnn/ExpandDims_33
ExpandDimspnn/strided_slice_81:output:0pnn/ExpandDims_33/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_33?

pnn/Mul_82Mulpnn/MatMul:output:0pnn/ExpandDims_33:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_82?
pnn/Sum_82/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_82/reduction_indices?

pnn/Sum_82Sumpnn/Mul_82:z:0%pnn/Sum_82/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_82}
pnn/Reshape_82/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_82/shape?
pnn/Reshape_82Reshapepnn/Sum_82:output:0pnn/Reshape_82/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_82?
pnn/ReadVariableOp_82ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_82?
pnn/strided_slice_82/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
pnn/strided_slice_82/stack?
pnn/strided_slice_82/stack_1Const*
_output_shapes
:*
dtype0*
valueB:!2
pnn/strided_slice_82/stack_1?
pnn/strided_slice_82/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_82/stack_2?
pnn/strided_slice_82StridedSlicepnn/ReadVariableOp_82:value:0#pnn/strided_slice_82/stack:output:0%pnn/strided_slice_82/stack_1:output:0%pnn/strided_slice_82/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_82p
pnn/ExpandDims_34/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_34/dim?
pnn/ExpandDims_34
ExpandDimspnn/strided_slice_82:output:0pnn/ExpandDims_34/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_34?

pnn/Mul_83Mulpnn/MatMul:output:0pnn/ExpandDims_34:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_83?
pnn/Sum_83/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_83/reduction_indices?

pnn/Sum_83Sumpnn/Mul_83:z:0%pnn/Sum_83/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_83}
pnn/Reshape_83/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_83/shape?
pnn/Reshape_83Reshapepnn/Sum_83:output:0pnn/Reshape_83/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_83?
pnn/ReadVariableOp_83ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_83?
pnn/strided_slice_83/stackConst*
_output_shapes
:*
dtype0*
valueB:!2
pnn/strided_slice_83/stack?
pnn/strided_slice_83/stack_1Const*
_output_shapes
:*
dtype0*
valueB:"2
pnn/strided_slice_83/stack_1?
pnn/strided_slice_83/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_83/stack_2?
pnn/strided_slice_83StridedSlicepnn/ReadVariableOp_83:value:0#pnn/strided_slice_83/stack:output:0%pnn/strided_slice_83/stack_1:output:0%pnn/strided_slice_83/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_83p
pnn/ExpandDims_35/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_35/dim?
pnn/ExpandDims_35
ExpandDimspnn/strided_slice_83:output:0pnn/ExpandDims_35/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_35?

pnn/Mul_84Mulpnn/MatMul:output:0pnn/ExpandDims_35:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_84?
pnn/Sum_84/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_84/reduction_indices?

pnn/Sum_84Sumpnn/Mul_84:z:0%pnn/Sum_84/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_84}
pnn/Reshape_84/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_84/shape?
pnn/Reshape_84Reshapepnn/Sum_84:output:0pnn/Reshape_84/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_84?
pnn/ReadVariableOp_84ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_84?
pnn/strided_slice_84/stackConst*
_output_shapes
:*
dtype0*
valueB:"2
pnn/strided_slice_84/stack?
pnn/strided_slice_84/stack_1Const*
_output_shapes
:*
dtype0*
valueB:#2
pnn/strided_slice_84/stack_1?
pnn/strided_slice_84/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_84/stack_2?
pnn/strided_slice_84StridedSlicepnn/ReadVariableOp_84:value:0#pnn/strided_slice_84/stack:output:0%pnn/strided_slice_84/stack_1:output:0%pnn/strided_slice_84/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_84p
pnn/ExpandDims_36/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_36/dim?
pnn/ExpandDims_36
ExpandDimspnn/strided_slice_84:output:0pnn/ExpandDims_36/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_36?

pnn/Mul_85Mulpnn/MatMul:output:0pnn/ExpandDims_36:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_85?
pnn/Sum_85/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_85/reduction_indices?

pnn/Sum_85Sumpnn/Mul_85:z:0%pnn/Sum_85/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_85}
pnn/Reshape_85/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_85/shape?
pnn/Reshape_85Reshapepnn/Sum_85:output:0pnn/Reshape_85/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_85?
pnn/ReadVariableOp_85ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_85?
pnn/strided_slice_85/stackConst*
_output_shapes
:*
dtype0*
valueB:#2
pnn/strided_slice_85/stack?
pnn/strided_slice_85/stack_1Const*
_output_shapes
:*
dtype0*
valueB:$2
pnn/strided_slice_85/stack_1?
pnn/strided_slice_85/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_85/stack_2?
pnn/strided_slice_85StridedSlicepnn/ReadVariableOp_85:value:0#pnn/strided_slice_85/stack:output:0%pnn/strided_slice_85/stack_1:output:0%pnn/strided_slice_85/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_85p
pnn/ExpandDims_37/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_37/dim?
pnn/ExpandDims_37
ExpandDimspnn/strided_slice_85:output:0pnn/ExpandDims_37/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_37?

pnn/Mul_86Mulpnn/MatMul:output:0pnn/ExpandDims_37:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_86?
pnn/Sum_86/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_86/reduction_indices?

pnn/Sum_86Sumpnn/Mul_86:z:0%pnn/Sum_86/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_86}
pnn/Reshape_86/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_86/shape?
pnn/Reshape_86Reshapepnn/Sum_86:output:0pnn/Reshape_86/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_86?
pnn/ReadVariableOp_86ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_86?
pnn/strided_slice_86/stackConst*
_output_shapes
:*
dtype0*
valueB:$2
pnn/strided_slice_86/stack?
pnn/strided_slice_86/stack_1Const*
_output_shapes
:*
dtype0*
valueB:%2
pnn/strided_slice_86/stack_1?
pnn/strided_slice_86/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_86/stack_2?
pnn/strided_slice_86StridedSlicepnn/ReadVariableOp_86:value:0#pnn/strided_slice_86/stack:output:0%pnn/strided_slice_86/stack_1:output:0%pnn/strided_slice_86/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_86p
pnn/ExpandDims_38/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_38/dim?
pnn/ExpandDims_38
ExpandDimspnn/strided_slice_86:output:0pnn/ExpandDims_38/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_38?

pnn/Mul_87Mulpnn/MatMul:output:0pnn/ExpandDims_38:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_87?
pnn/Sum_87/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_87/reduction_indices?

pnn/Sum_87Sumpnn/Mul_87:z:0%pnn/Sum_87/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_87}
pnn/Reshape_87/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_87/shape?
pnn/Reshape_87Reshapepnn/Sum_87:output:0pnn/Reshape_87/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_87?
pnn/ReadVariableOp_87ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_87?
pnn/strided_slice_87/stackConst*
_output_shapes
:*
dtype0*
valueB:%2
pnn/strided_slice_87/stack?
pnn/strided_slice_87/stack_1Const*
_output_shapes
:*
dtype0*
valueB:&2
pnn/strided_slice_87/stack_1?
pnn/strided_slice_87/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_87/stack_2?
pnn/strided_slice_87StridedSlicepnn/ReadVariableOp_87:value:0#pnn/strided_slice_87/stack:output:0%pnn/strided_slice_87/stack_1:output:0%pnn/strided_slice_87/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_87p
pnn/ExpandDims_39/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_39/dim?
pnn/ExpandDims_39
ExpandDimspnn/strided_slice_87:output:0pnn/ExpandDims_39/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_39?

pnn/Mul_88Mulpnn/MatMul:output:0pnn/ExpandDims_39:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_88?
pnn/Sum_88/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_88/reduction_indices?

pnn/Sum_88Sumpnn/Mul_88:z:0%pnn/Sum_88/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_88}
pnn/Reshape_88/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_88/shape?
pnn/Reshape_88Reshapepnn/Sum_88:output:0pnn/Reshape_88/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_88?
pnn/ReadVariableOp_88ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_88?
pnn/strided_slice_88/stackConst*
_output_shapes
:*
dtype0*
valueB:&2
pnn/strided_slice_88/stack?
pnn/strided_slice_88/stack_1Const*
_output_shapes
:*
dtype0*
valueB:'2
pnn/strided_slice_88/stack_1?
pnn/strided_slice_88/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_88/stack_2?
pnn/strided_slice_88StridedSlicepnn/ReadVariableOp_88:value:0#pnn/strided_slice_88/stack:output:0%pnn/strided_slice_88/stack_1:output:0%pnn/strided_slice_88/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_88p
pnn/ExpandDims_40/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_40/dim?
pnn/ExpandDims_40
ExpandDimspnn/strided_slice_88:output:0pnn/ExpandDims_40/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_40?

pnn/Mul_89Mulpnn/MatMul:output:0pnn/ExpandDims_40:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_89?
pnn/Sum_89/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_89/reduction_indices?

pnn/Sum_89Sumpnn/Mul_89:z:0%pnn/Sum_89/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_89}
pnn/Reshape_89/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_89/shape?
pnn/Reshape_89Reshapepnn/Sum_89:output:0pnn/Reshape_89/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_89?
pnn/ReadVariableOp_89ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_89?
pnn/strided_slice_89/stackConst*
_output_shapes
:*
dtype0*
valueB:'2
pnn/strided_slice_89/stack?
pnn/strided_slice_89/stack_1Const*
_output_shapes
:*
dtype0*
valueB:(2
pnn/strided_slice_89/stack_1?
pnn/strided_slice_89/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_89/stack_2?
pnn/strided_slice_89StridedSlicepnn/ReadVariableOp_89:value:0#pnn/strided_slice_89/stack:output:0%pnn/strided_slice_89/stack_1:output:0%pnn/strided_slice_89/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_89p
pnn/ExpandDims_41/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_41/dim?
pnn/ExpandDims_41
ExpandDimspnn/strided_slice_89:output:0pnn/ExpandDims_41/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_41?

pnn/Mul_90Mulpnn/MatMul:output:0pnn/ExpandDims_41:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_90?
pnn/Sum_90/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_90/reduction_indices?

pnn/Sum_90Sumpnn/Mul_90:z:0%pnn/Sum_90/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_90}
pnn/Reshape_90/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_90/shape?
pnn/Reshape_90Reshapepnn/Sum_90:output:0pnn/Reshape_90/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_90?
pnn/ReadVariableOp_90ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_90?
pnn/strided_slice_90/stackConst*
_output_shapes
:*
dtype0*
valueB:(2
pnn/strided_slice_90/stack?
pnn/strided_slice_90/stack_1Const*
_output_shapes
:*
dtype0*
valueB:)2
pnn/strided_slice_90/stack_1?
pnn/strided_slice_90/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_90/stack_2?
pnn/strided_slice_90StridedSlicepnn/ReadVariableOp_90:value:0#pnn/strided_slice_90/stack:output:0%pnn/strided_slice_90/stack_1:output:0%pnn/strided_slice_90/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_90p
pnn/ExpandDims_42/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_42/dim?
pnn/ExpandDims_42
ExpandDimspnn/strided_slice_90:output:0pnn/ExpandDims_42/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_42?

pnn/Mul_91Mulpnn/MatMul:output:0pnn/ExpandDims_42:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_91?
pnn/Sum_91/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_91/reduction_indices?

pnn/Sum_91Sumpnn/Mul_91:z:0%pnn/Sum_91/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_91}
pnn/Reshape_91/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_91/shape?
pnn/Reshape_91Reshapepnn/Sum_91:output:0pnn/Reshape_91/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_91?
pnn/ReadVariableOp_91ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_91?
pnn/strided_slice_91/stackConst*
_output_shapes
:*
dtype0*
valueB:)2
pnn/strided_slice_91/stack?
pnn/strided_slice_91/stack_1Const*
_output_shapes
:*
dtype0*
valueB:*2
pnn/strided_slice_91/stack_1?
pnn/strided_slice_91/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_91/stack_2?
pnn/strided_slice_91StridedSlicepnn/ReadVariableOp_91:value:0#pnn/strided_slice_91/stack:output:0%pnn/strided_slice_91/stack_1:output:0%pnn/strided_slice_91/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_91p
pnn/ExpandDims_43/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_43/dim?
pnn/ExpandDims_43
ExpandDimspnn/strided_slice_91:output:0pnn/ExpandDims_43/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_43?

pnn/Mul_92Mulpnn/MatMul:output:0pnn/ExpandDims_43:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_92?
pnn/Sum_92/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_92/reduction_indices?

pnn/Sum_92Sumpnn/Mul_92:z:0%pnn/Sum_92/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_92}
pnn/Reshape_92/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_92/shape?
pnn/Reshape_92Reshapepnn/Sum_92:output:0pnn/Reshape_92/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_92?
pnn/ReadVariableOp_92ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_92?
pnn/strided_slice_92/stackConst*
_output_shapes
:*
dtype0*
valueB:*2
pnn/strided_slice_92/stack?
pnn/strided_slice_92/stack_1Const*
_output_shapes
:*
dtype0*
valueB:+2
pnn/strided_slice_92/stack_1?
pnn/strided_slice_92/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_92/stack_2?
pnn/strided_slice_92StridedSlicepnn/ReadVariableOp_92:value:0#pnn/strided_slice_92/stack:output:0%pnn/strided_slice_92/stack_1:output:0%pnn/strided_slice_92/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_92p
pnn/ExpandDims_44/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_44/dim?
pnn/ExpandDims_44
ExpandDimspnn/strided_slice_92:output:0pnn/ExpandDims_44/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_44?

pnn/Mul_93Mulpnn/MatMul:output:0pnn/ExpandDims_44:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_93?
pnn/Sum_93/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_93/reduction_indices?

pnn/Sum_93Sumpnn/Mul_93:z:0%pnn/Sum_93/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_93}
pnn/Reshape_93/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_93/shape?
pnn/Reshape_93Reshapepnn/Sum_93:output:0pnn/Reshape_93/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_93?
pnn/ReadVariableOp_93ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_93?
pnn/strided_slice_93/stackConst*
_output_shapes
:*
dtype0*
valueB:+2
pnn/strided_slice_93/stack?
pnn/strided_slice_93/stack_1Const*
_output_shapes
:*
dtype0*
valueB:,2
pnn/strided_slice_93/stack_1?
pnn/strided_slice_93/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_93/stack_2?
pnn/strided_slice_93StridedSlicepnn/ReadVariableOp_93:value:0#pnn/strided_slice_93/stack:output:0%pnn/strided_slice_93/stack_1:output:0%pnn/strided_slice_93/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_93p
pnn/ExpandDims_45/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_45/dim?
pnn/ExpandDims_45
ExpandDimspnn/strided_slice_93:output:0pnn/ExpandDims_45/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_45?

pnn/Mul_94Mulpnn/MatMul:output:0pnn/ExpandDims_45:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_94?
pnn/Sum_94/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_94/reduction_indices?

pnn/Sum_94Sumpnn/Mul_94:z:0%pnn/Sum_94/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_94}
pnn/Reshape_94/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_94/shape?
pnn/Reshape_94Reshapepnn/Sum_94:output:0pnn/Reshape_94/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_94?
pnn/ReadVariableOp_94ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_94?
pnn/strided_slice_94/stackConst*
_output_shapes
:*
dtype0*
valueB:,2
pnn/strided_slice_94/stack?
pnn/strided_slice_94/stack_1Const*
_output_shapes
:*
dtype0*
valueB:-2
pnn/strided_slice_94/stack_1?
pnn/strided_slice_94/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_94/stack_2?
pnn/strided_slice_94StridedSlicepnn/ReadVariableOp_94:value:0#pnn/strided_slice_94/stack:output:0%pnn/strided_slice_94/stack_1:output:0%pnn/strided_slice_94/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_94p
pnn/ExpandDims_46/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_46/dim?
pnn/ExpandDims_46
ExpandDimspnn/strided_slice_94:output:0pnn/ExpandDims_46/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_46?

pnn/Mul_95Mulpnn/MatMul:output:0pnn/ExpandDims_46:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_95?
pnn/Sum_95/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_95/reduction_indices?

pnn/Sum_95Sumpnn/Mul_95:z:0%pnn/Sum_95/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_95}
pnn/Reshape_95/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_95/shape?
pnn/Reshape_95Reshapepnn/Sum_95:output:0pnn/Reshape_95/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_95?
pnn/ReadVariableOp_95ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_95?
pnn/strided_slice_95/stackConst*
_output_shapes
:*
dtype0*
valueB:-2
pnn/strided_slice_95/stack?
pnn/strided_slice_95/stack_1Const*
_output_shapes
:*
dtype0*
valueB:.2
pnn/strided_slice_95/stack_1?
pnn/strided_slice_95/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_95/stack_2?
pnn/strided_slice_95StridedSlicepnn/ReadVariableOp_95:value:0#pnn/strided_slice_95/stack:output:0%pnn/strided_slice_95/stack_1:output:0%pnn/strided_slice_95/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_95p
pnn/ExpandDims_47/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_47/dim?
pnn/ExpandDims_47
ExpandDimspnn/strided_slice_95:output:0pnn/ExpandDims_47/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_47?

pnn/Mul_96Mulpnn/MatMul:output:0pnn/ExpandDims_47:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_96?
pnn/Sum_96/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_96/reduction_indices?

pnn/Sum_96Sumpnn/Mul_96:z:0%pnn/Sum_96/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_96}
pnn/Reshape_96/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_96/shape?
pnn/Reshape_96Reshapepnn/Sum_96:output:0pnn/Reshape_96/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_96?
pnn/ReadVariableOp_96ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_96?
pnn/strided_slice_96/stackConst*
_output_shapes
:*
dtype0*
valueB:.2
pnn/strided_slice_96/stack?
pnn/strided_slice_96/stack_1Const*
_output_shapes
:*
dtype0*
valueB:/2
pnn/strided_slice_96/stack_1?
pnn/strided_slice_96/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_96/stack_2?
pnn/strided_slice_96StridedSlicepnn/ReadVariableOp_96:value:0#pnn/strided_slice_96/stack:output:0%pnn/strided_slice_96/stack_1:output:0%pnn/strided_slice_96/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_96p
pnn/ExpandDims_48/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_48/dim?
pnn/ExpandDims_48
ExpandDimspnn/strided_slice_96:output:0pnn/ExpandDims_48/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_48?

pnn/Mul_97Mulpnn/MatMul:output:0pnn/ExpandDims_48:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_97?
pnn/Sum_97/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_97/reduction_indices?

pnn/Sum_97Sumpnn/Mul_97:z:0%pnn/Sum_97/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_97}
pnn/Reshape_97/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_97/shape?
pnn/Reshape_97Reshapepnn/Sum_97:output:0pnn/Reshape_97/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_97?
pnn/ReadVariableOp_97ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_97?
pnn/strided_slice_97/stackConst*
_output_shapes
:*
dtype0*
valueB:/2
pnn/strided_slice_97/stack?
pnn/strided_slice_97/stack_1Const*
_output_shapes
:*
dtype0*
valueB:02
pnn/strided_slice_97/stack_1?
pnn/strided_slice_97/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_97/stack_2?
pnn/strided_slice_97StridedSlicepnn/ReadVariableOp_97:value:0#pnn/strided_slice_97/stack:output:0%pnn/strided_slice_97/stack_1:output:0%pnn/strided_slice_97/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_97p
pnn/ExpandDims_49/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_49/dim?
pnn/ExpandDims_49
ExpandDimspnn/strided_slice_97:output:0pnn/ExpandDims_49/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_49?

pnn/Mul_98Mulpnn/MatMul:output:0pnn/ExpandDims_49:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_98?
pnn/Sum_98/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_98/reduction_indices?

pnn/Sum_98Sumpnn/Mul_98:z:0%pnn/Sum_98/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_98}
pnn/Reshape_98/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_98/shape?
pnn/Reshape_98Reshapepnn/Sum_98:output:0pnn/Reshape_98/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_98?
pnn/ReadVariableOp_98ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_98?
pnn/strided_slice_98/stackConst*
_output_shapes
:*
dtype0*
valueB:02
pnn/strided_slice_98/stack?
pnn/strided_slice_98/stack_1Const*
_output_shapes
:*
dtype0*
valueB:12
pnn/strided_slice_98/stack_1?
pnn/strided_slice_98/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_98/stack_2?
pnn/strided_slice_98StridedSlicepnn/ReadVariableOp_98:value:0#pnn/strided_slice_98/stack:output:0%pnn/strided_slice_98/stack_1:output:0%pnn/strided_slice_98/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_98p
pnn/ExpandDims_50/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_50/dim?
pnn/ExpandDims_50
ExpandDimspnn/strided_slice_98:output:0pnn/ExpandDims_50/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_50?

pnn/Mul_99Mulpnn/MatMul:output:0pnn/ExpandDims_50:output:0*
T0*+
_output_shapes
:?????????2

pnn/Mul_99?
pnn/Sum_99/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_99/reduction_indices?

pnn/Sum_99Sumpnn/Mul_99:z:0%pnn/Sum_99/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2

pnn/Sum_99}
pnn/Reshape_99/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_99/shape?
pnn/Reshape_99Reshapepnn/Sum_99:output:0pnn/Reshape_99/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_99?
pnn/ReadVariableOp_99ReadVariableOppnn_readvariableop_50_resource*"
_output_shapes
:2*
dtype02
pnn/ReadVariableOp_99?
pnn/strided_slice_99/stackConst*
_output_shapes
:*
dtype0*
valueB:12
pnn/strided_slice_99/stack?
pnn/strided_slice_99/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
pnn/strided_slice_99/stack_1?
pnn/strided_slice_99/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
pnn/strided_slice_99/stack_2?
pnn/strided_slice_99StridedSlicepnn/ReadVariableOp_99:value:0#pnn/strided_slice_99/stack:output:0%pnn/strided_slice_99/stack_1:output:0%pnn/strided_slice_99/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
pnn/strided_slice_99p
pnn/ExpandDims_51/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
pnn/ExpandDims_51/dim?
pnn/ExpandDims_51
ExpandDimspnn/strided_slice_99:output:0pnn/ExpandDims_51/dim:output:0*
T0*"
_output_shapes
:2
pnn/ExpandDims_51?
pnn/Mul_100Mulpnn/MatMul:output:0pnn/ExpandDims_51:output:0*
T0*+
_output_shapes
:?????????2
pnn/Mul_100?
pnn/Sum_100/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
pnn/Sum_100/reduction_indices?
pnn/Sum_100Sumpnn/Mul_100:z:0&pnn/Sum_100/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
pnn/Sum_100
pnn/Reshape_100/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
pnn/Reshape_100/shape?
pnn/Reshape_100Reshapepnn/Sum_100:output:0pnn/Reshape_100/shape:output:0*
T0*'
_output_shapes
:?????????2
pnn/Reshape_100h
pnn/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
pnn/concat_1/axis?

pnn/concat_1ConcatV2pnn/Reshape_51:output:0pnn/Reshape_52:output:0pnn/Reshape_53:output:0pnn/Reshape_54:output:0pnn/Reshape_55:output:0pnn/Reshape_56:output:0pnn/Reshape_57:output:0pnn/Reshape_58:output:0pnn/Reshape_59:output:0pnn/Reshape_60:output:0pnn/Reshape_61:output:0pnn/Reshape_62:output:0pnn/Reshape_63:output:0pnn/Reshape_64:output:0pnn/Reshape_65:output:0pnn/Reshape_66:output:0pnn/Reshape_67:output:0pnn/Reshape_68:output:0pnn/Reshape_69:output:0pnn/Reshape_70:output:0pnn/Reshape_71:output:0pnn/Reshape_72:output:0pnn/Reshape_73:output:0pnn/Reshape_74:output:0pnn/Reshape_75:output:0pnn/Reshape_76:output:0pnn/Reshape_77:output:0pnn/Reshape_78:output:0pnn/Reshape_79:output:0pnn/Reshape_80:output:0pnn/Reshape_81:output:0pnn/Reshape_82:output:0pnn/Reshape_83:output:0pnn/Reshape_84:output:0pnn/Reshape_85:output:0pnn/Reshape_86:output:0pnn/Reshape_87:output:0pnn/Reshape_88:output:0pnn/Reshape_89:output:0pnn/Reshape_90:output:0pnn/Reshape_91:output:0pnn/Reshape_92:output:0pnn/Reshape_93:output:0pnn/Reshape_94:output:0pnn/Reshape_95:output:0pnn/Reshape_96:output:0pnn/Reshape_97:output:0pnn/Reshape_98:output:0pnn/Reshape_99:output:0pnn/Reshape_100:output:0pnn/concat_1/axis:output:0*
N2*
T0*'
_output_shapes
:?????????22
pnn/concat_1y
pnn/addAddV2pnn/concat:output:0pnn/concat_1:output:0*
T0*'
_output_shapes
:?????????22	
pnn/add?
pnn/add_1/ReadVariableOpReadVariableOp!pnn_add_1_readvariableop_resource*
_output_shapes
:2*
dtype02
pnn/add_1/ReadVariableOp?
	pnn/add_1AddV2pnn/add:z:0 pnn/add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
	pnn/add_1]
pnn/ReluRelupnn/add_1:z:0*
T0*'
_output_shapes
:?????????22

pnn/Reluk
pnn/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pnn/dropout/Const?
pnn/dropout/MulMulpnn/Relu:activations:0pnn/dropout/Const:output:0*
T0*'
_output_shapes
:?????????22
pnn/dropout/Mull
pnn/dropout/ShapeShapepnn/Relu:activations:0*
T0*
_output_shapes
:2
pnn/dropout/Shape?
(pnn/dropout/random_uniform/RandomUniformRandomUniformpnn/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed?2*
(pnn/dropout/random_uniform/RandomUniform}
pnn/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
pnn/dropout/GreaterEqual/y?
pnn/dropout/GreaterEqualGreaterEqual1pnn/dropout/random_uniform/RandomUniform:output:0#pnn/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22
pnn/dropout/GreaterEqual?
pnn/dropout/CastCastpnn/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
pnn/dropout/Cast?
pnn/dropout/Mul_1Mulpnn/dropout/Mul:z:0pnn/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????22
pnn/dropout/Mul_1?
pnn/MatMul_1/ReadVariableOpReadVariableOp$pnn_matmul_1_readvariableop_resource*
_output_shapes

:2 *
dtype02
pnn/MatMul_1/ReadVariableOp?
pnn/MatMul_1MatMulpnn/dropout/Mul_1:z:0#pnn/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
pnn/MatMul_1?
pnn/add_2/ReadVariableOpReadVariableOp!pnn_add_2_readvariableop_resource*
_output_shapes

: *
dtype02
pnn/add_2/ReadVariableOp?
	pnn/add_2AddV2pnn/MatMul_1:product:0 pnn/add_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
	pnn/add_2?
"pnn/moments/mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2$
"pnn/moments/mean/reduction_indices?
pnn/moments/meanMeanpnn/add_2:z:0+pnn/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
pnn/moments/mean?
pnn/moments/StopGradientStopGradientpnn/moments/mean:output:0*
T0*
_output_shapes

: 2
pnn/moments/StopGradient?
pnn/moments/SquaredDifferenceSquaredDifferencepnn/add_2:z:0!pnn/moments/StopGradient:output:0*
T0*'
_output_shapes
:????????? 2
pnn/moments/SquaredDifference?
&pnn/moments/variance/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2(
&pnn/moments/variance/reduction_indices?
pnn/moments/varianceMean!pnn/moments/SquaredDifference:z:0/pnn/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
pnn/moments/variance?
pnn/moments/SqueezeSqueezepnn/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
pnn/moments/Squeeze?
pnn/moments/Squeeze_1Squeezepnn/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
pnn/moments/Squeeze_1o
pnn/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2
pnn/batchnorm/add/y?
pnn/batchnorm/addAddV2pnn/moments/Squeeze_1:output:0pnn/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
pnn/batchnorm/addo
pnn/batchnorm/RsqrtRsqrtpnn/batchnorm/add:z:0*
T0*
_output_shapes
: 2
pnn/batchnorm/Rsqrt?
 pnn/batchnorm/mul/ReadVariableOpReadVariableOp)pnn_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02"
 pnn/batchnorm/mul/ReadVariableOp?
pnn/batchnorm/mulMulpnn/batchnorm/Rsqrt:y:0(pnn/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
pnn/batchnorm/mul?
pnn/batchnorm/mul_1Mulpnn/add_2:z:0pnn/batchnorm/mul:z:0*
T0*'
_output_shapes
:????????? 2
pnn/batchnorm/mul_1?
pnn/batchnorm/mul_2Mulpnn/moments/Squeeze:output:0pnn/batchnorm/mul:z:0*
T0*
_output_shapes
: 2
pnn/batchnorm/mul_2?
pnn/batchnorm/ReadVariableOpReadVariableOp%pnn_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
pnn/batchnorm/ReadVariableOp?
pnn/batchnorm/subSub$pnn/batchnorm/ReadVariableOp:value:0pnn/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
pnn/batchnorm/sub?
pnn/batchnorm/add_1AddV2pnn/batchnorm/mul_1:z:0pnn/batchnorm/sub:z:0*
T0*'
_output_shapes
:????????? 2
pnn/batchnorm/add_1k

pnn/Relu_1Relupnn/batchnorm/add_1:z:0*
T0*'
_output_shapes
:????????? 2

pnn/Relu_1o
pnn/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pnn/dropout_1/Const?
pnn/dropout_1/MulMulpnn/Relu_1:activations:0pnn/dropout_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
pnn/dropout_1/Mulr
pnn/dropout_1/ShapeShapepnn/Relu_1:activations:0*
T0*
_output_shapes
:2
pnn/dropout_1/Shape?
*pnn/dropout_1/random_uniform/RandomUniformRandomUniformpnn/dropout_1/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed?*
seed22,
*pnn/dropout_1/random_uniform/RandomUniform?
pnn/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
pnn/dropout_1/GreaterEqual/y?
pnn/dropout_1/GreaterEqualGreaterEqual3pnn/dropout_1/random_uniform/RandomUniform:output:0%pnn/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
pnn/dropout_1/GreaterEqual?
pnn/dropout_1/CastCastpnn/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
pnn/dropout_1/Cast?
pnn/dropout_1/Mul_1Mulpnn/dropout_1/Mul:z:0pnn/dropout_1/Cast:y:0*
T0*'
_output_shapes
:????????? 2
pnn/dropout_1/Mul_1?
pnn/MatMul_2/ReadVariableOpReadVariableOp$pnn_matmul_2_readvariableop_resource*
_output_shapes

:  *
dtype02
pnn/MatMul_2/ReadVariableOp?
pnn/MatMul_2MatMulpnn/dropout_1/Mul_1:z:0#pnn/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
pnn/MatMul_2?
pnn/add_3/ReadVariableOpReadVariableOp!pnn_add_3_readvariableop_resource*
_output_shapes

: *
dtype02
pnn/add_3/ReadVariableOp?
	pnn/add_3AddV2pnn/MatMul_2:product:0 pnn/add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
	pnn/add_3?
$pnn/moments_1/mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2&
$pnn/moments_1/mean/reduction_indices?
pnn/moments_1/meanMeanpnn/add_3:z:0-pnn/moments_1/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
pnn/moments_1/mean?
pnn/moments_1/StopGradientStopGradientpnn/moments_1/mean:output:0*
T0*
_output_shapes

: 2
pnn/moments_1/StopGradient?
pnn/moments_1/SquaredDifferenceSquaredDifferencepnn/add_3:z:0#pnn/moments_1/StopGradient:output:0*
T0*'
_output_shapes
:????????? 2!
pnn/moments_1/SquaredDifference?
(pnn/moments_1/variance/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2*
(pnn/moments_1/variance/reduction_indices?
pnn/moments_1/varianceMean#pnn/moments_1/SquaredDifference:z:01pnn/moments_1/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
pnn/moments_1/variance?
pnn/moments_1/SqueezeSqueezepnn/moments_1/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
pnn/moments_1/Squeeze?
pnn/moments_1/Squeeze_1Squeezepnn/moments_1/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
pnn/moments_1/Squeeze_1s
pnn/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2
pnn/batchnorm_1/add/y?
pnn/batchnorm_1/addAddV2 pnn/moments_1/Squeeze_1:output:0pnn/batchnorm_1/add/y:output:0*
T0*
_output_shapes
: 2
pnn/batchnorm_1/addu
pnn/batchnorm_1/RsqrtRsqrtpnn/batchnorm_1/add:z:0*
T0*
_output_shapes
: 2
pnn/batchnorm_1/Rsqrt?
"pnn/batchnorm_1/mul/ReadVariableOpReadVariableOp)pnn_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02$
"pnn/batchnorm_1/mul/ReadVariableOp?
pnn/batchnorm_1/mulMulpnn/batchnorm_1/Rsqrt:y:0*pnn/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
pnn/batchnorm_1/mul?
pnn/batchnorm_1/mul_1Mulpnn/add_3:z:0pnn/batchnorm_1/mul:z:0*
T0*'
_output_shapes
:????????? 2
pnn/batchnorm_1/mul_1?
pnn/batchnorm_1/mul_2Mulpnn/moments_1/Squeeze:output:0pnn/batchnorm_1/mul:z:0*
T0*
_output_shapes
: 2
pnn/batchnorm_1/mul_2?
pnn/batchnorm_1/ReadVariableOpReadVariableOp%pnn_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02 
pnn/batchnorm_1/ReadVariableOp?
pnn/batchnorm_1/subSub&pnn/batchnorm_1/ReadVariableOp:value:0pnn/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
: 2
pnn/batchnorm_1/sub?
pnn/batchnorm_1/add_1AddV2pnn/batchnorm_1/mul_1:z:0pnn/batchnorm_1/sub:z:0*
T0*'
_output_shapes
:????????? 2
pnn/batchnorm_1/add_1m

pnn/Relu_2Relupnn/batchnorm_1/add_1:z:0*
T0*'
_output_shapes
:????????? 2

pnn/Relu_2o
pnn/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pnn/dropout_2/Const?
pnn/dropout_2/MulMulpnn/Relu_2:activations:0pnn/dropout_2/Const:output:0*
T0*'
_output_shapes
:????????? 2
pnn/dropout_2/Mulr
pnn/dropout_2/ShapeShapepnn/Relu_2:activations:0*
T0*
_output_shapes
:2
pnn/dropout_2/Shape?
*pnn/dropout_2/random_uniform/RandomUniformRandomUniformpnn/dropout_2/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed?*
seed22,
*pnn/dropout_2/random_uniform/RandomUniform?
pnn/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
pnn/dropout_2/GreaterEqual/y?
pnn/dropout_2/GreaterEqualGreaterEqual3pnn/dropout_2/random_uniform/RandomUniform:output:0%pnn/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
pnn/dropout_2/GreaterEqual?
pnn/dropout_2/CastCastpnn/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
pnn/dropout_2/Cast?
pnn/dropout_2/Mul_1Mulpnn/dropout_2/Mul:z:0pnn/dropout_2/Cast:y:0*
T0*'
_output_shapes
:????????? 2
pnn/dropout_2/Mul_1?
pnn/MatMul_3/ReadVariableOpReadVariableOp$pnn_matmul_3_readvariableop_resource*
_output_shapes

: *
dtype02
pnn/MatMul_3/ReadVariableOp?
pnn/MatMul_3MatMulpnn/dropout_2/Mul_1:z:0#pnn/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
pnn/MatMul_3?
pnn/add_4/ReadVariableOpReadVariableOp!pnn_add_4_readvariableop_resource*
_output_shapes
: *
dtype02
pnn/add_4/ReadVariableOp?
	pnn/add_4AddV2pnn/MatMul_3:product:0 pnn/add_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	pnn/add_4f
pnn/SigmoidSigmoidpnn/add_4:z:0*
T0*'
_output_shapes
:?????????2
pnn/Sigmoid?
IdentityIdentitypnn/Sigmoid:y:0^pnn/MatMul_1/ReadVariableOp^pnn/MatMul_2/ReadVariableOp^pnn/MatMul_3/ReadVariableOp^pnn/ReadVariableOp^pnn/ReadVariableOp_1^pnn/ReadVariableOp_10^pnn/ReadVariableOp_11^pnn/ReadVariableOp_12^pnn/ReadVariableOp_13^pnn/ReadVariableOp_14^pnn/ReadVariableOp_15^pnn/ReadVariableOp_16^pnn/ReadVariableOp_17^pnn/ReadVariableOp_18^pnn/ReadVariableOp_19^pnn/ReadVariableOp_2^pnn/ReadVariableOp_20^pnn/ReadVariableOp_21^pnn/ReadVariableOp_22^pnn/ReadVariableOp_23^pnn/ReadVariableOp_24^pnn/ReadVariableOp_25^pnn/ReadVariableOp_26^pnn/ReadVariableOp_27^pnn/ReadVariableOp_28^pnn/ReadVariableOp_29^pnn/ReadVariableOp_3^pnn/ReadVariableOp_30^pnn/ReadVariableOp_31^pnn/ReadVariableOp_32^pnn/ReadVariableOp_33^pnn/ReadVariableOp_34^pnn/ReadVariableOp_35^pnn/ReadVariableOp_36^pnn/ReadVariableOp_37^pnn/ReadVariableOp_38^pnn/ReadVariableOp_39^pnn/ReadVariableOp_4^pnn/ReadVariableOp_40^pnn/ReadVariableOp_41^pnn/ReadVariableOp_42^pnn/ReadVariableOp_43^pnn/ReadVariableOp_44^pnn/ReadVariableOp_45^pnn/ReadVariableOp_46^pnn/ReadVariableOp_47^pnn/ReadVariableOp_48^pnn/ReadVariableOp_49^pnn/ReadVariableOp_5^pnn/ReadVariableOp_50^pnn/ReadVariableOp_51^pnn/ReadVariableOp_52^pnn/ReadVariableOp_53^pnn/ReadVariableOp_54^pnn/ReadVariableOp_55^pnn/ReadVariableOp_56^pnn/ReadVariableOp_57^pnn/ReadVariableOp_58^pnn/ReadVariableOp_59^pnn/ReadVariableOp_6^pnn/ReadVariableOp_60^pnn/ReadVariableOp_61^pnn/ReadVariableOp_62^pnn/ReadVariableOp_63^pnn/ReadVariableOp_64^pnn/ReadVariableOp_65^pnn/ReadVariableOp_66^pnn/ReadVariableOp_67^pnn/ReadVariableOp_68^pnn/ReadVariableOp_69^pnn/ReadVariableOp_7^pnn/ReadVariableOp_70^pnn/ReadVariableOp_71^pnn/ReadVariableOp_72^pnn/ReadVariableOp_73^pnn/ReadVariableOp_74^pnn/ReadVariableOp_75^pnn/ReadVariableOp_76^pnn/ReadVariableOp_77^pnn/ReadVariableOp_78^pnn/ReadVariableOp_79^pnn/ReadVariableOp_8^pnn/ReadVariableOp_80^pnn/ReadVariableOp_81^pnn/ReadVariableOp_82^pnn/ReadVariableOp_83^pnn/ReadVariableOp_84^pnn/ReadVariableOp_85^pnn/ReadVariableOp_86^pnn/ReadVariableOp_87^pnn/ReadVariableOp_88^pnn/ReadVariableOp_89^pnn/ReadVariableOp_9^pnn/ReadVariableOp_90^pnn/ReadVariableOp_91^pnn/ReadVariableOp_92^pnn/ReadVariableOp_93^pnn/ReadVariableOp_94^pnn/ReadVariableOp_95^pnn/ReadVariableOp_96^pnn/ReadVariableOp_97^pnn/ReadVariableOp_98^pnn/ReadVariableOp_99^pnn/add_1/ReadVariableOp^pnn/add_2/ReadVariableOp^pnn/add_3/ReadVariableOp^pnn/add_4/ReadVariableOp^pnn/batchnorm/ReadVariableOp!^pnn/batchnorm/mul/ReadVariableOp^pnn/batchnorm_1/ReadVariableOp#^pnn/batchnorm_1/mul/ReadVariableOp^pnn/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:??????????:??????????::::::::::::2:
pnn/MatMul_1/ReadVariableOppnn/MatMul_1/ReadVariableOp2:
pnn/MatMul_2/ReadVariableOppnn/MatMul_2/ReadVariableOp2:
pnn/MatMul_3/ReadVariableOppnn/MatMul_3/ReadVariableOp2(
pnn/ReadVariableOppnn/ReadVariableOp2,
pnn/ReadVariableOp_1pnn/ReadVariableOp_12.
pnn/ReadVariableOp_10pnn/ReadVariableOp_102.
pnn/ReadVariableOp_11pnn/ReadVariableOp_112.
pnn/ReadVariableOp_12pnn/ReadVariableOp_122.
pnn/ReadVariableOp_13pnn/ReadVariableOp_132.
pnn/ReadVariableOp_14pnn/ReadVariableOp_142.
pnn/ReadVariableOp_15pnn/ReadVariableOp_152.
pnn/ReadVariableOp_16pnn/ReadVariableOp_162.
pnn/ReadVariableOp_17pnn/ReadVariableOp_172.
pnn/ReadVariableOp_18pnn/ReadVariableOp_182.
pnn/ReadVariableOp_19pnn/ReadVariableOp_192,
pnn/ReadVariableOp_2pnn/ReadVariableOp_22.
pnn/ReadVariableOp_20pnn/ReadVariableOp_202.
pnn/ReadVariableOp_21pnn/ReadVariableOp_212.
pnn/ReadVariableOp_22pnn/ReadVariableOp_222.
pnn/ReadVariableOp_23pnn/ReadVariableOp_232.
pnn/ReadVariableOp_24pnn/ReadVariableOp_242.
pnn/ReadVariableOp_25pnn/ReadVariableOp_252.
pnn/ReadVariableOp_26pnn/ReadVariableOp_262.
pnn/ReadVariableOp_27pnn/ReadVariableOp_272.
pnn/ReadVariableOp_28pnn/ReadVariableOp_282.
pnn/ReadVariableOp_29pnn/ReadVariableOp_292,
pnn/ReadVariableOp_3pnn/ReadVariableOp_32.
pnn/ReadVariableOp_30pnn/ReadVariableOp_302.
pnn/ReadVariableOp_31pnn/ReadVariableOp_312.
pnn/ReadVariableOp_32pnn/ReadVariableOp_322.
pnn/ReadVariableOp_33pnn/ReadVariableOp_332.
pnn/ReadVariableOp_34pnn/ReadVariableOp_342.
pnn/ReadVariableOp_35pnn/ReadVariableOp_352.
pnn/ReadVariableOp_36pnn/ReadVariableOp_362.
pnn/ReadVariableOp_37pnn/ReadVariableOp_372.
pnn/ReadVariableOp_38pnn/ReadVariableOp_382.
pnn/ReadVariableOp_39pnn/ReadVariableOp_392,
pnn/ReadVariableOp_4pnn/ReadVariableOp_42.
pnn/ReadVariableOp_40pnn/ReadVariableOp_402.
pnn/ReadVariableOp_41pnn/ReadVariableOp_412.
pnn/ReadVariableOp_42pnn/ReadVariableOp_422.
pnn/ReadVariableOp_43pnn/ReadVariableOp_432.
pnn/ReadVariableOp_44pnn/ReadVariableOp_442.
pnn/ReadVariableOp_45pnn/ReadVariableOp_452.
pnn/ReadVariableOp_46pnn/ReadVariableOp_462.
pnn/ReadVariableOp_47pnn/ReadVariableOp_472.
pnn/ReadVariableOp_48pnn/ReadVariableOp_482.
pnn/ReadVariableOp_49pnn/ReadVariableOp_492,
pnn/ReadVariableOp_5pnn/ReadVariableOp_52.
pnn/ReadVariableOp_50pnn/ReadVariableOp_502.
pnn/ReadVariableOp_51pnn/ReadVariableOp_512.
pnn/ReadVariableOp_52pnn/ReadVariableOp_522.
pnn/ReadVariableOp_53pnn/ReadVariableOp_532.
pnn/ReadVariableOp_54pnn/ReadVariableOp_542.
pnn/ReadVariableOp_55pnn/ReadVariableOp_552.
pnn/ReadVariableOp_56pnn/ReadVariableOp_562.
pnn/ReadVariableOp_57pnn/ReadVariableOp_572.
pnn/ReadVariableOp_58pnn/ReadVariableOp_582.
pnn/ReadVariableOp_59pnn/ReadVariableOp_592,
pnn/ReadVariableOp_6pnn/ReadVariableOp_62.
pnn/ReadVariableOp_60pnn/ReadVariableOp_602.
pnn/ReadVariableOp_61pnn/ReadVariableOp_612.
pnn/ReadVariableOp_62pnn/ReadVariableOp_622.
pnn/ReadVariableOp_63pnn/ReadVariableOp_632.
pnn/ReadVariableOp_64pnn/ReadVariableOp_642.
pnn/ReadVariableOp_65pnn/ReadVariableOp_652.
pnn/ReadVariableOp_66pnn/ReadVariableOp_662.
pnn/ReadVariableOp_67pnn/ReadVariableOp_672.
pnn/ReadVariableOp_68pnn/ReadVariableOp_682.
pnn/ReadVariableOp_69pnn/ReadVariableOp_692,
pnn/ReadVariableOp_7pnn/ReadVariableOp_72.
pnn/ReadVariableOp_70pnn/ReadVariableOp_702.
pnn/ReadVariableOp_71pnn/ReadVariableOp_712.
pnn/ReadVariableOp_72pnn/ReadVariableOp_722.
pnn/ReadVariableOp_73pnn/ReadVariableOp_732.
pnn/ReadVariableOp_74pnn/ReadVariableOp_742.
pnn/ReadVariableOp_75pnn/ReadVariableOp_752.
pnn/ReadVariableOp_76pnn/ReadVariableOp_762.
pnn/ReadVariableOp_77pnn/ReadVariableOp_772.
pnn/ReadVariableOp_78pnn/ReadVariableOp_782.
pnn/ReadVariableOp_79pnn/ReadVariableOp_792,
pnn/ReadVariableOp_8pnn/ReadVariableOp_82.
pnn/ReadVariableOp_80pnn/ReadVariableOp_802.
pnn/ReadVariableOp_81pnn/ReadVariableOp_812.
pnn/ReadVariableOp_82pnn/ReadVariableOp_822.
pnn/ReadVariableOp_83pnn/ReadVariableOp_832.
pnn/ReadVariableOp_84pnn/ReadVariableOp_842.
pnn/ReadVariableOp_85pnn/ReadVariableOp_852.
pnn/ReadVariableOp_86pnn/ReadVariableOp_862.
pnn/ReadVariableOp_87pnn/ReadVariableOp_872.
pnn/ReadVariableOp_88pnn/ReadVariableOp_882.
pnn/ReadVariableOp_89pnn/ReadVariableOp_892,
pnn/ReadVariableOp_9pnn/ReadVariableOp_92.
pnn/ReadVariableOp_90pnn/ReadVariableOp_902.
pnn/ReadVariableOp_91pnn/ReadVariableOp_912.
pnn/ReadVariableOp_92pnn/ReadVariableOp_922.
pnn/ReadVariableOp_93pnn/ReadVariableOp_932.
pnn/ReadVariableOp_94pnn/ReadVariableOp_942.
pnn/ReadVariableOp_95pnn/ReadVariableOp_952.
pnn/ReadVariableOp_96pnn/ReadVariableOp_962.
pnn/ReadVariableOp_97pnn/ReadVariableOp_972.
pnn/ReadVariableOp_98pnn/ReadVariableOp_982.
pnn/ReadVariableOp_99pnn/ReadVariableOp_9924
pnn/add_1/ReadVariableOppnn/add_1/ReadVariableOp24
pnn/add_2/ReadVariableOppnn/add_2/ReadVariableOp24
pnn/add_3/ReadVariableOppnn/add_3/ReadVariableOp24
pnn/add_4/ReadVariableOppnn/add_4/ReadVariableOp2<
pnn/batchnorm/ReadVariableOppnn/batchnorm/ReadVariableOp2D
 pnn/batchnorm/mul/ReadVariableOp pnn/batchnorm/mul/ReadVariableOp2@
pnn/batchnorm_1/ReadVariableOppnn/batchnorm_1/ReadVariableOp2H
"pnn/batchnorm_1/mul/ReadVariableOp"pnn/batchnorm_1/mul/ReadVariableOp2,
pnn/embedding_lookuppnn/embedding_lookup:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1:QM
(
_output_shapes
:??????????
!
_user_specified_name	input_2
?	
?
&__inference_signature_wrapper_16075377
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_160740992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:??????????:??????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1:QM
(
_output_shapes
:??????????
!
_user_specified_name	input_2"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input_11
serving_default_input_1:0??????????
<
input_21
serving_default_input_2:0??????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?
?
deep_layers
dropout_deep
_weights
regularization_losses
trainable_variables
	variables
	keras_api

signatures
*&call_and_return_all_conditional_losses
_default_save_signature
__call__"?
_tf_keras_model?{"class_name": "PNN", "name": "pnn", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "PNN"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	feature_embeddings

feature_bias
product-linear
product-bias
product-quadratic-outer
layer_0

bias_0
layer_1

bias_1
batch_norm_scale_0
batch_norm_offset_0

output
output_bias"
trackable_dict_wrapper
 "
trackable_list_wrapper
~
	0

1
2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
~
	0

1
2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
?
regularization_losses
trainable_variables

layers
layer_metrics
non_trainable_variables
layer_regularization_losses
	variables
metrics
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
&:$
??2feature_embeddings
 :
??2feature_bias
:2?2Variable
:22Variable
:22Variable
:2 2Variable
: 2Variable
:  2Variable
: 2Variable
:2Variable
:2Variable
: 2Variable
: 2Variable
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
A__inference_pnn_layer_call_and_return_conditional_losses_16075314?
???
FullArgSpec
args?
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *P?M
K?H
"?
input_1??????????
"?
input_2??????????
?2?
#__inference__wrapped_model_16074099?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *P?M
K?H
"?
input_1??????????
"?
input_2??????????
?2?
&__inference_pnn_layer_call_fn_16075345?
???
FullArgSpec
args?
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *P?M
K?H
"?
input_1??????????
"?
input_2??????????
?B?
&__inference_signature_wrapper_16075377input_1input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_16074099?	Z?W
P?M
K?H
"?
input_1??????????
"?
input_2??????????
? "3?0
.
output_1"?
output_1??????????
A__inference_pnn_layer_call_and_return_conditional_losses_16075314?	Z?W
P?M
K?H
"?
input_1??????????
"?
input_2??????????
? "%?"
?
0?????????
? ?
&__inference_pnn_layer_call_fn_16075345?	Z?W
P?M
K?H
"?
input_1??????????
"?
input_2??????????
? "???????????
&__inference_signature_wrapper_16075377?	k?h
? 
a?^
-
input_1"?
input_1??????????
-
input_2"?
input_2??????????"3?0
.
output_1"?
output_1?????????