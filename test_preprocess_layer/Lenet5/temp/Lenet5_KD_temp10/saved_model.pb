??
??
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

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
?
If
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 ?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
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
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?x*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?x*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:x*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xT*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:xT*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:T*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T
*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:T
*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:
*
dtype0
?
random_crop/cond/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:**
shared_namerandom_crop/cond/StateVar
?
-random_crop/cond/StateVar/Read/ReadVariableOpReadVariableOprandom_crop/cond/StateVar*
_output_shapes
:*
dtype0	
?
,sequential_1/sequential/random_flip/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*=
shared_name.,sequential_1/sequential/random_flip/StateVar
?
@sequential_1/sequential/random_flip/StateVar/Read/ReadVariableOpReadVariableOp,sequential_1/sequential/random_flip/StateVar*
_output_shapes
:*
dtype0	

NoOpNoOp
?=
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?=
value?=B?= B?=
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
?
layer-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*
?
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses* 
?

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses*
?
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses* 
?
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses* 
?

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses*
?

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses*
?

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses*
J
0
1
(2
)3
<4
=5
D6
E7
L8
M9*
J
0
1
(2
)3
<4
=5
D6
E7
L8
M9*
* 
?
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Yserving_default* 
?
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^_random_generator
___call__
*`&call_and_return_all_conditional_losses*
?
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e_random_generator
f__call__
*g&call_and_return_all_conditional_losses*
* 
* 
* 
?
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

(0
)1*

(0
)1*
* 
?
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 
* 
* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

<0
=1*

<0
=1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

D0
E1*

D0
E1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

L0
M1*

L0
M1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*
* 
* 
* 
C
0
1
2
3
4
5
6
7
	8*
* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses* 

?
_generator*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
a	variables
btrainable_variables
cregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses* 

?
_generator*
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?
_state_var*
* 
* 
* 
* 
* 

?
_state_var*
?
VARIABLE_VALUErandom_crop/cond/StateVarRlayer-0/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,sequential_1/sequential/random_flip/StateVarRlayer-0/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
?
 serving_default_sequential_inputPlaceholder*/
_output_shapes
:?????????  *
dtype0*$
shape:?????????  
?
StatefulPartitionedCallStatefulPartitionedCall serving_default_sequential_inputrandom_crop/cond/StateVarconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_7018273
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp-random_crop/cond/StateVar/Read/ReadVariableOp@sequential_1/sequential/random_flip/StateVar/Read/ReadVariableOpConst*
Tin
2		*
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
GPU 2J 8? *)
f$R"
 __inference__traced_save_7019294
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasrandom_crop/cond/StateVar,sequential_1/sequential/random_flip/StateVar*
Tin
2*
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
GPU 2J 8? *,
f'R%
#__inference__traced_restore_7019340??
??
?
5sequential_1_sequential_random_crop_cond_true_7016653_
Qsequential_1_sequential_random_crop_cond_stateful_uniform_rngreadandskip_resource:	X
Tsequential_1_sequential_random_crop_cond_add_sequential_1_sequential_random_crop_sub\
Xsequential_1_sequential_random_crop_cond_add_1_sequential_1_sequential_random_crop_sub_1X
Tsequential_1_sequential_random_crop_cond_crop_to_bounding_box_shape_sequential_input5
1sequential_1_sequential_random_crop_cond_identity??Ksequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert/Assert?Msequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_1/Assert?Msequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_2/Assert?Msequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_3/Assert?Hsequential_1/sequential/random_crop/cond/stateful_uniform/RngReadAndSkip?
?sequential_1/sequential/random_crop/cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:
=sequential_1/sequential/random_crop/cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : ?
=sequential_1/sequential/random_crop/cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
?sequential_1/sequential/random_crop/cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
>sequential_1/sequential/random_crop/cond/stateful_uniform/ProdProdHsequential_1/sequential/random_crop/cond/stateful_uniform/shape:output:0Hsequential_1/sequential/random_crop/cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: ?
@sequential_1/sequential/random_crop/cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
@sequential_1/sequential/random_crop/cond/stateful_uniform/Cast_1CastGsequential_1/sequential/random_crop/cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
Hsequential_1/sequential/random_crop/cond/stateful_uniform/RngReadAndSkipRngReadAndSkipQsequential_1_sequential_random_crop_cond_stateful_uniform_rngreadandskip_resourceIsequential_1/sequential/random_crop/cond/stateful_uniform/Cast/x:output:0Dsequential_1/sequential/random_crop/cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:?
Msequential_1/sequential/random_crop/cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Osequential_1/sequential/random_crop/cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Osequential_1/sequential/random_crop/cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Gsequential_1/sequential/random_crop/cond/stateful_uniform/strided_sliceStridedSlicePsequential_1/sequential/random_crop/cond/stateful_uniform/RngReadAndSkip:value:0Vsequential_1/sequential/random_crop/cond/stateful_uniform/strided_slice/stack:output:0Xsequential_1/sequential/random_crop/cond/stateful_uniform/strided_slice/stack_1:output:0Xsequential_1/sequential/random_crop/cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
Asequential_1/sequential/random_crop/cond/stateful_uniform/BitcastBitcastPsequential_1/sequential/random_crop/cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0?
Osequential_1/sequential/random_crop/cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Qsequential_1/sequential/random_crop/cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Qsequential_1/sequential/random_crop/cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Isequential_1/sequential/random_crop/cond/stateful_uniform/strided_slice_1StridedSlicePsequential_1/sequential/random_crop/cond/stateful_uniform/RngReadAndSkip:value:0Xsequential_1/sequential/random_crop/cond/stateful_uniform/strided_slice_1/stack:output:0Zsequential_1/sequential/random_crop/cond/stateful_uniform/strided_slice_1/stack_1:output:0Zsequential_1/sequential/random_crop/cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
Csequential_1/sequential/random_crop/cond/stateful_uniform/Bitcast_1BitcastRsequential_1/sequential/random_crop/cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
=sequential_1/sequential/random_crop/cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
9sequential_1/sequential/random_crop/cond/stateful_uniformStatelessRandomUniformIntV2Hsequential_1/sequential/random_crop/cond/stateful_uniform/shape:output:0Lsequential_1/sequential/random_crop/cond/stateful_uniform/Bitcast_1:output:0Jsequential_1/sequential/random_crop/cond/stateful_uniform/Bitcast:output:0Fsequential_1/sequential/random_crop/cond/stateful_uniform/alg:output:0Fsequential_1/sequential/random_crop/cond/stateful_uniform/min:output:0Fsequential_1/sequential/random_crop/cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0?
<sequential_1/sequential/random_crop/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
>sequential_1/sequential/random_crop/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
>sequential_1/sequential/random_crop/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential_1/sequential/random_crop/cond/strided_sliceStridedSliceBsequential_1/sequential/random_crop/cond/stateful_uniform:output:0Esequential_1/sequential/random_crop/cond/strided_slice/stack:output:0Gsequential_1/sequential/random_crop/cond/strided_slice/stack_1:output:0Gsequential_1/sequential/random_crop/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.sequential_1/sequential/random_crop/cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
,sequential_1/sequential/random_crop/cond/addAddV2Tsequential_1_sequential_random_crop_cond_add_sequential_1_sequential_random_crop_sub7sequential_1/sequential/random_crop/cond/add/y:output:0*
T0*
_output_shapes
: ?
,sequential_1/sequential/random_crop/cond/modFloorMod?sequential_1/sequential/random_crop/cond/strided_slice:output:00sequential_1/sequential/random_crop/cond/add:z:0*
T0*
_output_shapes
: ?
>sequential_1/sequential/random_crop/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
@sequential_1/sequential/random_crop/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@sequential_1/sequential/random_crop/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_1/sequential/random_crop/cond/strided_slice_1StridedSliceBsequential_1/sequential/random_crop/cond/stateful_uniform:output:0Gsequential_1/sequential/random_crop/cond/strided_slice_1/stack:output:0Isequential_1/sequential/random_crop/cond/strided_slice_1/stack_1:output:0Isequential_1/sequential/random_crop/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0sequential_1/sequential/random_crop/cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
.sequential_1/sequential/random_crop/cond/add_1AddV2Xsequential_1_sequential_random_crop_cond_add_1_sequential_1_sequential_random_crop_sub_19sequential_1/sequential/random_crop/cond/add_1/y:output:0*
T0*
_output_shapes
: ?
.sequential_1/sequential/random_crop/cond/mod_1FloorModAsequential_1/sequential/random_crop/cond/strided_slice_1:output:02sequential_1/sequential/random_crop/cond/add_1:z:0*
T0*
_output_shapes
: ?
Csequential_1/sequential/random_crop/cond/crop_to_bounding_box/ShapeShapeTsequential_1_sequential_random_crop_cond_crop_to_bounding_box_shape_sequential_input*
T0*
_output_shapes
:?
Esequential_1/sequential/random_crop/cond/crop_to_bounding_box/unstackUnpackLsequential_1/sequential/random_crop/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
num?
Lsequential_1/sequential/random_crop/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Jsequential_1/sequential/random_crop/cond/crop_to_bounding_box/GreaterEqualGreaterEqual2sequential_1/sequential/random_crop/cond/mod_1:z:0Usequential_1/sequential/random_crop/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
Jsequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
Rsequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
Ksequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert/AssertAssertNsequential_1/sequential/random_crop/cond/crop_to_bounding_box/GreaterEqual:z:0[sequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 ?
Nsequential_1/sequential/random_crop/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Lsequential_1/sequential/random_crop/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqual0sequential_1/sequential/random_crop/cond/mod:z:0Wsequential_1/sequential/random_crop/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
Lsequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
Tsequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
Msequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_1/AssertAssertPsequential_1/sequential/random_crop/cond/crop_to_bounding_box/GreaterEqual_1:z:0]sequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0L^sequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 ?
Csequential_1/sequential/random_crop/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B : ?
Asequential_1/sequential/random_crop/cond/crop_to_bounding_box/addAddV2Lsequential_1/sequential/random_crop/cond/crop_to_bounding_box/add/x:output:02sequential_1/sequential/random_crop/cond/mod_1:z:0*
T0*
_output_shapes
: ?
Isequential_1/sequential/random_crop/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Gsequential_1/sequential/random_crop/cond/crop_to_bounding_box/LessEqual	LessEqualEsequential_1/sequential/random_crop/cond/crop_to_bounding_box/add:z:0Rsequential_1/sequential/random_crop/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
Lsequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
Tsequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
Msequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_2/AssertAssertKsequential_1/sequential/random_crop/cond/crop_to_bounding_box/LessEqual:z:0]sequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0N^sequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 ?
Esequential_1/sequential/random_crop/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : ?
Csequential_1/sequential/random_crop/cond/crop_to_bounding_box/add_1AddV2Nsequential_1/sequential/random_crop/cond/crop_to_bounding_box/add_1/x:output:00sequential_1/sequential/random_crop/cond/mod:z:0*
T0*
_output_shapes
: ?
Ksequential_1/sequential/random_crop/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Isequential_1/sequential/random_crop/cond/crop_to_bounding_box/LessEqual_1	LessEqualGsequential_1/sequential/random_crop/cond/crop_to_bounding_box/add_1:z:0Tsequential_1/sequential/random_crop/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
Lsequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
Tsequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
Msequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_3/AssertAssertMsequential_1/sequential/random_crop/cond/crop_to_bounding_box/LessEqual_1:z:0]sequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0N^sequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
Psequential_1/sequential/random_crop/cond/crop_to_bounding_box/control_dependencyIdentityTsequential_1_sequential_random_crop_cond_crop_to_bounding_box_shape_sequential_inputL^sequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert/AssertN^sequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_1/AssertN^sequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_2/AssertN^sequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_3/Assert*
T0*#
_class
loc:@sequential_input*/
_output_shapes
:?????????  ?
Esequential_1/sequential/random_crop/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : ?
Esequential_1/sequential/random_crop/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
Csequential_1/sequential/random_crop/cond/crop_to_bounding_box/stackPackNsequential_1/sequential/random_crop/cond/crop_to_bounding_box/stack/0:output:00sequential_1/sequential/random_crop/cond/mod:z:02sequential_1/sequential/random_crop/cond/mod_1:z:0Nsequential_1/sequential/random_crop/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
Esequential_1/sequential/random_crop/cond/crop_to_bounding_box/Shape_1ShapeYsequential_1/sequential/random_crop/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
Qsequential_1/sequential/random_crop/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Ssequential_1/sequential/random_crop/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Ssequential_1/sequential/random_crop/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Ksequential_1/sequential/random_crop/cond/crop_to_bounding_box/strided_sliceStridedSliceNsequential_1/sequential/random_crop/cond/crop_to_bounding_box/Shape_1:output:0Zsequential_1/sequential/random_crop/cond/crop_to_bounding_box/strided_slice/stack:output:0\sequential_1/sequential/random_crop/cond/crop_to_bounding_box/strided_slice/stack_1:output:0\sequential_1/sequential/random_crop/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Esequential_1/sequential/random_crop/cond/crop_to_bounding_box/Shape_2ShapeYsequential_1/sequential/random_crop/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
Ssequential_1/sequential/random_crop/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Usequential_1/sequential/random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Usequential_1/sequential/random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Msequential_1/sequential/random_crop/cond/crop_to_bounding_box/strided_slice_1StridedSliceNsequential_1/sequential/random_crop/cond/crop_to_bounding_box/Shape_2:output:0\sequential_1/sequential/random_crop/cond/crop_to_bounding_box/strided_slice_1/stack:output:0^sequential_1/sequential/random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0^sequential_1/sequential/random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Gsequential_1/sequential/random_crop/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Gsequential_1/sequential/random_crop/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : ?
Esequential_1/sequential/random_crop/cond/crop_to_bounding_box/stack_1PackTsequential_1/sequential/random_crop/cond/crop_to_bounding_box/strided_slice:output:0Psequential_1/sequential/random_crop/cond/crop_to_bounding_box/stack_1/1:output:0Psequential_1/sequential/random_crop/cond/crop_to_bounding_box/stack_1/2:output:0Vsequential_1/sequential/random_crop/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
Csequential_1/sequential/random_crop/cond/crop_to_bounding_box/SliceSliceYsequential_1/sequential/random_crop/cond/crop_to_bounding_box/control_dependency:output:0Lsequential_1/sequential/random_crop/cond/crop_to_bounding_box/stack:output:0Nsequential_1/sequential/random_crop/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:?????????  ?
1sequential_1/sequential/random_crop/cond/IdentityIdentityLsequential_1/sequential/random_crop/cond/crop_to_bounding_box/Slice:output:0.^sequential_1/sequential/random_crop/cond/NoOp*
T0*/
_output_shapes
:?????????  ?
-sequential_1/sequential/random_crop/cond/NoOpNoOpL^sequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert/AssertN^sequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_1/AssertN^sequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_2/AssertN^sequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_3/AssertI^sequential_1/sequential/random_crop/cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "o
1sequential_1_sequential_random_crop_cond_identity:sequential_1/sequential/random_crop/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :?????????  2?
Ksequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert/AssertKsequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert/Assert2?
Msequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_1/AssertMsequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_1/Assert2?
Msequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_2/AssertMsequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_2/Assert2?
Msequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_3/AssertMsequential_1/sequential/random_crop/cond/crop_to_bounding_box/Assert_3/Assert2?
Hsequential_1/sequential/random_crop/cond/stateful_uniform/RngReadAndSkipHsequential_1/sequential/random_crop/cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????  
?0
?
random_crop_cond_false_7018319 
random_crop_cond_placeholder"
random_crop_cond_placeholder_1"
random_crop_cond_placeholder_2!
random_crop_cond_shape_inputs
random_crop_cond_identityc
random_crop/cond/ShapeShaperandom_crop_cond_shape_inputs*
T0*
_output_shapes
:w
$random_crop/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????y
&random_crop/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????p
&random_crop/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
random_crop/cond/strided_sliceStridedSlicerandom_crop/cond/Shape:output:0-random_crop/cond/strided_slice/stack:output:0/random_crop/cond/strided_slice/stack_1:output:0/random_crop/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
&random_crop/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
(random_crop/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????r
(random_crop/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 random_crop/cond/strided_slice_1StridedSlicerandom_crop/cond/Shape:output:0/random_crop/cond/strided_slice_1/stack:output:01random_crop/cond/strided_slice_1/stack_1:output:01random_crop/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
random_crop/cond/mul/yConst*
_output_shapes
: *
dtype0*
value	B : ?
random_crop/cond/mulMul)random_crop/cond/strided_slice_1:output:0random_crop/cond/mul/y:output:0*
T0*
_output_shapes
: g
random_crop/cond/CastCastrandom_crop/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: _
random_crop/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B?
random_crop/cond/truedivRealDivrandom_crop/cond/Cast:y:0#random_crop/cond/truediv/y:output:0*
T0*
_output_shapes
: m
random_crop/cond/Cast_1Castrandom_crop/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: Z
random_crop/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
random_crop/cond/mul_1Mul'random_crop/cond/strided_slice:output:0!random_crop/cond/mul_1/y:output:0*
T0*
_output_shapes
: k
random_crop/cond/Cast_2Castrandom_crop/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: a
random_crop/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B?
random_crop/cond/truediv_1RealDivrandom_crop/cond/Cast_2:y:0%random_crop/cond/truediv_1/y:output:0*
T0*
_output_shapes
: o
random_crop/cond/Cast_3Castrandom_crop/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
random_crop/cond/MinimumMinimum'random_crop/cond/strided_slice:output:0random_crop/cond/Cast_1:y:0*
T0*
_output_shapes
: ?
random_crop/cond/Minimum_1Minimum)random_crop/cond/strided_slice_1:output:0random_crop/cond/Cast_3:y:0*
T0*
_output_shapes
: ?
random_crop/cond/subSub'random_crop/cond/strided_slice:output:0random_crop/cond/Minimum:z:0*
T0*
_output_shapes
: i
random_crop/cond/Cast_4Castrandom_crop/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: a
random_crop/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
random_crop/cond/truediv_2RealDivrandom_crop/cond/Cast_4:y:0%random_crop/cond/truediv_2/y:output:0*
T0*
_output_shapes
: o
random_crop/cond/Cast_5Castrandom_crop/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
random_crop/cond/sub_1Sub)random_crop/cond/strided_slice_1:output:0random_crop/cond/Minimum_1:z:0*
T0*
_output_shapes
: k
random_crop/cond/Cast_6Castrandom_crop/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: a
random_crop/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
random_crop/cond/truediv_3RealDivrandom_crop/cond/Cast_6:y:0%random_crop/cond/truediv_3/y:output:0*
T0*
_output_shapes
: o
random_crop/cond/Cast_7Castrandom_crop/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: Z
random_crop/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : Z
random_crop/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
random_crop/cond/stackPack!random_crop/cond/stack/0:output:0random_crop/cond/Cast_5:y:0random_crop/cond/Cast_7:y:0!random_crop/cond/stack/3:output:0*
N*
T0*
_output_shapes
:e
random_crop/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????e
random_crop/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
random_crop/cond/stack_1Pack#random_crop/cond/stack_1/0:output:0random_crop/cond/Minimum:z:0random_crop/cond/Minimum_1:z:0#random_crop/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?
random_crop/cond/SliceSlicerandom_crop_cond_shape_inputsrandom_crop/cond/stack:output:0!random_crop/cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"?????????  ?????????m
random_crop/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        ?
&random_crop/cond/resize/ResizeBilinearResizeBilinearrandom_crop/cond/Slice:output:0%random_crop/cond/resize/size:output:0*
T0*/
_output_shapes
:?????????  *
half_pixel_centers(?
random_crop/cond/IdentityIdentity7random_crop/cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:?????????  "?
random_crop_cond_identity"random_crop/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :?????????  :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????  
?
j
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7018705

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_2_layer_call_and_return_conditional_losses_7018805

inputs0
matmul_readvariableop_resource:T
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????T: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
|
,__inference_sequential_layer_call_fn_7018280

inputs
unknown:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_7017012w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????  : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
H__inference_random_crop_layer_call_and_return_conditional_losses_7017264

inputs
cond_input_0:	
identity??cond;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0*
value	B : S
subSubstrided_slice:output:0sub/y:output:0*
T0*
_output_shapes
: h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0*
value	B : Y
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0*
_output_shapes
: P
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : _
GreaterEqualGreaterEqualsub:z:0GreaterEqual/y:output:0*
T0*
_output_shapes
: R
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : e
GreaterEqual_1GreaterEqual	sub_1:z:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
: O
Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z
Rank/packedPackRank/packed/0:output:0GreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:M
All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z{
	All/inputPackAll/input/0:output:0GreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
AllAllAll/input:output:0range:output:0*
_output_shapes
: ?
condIfAll:output:0cond_input_0sub:z:0	sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *%
else_branchR
cond_false_7017128*.
output_shapes
:?????????  *$
then_branchR
cond_true_7017127b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:?????????  m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:?????????  M
NoOpNoOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????  : 2
condcond:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?	
?
D__inference_dense_2_layer_call_and_return_conditional_losses_7017444

inputs0
matmul_readvariableop_resource:T
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????T: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_7017288

inputs!
random_crop_7017281:	!
random_flip_7017284:	
identity??#random_crop/StatefulPartitionedCall?#random_flip/StatefulPartitionedCall?
#random_crop/StatefulPartitionedCallStatefulPartitionedCallinputsrandom_crop_7017281*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_random_crop_layer_call_and_return_conditional_losses_7017264?
#random_flip/StatefulPartitionedCallStatefulPartitionedCall,random_crop/StatefulPartitionedCall:output:0random_flip_7017284*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_7017083?
IdentityIdentity,random_flip/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  ?
NoOpNoOp$^random_crop/StatefulPartitionedCall$^random_flip/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 2J
#random_crop/StatefulPartitionedCall#random_crop/StatefulPartitionedCall2J
#random_flip/StatefulPartitionedCall#random_flip/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?M
?
H__inference_random_flip_layer_call_and_return_conditional_losses_7019225

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity??(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :?
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask?
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????  ?
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:~
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:?
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::?
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :?
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0_stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:??????????
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ?
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:??????????
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:?????????r
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:?????????y
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:?
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:?????????  ?
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:?????????  k
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:??????????
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:?????????  ?
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:?????????  
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*/
_output_shapes
:?????????  q
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????  : 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
j
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7017331

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_random_crop_layer_call_and_return_conditional_losses_7019151

inputs
cond_input_0:	
identity??cond;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0*
value	B : S
subSubstrided_slice:output:0sub/y:output:0*
T0*
_output_shapes
: h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0*
value	B : Y
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0*
_output_shapes
: P
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : _
GreaterEqualGreaterEqualsub:z:0GreaterEqual/y:output:0*
T0*
_output_shapes
: R
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : e
GreaterEqual_1GreaterEqual	sub_1:z:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
: O
Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z
Rank/packedPackRank/packed/0:output:0GreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:M
All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z{
	All/inputPackAll/input/0:output:0GreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
AllAllAll/input:output:0range:output:0*
_output_shapes
: ?
condIfAll:output:0cond_input_0sub:z:0	sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *%
else_branchR
cond_false_7019015*.
output_shapes
:?????????  *$
then_branchR
cond_true_7019014b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:?????????  m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:?????????  M
NoOpNoOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????  : 2
condcond:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
%__inference_signature_wrapper_7018273
sequential_input
unknown:	#
	unknown_0:
	unknown_1:#
	unknown_2:
	unknown_3:
	unknown_4:	?x
	unknown_5:x
	unknown_6:xT
	unknown_7:T
	unknown_8:T

	unknown_9:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsequential_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_7016828o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????  : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:?????????  
*
_user_specified_namesequential_input
?'
?
cond_false_7018849
cond_placeholder
cond_placeholder_1
cond_placeholder_2
cond_shape_inputs
cond_identityK

cond/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:k
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????m
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/Shape:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

cond/mul/yConst*
_output_shapes
: *
dtype0*
value	B : d
cond/mulMulcond/strided_slice_1:output:0cond/mul/y:output:0*
T0*
_output_shapes
: O
	cond/CastCastcond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: S
cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B`
cond/truedivRealDivcond/Cast:y:0cond/truediv/y:output:0*
T0*
_output_shapes
: U
cond/Cast_1Castcond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: N
cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B : f

cond/mul_1Mulcond/strided_slice:output:0cond/mul_1/y:output:0*
T0*
_output_shapes
: S
cond/Cast_2Castcond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Bf
cond/truediv_1RealDivcond/Cast_2:y:0cond/truediv_1/y:output:0*
T0*
_output_shapes
: W
cond/Cast_3Castcond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
cond/MinimumMinimumcond/strided_slice:output:0cond/Cast_1:y:0*
T0*
_output_shapes
: j
cond/Minimum_1Minimumcond/strided_slice_1:output:0cond/Cast_3:y:0*
T0*
_output_shapes
: _
cond/subSubcond/strided_slice:output:0cond/Minimum:z:0*
T0*
_output_shapes
: Q
cond/Cast_4Castcond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_2RealDivcond/Cast_4:y:0cond/truediv_2/y:output:0*
T0*
_output_shapes
: W
cond/Cast_5Castcond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: e

cond/sub_1Subcond/strided_slice_1:output:0cond/Minimum_1:z:0*
T0*
_output_shapes
: S
cond/Cast_6Castcond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_3RealDivcond/Cast_6:y:0cond/truediv_3/y:output:0*
T0*
_output_shapes
: W
cond/Cast_7Castcond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: N
cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : N
cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?

cond/stackPackcond/stack/0:output:0cond/Cast_5:y:0cond/Cast_7:y:0cond/stack/3:output:0*
N*
T0*
_output_shapes
:Y
cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????Y
cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
cond/stack_1Packcond/stack_1/0:output:0cond/Minimum:z:0cond/Minimum_1:z:0cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?

cond/SliceSlicecond_shape_inputscond/stack:output:0cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"?????????  ?????????a
cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        ?
cond/resize/ResizeBilinearResizeBilinearcond/Slice:output:0cond/resize/size:output:0*
T0*/
_output_shapes
:?????????  *
half_pixel_centers(?
cond/IdentityIdentity+cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:?????????  "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :?????????  :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????  
?9
?
)sequential_random_crop_cond_false_7018016+
'sequential_random_crop_cond_placeholder-
)sequential_random_crop_cond_placeholder_1-
)sequential_random_crop_cond_placeholder_2,
(sequential_random_crop_cond_shape_inputs(
$sequential_random_crop_cond_identityy
!sequential/random_crop/cond/ShapeShape(sequential_random_crop_cond_shape_inputs*
T0*
_output_shapes
:?
/sequential/random_crop/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
1sequential/random_crop/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????{
1sequential/random_crop/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential/random_crop/cond/strided_sliceStridedSlice*sequential/random_crop/cond/Shape:output:08sequential/random_crop/cond/strided_slice/stack:output:0:sequential/random_crop/cond/strided_slice/stack_1:output:0:sequential/random_crop/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1sequential/random_crop/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
3sequential/random_crop/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????}
3sequential/random_crop/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+sequential/random_crop/cond/strided_slice_1StridedSlice*sequential/random_crop/cond/Shape:output:0:sequential/random_crop/cond/strided_slice_1/stack:output:0<sequential/random_crop/cond/strided_slice_1/stack_1:output:0<sequential/random_crop/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!sequential/random_crop/cond/mul/yConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential/random_crop/cond/mulMul4sequential/random_crop/cond/strided_slice_1:output:0*sequential/random_crop/cond/mul/y:output:0*
T0*
_output_shapes
: }
 sequential/random_crop/cond/CastCast#sequential/random_crop/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: j
%sequential/random_crop/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B?
#sequential/random_crop/cond/truedivRealDiv$sequential/random_crop/cond/Cast:y:0.sequential/random_crop/cond/truediv/y:output:0*
T0*
_output_shapes
: ?
"sequential/random_crop/cond/Cast_1Cast'sequential/random_crop/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: e
#sequential/random_crop/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
!sequential/random_crop/cond/mul_1Mul2sequential/random_crop/cond/strided_slice:output:0,sequential/random_crop/cond/mul_1/y:output:0*
T0*
_output_shapes
: ?
"sequential/random_crop/cond/Cast_2Cast%sequential/random_crop/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: l
'sequential/random_crop/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B?
%sequential/random_crop/cond/truediv_1RealDiv&sequential/random_crop/cond/Cast_2:y:00sequential/random_crop/cond/truediv_1/y:output:0*
T0*
_output_shapes
: ?
"sequential/random_crop/cond/Cast_3Cast)sequential/random_crop/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
#sequential/random_crop/cond/MinimumMinimum2sequential/random_crop/cond/strided_slice:output:0&sequential/random_crop/cond/Cast_1:y:0*
T0*
_output_shapes
: ?
%sequential/random_crop/cond/Minimum_1Minimum4sequential/random_crop/cond/strided_slice_1:output:0&sequential/random_crop/cond/Cast_3:y:0*
T0*
_output_shapes
: ?
sequential/random_crop/cond/subSub2sequential/random_crop/cond/strided_slice:output:0'sequential/random_crop/cond/Minimum:z:0*
T0*
_output_shapes
: 
"sequential/random_crop/cond/Cast_4Cast#sequential/random_crop/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: l
'sequential/random_crop/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
%sequential/random_crop/cond/truediv_2RealDiv&sequential/random_crop/cond/Cast_4:y:00sequential/random_crop/cond/truediv_2/y:output:0*
T0*
_output_shapes
: ?
"sequential/random_crop/cond/Cast_5Cast)sequential/random_crop/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
!sequential/random_crop/cond/sub_1Sub4sequential/random_crop/cond/strided_slice_1:output:0)sequential/random_crop/cond/Minimum_1:z:0*
T0*
_output_shapes
: ?
"sequential/random_crop/cond/Cast_6Cast%sequential/random_crop/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: l
'sequential/random_crop/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
%sequential/random_crop/cond/truediv_3RealDiv&sequential/random_crop/cond/Cast_6:y:00sequential/random_crop/cond/truediv_3/y:output:0*
T0*
_output_shapes
: ?
"sequential/random_crop/cond/Cast_7Cast)sequential/random_crop/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: e
#sequential/random_crop/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : e
#sequential/random_crop/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
!sequential/random_crop/cond/stackPack,sequential/random_crop/cond/stack/0:output:0&sequential/random_crop/cond/Cast_5:y:0&sequential/random_crop/cond/Cast_7:y:0,sequential/random_crop/cond/stack/3:output:0*
N*
T0*
_output_shapes
:p
%sequential/random_crop/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????p
%sequential/random_crop/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
#sequential/random_crop/cond/stack_1Pack.sequential/random_crop/cond/stack_1/0:output:0'sequential/random_crop/cond/Minimum:z:0)sequential/random_crop/cond/Minimum_1:z:0.sequential/random_crop/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?
!sequential/random_crop/cond/SliceSlice(sequential_random_crop_cond_shape_inputs*sequential/random_crop/cond/stack:output:0,sequential/random_crop/cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"?????????  ?????????x
'sequential/random_crop/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        ?
1sequential/random_crop/cond/resize/ResizeBilinearResizeBilinear*sequential/random_crop/cond/Slice:output:00sequential/random_crop/cond/resize/size:output:0*
T0*/
_output_shapes
:?????????  *
half_pixel_centers(?
$sequential/random_crop/cond/IdentityIdentityBsequential/random_crop/cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:?????????  "U
$sequential_random_crop_cond_identity-sequential/random_crop/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :?????????  :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????  
?
?
,__inference_sequential_layer_call_fn_7017304
random_crop_input
unknown:	
	unknown_0:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallrandom_crop_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_7017288w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
/
_output_shapes
:?????????  
+
_user_specified_namerandom_crop_input
?
d
H__inference_random_flip_layer_call_and_return_conditional_losses_7019167

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
E
)__inference_flatten_layer_call_fn_7018740

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_7017398a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?g
?
cond_true_7018848;
-cond_stateful_uniform_rngreadandskip_resource:	
cond_add_sub
cond_add_1_sub_1*
&cond_crop_to_bounding_box_shape_inputs
cond_identity??'cond/crop_to_bounding_box/Assert/Assert?)cond/crop_to_bounding_box/Assert_1/Assert?)cond/crop_to_bounding_box/Assert_2/Assert?)cond/crop_to_bounding_box/Assert_3/Assert?$cond/stateful_uniform/RngReadAndSkipe
cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:[
cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : _
cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :????e
cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
cond/stateful_uniform/ProdProd$cond/stateful_uniform/shape:output:0$cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: ^
cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :y
cond/stateful_uniform/Cast_1Cast#cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$cond/stateful_uniform/RngReadAndSkipRngReadAndSkip-cond_stateful_uniform_rngreadandskip_resource%cond/stateful_uniform/Cast/x:output:0 cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:s
)cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#cond/stateful_uniform/strided_sliceStridedSlice,cond/stateful_uniform/RngReadAndSkip:value:02cond/stateful_uniform/strided_slice/stack:output:04cond/stateful_uniform/strided_slice/stack_1:output:04cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
cond/stateful_uniform/BitcastBitcast,cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0u
+cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%cond/stateful_uniform/strided_slice_1StridedSlice,cond/stateful_uniform/RngReadAndSkip:value:04cond/stateful_uniform/strided_slice_1/stack:output:06cond/stateful_uniform/strided_slice_1/stack_1:output:06cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
cond/stateful_uniform/Bitcast_1Bitcast.cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0[
cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
cond/stateful_uniformStatelessRandomUniformIntV2$cond/stateful_uniform/shape:output:0(cond/stateful_uniform/Bitcast_1:output:0&cond/stateful_uniform/Bitcast:output:0"cond/stateful_uniform/alg:output:0"cond/stateful_uniform/min:output:0"cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0b
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/stateful_uniform:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :U
cond/addAddV2cond_add_subcond/add/y:output:0*
T0*
_output_shapes
: `
cond/modFloorModcond/strided_slice:output:0cond/add:z:0*
T0*
_output_shapes
: d
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/stateful_uniform:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :]

cond/add_1AddV2cond_add_1_sub_1cond/add_1/y:output:0*
T0*
_output_shapes
: f

cond/mod_1FloorModcond/strided_slice_1:output:0cond/add_1:z:0*
T0*
_output_shapes
: u
cond/crop_to_bounding_box/ShapeShape&cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/unstackUnpack(cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numj
(cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
&cond/crop_to_bounding_box/GreaterEqualGreaterEqualcond/mod_1:z:01cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
&cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
.cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
'cond/crop_to_bounding_box/Assert/AssertAssert*cond/crop_to_bounding_box/GreaterEqual:z:07cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 l
*cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
(cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcond/mod:z:03cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
0cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
)cond/crop_to_bounding_box/Assert_1/AssertAssert,cond/crop_to_bounding_box/GreaterEqual_1:z:09cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0(^cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 a
cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/addAddV2(cond/crop_to_bounding_box/add/x:output:0cond/mod_1:z:0*
T0*
_output_shapes
: g
%cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
#cond/crop_to_bounding_box/LessEqual	LessEqual!cond/crop_to_bounding_box/add:z:0.cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_2/AssertAssert'cond/crop_to_bounding_box/LessEqual:z:09cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 c
!cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/add_1AddV2*cond/crop_to_bounding_box/add_1/x:output:0cond/mod:z:0*
T0*
_output_shapes
: i
'cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
%cond/crop_to_bounding_box/LessEqual_1	LessEqual#cond/crop_to_bounding_box/add_1:z:00cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_3/AssertAssert)cond/crop_to_bounding_box/LessEqual_1:z:09cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
,cond/crop_to_bounding_box/control_dependencyIdentity&cond_crop_to_bounding_box_shape_inputs(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????  c
!cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : c
!cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/stackPack*cond/crop_to_bounding_box/stack/0:output:0cond/mod:z:0cond/mod_1:z:0*cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/Shape_1Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:w
-cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'cond/crop_to_bounding_box/strided_sliceStridedSlice*cond/crop_to_bounding_box/Shape_1:output:06cond/crop_to_bounding_box/strided_slice/stack:output:08cond/crop_to_bounding_box/strided_slice/stack_1:output:08cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!cond/crop_to_bounding_box/Shape_2Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:y
/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)cond/crop_to_bounding_box/strided_slice_1StridedSlice*cond/crop_to_bounding_box/Shape_2:output:08cond/crop_to_bounding_box/strided_slice_1/stack:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : e
#cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : ?
!cond/crop_to_bounding_box/stack_1Pack0cond/crop_to_bounding_box/strided_slice:output:0,cond/crop_to_bounding_box/stack_1/1:output:0,cond/crop_to_bounding_box/stack_1/2:output:02cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
cond/crop_to_bounding_box/SliceSlice5cond/crop_to_bounding_box/control_dependency:output:0(cond/crop_to_bounding_box/stack:output:0*cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:?????????  ?
cond/IdentityIdentity(cond/crop_to_bounding_box/Slice:output:0
^cond/NoOp*
T0*/
_output_shapes
:?????????  ?
	cond/NoOpNoOp(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert%^cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :?????????  2R
'cond/crop_to_bounding_box/Assert/Assert'cond/crop_to_bounding_box/Assert/Assert2V
)cond/crop_to_bounding_box/Assert_1/Assert)cond/crop_to_bounding_box/Assert_1/Assert2V
)cond/crop_to_bounding_box/Assert_2/Assert)cond/crop_to_bounding_box/Assert_2/Assert2V
)cond/crop_to_bounding_box/Assert_3/Assert)cond/crop_to_bounding_box/Assert_3/Assert2L
$cond/stateful_uniform/RngReadAndSkip$cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????  
?
?
'__inference_dense_layer_call_fn_7018755

inputs
unknown:	?x
	unknown_0:x
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_7017411o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????x`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_layer_call_fn_7017017
random_crop_input
unknown:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallrandom_crop_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_7017012w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????  : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
/
_output_shapes
:?????????  
+
_user_specified_namerandom_crop_input
?g
?
cond_true_7019014;
-cond_stateful_uniform_rngreadandskip_resource:	
cond_add_sub
cond_add_1_sub_1*
&cond_crop_to_bounding_box_shape_inputs
cond_identity??'cond/crop_to_bounding_box/Assert/Assert?)cond/crop_to_bounding_box/Assert_1/Assert?)cond/crop_to_bounding_box/Assert_2/Assert?)cond/crop_to_bounding_box/Assert_3/Assert?$cond/stateful_uniform/RngReadAndSkipe
cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:[
cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : _
cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :????e
cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
cond/stateful_uniform/ProdProd$cond/stateful_uniform/shape:output:0$cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: ^
cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :y
cond/stateful_uniform/Cast_1Cast#cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$cond/stateful_uniform/RngReadAndSkipRngReadAndSkip-cond_stateful_uniform_rngreadandskip_resource%cond/stateful_uniform/Cast/x:output:0 cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:s
)cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#cond/stateful_uniform/strided_sliceStridedSlice,cond/stateful_uniform/RngReadAndSkip:value:02cond/stateful_uniform/strided_slice/stack:output:04cond/stateful_uniform/strided_slice/stack_1:output:04cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
cond/stateful_uniform/BitcastBitcast,cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0u
+cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%cond/stateful_uniform/strided_slice_1StridedSlice,cond/stateful_uniform/RngReadAndSkip:value:04cond/stateful_uniform/strided_slice_1/stack:output:06cond/stateful_uniform/strided_slice_1/stack_1:output:06cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
cond/stateful_uniform/Bitcast_1Bitcast.cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0[
cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
cond/stateful_uniformStatelessRandomUniformIntV2$cond/stateful_uniform/shape:output:0(cond/stateful_uniform/Bitcast_1:output:0&cond/stateful_uniform/Bitcast:output:0"cond/stateful_uniform/alg:output:0"cond/stateful_uniform/min:output:0"cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0b
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/stateful_uniform:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :U
cond/addAddV2cond_add_subcond/add/y:output:0*
T0*
_output_shapes
: `
cond/modFloorModcond/strided_slice:output:0cond/add:z:0*
T0*
_output_shapes
: d
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/stateful_uniform:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :]

cond/add_1AddV2cond_add_1_sub_1cond/add_1/y:output:0*
T0*
_output_shapes
: f

cond/mod_1FloorModcond/strided_slice_1:output:0cond/add_1:z:0*
T0*
_output_shapes
: u
cond/crop_to_bounding_box/ShapeShape&cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/unstackUnpack(cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numj
(cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
&cond/crop_to_bounding_box/GreaterEqualGreaterEqualcond/mod_1:z:01cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
&cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
.cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
'cond/crop_to_bounding_box/Assert/AssertAssert*cond/crop_to_bounding_box/GreaterEqual:z:07cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 l
*cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
(cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcond/mod:z:03cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
0cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
)cond/crop_to_bounding_box/Assert_1/AssertAssert,cond/crop_to_bounding_box/GreaterEqual_1:z:09cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0(^cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 a
cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/addAddV2(cond/crop_to_bounding_box/add/x:output:0cond/mod_1:z:0*
T0*
_output_shapes
: g
%cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
#cond/crop_to_bounding_box/LessEqual	LessEqual!cond/crop_to_bounding_box/add:z:0.cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_2/AssertAssert'cond/crop_to_bounding_box/LessEqual:z:09cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 c
!cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/add_1AddV2*cond/crop_to_bounding_box/add_1/x:output:0cond/mod:z:0*
T0*
_output_shapes
: i
'cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
%cond/crop_to_bounding_box/LessEqual_1	LessEqual#cond/crop_to_bounding_box/add_1:z:00cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_3/AssertAssert)cond/crop_to_bounding_box/LessEqual_1:z:09cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
,cond/crop_to_bounding_box/control_dependencyIdentity&cond_crop_to_bounding_box_shape_inputs(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????  c
!cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : c
!cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/stackPack*cond/crop_to_bounding_box/stack/0:output:0cond/mod:z:0cond/mod_1:z:0*cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/Shape_1Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:w
-cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'cond/crop_to_bounding_box/strided_sliceStridedSlice*cond/crop_to_bounding_box/Shape_1:output:06cond/crop_to_bounding_box/strided_slice/stack:output:08cond/crop_to_bounding_box/strided_slice/stack_1:output:08cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!cond/crop_to_bounding_box/Shape_2Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:y
/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)cond/crop_to_bounding_box/strided_slice_1StridedSlice*cond/crop_to_bounding_box/Shape_2:output:08cond/crop_to_bounding_box/strided_slice_1/stack:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : e
#cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : ?
!cond/crop_to_bounding_box/stack_1Pack0cond/crop_to_bounding_box/strided_slice:output:0,cond/crop_to_bounding_box/stack_1/1:output:0,cond/crop_to_bounding_box/stack_1/2:output:02cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
cond/crop_to_bounding_box/SliceSlice5cond/crop_to_bounding_box/control_dependency:output:0(cond/crop_to_bounding_box/stack:output:0*cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:?????????  ?
cond/IdentityIdentity(cond/crop_to_bounding_box/Slice:output:0
^cond/NoOp*
T0*/
_output_shapes
:?????????  ?
	cond/NoOpNoOp(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert%^cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :?????????  2R
'cond/crop_to_bounding_box/Assert/Assert'cond/crop_to_bounding_box/Assert/Assert2V
)cond/crop_to_bounding_box/Assert_1/Assert)cond/crop_to_bounding_box/Assert_1/Assert2V
)cond/crop_to_bounding_box/Assert_2/Assert)cond/crop_to_bounding_box/Assert_2/Assert2V
)cond/crop_to_bounding_box/Assert_3/Assert)cond/crop_to_bounding_box/Assert_3/Assert2L
$cond/stateful_uniform/RngReadAndSkip$cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????  
?
l
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_7017343

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_random_crop_layer_call_and_return_conditional_losses_7017001

inputs
cond_input_0:	
identity??cond;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0*
value	B : S
subSubstrided_slice:output:0sub/y:output:0*
T0*
_output_shapes
: h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0*
value	B : Y
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0*
_output_shapes
: P
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : _
GreaterEqualGreaterEqualsub:z:0GreaterEqual/y:output:0*
T0*
_output_shapes
: R
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : e
GreaterEqual_1GreaterEqual	sub_1:z:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
: O
Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z 
Rank/packedPackRank/packed/0:output:0GreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:M
All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z {
	All/inputPackAll/input/0:output:0GreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
AllAllAll/input:output:0range:output:0*
_output_shapes
: ?
condIfAll:output:0cond_input_0sub:z:0	sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *%
else_branchR
cond_false_7016865*.
output_shapes
:?????????  *$
then_branchR
cond_true_7016864b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:?????????  m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:?????????  M
NoOpNoOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????  : 2
condcond:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?z
?
random_crop_cond_true_7018318G
9random_crop_cond_stateful_uniform_rngreadandskip_resource:	(
$random_crop_cond_add_random_crop_sub,
(random_crop_cond_add_1_random_crop_sub_16
2random_crop_cond_crop_to_bounding_box_shape_inputs
random_crop_cond_identity??3random_crop/cond/crop_to_bounding_box/Assert/Assert?5random_crop/cond/crop_to_bounding_box/Assert_1/Assert?5random_crop/cond/crop_to_bounding_box/Assert_2/Assert?5random_crop/cond/crop_to_bounding_box/Assert_3/Assert?0random_crop/cond/stateful_uniform/RngReadAndSkipq
'random_crop/cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:g
%random_crop/cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : k
%random_crop/cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :????q
'random_crop/cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
&random_crop/cond/stateful_uniform/ProdProd0random_crop/cond/stateful_uniform/shape:output:00random_crop/cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: j
(random_crop/cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
(random_crop/cond/stateful_uniform/Cast_1Cast/random_crop/cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
0random_crop/cond/stateful_uniform/RngReadAndSkipRngReadAndSkip9random_crop_cond_stateful_uniform_rngreadandskip_resource1random_crop/cond/stateful_uniform/Cast/x:output:0,random_crop/cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:
5random_crop/cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7random_crop/cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7random_crop/cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/random_crop/cond/stateful_uniform/strided_sliceStridedSlice8random_crop/cond/stateful_uniform/RngReadAndSkip:value:0>random_crop/cond/stateful_uniform/strided_slice/stack:output:0@random_crop/cond/stateful_uniform/strided_slice/stack_1:output:0@random_crop/cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
)random_crop/cond/stateful_uniform/BitcastBitcast8random_crop/cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0?
7random_crop/cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9random_crop/cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9random_crop/cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1random_crop/cond/stateful_uniform/strided_slice_1StridedSlice8random_crop/cond/stateful_uniform/RngReadAndSkip:value:0@random_crop/cond/stateful_uniform/strided_slice_1/stack:output:0Brandom_crop/cond/stateful_uniform/strided_slice_1/stack_1:output:0Brandom_crop/cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
+random_crop/cond/stateful_uniform/Bitcast_1Bitcast:random_crop/cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0g
%random_crop/cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
!random_crop/cond/stateful_uniformStatelessRandomUniformIntV20random_crop/cond/stateful_uniform/shape:output:04random_crop/cond/stateful_uniform/Bitcast_1:output:02random_crop/cond/stateful_uniform/Bitcast:output:0.random_crop/cond/stateful_uniform/alg:output:0.random_crop/cond/stateful_uniform/min:output:0.random_crop/cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0n
$random_crop/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&random_crop/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&random_crop/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
random_crop/cond/strided_sliceStridedSlice*random_crop/cond/stateful_uniform:output:0-random_crop/cond/strided_slice/stack:output:0/random_crop/cond/strided_slice/stack_1:output:0/random_crop/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
random_crop/cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
random_crop/cond/addAddV2$random_crop_cond_add_random_crop_subrandom_crop/cond/add/y:output:0*
T0*
_output_shapes
: ?
random_crop/cond/modFloorMod'random_crop/cond/strided_slice:output:0random_crop/cond/add:z:0*
T0*
_output_shapes
: p
&random_crop/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(random_crop/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(random_crop/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 random_crop/cond/strided_slice_1StridedSlice*random_crop/cond/stateful_uniform:output:0/random_crop/cond/strided_slice_1/stack:output:01random_crop/cond/strided_slice_1/stack_1:output:01random_crop/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
random_crop/cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
random_crop/cond/add_1AddV2(random_crop_cond_add_1_random_crop_sub_1!random_crop/cond/add_1/y:output:0*
T0*
_output_shapes
: ?
random_crop/cond/mod_1FloorMod)random_crop/cond/strided_slice_1:output:0random_crop/cond/add_1:z:0*
T0*
_output_shapes
: ?
+random_crop/cond/crop_to_bounding_box/ShapeShape2random_crop_cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:?
-random_crop/cond/crop_to_bounding_box/unstackUnpack4random_crop/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numv
4random_crop/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
2random_crop/cond/crop_to_bounding_box/GreaterEqualGreaterEqualrandom_crop/cond/mod_1:z:0=random_crop/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
2random_crop/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
:random_crop/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
3random_crop/cond/crop_to_bounding_box/Assert/AssertAssert6random_crop/cond/crop_to_bounding_box/GreaterEqual:z:0Crandom_crop/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 x
6random_crop/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
4random_crop/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualrandom_crop/cond/mod:z:0?random_crop/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
4random_crop/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
<random_crop/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
5random_crop/cond/crop_to_bounding_box/Assert_1/AssertAssert8random_crop/cond/crop_to_bounding_box/GreaterEqual_1:z:0Erandom_crop/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:04^random_crop/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 m
+random_crop/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B : ?
)random_crop/cond/crop_to_bounding_box/addAddV24random_crop/cond/crop_to_bounding_box/add/x:output:0random_crop/cond/mod_1:z:0*
T0*
_output_shapes
: s
1random_crop/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
/random_crop/cond/crop_to_bounding_box/LessEqual	LessEqual-random_crop/cond/crop_to_bounding_box/add:z:0:random_crop/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
4random_crop/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
<random_crop/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
5random_crop/cond/crop_to_bounding_box/Assert_2/AssertAssert3random_crop/cond/crop_to_bounding_box/LessEqual:z:0Erandom_crop/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:06^random_crop/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 o
-random_crop/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : ?
+random_crop/cond/crop_to_bounding_box/add_1AddV26random_crop/cond/crop_to_bounding_box/add_1/x:output:0random_crop/cond/mod:z:0*
T0*
_output_shapes
: u
3random_crop/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
1random_crop/cond/crop_to_bounding_box/LessEqual_1	LessEqual/random_crop/cond/crop_to_bounding_box/add_1:z:0<random_crop/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
4random_crop/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
<random_crop/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
5random_crop/cond/crop_to_bounding_box/Assert_3/AssertAssert5random_crop/cond/crop_to_bounding_box/LessEqual_1:z:0Erandom_crop/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:06^random_crop/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
8random_crop/cond/crop_to_bounding_box/control_dependencyIdentity2random_crop_cond_crop_to_bounding_box_shape_inputs4^random_crop/cond/crop_to_bounding_box/Assert/Assert6^random_crop/cond/crop_to_bounding_box/Assert_1/Assert6^random_crop/cond/crop_to_bounding_box/Assert_2/Assert6^random_crop/cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????  o
-random_crop/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : o
-random_crop/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
+random_crop/cond/crop_to_bounding_box/stackPack6random_crop/cond/crop_to_bounding_box/stack/0:output:0random_crop/cond/mod:z:0random_crop/cond/mod_1:z:06random_crop/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
-random_crop/cond/crop_to_bounding_box/Shape_1ShapeArandom_crop/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
9random_crop/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
;random_crop/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;random_crop/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3random_crop/cond/crop_to_bounding_box/strided_sliceStridedSlice6random_crop/cond/crop_to_bounding_box/Shape_1:output:0Brandom_crop/cond/crop_to_bounding_box/strided_slice/stack:output:0Drandom_crop/cond/crop_to_bounding_box/strided_slice/stack_1:output:0Drandom_crop/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
-random_crop/cond/crop_to_bounding_box/Shape_2ShapeArandom_crop/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
;random_crop/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
=random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5random_crop/cond/crop_to_bounding_box/strided_slice_1StridedSlice6random_crop/cond/crop_to_bounding_box/Shape_2:output:0Drandom_crop/cond/crop_to_bounding_box/strided_slice_1/stack:output:0Frandom_crop/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0Frandom_crop/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/random_crop/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : q
/random_crop/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : ?
-random_crop/cond/crop_to_bounding_box/stack_1Pack<random_crop/cond/crop_to_bounding_box/strided_slice:output:08random_crop/cond/crop_to_bounding_box/stack_1/1:output:08random_crop/cond/crop_to_bounding_box/stack_1/2:output:0>random_crop/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
+random_crop/cond/crop_to_bounding_box/SliceSliceArandom_crop/cond/crop_to_bounding_box/control_dependency:output:04random_crop/cond/crop_to_bounding_box/stack:output:06random_crop/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:?????????  ?
random_crop/cond/IdentityIdentity4random_crop/cond/crop_to_bounding_box/Slice:output:0^random_crop/cond/NoOp*
T0*/
_output_shapes
:?????????  ?
random_crop/cond/NoOpNoOp4^random_crop/cond/crop_to_bounding_box/Assert/Assert6^random_crop/cond/crop_to_bounding_box/Assert_1/Assert6^random_crop/cond/crop_to_bounding_box/Assert_2/Assert6^random_crop/cond/crop_to_bounding_box/Assert_3/Assert1^random_crop/cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "?
random_crop_cond_identity"random_crop/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :?????????  2j
3random_crop/cond/crop_to_bounding_box/Assert/Assert3random_crop/cond/crop_to_bounding_box/Assert/Assert2n
5random_crop/cond/crop_to_bounding_box/Assert_1/Assert5random_crop/cond/crop_to_bounding_box/Assert_1/Assert2n
5random_crop/cond/crop_to_bounding_box/Assert_2/Assert5random_crop/cond/crop_to_bounding_box/Assert_2/Assert2n
5random_crop/cond/crop_to_bounding_box/Assert_3/Assert5random_crop/cond/crop_to_bounding_box/Assert_3/Assert2d
0random_crop/cond/stateful_uniform/RngReadAndSkip0random_crop/cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????  
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_7017398

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
Q
5__inference_average_pooling2d_1_layer_call_fn_7018730

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_7017343?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?'
?
cond_false_7016865
cond_placeholder
cond_placeholder_1
cond_placeholder_2
cond_shape_inputs
cond_identityK

cond/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:k
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????m
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/Shape:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

cond/mul/yConst*
_output_shapes
: *
dtype0*
value	B : d
cond/mulMulcond/strided_slice_1:output:0cond/mul/y:output:0*
T0*
_output_shapes
: O
	cond/CastCastcond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: S
cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B`
cond/truedivRealDivcond/Cast:y:0cond/truediv/y:output:0*
T0*
_output_shapes
: U
cond/Cast_1Castcond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: N
cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B : f

cond/mul_1Mulcond/strided_slice:output:0cond/mul_1/y:output:0*
T0*
_output_shapes
: S
cond/Cast_2Castcond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Bf
cond/truediv_1RealDivcond/Cast_2:y:0cond/truediv_1/y:output:0*
T0*
_output_shapes
: W
cond/Cast_3Castcond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
cond/MinimumMinimumcond/strided_slice:output:0cond/Cast_1:y:0*
T0*
_output_shapes
: j
cond/Minimum_1Minimumcond/strided_slice_1:output:0cond/Cast_3:y:0*
T0*
_output_shapes
: _
cond/subSubcond/strided_slice:output:0cond/Minimum:z:0*
T0*
_output_shapes
: Q
cond/Cast_4Castcond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_2RealDivcond/Cast_4:y:0cond/truediv_2/y:output:0*
T0*
_output_shapes
: W
cond/Cast_5Castcond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: e

cond/sub_1Subcond/strided_slice_1:output:0cond/Minimum_1:z:0*
T0*
_output_shapes
: S
cond/Cast_6Castcond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_3RealDivcond/Cast_6:y:0cond/truediv_3/y:output:0*
T0*
_output_shapes
: W
cond/Cast_7Castcond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: N
cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : N
cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?

cond/stackPackcond/stack/0:output:0cond/Cast_5:y:0cond/Cast_7:y:0cond/stack/3:output:0*
N*
T0*
_output_shapes
:Y
cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????Y
cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
cond/stack_1Packcond/stack_1/0:output:0cond/Minimum:z:0cond/Minimum_1:z:0cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?

cond/SliceSlicecond_shape_inputscond/stack:output:0cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"?????????  ?????????a
cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        ?
cond/resize/ResizeBilinearResizeBilinearcond/Slice:output:0cond/resize/size:output:0*
T0*/
_output_shapes
:?????????  *
half_pixel_centers(?
cond/IdentityIdentity+cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:?????????  "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :?????????  :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????  
?
d
H__inference_random_flip_layer_call_and_return_conditional_losses_7017009

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?9
?
)sequential_random_crop_cond_false_7017812+
'sequential_random_crop_cond_placeholder-
)sequential_random_crop_cond_placeholder_1-
)sequential_random_crop_cond_placeholder_2,
(sequential_random_crop_cond_shape_inputs(
$sequential_random_crop_cond_identityy
!sequential/random_crop/cond/ShapeShape(sequential_random_crop_cond_shape_inputs*
T0*
_output_shapes
:?
/sequential/random_crop/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
1sequential/random_crop/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????{
1sequential/random_crop/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential/random_crop/cond/strided_sliceStridedSlice*sequential/random_crop/cond/Shape:output:08sequential/random_crop/cond/strided_slice/stack:output:0:sequential/random_crop/cond/strided_slice/stack_1:output:0:sequential/random_crop/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1sequential/random_crop/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
3sequential/random_crop/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????}
3sequential/random_crop/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+sequential/random_crop/cond/strided_slice_1StridedSlice*sequential/random_crop/cond/Shape:output:0:sequential/random_crop/cond/strided_slice_1/stack:output:0<sequential/random_crop/cond/strided_slice_1/stack_1:output:0<sequential/random_crop/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!sequential/random_crop/cond/mul/yConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential/random_crop/cond/mulMul4sequential/random_crop/cond/strided_slice_1:output:0*sequential/random_crop/cond/mul/y:output:0*
T0*
_output_shapes
: }
 sequential/random_crop/cond/CastCast#sequential/random_crop/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: j
%sequential/random_crop/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B?
#sequential/random_crop/cond/truedivRealDiv$sequential/random_crop/cond/Cast:y:0.sequential/random_crop/cond/truediv/y:output:0*
T0*
_output_shapes
: ?
"sequential/random_crop/cond/Cast_1Cast'sequential/random_crop/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: e
#sequential/random_crop/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
!sequential/random_crop/cond/mul_1Mul2sequential/random_crop/cond/strided_slice:output:0,sequential/random_crop/cond/mul_1/y:output:0*
T0*
_output_shapes
: ?
"sequential/random_crop/cond/Cast_2Cast%sequential/random_crop/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: l
'sequential/random_crop/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B?
%sequential/random_crop/cond/truediv_1RealDiv&sequential/random_crop/cond/Cast_2:y:00sequential/random_crop/cond/truediv_1/y:output:0*
T0*
_output_shapes
: ?
"sequential/random_crop/cond/Cast_3Cast)sequential/random_crop/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
#sequential/random_crop/cond/MinimumMinimum2sequential/random_crop/cond/strided_slice:output:0&sequential/random_crop/cond/Cast_1:y:0*
T0*
_output_shapes
: ?
%sequential/random_crop/cond/Minimum_1Minimum4sequential/random_crop/cond/strided_slice_1:output:0&sequential/random_crop/cond/Cast_3:y:0*
T0*
_output_shapes
: ?
sequential/random_crop/cond/subSub2sequential/random_crop/cond/strided_slice:output:0'sequential/random_crop/cond/Minimum:z:0*
T0*
_output_shapes
: 
"sequential/random_crop/cond/Cast_4Cast#sequential/random_crop/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: l
'sequential/random_crop/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
%sequential/random_crop/cond/truediv_2RealDiv&sequential/random_crop/cond/Cast_4:y:00sequential/random_crop/cond/truediv_2/y:output:0*
T0*
_output_shapes
: ?
"sequential/random_crop/cond/Cast_5Cast)sequential/random_crop/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
!sequential/random_crop/cond/sub_1Sub4sequential/random_crop/cond/strided_slice_1:output:0)sequential/random_crop/cond/Minimum_1:z:0*
T0*
_output_shapes
: ?
"sequential/random_crop/cond/Cast_6Cast%sequential/random_crop/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: l
'sequential/random_crop/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
%sequential/random_crop/cond/truediv_3RealDiv&sequential/random_crop/cond/Cast_6:y:00sequential/random_crop/cond/truediv_3/y:output:0*
T0*
_output_shapes
: ?
"sequential/random_crop/cond/Cast_7Cast)sequential/random_crop/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: e
#sequential/random_crop/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : e
#sequential/random_crop/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
!sequential/random_crop/cond/stackPack,sequential/random_crop/cond/stack/0:output:0&sequential/random_crop/cond/Cast_5:y:0&sequential/random_crop/cond/Cast_7:y:0,sequential/random_crop/cond/stack/3:output:0*
N*
T0*
_output_shapes
:p
%sequential/random_crop/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????p
%sequential/random_crop/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
#sequential/random_crop/cond/stack_1Pack.sequential/random_crop/cond/stack_1/0:output:0'sequential/random_crop/cond/Minimum:z:0)sequential/random_crop/cond/Minimum_1:z:0.sequential/random_crop/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?
!sequential/random_crop/cond/SliceSlice(sequential_random_crop_cond_shape_inputs*sequential/random_crop/cond/stack:output:0,sequential/random_crop/cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"?????????  ?????????x
'sequential/random_crop/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        ?
1sequential/random_crop/cond/resize/ResizeBilinearResizeBilinear*sequential/random_crop/cond/Slice:output:00sequential/random_crop/cond/resize/size:output:0*
T0*/
_output_shapes
:?????????  *
half_pixel_centers(?
$sequential/random_crop/cond/IdentityIdentityBsequential/random_crop/cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:?????????  "U
$sequential_random_crop_cond_identity-sequential/random_crop/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :?????????  :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????  
?D
?
6sequential_1_sequential_random_crop_cond_false_70166548
4sequential_1_sequential_random_crop_cond_placeholder:
6sequential_1_sequential_random_crop_cond_placeholder_1:
6sequential_1_sequential_random_crop_cond_placeholder_2C
?sequential_1_sequential_random_crop_cond_shape_sequential_input5
1sequential_1_sequential_random_crop_cond_identity?
.sequential_1/sequential/random_crop/cond/ShapeShape?sequential_1_sequential_random_crop_cond_shape_sequential_input*
T0*
_output_shapes
:?
<sequential_1/sequential/random_crop/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
>sequential_1/sequential/random_crop/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
>sequential_1/sequential/random_crop/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential_1/sequential/random_crop/cond/strided_sliceStridedSlice7sequential_1/sequential/random_crop/cond/Shape:output:0Esequential_1/sequential/random_crop/cond/strided_slice/stack:output:0Gsequential_1/sequential/random_crop/cond/strided_slice/stack_1:output:0Gsequential_1/sequential/random_crop/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
>sequential_1/sequential/random_crop/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
@sequential_1/sequential/random_crop/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
@sequential_1/sequential/random_crop/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_1/sequential/random_crop/cond/strided_slice_1StridedSlice7sequential_1/sequential/random_crop/cond/Shape:output:0Gsequential_1/sequential/random_crop/cond/strided_slice_1/stack:output:0Isequential_1/sequential/random_crop/cond/strided_slice_1/stack_1:output:0Isequential_1/sequential/random_crop/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.sequential_1/sequential/random_crop/cond/mul/yConst*
_output_shapes
: *
dtype0*
value	B : ?
,sequential_1/sequential/random_crop/cond/mulMulAsequential_1/sequential/random_crop/cond/strided_slice_1:output:07sequential_1/sequential/random_crop/cond/mul/y:output:0*
T0*
_output_shapes
: ?
-sequential_1/sequential/random_crop/cond/CastCast0sequential_1/sequential/random_crop/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: w
2sequential_1/sequential/random_crop/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B?
0sequential_1/sequential/random_crop/cond/truedivRealDiv1sequential_1/sequential/random_crop/cond/Cast:y:0;sequential_1/sequential/random_crop/cond/truediv/y:output:0*
T0*
_output_shapes
: ?
/sequential_1/sequential/random_crop/cond/Cast_1Cast4sequential_1/sequential/random_crop/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: r
0sequential_1/sequential/random_crop/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
.sequential_1/sequential/random_crop/cond/mul_1Mul?sequential_1/sequential/random_crop/cond/strided_slice:output:09sequential_1/sequential/random_crop/cond/mul_1/y:output:0*
T0*
_output_shapes
: ?
/sequential_1/sequential/random_crop/cond/Cast_2Cast2sequential_1/sequential/random_crop/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: y
4sequential_1/sequential/random_crop/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B?
2sequential_1/sequential/random_crop/cond/truediv_1RealDiv3sequential_1/sequential/random_crop/cond/Cast_2:y:0=sequential_1/sequential/random_crop/cond/truediv_1/y:output:0*
T0*
_output_shapes
: ?
/sequential_1/sequential/random_crop/cond/Cast_3Cast6sequential_1/sequential/random_crop/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
0sequential_1/sequential/random_crop/cond/MinimumMinimum?sequential_1/sequential/random_crop/cond/strided_slice:output:03sequential_1/sequential/random_crop/cond/Cast_1:y:0*
T0*
_output_shapes
: ?
2sequential_1/sequential/random_crop/cond/Minimum_1MinimumAsequential_1/sequential/random_crop/cond/strided_slice_1:output:03sequential_1/sequential/random_crop/cond/Cast_3:y:0*
T0*
_output_shapes
: ?
,sequential_1/sequential/random_crop/cond/subSub?sequential_1/sequential/random_crop/cond/strided_slice:output:04sequential_1/sequential/random_crop/cond/Minimum:z:0*
T0*
_output_shapes
: ?
/sequential_1/sequential/random_crop/cond/Cast_4Cast0sequential_1/sequential/random_crop/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: y
4sequential_1/sequential/random_crop/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
2sequential_1/sequential/random_crop/cond/truediv_2RealDiv3sequential_1/sequential/random_crop/cond/Cast_4:y:0=sequential_1/sequential/random_crop/cond/truediv_2/y:output:0*
T0*
_output_shapes
: ?
/sequential_1/sequential/random_crop/cond/Cast_5Cast6sequential_1/sequential/random_crop/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
.sequential_1/sequential/random_crop/cond/sub_1SubAsequential_1/sequential/random_crop/cond/strided_slice_1:output:06sequential_1/sequential/random_crop/cond/Minimum_1:z:0*
T0*
_output_shapes
: ?
/sequential_1/sequential/random_crop/cond/Cast_6Cast2sequential_1/sequential/random_crop/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: y
4sequential_1/sequential/random_crop/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
2sequential_1/sequential/random_crop/cond/truediv_3RealDiv3sequential_1/sequential/random_crop/cond/Cast_6:y:0=sequential_1/sequential/random_crop/cond/truediv_3/y:output:0*
T0*
_output_shapes
: ?
/sequential_1/sequential/random_crop/cond/Cast_7Cast6sequential_1/sequential/random_crop/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: r
0sequential_1/sequential/random_crop/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : r
0sequential_1/sequential/random_crop/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
.sequential_1/sequential/random_crop/cond/stackPack9sequential_1/sequential/random_crop/cond/stack/0:output:03sequential_1/sequential/random_crop/cond/Cast_5:y:03sequential_1/sequential/random_crop/cond/Cast_7:y:09sequential_1/sequential/random_crop/cond/stack/3:output:0*
N*
T0*
_output_shapes
:}
2sequential_1/sequential/random_crop/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????}
2sequential_1/sequential/random_crop/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
0sequential_1/sequential/random_crop/cond/stack_1Pack;sequential_1/sequential/random_crop/cond/stack_1/0:output:04sequential_1/sequential/random_crop/cond/Minimum:z:06sequential_1/sequential/random_crop/cond/Minimum_1:z:0;sequential_1/sequential/random_crop/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?
.sequential_1/sequential/random_crop/cond/SliceSlice?sequential_1_sequential_random_crop_cond_shape_sequential_input7sequential_1/sequential/random_crop/cond/stack:output:09sequential_1/sequential/random_crop/cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"?????????  ??????????
4sequential_1/sequential/random_crop/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        ?
>sequential_1/sequential/random_crop/cond/resize/ResizeBilinearResizeBilinear7sequential_1/sequential/random_crop/cond/Slice:output:0=sequential_1/sequential/random_crop/cond/resize/size:output:0*
T0*/
_output_shapes
:?????????  *
half_pixel_centers(?
1sequential_1/sequential/random_crop/cond/IdentityIdentityOsequential_1/sequential/random_crop/cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:?????????  "o
1sequential_1_sequential_random_crop_cond_identity:sequential_1/sequential/random_crop/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :?????????  :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????  
?
?
H__inference_random_crop_layer_call_and_return_conditional_losses_7018985

inputs
cond_input_0:	
identity??cond;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0*
value	B : S
subSubstrided_slice:output:0sub/y:output:0*
T0*
_output_shapes
: h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0*
value	B : Y
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0*
_output_shapes
: P
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : _
GreaterEqualGreaterEqualsub:z:0GreaterEqual/y:output:0*
T0*
_output_shapes
: R
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : e
GreaterEqual_1GreaterEqual	sub_1:z:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
: O
Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z 
Rank/packedPackRank/packed/0:output:0GreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:M
All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z {
	All/inputPackAll/input/0:output:0GreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
AllAllAll/input:output:0range:output:0*
_output_shapes
: ?
condIfAll:output:0cond_input_0sub:z:0	sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *%
else_branchR
cond_false_7018849*.
output_shapes
:?????????  *$
then_branchR
cond_true_7018848b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:?????????  m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:?????????  M
NoOpNoOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????  : 2
condcond:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
B__inference_dense_layer_call_and_return_conditional_losses_7018766

inputs1
matmul_readvariableop_resource:	?x-
biasadd_readvariableop_resource:x
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????xr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????xP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????xa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????xw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?0
?
random_crop_cond_false_7018485 
random_crop_cond_placeholder"
random_crop_cond_placeholder_1"
random_crop_cond_placeholder_2!
random_crop_cond_shape_inputs
random_crop_cond_identityc
random_crop/cond/ShapeShaperandom_crop_cond_shape_inputs*
T0*
_output_shapes
:w
$random_crop/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????y
&random_crop/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????p
&random_crop/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
random_crop/cond/strided_sliceStridedSlicerandom_crop/cond/Shape:output:0-random_crop/cond/strided_slice/stack:output:0/random_crop/cond/strided_slice/stack_1:output:0/random_crop/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
&random_crop/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
(random_crop/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????r
(random_crop/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 random_crop/cond/strided_slice_1StridedSlicerandom_crop/cond/Shape:output:0/random_crop/cond/strided_slice_1/stack:output:01random_crop/cond/strided_slice_1/stack_1:output:01random_crop/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
random_crop/cond/mul/yConst*
_output_shapes
: *
dtype0*
value	B : ?
random_crop/cond/mulMul)random_crop/cond/strided_slice_1:output:0random_crop/cond/mul/y:output:0*
T0*
_output_shapes
: g
random_crop/cond/CastCastrandom_crop/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: _
random_crop/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B?
random_crop/cond/truedivRealDivrandom_crop/cond/Cast:y:0#random_crop/cond/truediv/y:output:0*
T0*
_output_shapes
: m
random_crop/cond/Cast_1Castrandom_crop/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: Z
random_crop/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
random_crop/cond/mul_1Mul'random_crop/cond/strided_slice:output:0!random_crop/cond/mul_1/y:output:0*
T0*
_output_shapes
: k
random_crop/cond/Cast_2Castrandom_crop/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: a
random_crop/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B?
random_crop/cond/truediv_1RealDivrandom_crop/cond/Cast_2:y:0%random_crop/cond/truediv_1/y:output:0*
T0*
_output_shapes
: o
random_crop/cond/Cast_3Castrandom_crop/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
random_crop/cond/MinimumMinimum'random_crop/cond/strided_slice:output:0random_crop/cond/Cast_1:y:0*
T0*
_output_shapes
: ?
random_crop/cond/Minimum_1Minimum)random_crop/cond/strided_slice_1:output:0random_crop/cond/Cast_3:y:0*
T0*
_output_shapes
: ?
random_crop/cond/subSub'random_crop/cond/strided_slice:output:0random_crop/cond/Minimum:z:0*
T0*
_output_shapes
: i
random_crop/cond/Cast_4Castrandom_crop/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: a
random_crop/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
random_crop/cond/truediv_2RealDivrandom_crop/cond/Cast_4:y:0%random_crop/cond/truediv_2/y:output:0*
T0*
_output_shapes
: o
random_crop/cond/Cast_5Castrandom_crop/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
random_crop/cond/sub_1Sub)random_crop/cond/strided_slice_1:output:0random_crop/cond/Minimum_1:z:0*
T0*
_output_shapes
: k
random_crop/cond/Cast_6Castrandom_crop/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: a
random_crop/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
random_crop/cond/truediv_3RealDivrandom_crop/cond/Cast_6:y:0%random_crop/cond/truediv_3/y:output:0*
T0*
_output_shapes
: o
random_crop/cond/Cast_7Castrandom_crop/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: Z
random_crop/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : Z
random_crop/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
random_crop/cond/stackPack!random_crop/cond/stack/0:output:0random_crop/cond/Cast_5:y:0random_crop/cond/Cast_7:y:0!random_crop/cond/stack/3:output:0*
N*
T0*
_output_shapes
:e
random_crop/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????e
random_crop/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
random_crop/cond/stack_1Pack#random_crop/cond/stack_1/0:output:0random_crop/cond/Minimum:z:0random_crop/cond/Minimum_1:z:0#random_crop/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?
random_crop/cond/SliceSlicerandom_crop_cond_shape_inputsrandom_crop/cond/stack:output:0!random_crop/cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"?????????  ?????????m
random_crop/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        ?
&random_crop/cond/resize/ResizeBilinearResizeBilinearrandom_crop/cond/Slice:output:0%random_crop/cond/resize/size:output:0*
T0*/
_output_shapes
:?????????  *
half_pixel_centers(?
random_crop/cond/IdentityIdentity7random_crop/cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:?????????  "?
random_crop_cond_identity"random_crop/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :?????????  :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????  
?)
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_7017726
sequential_input 
sequential_7017692:	 
sequential_7017694:	(
conv2d_7017697:
conv2d_7017699:*
conv2d_1_7017703:
conv2d_1_7017705: 
dense_7017710:	?x
dense_7017712:x!
dense_1_7017715:xT
dense_1_7017717:T!
dense_2_7017720:T

dense_2_7017722:

identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallsequential_inputsequential_7017692sequential_7017694*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_7017288?
conv2d/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0conv2d_7017697conv2d_7017699*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_7017367?
!average_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7017331?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_1_7017703conv2d_1_7017705*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_7017385?
#average_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_7017343?
flatten/PartitionedCallPartitionedCall,average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_7017398?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_7017710dense_7017712*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_7017411?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_7017715dense_1_7017717*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_7017428?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_7017720dense_2_7017722*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_7017444w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:a ]
/
_output_shapes
:?????????  
*
_user_specified_namesequential_input
?w
?
G__inference_sequential_layer_call_and_return_conditional_losses_7018675

inputs&
random_crop_cond_input_0:	K
=random_flip_stateful_uniform_full_int_rngreadandskip_resource:	
identity??random_crop/cond?4random_flip/stateful_uniform_full_int/RngReadAndSkipG
random_crop/ShapeShapeinputs*
T0*
_output_shapes
:r
random_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????t
!random_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????k
!random_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
random_crop/strided_sliceStridedSlicerandom_crop/Shape:output:0(random_crop/strided_slice/stack:output:0*random_crop/strided_slice/stack_1:output:0*random_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskS
random_crop/sub/yConst*
_output_shapes
: *
dtype0*
value	B : w
random_crop/subSub"random_crop/strided_slice:output:0random_crop/sub/y:output:0*
T0*
_output_shapes
: t
!random_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????v
#random_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#random_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
random_crop/strided_slice_1StridedSlicerandom_crop/Shape:output:0*random_crop/strided_slice_1/stack:output:0,random_crop/strided_slice_1/stack_1:output:0,random_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
random_crop/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B : }
random_crop/sub_1Sub$random_crop/strided_slice_1:output:0random_crop/sub_1/y:output:0*
T0*
_output_shapes
: \
random_crop/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
random_crop/GreaterEqualGreaterEqualrandom_crop/sub:z:0#random_crop/GreaterEqual/y:output:0*
T0*
_output_shapes
: ^
random_crop/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
random_crop/GreaterEqual_1GreaterEqualrandom_crop/sub_1:z:0%random_crop/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: [
random_crop/Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z?
random_crop/Rank/packedPack"random_crop/Rank/packed/0:output:0random_crop/GreaterEqual:z:0random_crop/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:R
random_crop/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
random_crop/range/startConst*
_output_shapes
: *
dtype0*
value	B : Y
random_crop/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
random_crop/rangeRange random_crop/range/start:output:0random_crop/Rank:output:0 random_crop/range/delta:output:0*
_output_shapes
:Y
random_crop/All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z?
random_crop/All/inputPack random_crop/All/input/0:output:0random_crop/GreaterEqual:z:0random_crop/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:j
random_crop/AllAllrandom_crop/All/input:output:0random_crop/range:output:0*
_output_shapes
: ?
random_crop/condIfrandom_crop/All:output:0random_crop_cond_input_0random_crop/sub:z:0random_crop/sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *1
else_branch"R 
random_crop_cond_false_7018485*.
output_shapes
:?????????  *0
then_branch!R
random_crop_cond_true_7018484z
random_crop/cond/IdentityIdentityrandom_crop/cond:output:0*
T0*/
_output_shapes
:?????????  u
+random_flip/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:u
+random_flip/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
*random_flip/stateful_uniform_full_int/ProdProd4random_flip/stateful_uniform_full_int/shape:output:04random_flip/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: n
,random_flip/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
,random_flip/stateful_uniform_full_int/Cast_1Cast3random_flip/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
4random_flip/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip=random_flip_stateful_uniform_full_int_rngreadandskip_resource5random_flip/stateful_uniform_full_int/Cast/x:output:00random_flip/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:?
9random_flip/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
;random_flip/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;random_flip/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3random_flip/stateful_uniform_full_int/strided_sliceStridedSlice<random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Brandom_flip/stateful_uniform_full_int/strided_slice/stack:output:0Drandom_flip/stateful_uniform_full_int/strided_slice/stack_1:output:0Drandom_flip/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
-random_flip/stateful_uniform_full_int/BitcastBitcast<random_flip/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0?
;random_flip/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5random_flip/stateful_uniform_full_int/strided_slice_1StridedSlice<random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Drandom_flip/stateful_uniform_full_int/strided_slice_1/stack:output:0Frandom_flip/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Frandom_flip/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
/random_flip/stateful_uniform_full_int/Bitcast_1Bitcast>random_flip/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0k
)random_flip/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :?
%random_flip/stateful_uniform_full_intStatelessRandomUniformFullIntV24random_flip/stateful_uniform_full_int/shape:output:08random_flip/stateful_uniform_full_int/Bitcast_1:output:06random_flip/stateful_uniform_full_int/Bitcast:output:02random_flip/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	`
random_flip/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R ?
random_flip/stackPack.random_flip/stateful_uniform_full_int:output:0random_flip/zeros_like:output:0*
N*
T0	*
_output_shapes

:p
random_flip/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!random_flip/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!random_flip/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
random_flip/strided_sliceStridedSlicerandom_flip/stack:output:0(random_flip/strided_slice/stack:output:0*random_flip/strided_slice/stack_1:output:0*random_flip/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask?
?random_flip/stateless_random_flip_left_right/control_dependencyIdentity"random_crop/cond/Identity:output:0*
T0*,
_class"
 loc:@random_crop/cond/Identity*/
_output_shapes
:?????????  ?
2random_flip/stateless_random_flip_left_right/ShapeShapeHrandom_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:?
@random_flip/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:random_flip/stateless_random_flip_left_right/strided_sliceStridedSlice;random_flip/stateless_random_flip_left_right/Shape:output:0Irandom_flip/stateless_random_flip_left_right/strided_slice/stack:output:0Krandom_flip/stateless_random_flip_left_right/strided_slice/stack_1:output:0Krandom_flip/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Krandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shapePackCrandom_flip/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:?
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter"random_flip/strided_slice:output:0* 
_output_shapes
::?
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :?
^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Trandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0hrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0lrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0krandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:??????????
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/subSubRrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Rrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ?
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mulMulgrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Mrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:??????????
Erandom_flip/stateless_random_flip_left_right/stateless_random_uniformAddV2Mrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Rrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:?????????~
<random_flip/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
<random_flip/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :~
<random_flip/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
:random_flip/stateless_random_flip_left_right/Reshape/shapePackCrandom_flip/stateless_random_flip_left_right/strided_slice:output:0Erandom_flip/stateless_random_flip_left_right/Reshape/shape/1:output:0Erandom_flip/stateless_random_flip_left_right/Reshape/shape/2:output:0Erandom_flip/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
4random_flip/stateless_random_flip_left_right/ReshapeReshapeIrandom_flip/stateless_random_flip_left_right/stateless_random_uniform:z:0Crandom_flip/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
2random_flip/stateless_random_flip_left_right/RoundRound=random_flip/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:??????????
;random_flip/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:?
6random_flip/stateless_random_flip_left_right/ReverseV2	ReverseV2Hrandom_flip/stateless_random_flip_left_right/control_dependency:output:0Drandom_flip/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:?????????  ?
0random_flip/stateless_random_flip_left_right/mulMul6random_flip/stateless_random_flip_left_right/Round:y:0?random_flip/stateless_random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:?????????  w
2random_flip/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
0random_flip/stateless_random_flip_left_right/subSub;random_flip/stateless_random_flip_left_right/sub/x:output:06random_flip/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:??????????
2random_flip/stateless_random_flip_left_right/mul_1Mul4random_flip/stateless_random_flip_left_right/sub:z:0Hrandom_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:?????????  ?
0random_flip/stateless_random_flip_left_right/addAddV24random_flip/stateless_random_flip_left_right/mul:z:06random_flip/stateless_random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:?????????  ?
IdentityIdentity4random_flip/stateless_random_flip_left_right/add:z:0^NoOp*
T0*/
_output_shapes
:?????????  ?
NoOpNoOp^random_crop/cond5^random_flip/stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 2$
random_crop/condrandom_crop/cond2l
4random_flip/stateful_uniform_full_int/RngReadAndSkip4random_flip/stateful_uniform_full_int/RngReadAndSkip:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
.__inference_sequential_1_layer_call_fn_7017476
sequential_input
unknown:	#
	unknown_0:
	unknown_1:#
	unknown_2:
	unknown_3:
	unknown_4:	?x
	unknown_5:x
	unknown_6:xT
	unknown_7:T
	unknown_8:T

	unknown_9:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsequential_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_7017451o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????  : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:?????????  
*
_user_specified_namesequential_input
??
?	
I__inference_sequential_1_layer_call_and_return_conditional_losses_7018244

inputs1
#sequential_random_crop_cond_input_0:	V
Hsequential_random_flip_stateful_uniform_full_int_rngreadandskip_resource:	?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	?x3
%dense_biasadd_readvariableop_resource:x8
&dense_1_matmul_readvariableop_resource:xT5
'dense_1_biasadd_readvariableop_resource:T8
&dense_2_matmul_readvariableop_resource:T
5
'dense_2_biasadd_readvariableop_resource:

identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?sequential/random_crop/cond??sequential/random_flip/stateful_uniform_full_int/RngReadAndSkipR
sequential/random_crop/ShapeShapeinputs*
T0*
_output_shapes
:}
*sequential/random_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????
,sequential/random_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????v
,sequential/random_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$sequential/random_crop/strided_sliceStridedSlice%sequential/random_crop/Shape:output:03sequential/random_crop/strided_slice/stack:output:05sequential/random_crop/strided_slice/stack_1:output:05sequential/random_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
sequential/random_crop/sub/yConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential/random_crop/subSub-sequential/random_crop/strided_slice:output:0%sequential/random_crop/sub/y:output:0*
T0*
_output_shapes
: 
,sequential/random_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
.sequential/random_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????x
.sequential/random_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&sequential/random_crop/strided_slice_1StridedSlice%sequential/random_crop/Shape:output:05sequential/random_crop/strided_slice_1/stack:output:07sequential/random_crop/strided_slice_1/stack_1:output:07sequential/random_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
sequential/random_crop/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential/random_crop/sub_1Sub/sequential/random_crop/strided_slice_1:output:0'sequential/random_crop/sub_1/y:output:0*
T0*
_output_shapes
: g
%sequential/random_crop/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
#sequential/random_crop/GreaterEqualGreaterEqualsequential/random_crop/sub:z:0.sequential/random_crop/GreaterEqual/y:output:0*
T0*
_output_shapes
: i
'sequential/random_crop/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/random_crop/GreaterEqual_1GreaterEqual sequential/random_crop/sub_1:z:00sequential/random_crop/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: f
$sequential/random_crop/Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z?
"sequential/random_crop/Rank/packedPack-sequential/random_crop/Rank/packed/0:output:0'sequential/random_crop/GreaterEqual:z:0)sequential/random_crop/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:]
sequential/random_crop/RankConst*
_output_shapes
: *
dtype0*
value	B :d
"sequential/random_crop/range/startConst*
_output_shapes
: *
dtype0*
value	B : d
"sequential/random_crop/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
sequential/random_crop/rangeRange+sequential/random_crop/range/start:output:0$sequential/random_crop/Rank:output:0+sequential/random_crop/range/delta:output:0*
_output_shapes
:d
"sequential/random_crop/All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z?
 sequential/random_crop/All/inputPack+sequential/random_crop/All/input/0:output:0'sequential/random_crop/GreaterEqual:z:0)sequential/random_crop/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:?
sequential/random_crop/AllAll)sequential/random_crop/All/input:output:0%sequential/random_crop/range:output:0*
_output_shapes
: ?
sequential/random_crop/condIf#sequential/random_crop/All:output:0#sequential_random_crop_cond_input_0sequential/random_crop/sub:z:0 sequential/random_crop/sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *<
else_branch-R+
)sequential_random_crop_cond_false_7018016*.
output_shapes
:?????????  *;
then_branch,R*
(sequential_random_crop_cond_true_7018015?
$sequential/random_crop/cond/IdentityIdentity$sequential/random_crop/cond:output:0*
T0*/
_output_shapes
:?????????  ?
6sequential/random_flip/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
6sequential/random_flip/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
5sequential/random_flip/stateful_uniform_full_int/ProdProd?sequential/random_flip/stateful_uniform_full_int/shape:output:0?sequential/random_flip/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: y
7sequential/random_flip/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
7sequential/random_flip/stateful_uniform_full_int/Cast_1Cast>sequential/random_flip/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
?sequential/random_flip/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipHsequential_random_flip_stateful_uniform_full_int_rngreadandskip_resource@sequential/random_flip/stateful_uniform_full_int/Cast/x:output:0;sequential/random_flip/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:?
Dsequential/random_flip/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Fsequential/random_flip/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Fsequential/random_flip/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
>sequential/random_flip/stateful_uniform_full_int/strided_sliceStridedSliceGsequential/random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Msequential/random_flip/stateful_uniform_full_int/strided_slice/stack:output:0Osequential/random_flip/stateful_uniform_full_int/strided_slice/stack_1:output:0Osequential/random_flip/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
8sequential/random_flip/stateful_uniform_full_int/BitcastBitcastGsequential/random_flip/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0?
Fsequential/random_flip/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Hsequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
@sequential/random_flip/stateful_uniform_full_int/strided_slice_1StridedSliceGsequential/random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Osequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack:output:0Qsequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Qsequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
:sequential/random_flip/stateful_uniform_full_int/Bitcast_1BitcastIsequential/random_flip/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0v
4sequential/random_flip/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :?
0sequential/random_flip/stateful_uniform_full_intStatelessRandomUniformFullIntV2?sequential/random_flip/stateful_uniform_full_int/shape:output:0Csequential/random_flip/stateful_uniform_full_int/Bitcast_1:output:0Asequential/random_flip/stateful_uniform_full_int/Bitcast:output:0=sequential/random_flip/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	k
!sequential/random_flip/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R ?
sequential/random_flip/stackPack9sequential/random_flip/stateful_uniform_full_int:output:0*sequential/random_flip/zeros_like:output:0*
N*
T0	*
_output_shapes

:{
*sequential/random_flip/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,sequential/random_flip/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,sequential/random_flip/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
$sequential/random_flip/strided_sliceStridedSlice%sequential/random_flip/stack:output:03sequential/random_flip/strided_slice/stack:output:05sequential/random_flip/strided_slice/stack_1:output:05sequential/random_flip/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask?
Jsequential/random_flip/stateless_random_flip_left_right/control_dependencyIdentity-sequential/random_crop/cond/Identity:output:0*
T0*7
_class-
+)loc:@sequential/random_crop/cond/Identity*/
_output_shapes
:?????????  ?
=sequential/random_flip/stateless_random_flip_left_right/ShapeShapeSsequential/random_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:?
Ksequential/random_flip/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Msequential/random_flip/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Msequential/random_flip/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Esequential/random_flip/stateless_random_flip_left_right/strided_sliceStridedSliceFsequential/random_flip/stateless_random_flip_left_right/Shape:output:0Tsequential/random_flip/stateless_random_flip_left_right/strided_slice/stack:output:0Vsequential/random_flip/stateless_random_flip_left_right/strided_slice/stack_1:output:0Vsequential/random_flip/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Vsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/shapePackNsequential/random_flip/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:?
Tsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
Tsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
msequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter-sequential/random_flip/strided_slice:output:0* 
_output_shapes
::?
msequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :?
isequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2_sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0ssequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0wsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0vsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:??????????
Tsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/subSub]sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/max:output:0]sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ?
Tsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/mulMulrsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Xsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:??????????
Psequential/random_flip/stateless_random_flip_left_right/stateless_random_uniformAddV2Xsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0]sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:??????????
Gsequential/random_flip/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Gsequential/random_flip/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Gsequential/random_flip/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Esequential/random_flip/stateless_random_flip_left_right/Reshape/shapePackNsequential/random_flip/stateless_random_flip_left_right/strided_slice:output:0Psequential/random_flip/stateless_random_flip_left_right/Reshape/shape/1:output:0Psequential/random_flip/stateless_random_flip_left_right/Reshape/shape/2:output:0Psequential/random_flip/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
?sequential/random_flip/stateless_random_flip_left_right/ReshapeReshapeTsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform:z:0Nsequential/random_flip/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
=sequential/random_flip/stateless_random_flip_left_right/RoundRoundHsequential/random_flip/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:??????????
Fsequential/random_flip/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:?
Asequential/random_flip/stateless_random_flip_left_right/ReverseV2	ReverseV2Ssequential/random_flip/stateless_random_flip_left_right/control_dependency:output:0Osequential/random_flip/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:?????????  ?
;sequential/random_flip/stateless_random_flip_left_right/mulMulAsequential/random_flip/stateless_random_flip_left_right/Round:y:0Jsequential/random_flip/stateless_random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:?????????  ?
=sequential/random_flip/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
;sequential/random_flip/stateless_random_flip_left_right/subSubFsequential/random_flip/stateless_random_flip_left_right/sub/x:output:0Asequential/random_flip/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:??????????
=sequential/random_flip/stateless_random_flip_left_right/mul_1Mul?sequential/random_flip/stateless_random_flip_left_right/sub:z:0Ssequential/random_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:?????????  ?
;sequential/random_flip/stateless_random_flip_left_right/addAddV2?sequential/random_flip/stateless_random_flip_left_right/mul:z:0Asequential/random_flip/stateless_random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:?????????  ?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2D?sequential/random_flip/stateless_random_flip_left_right/add:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
average_pooling2d/AvgPoolAvgPoolconv2d/Relu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2D"average_pooling2d/AvgPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

?
average_pooling2d_1/AvgPoolAvgPoolconv2d_1/Relu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
flatten/ReshapeReshape$average_pooling2d_1/AvgPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????T?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^sequential/random_crop/cond@^sequential/random_flip/stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2:
sequential/random_crop/condsequential/random_crop/cond2?
?sequential/random_flip/stateful_uniform_full_int/RngReadAndSkip?sequential/random_flip/stateful_uniform_full_int/RngReadAndSkip:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_7018725

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????

i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????

w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_7017385

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????

i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????

w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
G__inference_sequential_layer_call_and_return_conditional_losses_7018455

inputs&
random_crop_cond_input_0:	
identity??random_crop/condG
random_crop/ShapeShapeinputs*
T0*
_output_shapes
:r
random_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????t
!random_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????k
!random_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
random_crop/strided_sliceStridedSlicerandom_crop/Shape:output:0(random_crop/strided_slice/stack:output:0*random_crop/strided_slice/stack_1:output:0*random_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskS
random_crop/sub/yConst*
_output_shapes
: *
dtype0*
value	B : w
random_crop/subSub"random_crop/strided_slice:output:0random_crop/sub/y:output:0*
T0*
_output_shapes
: t
!random_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????v
#random_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#random_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
random_crop/strided_slice_1StridedSlicerandom_crop/Shape:output:0*random_crop/strided_slice_1/stack:output:0,random_crop/strided_slice_1/stack_1:output:0,random_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
random_crop/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B : }
random_crop/sub_1Sub$random_crop/strided_slice_1:output:0random_crop/sub_1/y:output:0*
T0*
_output_shapes
: \
random_crop/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
random_crop/GreaterEqualGreaterEqualrandom_crop/sub:z:0#random_crop/GreaterEqual/y:output:0*
T0*
_output_shapes
: ^
random_crop/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
random_crop/GreaterEqual_1GreaterEqualrandom_crop/sub_1:z:0%random_crop/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: [
random_crop/Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z ?
random_crop/Rank/packedPack"random_crop/Rank/packed/0:output:0random_crop/GreaterEqual:z:0random_crop/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:R
random_crop/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
random_crop/range/startConst*
_output_shapes
: *
dtype0*
value	B : Y
random_crop/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
random_crop/rangeRange random_crop/range/start:output:0random_crop/Rank:output:0 random_crop/range/delta:output:0*
_output_shapes
:Y
random_crop/All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z ?
random_crop/All/inputPack random_crop/All/input/0:output:0random_crop/GreaterEqual:z:0random_crop/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:j
random_crop/AllAllrandom_crop/All/input:output:0random_crop/range:output:0*
_output_shapes
: ?
random_crop/condIfrandom_crop/All:output:0random_crop_cond_input_0random_crop/sub:z:0random_crop/sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *1
else_branch"R 
random_crop_cond_false_7018319*.
output_shapes
:?????????  *0
then_branch!R
random_crop_cond_true_7018318z
random_crop/cond/IdentityIdentityrandom_crop/cond:output:0*
T0*/
_output_shapes
:?????????  y
IdentityIdentity"random_crop/cond/Identity:output:0^NoOp*
T0*/
_output_shapes
:?????????  Y
NoOpNoOp^random_crop/cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????  : 2$
random_crop/condrandom_crop/cond:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_7017322
random_crop_input!
random_crop_7017315:	!
random_flip_7017318:	
identity??#random_crop/StatefulPartitionedCall?#random_flip/StatefulPartitionedCall?
#random_crop/StatefulPartitionedCallStatefulPartitionedCallrandom_crop_inputrandom_crop_7017315*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_random_crop_layer_call_and_return_conditional_losses_7017264?
#random_flip/StatefulPartitionedCallStatefulPartitionedCall,random_crop/StatefulPartitionedCall:output:0random_flip_7017318*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_7017083?
IdentityIdentity,random_flip/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  ?
NoOpNoOp$^random_crop/StatefulPartitionedCall$^random_flip/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 2J
#random_crop/StatefulPartitionedCall#random_crop/StatefulPartitionedCall2J
#random_flip/StatefulPartitionedCall#random_flip/StatefulPartitionedCall:b ^
/
_output_shapes
:?????????  
+
_user_specified_namerandom_crop_input
?
?
C__inference_conv2d_layer_call_and_return_conditional_losses_7017367

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?U
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_7017986

inputs1
#sequential_random_crop_cond_input_0:	?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	?x3
%dense_biasadd_readvariableop_resource:x8
&dense_1_matmul_readvariableop_resource:xT5
'dense_1_biasadd_readvariableop_resource:T8
&dense_2_matmul_readvariableop_resource:T
5
'dense_2_biasadd_readvariableop_resource:

identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?sequential/random_crop/condR
sequential/random_crop/ShapeShapeinputs*
T0*
_output_shapes
:}
*sequential/random_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????
,sequential/random_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????v
,sequential/random_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$sequential/random_crop/strided_sliceStridedSlice%sequential/random_crop/Shape:output:03sequential/random_crop/strided_slice/stack:output:05sequential/random_crop/strided_slice/stack_1:output:05sequential/random_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
sequential/random_crop/sub/yConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential/random_crop/subSub-sequential/random_crop/strided_slice:output:0%sequential/random_crop/sub/y:output:0*
T0*
_output_shapes
: 
,sequential/random_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
.sequential/random_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????x
.sequential/random_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&sequential/random_crop/strided_slice_1StridedSlice%sequential/random_crop/Shape:output:05sequential/random_crop/strided_slice_1/stack:output:07sequential/random_crop/strided_slice_1/stack_1:output:07sequential/random_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
sequential/random_crop/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential/random_crop/sub_1Sub/sequential/random_crop/strided_slice_1:output:0'sequential/random_crop/sub_1/y:output:0*
T0*
_output_shapes
: g
%sequential/random_crop/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
#sequential/random_crop/GreaterEqualGreaterEqualsequential/random_crop/sub:z:0.sequential/random_crop/GreaterEqual/y:output:0*
T0*
_output_shapes
: i
'sequential/random_crop/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/random_crop/GreaterEqual_1GreaterEqual sequential/random_crop/sub_1:z:00sequential/random_crop/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: f
$sequential/random_crop/Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z ?
"sequential/random_crop/Rank/packedPack-sequential/random_crop/Rank/packed/0:output:0'sequential/random_crop/GreaterEqual:z:0)sequential/random_crop/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:]
sequential/random_crop/RankConst*
_output_shapes
: *
dtype0*
value	B :d
"sequential/random_crop/range/startConst*
_output_shapes
: *
dtype0*
value	B : d
"sequential/random_crop/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
sequential/random_crop/rangeRange+sequential/random_crop/range/start:output:0$sequential/random_crop/Rank:output:0+sequential/random_crop/range/delta:output:0*
_output_shapes
:d
"sequential/random_crop/All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z ?
 sequential/random_crop/All/inputPack+sequential/random_crop/All/input/0:output:0'sequential/random_crop/GreaterEqual:z:0)sequential/random_crop/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:?
sequential/random_crop/AllAll)sequential/random_crop/All/input:output:0%sequential/random_crop/range:output:0*
_output_shapes
: ?
sequential/random_crop/condIf#sequential/random_crop/All:output:0#sequential_random_crop_cond_input_0sequential/random_crop/sub:z:0 sequential/random_crop/sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *<
else_branch-R+
)sequential_random_crop_cond_false_7017812*.
output_shapes
:?????????  *;
then_branch,R*
(sequential_random_crop_cond_true_7017811?
$sequential/random_crop/cond/IdentityIdentity$sequential/random_crop/cond:output:0*
T0*/
_output_shapes
:?????????  ?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2D-sequential/random_crop/cond/Identity:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
average_pooling2d/AvgPoolAvgPoolconv2d/Relu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2D"average_pooling2d/AvgPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

?
average_pooling2d_1/AvgPoolAvgPoolconv2d_1/Relu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
flatten/ReshapeReshape$average_pooling2d_1/AvgPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????T?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^sequential/random_crop/cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????  : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2:
sequential/random_crop/condsequential/random_crop/cond:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
O
3__inference_average_pooling2d_layer_call_fn_7018700

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7017331?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?g
?
cond_true_7017127;
-cond_stateful_uniform_rngreadandskip_resource:	
cond_add_sub
cond_add_1_sub_1*
&cond_crop_to_bounding_box_shape_inputs
cond_identity??'cond/crop_to_bounding_box/Assert/Assert?)cond/crop_to_bounding_box/Assert_1/Assert?)cond/crop_to_bounding_box/Assert_2/Assert?)cond/crop_to_bounding_box/Assert_3/Assert?$cond/stateful_uniform/RngReadAndSkipe
cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:[
cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : _
cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :????e
cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
cond/stateful_uniform/ProdProd$cond/stateful_uniform/shape:output:0$cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: ^
cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :y
cond/stateful_uniform/Cast_1Cast#cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$cond/stateful_uniform/RngReadAndSkipRngReadAndSkip-cond_stateful_uniform_rngreadandskip_resource%cond/stateful_uniform/Cast/x:output:0 cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:s
)cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#cond/stateful_uniform/strided_sliceStridedSlice,cond/stateful_uniform/RngReadAndSkip:value:02cond/stateful_uniform/strided_slice/stack:output:04cond/stateful_uniform/strided_slice/stack_1:output:04cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
cond/stateful_uniform/BitcastBitcast,cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0u
+cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%cond/stateful_uniform/strided_slice_1StridedSlice,cond/stateful_uniform/RngReadAndSkip:value:04cond/stateful_uniform/strided_slice_1/stack:output:06cond/stateful_uniform/strided_slice_1/stack_1:output:06cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
cond/stateful_uniform/Bitcast_1Bitcast.cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0[
cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
cond/stateful_uniformStatelessRandomUniformIntV2$cond/stateful_uniform/shape:output:0(cond/stateful_uniform/Bitcast_1:output:0&cond/stateful_uniform/Bitcast:output:0"cond/stateful_uniform/alg:output:0"cond/stateful_uniform/min:output:0"cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0b
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/stateful_uniform:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :U
cond/addAddV2cond_add_subcond/add/y:output:0*
T0*
_output_shapes
: `
cond/modFloorModcond/strided_slice:output:0cond/add:z:0*
T0*
_output_shapes
: d
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/stateful_uniform:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :]

cond/add_1AddV2cond_add_1_sub_1cond/add_1/y:output:0*
T0*
_output_shapes
: f

cond/mod_1FloorModcond/strided_slice_1:output:0cond/add_1:z:0*
T0*
_output_shapes
: u
cond/crop_to_bounding_box/ShapeShape&cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/unstackUnpack(cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numj
(cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
&cond/crop_to_bounding_box/GreaterEqualGreaterEqualcond/mod_1:z:01cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
&cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
.cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
'cond/crop_to_bounding_box/Assert/AssertAssert*cond/crop_to_bounding_box/GreaterEqual:z:07cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 l
*cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
(cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcond/mod:z:03cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
0cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
)cond/crop_to_bounding_box/Assert_1/AssertAssert,cond/crop_to_bounding_box/GreaterEqual_1:z:09cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0(^cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 a
cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/addAddV2(cond/crop_to_bounding_box/add/x:output:0cond/mod_1:z:0*
T0*
_output_shapes
: g
%cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
#cond/crop_to_bounding_box/LessEqual	LessEqual!cond/crop_to_bounding_box/add:z:0.cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_2/AssertAssert'cond/crop_to_bounding_box/LessEqual:z:09cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 c
!cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/add_1AddV2*cond/crop_to_bounding_box/add_1/x:output:0cond/mod:z:0*
T0*
_output_shapes
: i
'cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
%cond/crop_to_bounding_box/LessEqual_1	LessEqual#cond/crop_to_bounding_box/add_1:z:00cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_3/AssertAssert)cond/crop_to_bounding_box/LessEqual_1:z:09cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
,cond/crop_to_bounding_box/control_dependencyIdentity&cond_crop_to_bounding_box_shape_inputs(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????  c
!cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : c
!cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/stackPack*cond/crop_to_bounding_box/stack/0:output:0cond/mod:z:0cond/mod_1:z:0*cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/Shape_1Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:w
-cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'cond/crop_to_bounding_box/strided_sliceStridedSlice*cond/crop_to_bounding_box/Shape_1:output:06cond/crop_to_bounding_box/strided_slice/stack:output:08cond/crop_to_bounding_box/strided_slice/stack_1:output:08cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!cond/crop_to_bounding_box/Shape_2Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:y
/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)cond/crop_to_bounding_box/strided_slice_1StridedSlice*cond/crop_to_bounding_box/Shape_2:output:08cond/crop_to_bounding_box/strided_slice_1/stack:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : e
#cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : ?
!cond/crop_to_bounding_box/stack_1Pack0cond/crop_to_bounding_box/strided_slice:output:0,cond/crop_to_bounding_box/stack_1/1:output:0,cond/crop_to_bounding_box/stack_1/2:output:02cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
cond/crop_to_bounding_box/SliceSlice5cond/crop_to_bounding_box/control_dependency:output:0(cond/crop_to_bounding_box/stack:output:0*cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:?????????  ?
cond/IdentityIdentity(cond/crop_to_bounding_box/Slice:output:0
^cond/NoOp*
T0*/
_output_shapes
:?????????  ?
	cond/NoOpNoOp(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert%^cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :?????????  2R
'cond/crop_to_bounding_box/Assert/Assert'cond/crop_to_bounding_box/Assert/Assert2V
)cond/crop_to_bounding_box/Assert_1/Assert)cond/crop_to_bounding_box/Assert_1/Assert2V
)cond/crop_to_bounding_box/Assert_2/Assert)cond/crop_to_bounding_box/Assert_2/Assert2V
)cond/crop_to_bounding_box/Assert_3/Assert)cond/crop_to_bounding_box/Assert_3/Assert2L
$cond/stateful_uniform/RngReadAndSkip$cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????  
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_7018746

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
}
-__inference_random_flip_layer_call_fn_7019163

inputs
unknown:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_7017083w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????  : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
D__inference_dense_1_layer_call_and_return_conditional_losses_7018786

inputs0
matmul_readvariableop_resource:xT-
biasadd_readvariableop_resource:T
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Tr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????TP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????Ta
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????Tw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
l
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_7018735

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_7017312
random_crop_input!
random_crop_7017307:	
identity??#random_crop/StatefulPartitionedCall?
#random_crop/StatefulPartitionedCallStatefulPartitionedCallrandom_crop_inputrandom_crop_7017307*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_random_crop_layer_call_and_return_conditional_losses_7017001?
random_flip/PartitionedCallPartitionedCall,random_crop/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_7017009{
IdentityIdentity$random_flip/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  l
NoOpNoOp$^random_crop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????  : 2J
#random_crop/StatefulPartitionedCall#random_crop/StatefulPartitionedCall:b ^
/
_output_shapes
:?????????  
+
_user_specified_namerandom_crop_input
?
?
)__inference_dense_2_layer_call_fn_7018795

inputs
unknown:T

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_7017444o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????T: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?M
?
H__inference_random_flip_layer_call_and_return_conditional_losses_7017083

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity??(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :?
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask?
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????  ?
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:~
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:?
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::?
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :?
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0_stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:??????????
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ?
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:??????????
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:?????????r
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:?????????y
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:?
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:?????????  ?
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:?????????  k
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:??????????
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:?????????  ?
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:?????????  
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*/
_output_shapes
:?????????  q
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????  : 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?)
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_7017598

inputs 
sequential_7017564:	 
sequential_7017566:	(
conv2d_7017569:
conv2d_7017571:*
conv2d_1_7017575:
conv2d_1_7017577: 
dense_7017582:	?x
dense_7017584:x!
dense_1_7017587:xT
dense_1_7017589:T!
dense_2_7017592:T

dense_2_7017594:

identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_7017564sequential_7017566*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_7017288?
conv2d/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0conv2d_7017569conv2d_7017571*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_7017367?
!average_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7017331?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_1_7017575conv2d_1_7017577*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_7017385?
#average_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_7017343?
flatten/PartitionedCallPartitionedCall,average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_7017398?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_7017582dense_7017584*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_7017411?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_7017587dense_1_7017589*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_7017428?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_7017592dense_2_7017594*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_7017444w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
}
-__inference_random_crop_layer_call_fn_7018812

inputs
unknown:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_random_crop_layer_call_and_return_conditional_losses_7017001w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????  : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
(__inference_conv2d_layer_call_fn_7018684

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_7017367w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?$
?
 __inference__traced_save_7019294
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop8
4savev2_random_crop_cond_statevar_read_readvariableop	K
Gsavev2_sequential_1_sequential_random_flip_statevar_read_readvariableop	
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop4savev2_random_crop_cond_statevar_read_readvariableopGsavev2_sequential_1_sequential_random_flip_statevar_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2		?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapess
q: :::::	?x:x:xT:T:T
:
::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	?x: 

_output_shapes
:x:$ 

_output_shapes

:xT: 

_output_shapes
:T:$	 

_output_shapes

:T
: 


_output_shapes
:
: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?

?
B__inference_dense_layer_call_and_return_conditional_losses_7017411

inputs1
matmul_readvariableop_resource:	?x-
biasadd_readvariableop_resource:x
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????xr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????xP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????xa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????xw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_dense_1_layer_call_and_return_conditional_losses_7017428

inputs0
matmul_readvariableop_resource:xT-
biasadd_readvariableop_resource:T
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Tr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????TP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????Ta
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????Tw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
.__inference_sequential_1_layer_call_fn_7017782

inputs
unknown:	
	unknown_0:	#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:	?x
	unknown_6:x
	unknown_7:xT
	unknown_8:T
	unknown_9:T


unknown_10:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_7017598o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
??
?
(sequential_random_crop_cond_true_7017811R
Dsequential_random_crop_cond_stateful_uniform_rngreadandskip_resource:	>
:sequential_random_crop_cond_add_sequential_random_crop_subB
>sequential_random_crop_cond_add_1_sequential_random_crop_sub_1A
=sequential_random_crop_cond_crop_to_bounding_box_shape_inputs(
$sequential_random_crop_cond_identity??>sequential/random_crop/cond/crop_to_bounding_box/Assert/Assert?@sequential/random_crop/cond/crop_to_bounding_box/Assert_1/Assert?@sequential/random_crop/cond/crop_to_bounding_box/Assert_2/Assert?@sequential/random_crop/cond/crop_to_bounding_box/Assert_3/Assert?;sequential/random_crop/cond/stateful_uniform/RngReadAndSkip|
2sequential/random_crop/cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:r
0sequential/random_crop/cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : v
0sequential/random_crop/cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :????|
2sequential/random_crop/cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
1sequential/random_crop/cond/stateful_uniform/ProdProd;sequential/random_crop/cond/stateful_uniform/shape:output:0;sequential/random_crop/cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: u
3sequential/random_crop/cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
3sequential/random_crop/cond/stateful_uniform/Cast_1Cast:sequential/random_crop/cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
;sequential/random_crop/cond/stateful_uniform/RngReadAndSkipRngReadAndSkipDsequential_random_crop_cond_stateful_uniform_rngreadandskip_resource<sequential/random_crop/cond/stateful_uniform/Cast/x:output:07sequential/random_crop/cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:?
@sequential/random_crop/cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bsequential/random_crop/cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bsequential/random_crop/cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:sequential/random_crop/cond/stateful_uniform/strided_sliceStridedSliceCsequential/random_crop/cond/stateful_uniform/RngReadAndSkip:value:0Isequential/random_crop/cond/stateful_uniform/strided_slice/stack:output:0Ksequential/random_crop/cond/stateful_uniform/strided_slice/stack_1:output:0Ksequential/random_crop/cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
4sequential/random_crop/cond/stateful_uniform/BitcastBitcastCsequential/random_crop/cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0?
Bsequential/random_crop/cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/random_crop/cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/random_crop/cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential/random_crop/cond/stateful_uniform/strided_slice_1StridedSliceCsequential/random_crop/cond/stateful_uniform/RngReadAndSkip:value:0Ksequential/random_crop/cond/stateful_uniform/strided_slice_1/stack:output:0Msequential/random_crop/cond/stateful_uniform/strided_slice_1/stack_1:output:0Msequential/random_crop/cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
6sequential/random_crop/cond/stateful_uniform/Bitcast_1BitcastEsequential/random_crop/cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0r
0sequential/random_crop/cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
,sequential/random_crop/cond/stateful_uniformStatelessRandomUniformIntV2;sequential/random_crop/cond/stateful_uniform/shape:output:0?sequential/random_crop/cond/stateful_uniform/Bitcast_1:output:0=sequential/random_crop/cond/stateful_uniform/Bitcast:output:09sequential/random_crop/cond/stateful_uniform/alg:output:09sequential/random_crop/cond/stateful_uniform/min:output:09sequential/random_crop/cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0y
/sequential/random_crop/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential/random_crop/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential/random_crop/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential/random_crop/cond/strided_sliceStridedSlice5sequential/random_crop/cond/stateful_uniform:output:08sequential/random_crop/cond/strided_slice/stack:output:0:sequential/random_crop/cond/strided_slice/stack_1:output:0:sequential/random_crop/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!sequential/random_crop/cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
sequential/random_crop/cond/addAddV2:sequential_random_crop_cond_add_sequential_random_crop_sub*sequential/random_crop/cond/add/y:output:0*
T0*
_output_shapes
: ?
sequential/random_crop/cond/modFloorMod2sequential/random_crop/cond/strided_slice:output:0#sequential/random_crop/cond/add:z:0*
T0*
_output_shapes
: {
1sequential/random_crop/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3sequential/random_crop/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential/random_crop/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+sequential/random_crop/cond/strided_slice_1StridedSlice5sequential/random_crop/cond/stateful_uniform:output:0:sequential/random_crop/cond/strided_slice_1/stack:output:0<sequential/random_crop/cond/strided_slice_1/stack_1:output:0<sequential/random_crop/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sequential/random_crop/cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
!sequential/random_crop/cond/add_1AddV2>sequential_random_crop_cond_add_1_sequential_random_crop_sub_1,sequential/random_crop/cond/add_1/y:output:0*
T0*
_output_shapes
: ?
!sequential/random_crop/cond/mod_1FloorMod4sequential/random_crop/cond/strided_slice_1:output:0%sequential/random_crop/cond/add_1:z:0*
T0*
_output_shapes
: ?
6sequential/random_crop/cond/crop_to_bounding_box/ShapeShape=sequential_random_crop_cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:?
8sequential/random_crop/cond/crop_to_bounding_box/unstackUnpack?sequential/random_crop/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
num?
?sequential/random_crop/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
=sequential/random_crop/cond/crop_to_bounding_box/GreaterEqualGreaterEqual%sequential/random_crop/cond/mod_1:z:0Hsequential/random_crop/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
=sequential/random_crop/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
Esequential/random_crop/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
>sequential/random_crop/cond/crop_to_bounding_box/Assert/AssertAssertAsequential/random_crop/cond/crop_to_bounding_box/GreaterEqual:z:0Nsequential/random_crop/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 ?
Asequential/random_crop/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
?sequential/random_crop/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqual#sequential/random_crop/cond/mod:z:0Jsequential/random_crop/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
?sequential/random_crop/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
Gsequential/random_crop/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
@sequential/random_crop/cond/crop_to_bounding_box/Assert_1/AssertAssertCsequential/random_crop/cond/crop_to_bounding_box/GreaterEqual_1:z:0Psequential/random_crop/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0?^sequential/random_crop/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 x
6sequential/random_crop/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B : ?
4sequential/random_crop/cond/crop_to_bounding_box/addAddV2?sequential/random_crop/cond/crop_to_bounding_box/add/x:output:0%sequential/random_crop/cond/mod_1:z:0*
T0*
_output_shapes
: ~
<sequential/random_crop/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
:sequential/random_crop/cond/crop_to_bounding_box/LessEqual	LessEqual8sequential/random_crop/cond/crop_to_bounding_box/add:z:0Esequential/random_crop/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
?sequential/random_crop/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
Gsequential/random_crop/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
@sequential/random_crop/cond/crop_to_bounding_box/Assert_2/AssertAssert>sequential/random_crop/cond/crop_to_bounding_box/LessEqual:z:0Psequential/random_crop/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0A^sequential/random_crop/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 z
8sequential/random_crop/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : ?
6sequential/random_crop/cond/crop_to_bounding_box/add_1AddV2Asequential/random_crop/cond/crop_to_bounding_box/add_1/x:output:0#sequential/random_crop/cond/mod:z:0*
T0*
_output_shapes
: ?
>sequential/random_crop/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
<sequential/random_crop/cond/crop_to_bounding_box/LessEqual_1	LessEqual:sequential/random_crop/cond/crop_to_bounding_box/add_1:z:0Gsequential/random_crop/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
?sequential/random_crop/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
Gsequential/random_crop/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
@sequential/random_crop/cond/crop_to_bounding_box/Assert_3/AssertAssert@sequential/random_crop/cond/crop_to_bounding_box/LessEqual_1:z:0Psequential/random_crop/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0A^sequential/random_crop/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
Csequential/random_crop/cond/crop_to_bounding_box/control_dependencyIdentity=sequential_random_crop_cond_crop_to_bounding_box_shape_inputs?^sequential/random_crop/cond/crop_to_bounding_box/Assert/AssertA^sequential/random_crop/cond/crop_to_bounding_box/Assert_1/AssertA^sequential/random_crop/cond/crop_to_bounding_box/Assert_2/AssertA^sequential/random_crop/cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????  z
8sequential/random_crop/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : z
8sequential/random_crop/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
6sequential/random_crop/cond/crop_to_bounding_box/stackPackAsequential/random_crop/cond/crop_to_bounding_box/stack/0:output:0#sequential/random_crop/cond/mod:z:0%sequential/random_crop/cond/mod_1:z:0Asequential/random_crop/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
8sequential/random_crop/cond/crop_to_bounding_box/Shape_1ShapeLsequential/random_crop/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
Dsequential/random_crop/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Fsequential/random_crop/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Fsequential/random_crop/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
>sequential/random_crop/cond/crop_to_bounding_box/strided_sliceStridedSliceAsequential/random_crop/cond/crop_to_bounding_box/Shape_1:output:0Msequential/random_crop/cond/crop_to_bounding_box/strided_slice/stack:output:0Osequential/random_crop/cond/crop_to_bounding_box/strided_slice/stack_1:output:0Osequential/random_crop/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
8sequential/random_crop/cond/crop_to_bounding_box/Shape_2ShapeLsequential/random_crop/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
Fsequential/random_crop/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Hsequential/random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential/random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
@sequential/random_crop/cond/crop_to_bounding_box/strided_slice_1StridedSliceAsequential/random_crop/cond/crop_to_bounding_box/Shape_2:output:0Osequential/random_crop/cond/crop_to_bounding_box/strided_slice_1/stack:output:0Qsequential/random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0Qsequential/random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
:sequential/random_crop/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : |
:sequential/random_crop/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : ?
8sequential/random_crop/cond/crop_to_bounding_box/stack_1PackGsequential/random_crop/cond/crop_to_bounding_box/strided_slice:output:0Csequential/random_crop/cond/crop_to_bounding_box/stack_1/1:output:0Csequential/random_crop/cond/crop_to_bounding_box/stack_1/2:output:0Isequential/random_crop/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
6sequential/random_crop/cond/crop_to_bounding_box/SliceSliceLsequential/random_crop/cond/crop_to_bounding_box/control_dependency:output:0?sequential/random_crop/cond/crop_to_bounding_box/stack:output:0Asequential/random_crop/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:?????????  ?
$sequential/random_crop/cond/IdentityIdentity?sequential/random_crop/cond/crop_to_bounding_box/Slice:output:0!^sequential/random_crop/cond/NoOp*
T0*/
_output_shapes
:?????????  ?
 sequential/random_crop/cond/NoOpNoOp?^sequential/random_crop/cond/crop_to_bounding_box/Assert/AssertA^sequential/random_crop/cond/crop_to_bounding_box/Assert_1/AssertA^sequential/random_crop/cond/crop_to_bounding_box/Assert_2/AssertA^sequential/random_crop/cond/crop_to_bounding_box/Assert_3/Assert<^sequential/random_crop/cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "U
$sequential_random_crop_cond_identity-sequential/random_crop/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :?????????  2?
>sequential/random_crop/cond/crop_to_bounding_box/Assert/Assert>sequential/random_crop/cond/crop_to_bounding_box/Assert/Assert2?
@sequential/random_crop/cond/crop_to_bounding_box/Assert_1/Assert@sequential/random_crop/cond/crop_to_bounding_box/Assert_1/Assert2?
@sequential/random_crop/cond/crop_to_bounding_box/Assert_2/Assert@sequential/random_crop/cond/crop_to_bounding_box/Assert_2/Assert2?
@sequential/random_crop/cond/crop_to_bounding_box/Assert_3/Assert@sequential/random_crop/cond/crop_to_bounding_box/Assert_3/Assert2z
;sequential/random_crop/cond/stateful_uniform/RngReadAndSkip;sequential/random_crop/cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????  
?
?
,__inference_sequential_layer_call_fn_7018289

inputs
unknown:	
	unknown_0:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_7017288w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?)
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_7017689
sequential_input 
sequential_7017657:	(
conv2d_7017660:
conv2d_7017662:*
conv2d_1_7017666:
conv2d_1_7017668: 
dense_7017673:	?x
dense_7017675:x!
dense_1_7017678:xT
dense_1_7017680:T!
dense_2_7017683:T

dense_2_7017685:

identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallsequential_inputsequential_7017657*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_7017012?
conv2d/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0conv2d_7017660conv2d_7017662*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_7017367?
!average_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7017331?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_1_7017666conv2d_1_7017668*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_7017385?
#average_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_7017343?
flatten/PartitionedCallPartitionedCall,average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_7017398?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_7017673dense_7017675*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_7017411?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_7017678dense_1_7017680*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_7017428?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_7017683dense_2_7017685*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_7017444w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????  : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:a ]
/
_output_shapes
:?????????  
*
_user_specified_namesequential_input
?

?
.__inference_sequential_1_layer_call_fn_7017753

inputs
unknown:	#
	unknown_0:
	unknown_1:#
	unknown_2:
	unknown_3:
	unknown_4:	?x
	unknown_5:x
	unknown_6:xT
	unknown_7:T
	unknown_8:T

	unknown_9:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_7017451o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????  : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
C__inference_conv2d_layer_call_and_return_conditional_losses_7018695

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
I
-__inference_random_flip_layer_call_fn_7019156

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_7017009h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?g
?
cond_true_7016864;
-cond_stateful_uniform_rngreadandskip_resource:	
cond_add_sub
cond_add_1_sub_1*
&cond_crop_to_bounding_box_shape_inputs
cond_identity??'cond/crop_to_bounding_box/Assert/Assert?)cond/crop_to_bounding_box/Assert_1/Assert?)cond/crop_to_bounding_box/Assert_2/Assert?)cond/crop_to_bounding_box/Assert_3/Assert?$cond/stateful_uniform/RngReadAndSkipe
cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:[
cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : _
cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :????e
cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
cond/stateful_uniform/ProdProd$cond/stateful_uniform/shape:output:0$cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: ^
cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :y
cond/stateful_uniform/Cast_1Cast#cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$cond/stateful_uniform/RngReadAndSkipRngReadAndSkip-cond_stateful_uniform_rngreadandskip_resource%cond/stateful_uniform/Cast/x:output:0 cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:s
)cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#cond/stateful_uniform/strided_sliceStridedSlice,cond/stateful_uniform/RngReadAndSkip:value:02cond/stateful_uniform/strided_slice/stack:output:04cond/stateful_uniform/strided_slice/stack_1:output:04cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
cond/stateful_uniform/BitcastBitcast,cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0u
+cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%cond/stateful_uniform/strided_slice_1StridedSlice,cond/stateful_uniform/RngReadAndSkip:value:04cond/stateful_uniform/strided_slice_1/stack:output:06cond/stateful_uniform/strided_slice_1/stack_1:output:06cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
cond/stateful_uniform/Bitcast_1Bitcast.cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0[
cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
cond/stateful_uniformStatelessRandomUniformIntV2$cond/stateful_uniform/shape:output:0(cond/stateful_uniform/Bitcast_1:output:0&cond/stateful_uniform/Bitcast:output:0"cond/stateful_uniform/alg:output:0"cond/stateful_uniform/min:output:0"cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0b
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/stateful_uniform:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :U
cond/addAddV2cond_add_subcond/add/y:output:0*
T0*
_output_shapes
: `
cond/modFloorModcond/strided_slice:output:0cond/add:z:0*
T0*
_output_shapes
: d
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/stateful_uniform:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :]

cond/add_1AddV2cond_add_1_sub_1cond/add_1/y:output:0*
T0*
_output_shapes
: f

cond/mod_1FloorModcond/strided_slice_1:output:0cond/add_1:z:0*
T0*
_output_shapes
: u
cond/crop_to_bounding_box/ShapeShape&cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/unstackUnpack(cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numj
(cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
&cond/crop_to_bounding_box/GreaterEqualGreaterEqualcond/mod_1:z:01cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
&cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
.cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
'cond/crop_to_bounding_box/Assert/AssertAssert*cond/crop_to_bounding_box/GreaterEqual:z:07cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 l
*cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
(cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcond/mod:z:03cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
0cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
)cond/crop_to_bounding_box/Assert_1/AssertAssert,cond/crop_to_bounding_box/GreaterEqual_1:z:09cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0(^cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 a
cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/addAddV2(cond/crop_to_bounding_box/add/x:output:0cond/mod_1:z:0*
T0*
_output_shapes
: g
%cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
#cond/crop_to_bounding_box/LessEqual	LessEqual!cond/crop_to_bounding_box/add:z:0.cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_2/AssertAssert'cond/crop_to_bounding_box/LessEqual:z:09cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 c
!cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/add_1AddV2*cond/crop_to_bounding_box/add_1/x:output:0cond/mod:z:0*
T0*
_output_shapes
: i
'cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
%cond/crop_to_bounding_box/LessEqual_1	LessEqual#cond/crop_to_bounding_box/add_1:z:00cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_3/AssertAssert)cond/crop_to_bounding_box/LessEqual_1:z:09cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
,cond/crop_to_bounding_box/control_dependencyIdentity&cond_crop_to_bounding_box_shape_inputs(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????  c
!cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : c
!cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/stackPack*cond/crop_to_bounding_box/stack/0:output:0cond/mod:z:0cond/mod_1:z:0*cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/Shape_1Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:w
-cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'cond/crop_to_bounding_box/strided_sliceStridedSlice*cond/crop_to_bounding_box/Shape_1:output:06cond/crop_to_bounding_box/strided_slice/stack:output:08cond/crop_to_bounding_box/strided_slice/stack_1:output:08cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!cond/crop_to_bounding_box/Shape_2Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:y
/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)cond/crop_to_bounding_box/strided_slice_1StridedSlice*cond/crop_to_bounding_box/Shape_2:output:08cond/crop_to_bounding_box/strided_slice_1/stack:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : e
#cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : ?
!cond/crop_to_bounding_box/stack_1Pack0cond/crop_to_bounding_box/strided_slice:output:0,cond/crop_to_bounding_box/stack_1/1:output:0,cond/crop_to_bounding_box/stack_1/2:output:02cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
cond/crop_to_bounding_box/SliceSlice5cond/crop_to_bounding_box/control_dependency:output:0(cond/crop_to_bounding_box/stack:output:0*cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:?????????  ?
cond/IdentityIdentity(cond/crop_to_bounding_box/Slice:output:0
^cond/NoOp*
T0*/
_output_shapes
:?????????  ?
	cond/NoOpNoOp(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert%^cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :?????????  2R
'cond/crop_to_bounding_box/Assert/Assert'cond/crop_to_bounding_box/Assert/Assert2V
)cond/crop_to_bounding_box/Assert_1/Assert)cond/crop_to_bounding_box/Assert_1/Assert2V
)cond/crop_to_bounding_box/Assert_2/Assert)cond/crop_to_bounding_box/Assert_2/Assert2V
)cond/crop_to_bounding_box/Assert_3/Assert)cond/crop_to_bounding_box/Assert_3/Assert2L
$cond/stateful_uniform/RngReadAndSkip$cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????  
??
?
(sequential_random_crop_cond_true_7018015R
Dsequential_random_crop_cond_stateful_uniform_rngreadandskip_resource:	>
:sequential_random_crop_cond_add_sequential_random_crop_subB
>sequential_random_crop_cond_add_1_sequential_random_crop_sub_1A
=sequential_random_crop_cond_crop_to_bounding_box_shape_inputs(
$sequential_random_crop_cond_identity??>sequential/random_crop/cond/crop_to_bounding_box/Assert/Assert?@sequential/random_crop/cond/crop_to_bounding_box/Assert_1/Assert?@sequential/random_crop/cond/crop_to_bounding_box/Assert_2/Assert?@sequential/random_crop/cond/crop_to_bounding_box/Assert_3/Assert?;sequential/random_crop/cond/stateful_uniform/RngReadAndSkip|
2sequential/random_crop/cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:r
0sequential/random_crop/cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : v
0sequential/random_crop/cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :????|
2sequential/random_crop/cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
1sequential/random_crop/cond/stateful_uniform/ProdProd;sequential/random_crop/cond/stateful_uniform/shape:output:0;sequential/random_crop/cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: u
3sequential/random_crop/cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
3sequential/random_crop/cond/stateful_uniform/Cast_1Cast:sequential/random_crop/cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
;sequential/random_crop/cond/stateful_uniform/RngReadAndSkipRngReadAndSkipDsequential_random_crop_cond_stateful_uniform_rngreadandskip_resource<sequential/random_crop/cond/stateful_uniform/Cast/x:output:07sequential/random_crop/cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:?
@sequential/random_crop/cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bsequential/random_crop/cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bsequential/random_crop/cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:sequential/random_crop/cond/stateful_uniform/strided_sliceStridedSliceCsequential/random_crop/cond/stateful_uniform/RngReadAndSkip:value:0Isequential/random_crop/cond/stateful_uniform/strided_slice/stack:output:0Ksequential/random_crop/cond/stateful_uniform/strided_slice/stack_1:output:0Ksequential/random_crop/cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
4sequential/random_crop/cond/stateful_uniform/BitcastBitcastCsequential/random_crop/cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0?
Bsequential/random_crop/cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/random_crop/cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/random_crop/cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential/random_crop/cond/stateful_uniform/strided_slice_1StridedSliceCsequential/random_crop/cond/stateful_uniform/RngReadAndSkip:value:0Ksequential/random_crop/cond/stateful_uniform/strided_slice_1/stack:output:0Msequential/random_crop/cond/stateful_uniform/strided_slice_1/stack_1:output:0Msequential/random_crop/cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
6sequential/random_crop/cond/stateful_uniform/Bitcast_1BitcastEsequential/random_crop/cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0r
0sequential/random_crop/cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
,sequential/random_crop/cond/stateful_uniformStatelessRandomUniformIntV2;sequential/random_crop/cond/stateful_uniform/shape:output:0?sequential/random_crop/cond/stateful_uniform/Bitcast_1:output:0=sequential/random_crop/cond/stateful_uniform/Bitcast:output:09sequential/random_crop/cond/stateful_uniform/alg:output:09sequential/random_crop/cond/stateful_uniform/min:output:09sequential/random_crop/cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0y
/sequential/random_crop/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential/random_crop/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential/random_crop/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential/random_crop/cond/strided_sliceStridedSlice5sequential/random_crop/cond/stateful_uniform:output:08sequential/random_crop/cond/strided_slice/stack:output:0:sequential/random_crop/cond/strided_slice/stack_1:output:0:sequential/random_crop/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!sequential/random_crop/cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
sequential/random_crop/cond/addAddV2:sequential_random_crop_cond_add_sequential_random_crop_sub*sequential/random_crop/cond/add/y:output:0*
T0*
_output_shapes
: ?
sequential/random_crop/cond/modFloorMod2sequential/random_crop/cond/strided_slice:output:0#sequential/random_crop/cond/add:z:0*
T0*
_output_shapes
: {
1sequential/random_crop/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3sequential/random_crop/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential/random_crop/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+sequential/random_crop/cond/strided_slice_1StridedSlice5sequential/random_crop/cond/stateful_uniform:output:0:sequential/random_crop/cond/strided_slice_1/stack:output:0<sequential/random_crop/cond/strided_slice_1/stack_1:output:0<sequential/random_crop/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sequential/random_crop/cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
!sequential/random_crop/cond/add_1AddV2>sequential_random_crop_cond_add_1_sequential_random_crop_sub_1,sequential/random_crop/cond/add_1/y:output:0*
T0*
_output_shapes
: ?
!sequential/random_crop/cond/mod_1FloorMod4sequential/random_crop/cond/strided_slice_1:output:0%sequential/random_crop/cond/add_1:z:0*
T0*
_output_shapes
: ?
6sequential/random_crop/cond/crop_to_bounding_box/ShapeShape=sequential_random_crop_cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:?
8sequential/random_crop/cond/crop_to_bounding_box/unstackUnpack?sequential/random_crop/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
num?
?sequential/random_crop/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
=sequential/random_crop/cond/crop_to_bounding_box/GreaterEqualGreaterEqual%sequential/random_crop/cond/mod_1:z:0Hsequential/random_crop/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
=sequential/random_crop/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
Esequential/random_crop/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
>sequential/random_crop/cond/crop_to_bounding_box/Assert/AssertAssertAsequential/random_crop/cond/crop_to_bounding_box/GreaterEqual:z:0Nsequential/random_crop/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 ?
Asequential/random_crop/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
?sequential/random_crop/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqual#sequential/random_crop/cond/mod:z:0Jsequential/random_crop/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
?sequential/random_crop/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
Gsequential/random_crop/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
@sequential/random_crop/cond/crop_to_bounding_box/Assert_1/AssertAssertCsequential/random_crop/cond/crop_to_bounding_box/GreaterEqual_1:z:0Psequential/random_crop/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0?^sequential/random_crop/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 x
6sequential/random_crop/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B : ?
4sequential/random_crop/cond/crop_to_bounding_box/addAddV2?sequential/random_crop/cond/crop_to_bounding_box/add/x:output:0%sequential/random_crop/cond/mod_1:z:0*
T0*
_output_shapes
: ~
<sequential/random_crop/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
:sequential/random_crop/cond/crop_to_bounding_box/LessEqual	LessEqual8sequential/random_crop/cond/crop_to_bounding_box/add:z:0Esequential/random_crop/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
?sequential/random_crop/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
Gsequential/random_crop/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
@sequential/random_crop/cond/crop_to_bounding_box/Assert_2/AssertAssert>sequential/random_crop/cond/crop_to_bounding_box/LessEqual:z:0Psequential/random_crop/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0A^sequential/random_crop/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 z
8sequential/random_crop/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : ?
6sequential/random_crop/cond/crop_to_bounding_box/add_1AddV2Asequential/random_crop/cond/crop_to_bounding_box/add_1/x:output:0#sequential/random_crop/cond/mod:z:0*
T0*
_output_shapes
: ?
>sequential/random_crop/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
<sequential/random_crop/cond/crop_to_bounding_box/LessEqual_1	LessEqual:sequential/random_crop/cond/crop_to_bounding_box/add_1:z:0Gsequential/random_crop/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
?sequential/random_crop/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
Gsequential/random_crop/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
@sequential/random_crop/cond/crop_to_bounding_box/Assert_3/AssertAssert@sequential/random_crop/cond/crop_to_bounding_box/LessEqual_1:z:0Psequential/random_crop/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0A^sequential/random_crop/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
Csequential/random_crop/cond/crop_to_bounding_box/control_dependencyIdentity=sequential_random_crop_cond_crop_to_bounding_box_shape_inputs?^sequential/random_crop/cond/crop_to_bounding_box/Assert/AssertA^sequential/random_crop/cond/crop_to_bounding_box/Assert_1/AssertA^sequential/random_crop/cond/crop_to_bounding_box/Assert_2/AssertA^sequential/random_crop/cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????  z
8sequential/random_crop/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : z
8sequential/random_crop/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
6sequential/random_crop/cond/crop_to_bounding_box/stackPackAsequential/random_crop/cond/crop_to_bounding_box/stack/0:output:0#sequential/random_crop/cond/mod:z:0%sequential/random_crop/cond/mod_1:z:0Asequential/random_crop/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
8sequential/random_crop/cond/crop_to_bounding_box/Shape_1ShapeLsequential/random_crop/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
Dsequential/random_crop/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Fsequential/random_crop/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Fsequential/random_crop/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
>sequential/random_crop/cond/crop_to_bounding_box/strided_sliceStridedSliceAsequential/random_crop/cond/crop_to_bounding_box/Shape_1:output:0Msequential/random_crop/cond/crop_to_bounding_box/strided_slice/stack:output:0Osequential/random_crop/cond/crop_to_bounding_box/strided_slice/stack_1:output:0Osequential/random_crop/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
8sequential/random_crop/cond/crop_to_bounding_box/Shape_2ShapeLsequential/random_crop/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
Fsequential/random_crop/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Hsequential/random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential/random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
@sequential/random_crop/cond/crop_to_bounding_box/strided_slice_1StridedSliceAsequential/random_crop/cond/crop_to_bounding_box/Shape_2:output:0Osequential/random_crop/cond/crop_to_bounding_box/strided_slice_1/stack:output:0Qsequential/random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0Qsequential/random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
:sequential/random_crop/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : |
:sequential/random_crop/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : ?
8sequential/random_crop/cond/crop_to_bounding_box/stack_1PackGsequential/random_crop/cond/crop_to_bounding_box/strided_slice:output:0Csequential/random_crop/cond/crop_to_bounding_box/stack_1/1:output:0Csequential/random_crop/cond/crop_to_bounding_box/stack_1/2:output:0Isequential/random_crop/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
6sequential/random_crop/cond/crop_to_bounding_box/SliceSliceLsequential/random_crop/cond/crop_to_bounding_box/control_dependency:output:0?sequential/random_crop/cond/crop_to_bounding_box/stack:output:0Asequential/random_crop/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:?????????  ?
$sequential/random_crop/cond/IdentityIdentity?sequential/random_crop/cond/crop_to_bounding_box/Slice:output:0!^sequential/random_crop/cond/NoOp*
T0*/
_output_shapes
:?????????  ?
 sequential/random_crop/cond/NoOpNoOp?^sequential/random_crop/cond/crop_to_bounding_box/Assert/AssertA^sequential/random_crop/cond/crop_to_bounding_box/Assert_1/AssertA^sequential/random_crop/cond/crop_to_bounding_box/Assert_2/AssertA^sequential/random_crop/cond/crop_to_bounding_box/Assert_3/Assert<^sequential/random_crop/cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "U
$sequential_random_crop_cond_identity-sequential/random_crop/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :?????????  2?
>sequential/random_crop/cond/crop_to_bounding_box/Assert/Assert>sequential/random_crop/cond/crop_to_bounding_box/Assert/Assert2?
@sequential/random_crop/cond/crop_to_bounding_box/Assert_1/Assert@sequential/random_crop/cond/crop_to_bounding_box/Assert_1/Assert2?
@sequential/random_crop/cond/crop_to_bounding_box/Assert_2/Assert@sequential/random_crop/cond/crop_to_bounding_box/Assert_2/Assert2?
@sequential/random_crop/cond/crop_to_bounding_box/Assert_3/Assert@sequential/random_crop/cond/crop_to_bounding_box/Assert_3/Assert2z
;sequential/random_crop/cond/stateful_uniform/RngReadAndSkip;sequential/random_crop/cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????  
?
}
-__inference_random_crop_layer_call_fn_7018819

inputs
unknown:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_random_crop_layer_call_and_return_conditional_losses_7017264w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????  : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
)__inference_dense_1_layer_call_fn_7018775

inputs
unknown:xT
	unknown_0:T
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_7017428o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????T`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????x: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_7017012

inputs!
random_crop_7017002:	
identity??#random_crop/StatefulPartitionedCall?
#random_crop/StatefulPartitionedCallStatefulPartitionedCallinputsrandom_crop_7017002*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_random_crop_layer_call_and_return_conditional_losses_7017001?
random_flip/PartitionedCallPartitionedCall,random_crop/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_7017009{
IdentityIdentity$random_flip/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  l
NoOpNoOp$^random_crop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????  : 2J
#random_crop/StatefulPartitionedCall#random_crop/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?3
?
#__inference__traced_restore_7019340
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:<
"assignvariableop_2_conv2d_1_kernel:.
 assignvariableop_3_conv2d_1_bias:2
assignvariableop_4_dense_kernel:	?x+
assignvariableop_5_dense_bias:x3
!assignvariableop_6_dense_1_kernel:xT-
assignvariableop_7_dense_1_bias:T3
!assignvariableop_8_dense_2_kernel:T
-
assignvariableop_9_dense_2_bias:
;
-assignvariableop_10_random_crop_cond_statevar:	N
@assignvariableop_11_sequential_1_sequential_random_flip_statevar:	
identity_13??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp-assignvariableop_10_random_crop_cond_statevarIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp@assignvariableop_11_sequential_1_sequential_random_flip_statevarIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
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
?
?
*__inference_conv2d_1_layer_call_fn_7018714

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_7017385w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????

`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?(
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_7017451

inputs 
sequential_7017353:	(
conv2d_7017368:
conv2d_7017370:*
conv2d_1_7017386:
conv2d_1_7017388: 
dense_7017412:	?x
dense_7017414:x!
dense_1_7017429:xT
dense_1_7017431:T!
dense_2_7017445:T

dense_2_7017447:

identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_7017353*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_7017012?
conv2d/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0conv2d_7017368conv2d_7017370*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_7017367?
!average_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7017331?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_1_7017386conv2d_1_7017388*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_7017385?
#average_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_7017343?
flatten/PartitionedCallPartitionedCall,average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_7017398?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_7017412dense_7017414*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_7017411?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_7017429dense_1_7017431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_7017428?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_7017445dense_2_7017447*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_7017444w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????  : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?'
?
cond_false_7017128
cond_placeholder
cond_placeholder_1
cond_placeholder_2
cond_shape_inputs
cond_identityK

cond/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:k
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????m
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/Shape:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

cond/mul/yConst*
_output_shapes
: *
dtype0*
value	B : d
cond/mulMulcond/strided_slice_1:output:0cond/mul/y:output:0*
T0*
_output_shapes
: O
	cond/CastCastcond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: S
cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B`
cond/truedivRealDivcond/Cast:y:0cond/truediv/y:output:0*
T0*
_output_shapes
: U
cond/Cast_1Castcond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: N
cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B : f

cond/mul_1Mulcond/strided_slice:output:0cond/mul_1/y:output:0*
T0*
_output_shapes
: S
cond/Cast_2Castcond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Bf
cond/truediv_1RealDivcond/Cast_2:y:0cond/truediv_1/y:output:0*
T0*
_output_shapes
: W
cond/Cast_3Castcond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
cond/MinimumMinimumcond/strided_slice:output:0cond/Cast_1:y:0*
T0*
_output_shapes
: j
cond/Minimum_1Minimumcond/strided_slice_1:output:0cond/Cast_3:y:0*
T0*
_output_shapes
: _
cond/subSubcond/strided_slice:output:0cond/Minimum:z:0*
T0*
_output_shapes
: Q
cond/Cast_4Castcond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_2RealDivcond/Cast_4:y:0cond/truediv_2/y:output:0*
T0*
_output_shapes
: W
cond/Cast_5Castcond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: e

cond/sub_1Subcond/strided_slice_1:output:0cond/Minimum_1:z:0*
T0*
_output_shapes
: S
cond/Cast_6Castcond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_3RealDivcond/Cast_6:y:0cond/truediv_3/y:output:0*
T0*
_output_shapes
: W
cond/Cast_7Castcond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: N
cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : N
cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?

cond/stackPackcond/stack/0:output:0cond/Cast_5:y:0cond/Cast_7:y:0cond/stack/3:output:0*
N*
T0*
_output_shapes
:Y
cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????Y
cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
cond/stack_1Packcond/stack_1/0:output:0cond/Minimum:z:0cond/Minimum_1:z:0cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?

cond/SliceSlicecond_shape_inputscond/stack:output:0cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"?????????  ?????????a
cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        ?
cond/resize/ResizeBilinearResizeBilinearcond/Slice:output:0cond/resize/size:output:0*
T0*/
_output_shapes
:?????????  *
half_pixel_centers(?
cond/IdentityIdentity+cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:?????????  "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :?????????  :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????  
?'
?
cond_false_7019015
cond_placeholder
cond_placeholder_1
cond_placeholder_2
cond_shape_inputs
cond_identityK

cond/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:k
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????m
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/Shape:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

cond/mul/yConst*
_output_shapes
: *
dtype0*
value	B : d
cond/mulMulcond/strided_slice_1:output:0cond/mul/y:output:0*
T0*
_output_shapes
: O
	cond/CastCastcond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: S
cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B`
cond/truedivRealDivcond/Cast:y:0cond/truediv/y:output:0*
T0*
_output_shapes
: U
cond/Cast_1Castcond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: N
cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B : f

cond/mul_1Mulcond/strided_slice:output:0cond/mul_1/y:output:0*
T0*
_output_shapes
: S
cond/Cast_2Castcond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Bf
cond/truediv_1RealDivcond/Cast_2:y:0cond/truediv_1/y:output:0*
T0*
_output_shapes
: W
cond/Cast_3Castcond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
cond/MinimumMinimumcond/strided_slice:output:0cond/Cast_1:y:0*
T0*
_output_shapes
: j
cond/Minimum_1Minimumcond/strided_slice_1:output:0cond/Cast_3:y:0*
T0*
_output_shapes
: _
cond/subSubcond/strided_slice:output:0cond/Minimum:z:0*
T0*
_output_shapes
: Q
cond/Cast_4Castcond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_2RealDivcond/Cast_4:y:0cond/truediv_2/y:output:0*
T0*
_output_shapes
: W
cond/Cast_5Castcond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: e

cond/sub_1Subcond/strided_slice_1:output:0cond/Minimum_1:z:0*
T0*
_output_shapes
: S
cond/Cast_6Castcond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_3RealDivcond/Cast_6:y:0cond/truediv_3/y:output:0*
T0*
_output_shapes
: W
cond/Cast_7Castcond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: N
cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : N
cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?

cond/stackPackcond/stack/0:output:0cond/Cast_5:y:0cond/Cast_7:y:0cond/stack/3:output:0*
N*
T0*
_output_shapes
:Y
cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????Y
cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
cond/stack_1Packcond/stack_1/0:output:0cond/Minimum:z:0cond/Minimum_1:z:0cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?

cond/SliceSlicecond_shape_inputscond/stack:output:0cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"?????????  ?????????a
cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        ?
cond/resize/ResizeBilinearResizeBilinearcond/Slice:output:0cond/resize/size:output:0*
T0*/
_output_shapes
:?????????  *
half_pixel_centers(?
cond/IdentityIdentity+cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:?????????  "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :?????????  :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????  
?h
?

"__inference__wrapped_model_7016828
sequential_input>
0sequential_1_sequential_random_crop_cond_input_0:	L
2sequential_1_conv2d_conv2d_readvariableop_resource:A
3sequential_1_conv2d_biasadd_readvariableop_resource:N
4sequential_1_conv2d_1_conv2d_readvariableop_resource:C
5sequential_1_conv2d_1_biasadd_readvariableop_resource:D
1sequential_1_dense_matmul_readvariableop_resource:	?x@
2sequential_1_dense_biasadd_readvariableop_resource:xE
3sequential_1_dense_1_matmul_readvariableop_resource:xTB
4sequential_1_dense_1_biasadd_readvariableop_resource:TE
3sequential_1_dense_2_matmul_readvariableop_resource:T
B
4sequential_1_dense_2_biasadd_readvariableop_resource:

identity??*sequential_1/conv2d/BiasAdd/ReadVariableOp?)sequential_1/conv2d/Conv2D/ReadVariableOp?,sequential_1/conv2d_1/BiasAdd/ReadVariableOp?+sequential_1/conv2d_1/Conv2D/ReadVariableOp?)sequential_1/dense/BiasAdd/ReadVariableOp?(sequential_1/dense/MatMul/ReadVariableOp?+sequential_1/dense_1/BiasAdd/ReadVariableOp?*sequential_1/dense_1/MatMul/ReadVariableOp?+sequential_1/dense_2/BiasAdd/ReadVariableOp?*sequential_1/dense_2/MatMul/ReadVariableOp?(sequential_1/sequential/random_crop/condi
)sequential_1/sequential/random_crop/ShapeShapesequential_input*
T0*
_output_shapes
:?
7sequential_1/sequential/random_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
9sequential_1/sequential/random_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
9sequential_1/sequential/random_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1sequential_1/sequential/random_crop/strided_sliceStridedSlice2sequential_1/sequential/random_crop/Shape:output:0@sequential_1/sequential/random_crop/strided_slice/stack:output:0Bsequential_1/sequential/random_crop/strided_slice/stack_1:output:0Bsequential_1/sequential/random_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_1/sequential/random_crop/sub/yConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_1/sequential/random_crop/subSub:sequential_1/sequential/random_crop/strided_slice:output:02sequential_1/sequential/random_crop/sub/y:output:0*
T0*
_output_shapes
: ?
9sequential_1/sequential/random_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
;sequential_1/sequential/random_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
;sequential_1/sequential/random_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3sequential_1/sequential/random_crop/strided_slice_1StridedSlice2sequential_1/sequential/random_crop/Shape:output:0Bsequential_1/sequential/random_crop/strided_slice_1/stack:output:0Dsequential_1/sequential/random_crop/strided_slice_1/stack_1:output:0Dsequential_1/sequential/random_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+sequential_1/sequential/random_crop/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_1/sequential/random_crop/sub_1Sub<sequential_1/sequential/random_crop/strided_slice_1:output:04sequential_1/sequential/random_crop/sub_1/y:output:0*
T0*
_output_shapes
: t
2sequential_1/sequential/random_crop/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
0sequential_1/sequential/random_crop/GreaterEqualGreaterEqual+sequential_1/sequential/random_crop/sub:z:0;sequential_1/sequential/random_crop/GreaterEqual/y:output:0*
T0*
_output_shapes
: v
4sequential_1/sequential/random_crop/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
2sequential_1/sequential/random_crop/GreaterEqual_1GreaterEqual-sequential_1/sequential/random_crop/sub_1:z:0=sequential_1/sequential/random_crop/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: s
1sequential_1/sequential/random_crop/Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z ?
/sequential_1/sequential/random_crop/Rank/packedPack:sequential_1/sequential/random_crop/Rank/packed/0:output:04sequential_1/sequential/random_crop/GreaterEqual:z:06sequential_1/sequential/random_crop/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:j
(sequential_1/sequential/random_crop/RankConst*
_output_shapes
: *
dtype0*
value	B :q
/sequential_1/sequential/random_crop/range/startConst*
_output_shapes
: *
dtype0*
value	B : q
/sequential_1/sequential/random_crop/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
)sequential_1/sequential/random_crop/rangeRange8sequential_1/sequential/random_crop/range/start:output:01sequential_1/sequential/random_crop/Rank:output:08sequential_1/sequential/random_crop/range/delta:output:0*
_output_shapes
:q
/sequential_1/sequential/random_crop/All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z ?
-sequential_1/sequential/random_crop/All/inputPack8sequential_1/sequential/random_crop/All/input/0:output:04sequential_1/sequential/random_crop/GreaterEqual:z:06sequential_1/sequential/random_crop/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:?
'sequential_1/sequential/random_crop/AllAll6sequential_1/sequential/random_crop/All/input:output:02sequential_1/sequential/random_crop/range:output:0*
_output_shapes
: ?
(sequential_1/sequential/random_crop/condIf0sequential_1/sequential/random_crop/All:output:00sequential_1_sequential_random_crop_cond_input_0+sequential_1/sequential/random_crop/sub:z:0-sequential_1/sequential/random_crop/sub_1:z:0sequential_input*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *I
else_branch:R8
6sequential_1_sequential_random_crop_cond_false_7016654*.
output_shapes
:?????????  *H
then_branch9R7
5sequential_1_sequential_random_crop_cond_true_7016653?
1sequential_1/sequential/random_crop/cond/IdentityIdentity1sequential_1/sequential/random_crop/cond:output:0*
T0*/
_output_shapes
:?????????  ?
)sequential_1/conv2d/Conv2D/ReadVariableOpReadVariableOp2sequential_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_1/conv2d/Conv2DConv2D:sequential_1/sequential/random_crop/cond/Identity:output:01sequential_1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
*sequential_1/conv2d/BiasAdd/ReadVariableOpReadVariableOp3sequential_1_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_1/conv2d/BiasAddBiasAdd#sequential_1/conv2d/Conv2D:output:02sequential_1/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
sequential_1/conv2d/ReluRelu$sequential_1/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
&sequential_1/average_pooling2d/AvgPoolAvgPool&sequential_1/conv2d/Relu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
+sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_1/conv2d_1/Conv2DConv2D/sequential_1/average_pooling2d/AvgPool:output:03sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingVALID*
strides
?
,sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_1/conv2d_1/BiasAddBiasAdd%sequential_1/conv2d_1/Conv2D:output:04sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

?
sequential_1/conv2d_1/ReluRelu&sequential_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

?
(sequential_1/average_pooling2d_1/AvgPoolAvgPool(sequential_1/conv2d_1/Relu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
k
sequential_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
sequential_1/flatten/ReshapeReshape1sequential_1/average_pooling2d_1/AvgPool:output:0#sequential_1/flatten/Const:output:0*
T0*(
_output_shapes
:???????????
(sequential_1/dense/MatMul/ReadVariableOpReadVariableOp1sequential_1_dense_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype0?
sequential_1/dense/MatMulMatMul%sequential_1/flatten/Reshape:output:00sequential_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x?
)sequential_1/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0?
sequential_1/dense/BiasAddBiasAdd#sequential_1/dense/MatMul:product:01sequential_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????xv
sequential_1/dense/ReluRelu#sequential_1/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x?
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype0?
sequential_1/dense_1/MatMulMatMul%sequential_1/dense/Relu:activations:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T?
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype0?
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Tz
sequential_1/dense_1/ReluRelu%sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????T?
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype0?
sequential_1/dense_2/MatMulMatMul'sequential_1/dense_1/Relu:activations:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
t
IdentityIdentity%sequential_1/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp+^sequential_1/conv2d/BiasAdd/ReadVariableOp*^sequential_1/conv2d/Conv2D/ReadVariableOp-^sequential_1/conv2d_1/BiasAdd/ReadVariableOp,^sequential_1/conv2d_1/Conv2D/ReadVariableOp*^sequential_1/dense/BiasAdd/ReadVariableOp)^sequential_1/dense/MatMul/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp)^sequential_1/sequential/random_crop/cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????  : : : : : : : : : : : 2X
*sequential_1/conv2d/BiasAdd/ReadVariableOp*sequential_1/conv2d/BiasAdd/ReadVariableOp2V
)sequential_1/conv2d/Conv2D/ReadVariableOp)sequential_1/conv2d/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_1/BiasAdd/ReadVariableOp,sequential_1/conv2d_1/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_1/Conv2D/ReadVariableOp+sequential_1/conv2d_1/Conv2D/ReadVariableOp2V
)sequential_1/dense/BiasAdd/ReadVariableOp)sequential_1/dense/BiasAdd/ReadVariableOp2T
(sequential_1/dense/MatMul/ReadVariableOp(sequential_1/dense/MatMul/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2T
(sequential_1/sequential/random_crop/cond(sequential_1/sequential/random_crop/cond:a ]
/
_output_shapes
:?????????  
*
_user_specified_namesequential_input
?
?
.__inference_sequential_1_layer_call_fn_7017654
sequential_input
unknown:	
	unknown_0:	#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:	?x
	unknown_6:x
	unknown_7:xT
	unknown_8:T
	unknown_9:T


unknown_10:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsequential_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_7017598o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:?????????  
*
_user_specified_namesequential_input
?z
?
random_crop_cond_true_7018484G
9random_crop_cond_stateful_uniform_rngreadandskip_resource:	(
$random_crop_cond_add_random_crop_sub,
(random_crop_cond_add_1_random_crop_sub_16
2random_crop_cond_crop_to_bounding_box_shape_inputs
random_crop_cond_identity??3random_crop/cond/crop_to_bounding_box/Assert/Assert?5random_crop/cond/crop_to_bounding_box/Assert_1/Assert?5random_crop/cond/crop_to_bounding_box/Assert_2/Assert?5random_crop/cond/crop_to_bounding_box/Assert_3/Assert?0random_crop/cond/stateful_uniform/RngReadAndSkipq
'random_crop/cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:g
%random_crop/cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : k
%random_crop/cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :????q
'random_crop/cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
&random_crop/cond/stateful_uniform/ProdProd0random_crop/cond/stateful_uniform/shape:output:00random_crop/cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: j
(random_crop/cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
(random_crop/cond/stateful_uniform/Cast_1Cast/random_crop/cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
0random_crop/cond/stateful_uniform/RngReadAndSkipRngReadAndSkip9random_crop_cond_stateful_uniform_rngreadandskip_resource1random_crop/cond/stateful_uniform/Cast/x:output:0,random_crop/cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:
5random_crop/cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7random_crop/cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7random_crop/cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/random_crop/cond/stateful_uniform/strided_sliceStridedSlice8random_crop/cond/stateful_uniform/RngReadAndSkip:value:0>random_crop/cond/stateful_uniform/strided_slice/stack:output:0@random_crop/cond/stateful_uniform/strided_slice/stack_1:output:0@random_crop/cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
)random_crop/cond/stateful_uniform/BitcastBitcast8random_crop/cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0?
7random_crop/cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9random_crop/cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9random_crop/cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1random_crop/cond/stateful_uniform/strided_slice_1StridedSlice8random_crop/cond/stateful_uniform/RngReadAndSkip:value:0@random_crop/cond/stateful_uniform/strided_slice_1/stack:output:0Brandom_crop/cond/stateful_uniform/strided_slice_1/stack_1:output:0Brandom_crop/cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
+random_crop/cond/stateful_uniform/Bitcast_1Bitcast:random_crop/cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0g
%random_crop/cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
!random_crop/cond/stateful_uniformStatelessRandomUniformIntV20random_crop/cond/stateful_uniform/shape:output:04random_crop/cond/stateful_uniform/Bitcast_1:output:02random_crop/cond/stateful_uniform/Bitcast:output:0.random_crop/cond/stateful_uniform/alg:output:0.random_crop/cond/stateful_uniform/min:output:0.random_crop/cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0n
$random_crop/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&random_crop/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&random_crop/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
random_crop/cond/strided_sliceStridedSlice*random_crop/cond/stateful_uniform:output:0-random_crop/cond/strided_slice/stack:output:0/random_crop/cond/strided_slice/stack_1:output:0/random_crop/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
random_crop/cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
random_crop/cond/addAddV2$random_crop_cond_add_random_crop_subrandom_crop/cond/add/y:output:0*
T0*
_output_shapes
: ?
random_crop/cond/modFloorMod'random_crop/cond/strided_slice:output:0random_crop/cond/add:z:0*
T0*
_output_shapes
: p
&random_crop/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(random_crop/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(random_crop/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 random_crop/cond/strided_slice_1StridedSlice*random_crop/cond/stateful_uniform:output:0/random_crop/cond/strided_slice_1/stack:output:01random_crop/cond/strided_slice_1/stack_1:output:01random_crop/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
random_crop/cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
random_crop/cond/add_1AddV2(random_crop_cond_add_1_random_crop_sub_1!random_crop/cond/add_1/y:output:0*
T0*
_output_shapes
: ?
random_crop/cond/mod_1FloorMod)random_crop/cond/strided_slice_1:output:0random_crop/cond/add_1:z:0*
T0*
_output_shapes
: ?
+random_crop/cond/crop_to_bounding_box/ShapeShape2random_crop_cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:?
-random_crop/cond/crop_to_bounding_box/unstackUnpack4random_crop/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numv
4random_crop/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
2random_crop/cond/crop_to_bounding_box/GreaterEqualGreaterEqualrandom_crop/cond/mod_1:z:0=random_crop/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
2random_crop/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
:random_crop/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
3random_crop/cond/crop_to_bounding_box/Assert/AssertAssert6random_crop/cond/crop_to_bounding_box/GreaterEqual:z:0Crandom_crop/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 x
6random_crop/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
4random_crop/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualrandom_crop/cond/mod:z:0?random_crop/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
4random_crop/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
<random_crop/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
5random_crop/cond/crop_to_bounding_box/Assert_1/AssertAssert8random_crop/cond/crop_to_bounding_box/GreaterEqual_1:z:0Erandom_crop/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:04^random_crop/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 m
+random_crop/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B : ?
)random_crop/cond/crop_to_bounding_box/addAddV24random_crop/cond/crop_to_bounding_box/add/x:output:0random_crop/cond/mod_1:z:0*
T0*
_output_shapes
: s
1random_crop/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
/random_crop/cond/crop_to_bounding_box/LessEqual	LessEqual-random_crop/cond/crop_to_bounding_box/add:z:0:random_crop/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
4random_crop/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
<random_crop/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
5random_crop/cond/crop_to_bounding_box/Assert_2/AssertAssert3random_crop/cond/crop_to_bounding_box/LessEqual:z:0Erandom_crop/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:06^random_crop/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 o
-random_crop/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : ?
+random_crop/cond/crop_to_bounding_box/add_1AddV26random_crop/cond/crop_to_bounding_box/add_1/x:output:0random_crop/cond/mod:z:0*
T0*
_output_shapes
: u
3random_crop/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
1random_crop/cond/crop_to_bounding_box/LessEqual_1	LessEqual/random_crop/cond/crop_to_bounding_box/add_1:z:0<random_crop/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
4random_crop/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
<random_crop/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
5random_crop/cond/crop_to_bounding_box/Assert_3/AssertAssert5random_crop/cond/crop_to_bounding_box/LessEqual_1:z:0Erandom_crop/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:06^random_crop/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
8random_crop/cond/crop_to_bounding_box/control_dependencyIdentity2random_crop_cond_crop_to_bounding_box_shape_inputs4^random_crop/cond/crop_to_bounding_box/Assert/Assert6^random_crop/cond/crop_to_bounding_box/Assert_1/Assert6^random_crop/cond/crop_to_bounding_box/Assert_2/Assert6^random_crop/cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????  o
-random_crop/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : o
-random_crop/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
+random_crop/cond/crop_to_bounding_box/stackPack6random_crop/cond/crop_to_bounding_box/stack/0:output:0random_crop/cond/mod:z:0random_crop/cond/mod_1:z:06random_crop/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
-random_crop/cond/crop_to_bounding_box/Shape_1ShapeArandom_crop/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
9random_crop/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
;random_crop/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;random_crop/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3random_crop/cond/crop_to_bounding_box/strided_sliceStridedSlice6random_crop/cond/crop_to_bounding_box/Shape_1:output:0Brandom_crop/cond/crop_to_bounding_box/strided_slice/stack:output:0Drandom_crop/cond/crop_to_bounding_box/strided_slice/stack_1:output:0Drandom_crop/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
-random_crop/cond/crop_to_bounding_box/Shape_2ShapeArandom_crop/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
;random_crop/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
=random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5random_crop/cond/crop_to_bounding_box/strided_slice_1StridedSlice6random_crop/cond/crop_to_bounding_box/Shape_2:output:0Drandom_crop/cond/crop_to_bounding_box/strided_slice_1/stack:output:0Frandom_crop/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0Frandom_crop/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/random_crop/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : q
/random_crop/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : ?
-random_crop/cond/crop_to_bounding_box/stack_1Pack<random_crop/cond/crop_to_bounding_box/strided_slice:output:08random_crop/cond/crop_to_bounding_box/stack_1/1:output:08random_crop/cond/crop_to_bounding_box/stack_1/2:output:0>random_crop/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
+random_crop/cond/crop_to_bounding_box/SliceSliceArandom_crop/cond/crop_to_bounding_box/control_dependency:output:04random_crop/cond/crop_to_bounding_box/stack:output:06random_crop/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:?????????  ?
random_crop/cond/IdentityIdentity4random_crop/cond/crop_to_bounding_box/Slice:output:0^random_crop/cond/NoOp*
T0*/
_output_shapes
:?????????  ?
random_crop/cond/NoOpNoOp4^random_crop/cond/crop_to_bounding_box/Assert/Assert6^random_crop/cond/crop_to_bounding_box/Assert_1/Assert6^random_crop/cond/crop_to_bounding_box/Assert_2/Assert6^random_crop/cond/crop_to_bounding_box/Assert_3/Assert1^random_crop/cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "?
random_crop_cond_identity"random_crop/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :?????????  2j
3random_crop/cond/crop_to_bounding_box/Assert/Assert3random_crop/cond/crop_to_bounding_box/Assert/Assert2n
5random_crop/cond/crop_to_bounding_box/Assert_1/Assert5random_crop/cond/crop_to_bounding_box/Assert_1/Assert2n
5random_crop/cond/crop_to_bounding_box/Assert_2/Assert5random_crop/cond/crop_to_bounding_box/Assert_2/Assert2n
5random_crop/cond/crop_to_bounding_box/Assert_3/Assert5random_crop/cond/crop_to_bounding_box/Assert_3/Assert2d
0random_crop/cond/stateful_uniform/RngReadAndSkip0random_crop/cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????  "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
U
sequential_inputA
"serving_default_sequential_input:0?????????  ;
dense_20
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
?
layer-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
?
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
?

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
?
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
?
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
?

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_layer
f
0
1
(2
)3
<4
=5
D6
E7
L8
M9"
trackable_list_wrapper
f
0
1
(2
)3
<4
=5
D6
E7
L8
M9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_sequential_1_layer_call_fn_7017476
.__inference_sequential_1_layer_call_fn_7017753
.__inference_sequential_1_layer_call_fn_7017782
.__inference_sequential_1_layer_call_fn_7017654?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_1_layer_call_and_return_conditional_losses_7017986
I__inference_sequential_1_layer_call_and_return_conditional_losses_7018244
I__inference_sequential_1_layer_call_and_return_conditional_losses_7017689
I__inference_sequential_1_layer_call_and_return_conditional_losses_7017726?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference__wrapped_model_7016828sequential_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
Yserving_default"
signature_map
?
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^_random_generator
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
?
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e_random_generator
f__call__
*g&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_sequential_layer_call_fn_7017017
,__inference_sequential_layer_call_fn_7018280
,__inference_sequential_layer_call_fn_7018289
,__inference_sequential_layer_call_fn_7017304?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_layer_call_and_return_conditional_losses_7018455
G__inference_sequential_layer_call_and_return_conditional_losses_7018675
G__inference_sequential_layer_call_and_return_conditional_losses_7017312
G__inference_sequential_layer_call_and_return_conditional_losses_7017322?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
':%2conv2d/kernel
:2conv2d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_conv2d_layer_call_fn_7018684?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_layer_call_and_return_conditional_losses_7018695?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
?2?
3__inference_average_pooling2d_layer_call_fn_7018700?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7018705?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
):'2conv2d_1/kernel
:2conv2d_1/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_1_layer_call_fn_7018714?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_7018725?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
?2?
5__inference_average_pooling2d_1_layer_call_fn_7018730?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_7018735?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_flatten_layer_call_fn_7018740?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_layer_call_and_return_conditional_losses_7018746?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	?x2dense/kernel
:x2
dense/bias
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_dense_layer_call_fn_7018755?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_layer_call_and_return_conditional_losses_7018766?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 :xT2dense_1/kernel
:T2dense_1/bias
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_dense_1_layer_call_fn_7018775?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_1_layer_call_and_return_conditional_losses_7018786?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 :T
2dense_2/kernel
:
2dense_2/bias
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_dense_2_layer_call_fn_7018795?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_2_layer_call_and_return_conditional_losses_7018805?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_signature_wrapper_7018273sequential_input"?
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
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
/
?
_generator"
_generic_user_object
?2?
-__inference_random_crop_layer_call_fn_7018812
-__inference_random_crop_layer_call_fn_7018819?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_random_crop_layer_call_and_return_conditional_losses_7018985
H__inference_random_crop_layer_call_and_return_conditional_losses_7019151?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
a	variables
btrainable_variables
cregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
/
?
_generator"
_generic_user_object
?2?
-__inference_random_flip_layer_call_fn_7019156
-__inference_random_flip_layer_call_fn_7019163?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_random_flip_layer_call_and_return_conditional_losses_7019167
H__inference_random_flip_layer_call_and_return_conditional_losses_7019225?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
/
?
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
/
?
_state_var"
_generic_user_object
%:#	2random_crop/cond/StateVar
8:6	2,sequential_1/sequential/random_flip/StateVar?
"__inference__wrapped_model_7016828??()<=DELMA?>
7?4
2?/
sequential_input?????????  
? "1?.
,
dense_2!?
dense_2?????????
?
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_7018735?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
5__inference_average_pooling2d_1_layer_call_fn_7018730?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_7018705?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
3__inference_average_pooling2d_layer_call_fn_7018700?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
E__inference_conv2d_1_layer_call_and_return_conditional_losses_7018725l()7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????


? ?
*__inference_conv2d_1_layer_call_fn_7018714_()7?4
-?*
(?%
inputs?????????
? " ??????????

?
C__inference_conv2d_layer_call_and_return_conditional_losses_7018695l7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????
? ?
(__inference_conv2d_layer_call_fn_7018684_7?4
-?*
(?%
inputs?????????  
? " ???????????
D__inference_dense_1_layer_call_and_return_conditional_losses_7018786\DE/?,
%?"
 ?
inputs?????????x
? "%?"
?
0?????????T
? |
)__inference_dense_1_layer_call_fn_7018775ODE/?,
%?"
 ?
inputs?????????x
? "??????????T?
D__inference_dense_2_layer_call_and_return_conditional_losses_7018805\LM/?,
%?"
 ?
inputs?????????T
? "%?"
?
0?????????

? |
)__inference_dense_2_layer_call_fn_7018795OLM/?,
%?"
 ?
inputs?????????T
? "??????????
?
B__inference_dense_layer_call_and_return_conditional_losses_7018766]<=0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????x
? {
'__inference_dense_layer_call_fn_7018755P<=0?-
&?#
!?
inputs??????????
? "??????????x?
D__inference_flatten_layer_call_and_return_conditional_losses_7018746a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
)__inference_flatten_layer_call_fn_7018740T7?4
-?*
(?%
inputs?????????
? "????????????
H__inference_random_crop_layer_call_and_return_conditional_losses_7018985p?;?8
1?.
(?%
inputs?????????  
p 
? "-?*
#? 
0?????????  
? ?
H__inference_random_crop_layer_call_and_return_conditional_losses_7019151p?;?8
1?.
(?%
inputs?????????  
p
? "-?*
#? 
0?????????  
? ?
-__inference_random_crop_layer_call_fn_7018812c?;?8
1?.
(?%
inputs?????????  
p 
? " ??????????  ?
-__inference_random_crop_layer_call_fn_7018819c?;?8
1?.
(?%
inputs?????????  
p
? " ??????????  ?
H__inference_random_flip_layer_call_and_return_conditional_losses_7019167l;?8
1?.
(?%
inputs?????????  
p 
? "-?*
#? 
0?????????  
? ?
H__inference_random_flip_layer_call_and_return_conditional_losses_7019225p?;?8
1?.
(?%
inputs?????????  
p
? "-?*
#? 
0?????????  
? ?
-__inference_random_flip_layer_call_fn_7019156_;?8
1?.
(?%
inputs?????????  
p 
? " ??????????  ?
-__inference_random_flip_layer_call_fn_7019163c?;?8
1?.
(?%
inputs?????????  
p
? " ??????????  ?
I__inference_sequential_1_layer_call_and_return_conditional_losses_7017689??()<=DELMI?F
??<
2?/
sequential_input?????????  
p 

 
? "%?"
?
0?????????

? ?
I__inference_sequential_1_layer_call_and_return_conditional_losses_7017726???()<=DELMI?F
??<
2?/
sequential_input?????????  
p

 
? "%?"
?
0?????????

? ?
I__inference_sequential_1_layer_call_and_return_conditional_losses_7017986v?()<=DELM??<
5?2
(?%
inputs?????????  
p 

 
? "%?"
?
0?????????

? ?
I__inference_sequential_1_layer_call_and_return_conditional_losses_7018244x??()<=DELM??<
5?2
(?%
inputs?????????  
p

 
? "%?"
?
0?????????

? ?
.__inference_sequential_1_layer_call_fn_7017476s?()<=DELMI?F
??<
2?/
sequential_input?????????  
p 

 
? "??????????
?
.__inference_sequential_1_layer_call_fn_7017654u??()<=DELMI?F
??<
2?/
sequential_input?????????  
p

 
? "??????????
?
.__inference_sequential_1_layer_call_fn_7017753i?()<=DELM??<
5?2
(?%
inputs?????????  
p 

 
? "??????????
?
.__inference_sequential_1_layer_call_fn_7017782k??()<=DELM??<
5?2
(?%
inputs?????????  
p

 
? "??????????
?
G__inference_sequential_layer_call_and_return_conditional_losses_7017312?J?G
@?=
3?0
random_crop_input?????????  
p 

 
? "-?*
#? 
0?????????  
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_7017322???J?G
@?=
3?0
random_crop_input?????????  
p

 
? "-?*
#? 
0?????????  
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_7018455t???<
5?2
(?%
inputs?????????  
p 

 
? "-?*
#? 
0?????????  
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_7018675v????<
5?2
(?%
inputs?????????  
p

 
? "-?*
#? 
0?????????  
? ?
,__inference_sequential_layer_call_fn_7017017r?J?G
@?=
3?0
random_crop_input?????????  
p 

 
? " ??????????  ?
,__inference_sequential_layer_call_fn_7017304t??J?G
@?=
3?0
random_crop_input?????????  
p

 
? " ??????????  ?
,__inference_sequential_layer_call_fn_7018280g???<
5?2
(?%
inputs?????????  
p 

 
? " ??????????  ?
,__inference_sequential_layer_call_fn_7018289i????<
5?2
(?%
inputs?????????  
p

 
? " ??????????  ?
%__inference_signature_wrapper_7018273??()<=DELMU?R
? 
K?H
F
sequential_input2?/
sequential_input?????????  "1?.
,
dense_2!?
dense_2?????????
