ª
æ·
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
validate_shapebool( 
¼
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

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
º
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
 
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
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ã½

conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
:*
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
:*
dtype0

conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
:*
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
:*
dtype0
{
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	x* 
shared_namedense_15/kernel
t
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes
:	x*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:x*
dtype0
z
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xT* 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

:xT*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:T*
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T
* 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

:T
*
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:
*
dtype0

random_crop_5/cond/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*,
shared_namerandom_crop_5/cond/StateVar

/random_crop_5/cond/StateVar/Read/ReadVariableOpReadVariableOprandom_crop_5/cond/StateVar*
_output_shapes
:*
dtype0	
¼
2sequential_16/sequential_15/random_flip_5/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*C
shared_name42sequential_16/sequential_15/random_flip_5/StateVar
µ
Fsequential_16/sequential_15/random_flip_5/StateVar/Read/ReadVariableOpReadVariableOp2sequential_16/sequential_15/random_flip_5/StateVar*
_output_shapes
:*
dtype0	

NoOpNoOp
>
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ë=
valueÁ=B¾= B·=
´
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
ª
layer-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*

"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses* 
¦

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses*

0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses* 

6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses* 
¦

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses*
¦

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses*
¦

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
°
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
§
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^_random_generator
___call__
*`&call_and_return_all_conditional_losses*
§
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

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
`Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

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

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
`Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

(0
)1*

(0
)1*
* 

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

|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_15/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

<0
=1*

<0
=1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_16/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

D0
E1*

D0
E1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_17/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_17/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

L0
M1*

L0
M1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Z	variables
[trainable_variables
\regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses* 


_generator*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
a	variables
btrainable_variables
cregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses* 

 
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
¡
_state_var*
* 
* 
* 
* 
* 

¢
_state_var*

VARIABLE_VALUErandom_crop_5/cond/StateVarRlayer-0/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2sequential_16/sequential_15/random_flip_5/StateVarRlayer-0/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*

#serving_default_sequential_15_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ  

StatefulPartitionedCallStatefulPartitionedCall#serving_default_sequential_15_inputrandom_crop_5/cond/StateVarconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_22200583
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOp/random_crop_5/cond/StateVar/Read/ReadVariableOpFsequential_16/sequential_15/random_flip_5/StateVar/Read/ReadVariableOpConst*
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
GPU 2J 8 **
f%R#
!__inference__traced_save_22201604

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasrandom_crop_5/cond/StateVar2sequential_16/sequential_15/random_flip_5/StateVar*
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
GPU 2J 8 *-
f(R&
$__inference__traced_restore_22201650§×

g
K__inference_random_flip_5_layer_call_and_return_conditional_losses_22199319

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
·

K__inference_random_crop_5_layer_call_and_return_conditional_losses_22199574

inputs
cond_input_0:	
identity¢cond;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿh
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
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
þÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
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
: â
condIfAll:output:0cond_input_0sub:z:0	sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *&
else_branchR
cond_false_22199438*.
output_shapes
:ÿÿÿÿÿÿÿÿÿ  *%
then_branchR
cond_true_22199437b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  M
NoOpNoOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  : 2
condcond:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
£
n
R__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_22199653

inputs
identity«
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î=
­
/sequential_15_random_crop_5_cond_false_222001220
,sequential_15_random_crop_5_cond_placeholder2
.sequential_15_random_crop_5_cond_placeholder_12
.sequential_15_random_crop_5_cond_placeholder_21
-sequential_15_random_crop_5_cond_shape_inputs-
)sequential_15_random_crop_5_cond_identity
&sequential_15/random_crop_5/cond/ShapeShape-sequential_15_random_crop_5_cond_shape_inputs*
T0*
_output_shapes
:
4sequential_15/random_crop_5/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
6sequential_15/random_crop_5/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
6sequential_15/random_crop_5/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.sequential_15/random_crop_5/cond/strided_sliceStridedSlice/sequential_15/random_crop_5/cond/Shape:output:0=sequential_15/random_crop_5/cond/strided_slice/stack:output:0?sequential_15/random_crop_5/cond/strided_slice/stack_1:output:0?sequential_15/random_crop_5/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
6sequential_15/random_crop_5/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
8sequential_15/random_crop_5/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
8sequential_15/random_crop_5/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
0sequential_15/random_crop_5/cond/strided_slice_1StridedSlice/sequential_15/random_crop_5/cond/Shape:output:0?sequential_15/random_crop_5/cond/strided_slice_1/stack:output:0Asequential_15/random_crop_5/cond/strided_slice_1/stack_1:output:0Asequential_15/random_crop_5/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&sequential_15/random_crop_5/cond/mul/yConst*
_output_shapes
: *
dtype0*
value	B : ¸
$sequential_15/random_crop_5/cond/mulMul9sequential_15/random_crop_5/cond/strided_slice_1:output:0/sequential_15/random_crop_5/cond/mul/y:output:0*
T0*
_output_shapes
: 
%sequential_15/random_crop_5/cond/CastCast(sequential_15/random_crop_5/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: o
*sequential_15/random_crop_5/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B´
(sequential_15/random_crop_5/cond/truedivRealDiv)sequential_15/random_crop_5/cond/Cast:y:03sequential_15/random_crop_5/cond/truediv/y:output:0*
T0*
_output_shapes
: 
'sequential_15/random_crop_5/cond/Cast_1Cast,sequential_15/random_crop_5/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: j
(sequential_15/random_crop_5/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B : º
&sequential_15/random_crop_5/cond/mul_1Mul7sequential_15/random_crop_5/cond/strided_slice:output:01sequential_15/random_crop_5/cond/mul_1/y:output:0*
T0*
_output_shapes
: 
'sequential_15/random_crop_5/cond/Cast_2Cast*sequential_15/random_crop_5/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: q
,sequential_15/random_crop_5/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Bº
*sequential_15/random_crop_5/cond/truediv_1RealDiv+sequential_15/random_crop_5/cond/Cast_2:y:05sequential_15/random_crop_5/cond/truediv_1/y:output:0*
T0*
_output_shapes
: 
'sequential_15/random_crop_5/cond/Cast_3Cast.sequential_15/random_crop_5/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: º
(sequential_15/random_crop_5/cond/MinimumMinimum7sequential_15/random_crop_5/cond/strided_slice:output:0+sequential_15/random_crop_5/cond/Cast_1:y:0*
T0*
_output_shapes
: ¾
*sequential_15/random_crop_5/cond/Minimum_1Minimum9sequential_15/random_crop_5/cond/strided_slice_1:output:0+sequential_15/random_crop_5/cond/Cast_3:y:0*
T0*
_output_shapes
: ³
$sequential_15/random_crop_5/cond/subSub7sequential_15/random_crop_5/cond/strided_slice:output:0,sequential_15/random_crop_5/cond/Minimum:z:0*
T0*
_output_shapes
: 
'sequential_15/random_crop_5/cond/Cast_4Cast(sequential_15/random_crop_5/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: q
,sequential_15/random_crop_5/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @º
*sequential_15/random_crop_5/cond/truediv_2RealDiv+sequential_15/random_crop_5/cond/Cast_4:y:05sequential_15/random_crop_5/cond/truediv_2/y:output:0*
T0*
_output_shapes
: 
'sequential_15/random_crop_5/cond/Cast_5Cast.sequential_15/random_crop_5/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ¹
&sequential_15/random_crop_5/cond/sub_1Sub9sequential_15/random_crop_5/cond/strided_slice_1:output:0.sequential_15/random_crop_5/cond/Minimum_1:z:0*
T0*
_output_shapes
: 
'sequential_15/random_crop_5/cond/Cast_6Cast*sequential_15/random_crop_5/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: q
,sequential_15/random_crop_5/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @º
*sequential_15/random_crop_5/cond/truediv_3RealDiv+sequential_15/random_crop_5/cond/Cast_6:y:05sequential_15/random_crop_5/cond/truediv_3/y:output:0*
T0*
_output_shapes
: 
'sequential_15/random_crop_5/cond/Cast_7Cast.sequential_15/random_crop_5/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: j
(sequential_15/random_crop_5/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : j
(sequential_15/random_crop_5/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 
&sequential_15/random_crop_5/cond/stackPack1sequential_15/random_crop_5/cond/stack/0:output:0+sequential_15/random_crop_5/cond/Cast_5:y:0+sequential_15/random_crop_5/cond/Cast_7:y:01sequential_15/random_crop_5/cond/stack/3:output:0*
N*
T0*
_output_shapes
:u
*sequential_15/random_crop_5/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿu
*sequential_15/random_crop_5/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
(sequential_15/random_crop_5/cond/stack_1Pack3sequential_15/random_crop_5/cond/stack_1/0:output:0,sequential_15/random_crop_5/cond/Minimum:z:0.sequential_15/random_crop_5/cond/Minimum_1:z:03sequential_15/random_crop_5/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:
&sequential_15/random_crop_5/cond/SliceSlice-sequential_15_random_crop_5_cond_shape_inputs/sequential_15/random_crop_5/cond/stack:output:01sequential_15/random_crop_5/cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ  ÿÿÿÿÿÿÿÿÿ}
,sequential_15/random_crop_5/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        
6sequential_15/random_crop_5/cond/resize/ResizeBilinearResizeBilinear/sequential_15/random_crop_5/cond/Slice:output:05sequential_15/random_crop_5/cond/resize/size:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
half_pixel_centers(¸
)sequential_15/random_crop_5/cond/IdentityIdentityGsequential_15/random_crop_5/cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "_
)sequential_15_random_crop_5_cond_identity2sequential_15/random_crop_5/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :ÿÿÿÿÿÿÿÿÿ  :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
ª¾


K__inference_sequential_16_layer_call_and_return_conditional_losses_22200554

inputs6
(sequential_15_random_crop_5_cond_input_0:	[
Msequential_15_random_flip_5_stateful_uniform_full_int_rngreadandskip_resource:	B
(conv2d_10_conv2d_readvariableop_resource:7
)conv2d_10_biasadd_readvariableop_resource:B
(conv2d_11_conv2d_readvariableop_resource:7
)conv2d_11_biasadd_readvariableop_resource::
'dense_15_matmul_readvariableop_resource:	x6
(dense_15_biasadd_readvariableop_resource:x9
'dense_16_matmul_readvariableop_resource:xT6
(dense_16_biasadd_readvariableop_resource:T9
'dense_17_matmul_readvariableop_resource:T
6
(dense_17_biasadd_readvariableop_resource:

identity¢ conv2d_10/BiasAdd/ReadVariableOp¢conv2d_10/Conv2D/ReadVariableOp¢ conv2d_11/BiasAdd/ReadVariableOp¢conv2d_11/Conv2D/ReadVariableOp¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¢dense_16/BiasAdd/ReadVariableOp¢dense_16/MatMul/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢dense_17/MatMul/ReadVariableOp¢ sequential_15/random_crop_5/cond¢Dsequential_15/random_flip_5/stateful_uniform_full_int/RngReadAndSkipW
!sequential_15/random_crop_5/ShapeShapeinputs*
T0*
_output_shapes
:
/sequential_15/random_crop_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
1sequential_15/random_crop_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ{
1sequential_15/random_crop_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)sequential_15/random_crop_5/strided_sliceStridedSlice*sequential_15/random_crop_5/Shape:output:08sequential_15/random_crop_5/strided_slice/stack:output:0:sequential_15/random_crop_5/strided_slice/stack_1:output:0:sequential_15/random_crop_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!sequential_15/random_crop_5/sub/yConst*
_output_shapes
: *
dtype0*
value	B : §
sequential_15/random_crop_5/subSub2sequential_15/random_crop_5/strided_slice:output:0*sequential_15/random_crop_5/sub/y:output:0*
T0*
_output_shapes
: 
1sequential_15/random_crop_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
3sequential_15/random_crop_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ}
3sequential_15/random_crop_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+sequential_15/random_crop_5/strided_slice_1StridedSlice*sequential_15/random_crop_5/Shape:output:0:sequential_15/random_crop_5/strided_slice_1/stack:output:0<sequential_15/random_crop_5/strided_slice_1/stack_1:output:0<sequential_15/random_crop_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sequential_15/random_crop_5/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B : ­
!sequential_15/random_crop_5/sub_1Sub4sequential_15/random_crop_5/strided_slice_1:output:0,sequential_15/random_crop_5/sub_1/y:output:0*
T0*
_output_shapes
: l
*sequential_15/random_crop_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ³
(sequential_15/random_crop_5/GreaterEqualGreaterEqual#sequential_15/random_crop_5/sub:z:03sequential_15/random_crop_5/GreaterEqual/y:output:0*
T0*
_output_shapes
: n
,sequential_15/random_crop_5/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ¹
*sequential_15/random_crop_5/GreaterEqual_1GreaterEqual%sequential_15/random_crop_5/sub_1:z:05sequential_15/random_crop_5/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: k
)sequential_15/random_crop_5/Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Zï
'sequential_15/random_crop_5/Rank/packedPack2sequential_15/random_crop_5/Rank/packed/0:output:0,sequential_15/random_crop_5/GreaterEqual:z:0.sequential_15/random_crop_5/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:b
 sequential_15/random_crop_5/RankConst*
_output_shapes
: *
dtype0*
value	B :i
'sequential_15/random_crop_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : i
'sequential_15/random_crop_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Õ
!sequential_15/random_crop_5/rangeRange0sequential_15/random_crop_5/range/start:output:0)sequential_15/random_crop_5/Rank:output:00sequential_15/random_crop_5/range/delta:output:0*
_output_shapes
:i
'sequential_15/random_crop_5/All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Zë
%sequential_15/random_crop_5/All/inputPack0sequential_15/random_crop_5/All/input/0:output:0,sequential_15/random_crop_5/GreaterEqual:z:0.sequential_15/random_crop_5/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:
sequential_15/random_crop_5/AllAll.sequential_15/random_crop_5/All/input:output:0*sequential_15/random_crop_5/range:output:0*
_output_shapes
: ¦
 sequential_15/random_crop_5/condIf(sequential_15/random_crop_5/All:output:0(sequential_15_random_crop_5_cond_input_0#sequential_15/random_crop_5/sub:z:0%sequential_15/random_crop_5/sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *B
else_branch3R1
/sequential_15_random_crop_5_cond_false_22200326*.
output_shapes
:ÿÿÿÿÿÿÿÿÿ  *A
then_branch2R0
.sequential_15_random_crop_5_cond_true_22200325
)sequential_15/random_crop_5/cond/IdentityIdentity)sequential_15/random_crop_5/cond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
;sequential_15/random_flip_5/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:
;sequential_15/random_flip_5/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ï
:sequential_15/random_flip_5/stateful_uniform_full_int/ProdProdDsequential_15/random_flip_5/stateful_uniform_full_int/shape:output:0Dsequential_15/random_flip_5/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: ~
<sequential_15/random_flip_5/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :¹
<sequential_15/random_flip_5/stateful_uniform_full_int/Cast_1CastCsequential_15/random_flip_5/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Ê
Dsequential_15/random_flip_5/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipMsequential_15_random_flip_5_stateful_uniform_full_int_rngreadandskip_resourceEsequential_15/random_flip_5/stateful_uniform_full_int/Cast/x:output:0@sequential_15/random_flip_5/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
Isequential_15/random_flip_5/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ksequential_15/random_flip_5/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ksequential_15/random_flip_5/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
Csequential_15/random_flip_5/stateful_uniform_full_int/strided_sliceStridedSliceLsequential_15/random_flip_5/stateful_uniform_full_int/RngReadAndSkip:value:0Rsequential_15/random_flip_5/stateful_uniform_full_int/strided_slice/stack:output:0Tsequential_15/random_flip_5/stateful_uniform_full_int/strided_slice/stack_1:output:0Tsequential_15/random_flip_5/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskÇ
=sequential_15/random_flip_5/stateful_uniform_full_int/BitcastBitcastLsequential_15/random_flip_5/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Ksequential_15/random_flip_5/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Msequential_15/random_flip_5/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Msequential_15/random_flip_5/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
Esequential_15/random_flip_5/stateful_uniform_full_int/strided_slice_1StridedSliceLsequential_15/random_flip_5/stateful_uniform_full_int/RngReadAndSkip:value:0Tsequential_15/random_flip_5/stateful_uniform_full_int/strided_slice_1/stack:output:0Vsequential_15/random_flip_5/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Vsequential_15/random_flip_5/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ë
?sequential_15/random_flip_5/stateful_uniform_full_int/Bitcast_1BitcastNsequential_15/random_flip_5/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0{
9sequential_15/random_flip_5/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :
5sequential_15/random_flip_5/stateful_uniform_full_intStatelessRandomUniformFullIntV2Dsequential_15/random_flip_5/stateful_uniform_full_int/shape:output:0Hsequential_15/random_flip_5/stateful_uniform_full_int/Bitcast_1:output:0Fsequential_15/random_flip_5/stateful_uniform_full_int/Bitcast:output:0Bsequential_15/random_flip_5/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	p
&sequential_15/random_flip_5/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R Ì
!sequential_15/random_flip_5/stackPack>sequential_15/random_flip_5/stateful_uniform_full_int:output:0/sequential_15/random_flip_5/zeros_like:output:0*
N*
T0	*
_output_shapes

:
/sequential_15/random_flip_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1sequential_15/random_flip_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
1sequential_15/random_flip_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
)sequential_15/random_flip_5/strided_sliceStridedSlice*sequential_15/random_flip_5/stack:output:08sequential_15/random_flip_5/strided_slice/stack:output:0:sequential_15/random_flip_5/strided_slice/stack_1:output:0:sequential_15/random_flip_5/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask
Osequential_15/random_flip_5/stateless_random_flip_left_right/control_dependencyIdentity2sequential_15/random_crop_5/cond/Identity:output:0*
T0*<
_class2
0.loc:@sequential_15/random_crop_5/cond/Identity*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  Ê
Bsequential_15/random_flip_5/stateless_random_flip_left_right/ShapeShapeXsequential_15/random_flip_5/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:
Psequential_15/random_flip_5/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Rsequential_15/random_flip_5/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Rsequential_15/random_flip_5/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Jsequential_15/random_flip_5/stateless_random_flip_left_right/strided_sliceStridedSliceKsequential_15/random_flip_5/stateless_random_flip_left_right/Shape:output:0Ysequential_15/random_flip_5/stateless_random_flip_left_right/strided_slice/stack:output:0[sequential_15/random_flip_5/stateless_random_flip_left_right/strided_slice/stack_1:output:0[sequential_15/random_flip_5/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskæ
[sequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniform/shapePackSsequential_15/random_flip_5/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:
Ysequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
Ysequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?è
rsequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter2sequential_15/random_flip_5/strided_slice:output:0* 
_output_shapes
::´
rsequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
nsequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2dsequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0xsequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0|sequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0{sequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
Ysequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniform/subSubbsequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniform/max:output:0bsequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: æ
Ysequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniform/mulMulwsequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0]sequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
Usequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniformAddV2]sequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0bsequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Lsequential_15/random_flip_5/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
Lsequential_15/random_flip_5/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Lsequential_15/random_flip_5/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Ú
Jsequential_15/random_flip_5/stateless_random_flip_left_right/Reshape/shapePackSsequential_15/random_flip_5/stateless_random_flip_left_right/strided_slice:output:0Usequential_15/random_flip_5/stateless_random_flip_left_right/Reshape/shape/1:output:0Usequential_15/random_flip_5/stateless_random_flip_left_right/Reshape/shape/2:output:0Usequential_15/random_flip_5/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:¹
Dsequential_15/random_flip_5/stateless_random_flip_left_right/ReshapeReshapeYsequential_15/random_flip_5/stateless_random_flip_left_right/stateless_random_uniform:z:0Ssequential_15/random_flip_5/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
Bsequential_15/random_flip_5/stateless_random_flip_left_right/RoundRoundMsequential_15/random_flip_5/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ksequential_15/random_flip_5/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:½
Fsequential_15/random_flip_5/stateless_random_flip_left_right/ReverseV2	ReverseV2Xsequential_15/random_flip_5/stateless_random_flip_left_right/control_dependency:output:0Tsequential_15/random_flip_5/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
@sequential_15/random_flip_5/stateless_random_flip_left_right/mulMulFsequential_15/random_flip_5/stateless_random_flip_left_right/Round:y:0Osequential_15/random_flip_5/stateless_random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
Bsequential_15/random_flip_5/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
@sequential_15/random_flip_5/stateless_random_flip_left_right/subSubKsequential_15/random_flip_5/stateless_random_flip_left_right/sub/x:output:0Fsequential_15/random_flip_5/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
Bsequential_15/random_flip_5/stateless_random_flip_left_right/mul_1MulDsequential_15/random_flip_5/stateless_random_flip_left_right/sub:z:0Xsequential_15/random_flip_5/stateless_random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
@sequential_15/random_flip_5/stateless_random_flip_left_right/addAddV2Dsequential_15/random_flip_5/stateless_random_flip_left_right/mul:z:0Fsequential_15/random_flip_5/stateless_random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ì
conv2d_10/Conv2DConv2DDsequential_15/random_flip_5/stateless_random_flip_left_right/add:z:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
average_pooling2d_10/AvgPoolAvgPoolconv2d_10/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Í
conv2d_11/Conv2DConv2D%average_pooling2d_10/AvgPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
paddingVALID*
strides

 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

l
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

»
average_pooling2d_11/AvgPoolAvgPoolconv2d_11/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
`
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  
flatten_5/ReshapeReshape%average_pooling2d_11/AvgPool:output:0flatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	x*
dtype0
dense_15/MatMulMatMulflatten_5/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxb
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype0
dense_16/MatMulMatMuldense_15/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype0
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTb
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype0
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
IdentityIdentitydense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp!^sequential_15/random_crop_5/condE^sequential_15/random_flip_5/stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2D
 sequential_15/random_crop_5/cond sequential_15/random_crop_5/cond2
Dsequential_15/random_flip_5/stateful_uniform_full_int/RngReadAndSkipDsequential_15/random_flip_5/stateful_uniform_full_int/RngReadAndSkip:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
½
¸
K__inference_sequential_15_layer_call_and_return_conditional_losses_22199322

inputs$
random_crop_5_22199312:	
identity¢%random_crop_5/StatefulPartitionedCallõ
%random_crop_5/StatefulPartitionedCallStatefulPartitionedCallinputsrandom_crop_5_22199312*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_random_crop_5_layer_call_and_return_conditional_losses_22199311ô
random_flip_5/PartitionedCallPartitionedCall.random_crop_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_random_flip_5_layer_call_and_return_conditional_losses_22199319}
IdentityIdentity&random_flip_5/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  n
NoOpNoOp&^random_crop_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  : 2N
%random_crop_5/StatefulPartitionedCall%random_crop_5/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

¦
0__inference_sequential_15_layer_call_fn_22199614
random_crop_5_input
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallrandom_crop_5_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_15_layer_call_and_return_conditional_losses_22199598w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
-
_user_specified_namerandom_crop_5_input


G__inference_conv2d_11_layer_call_and_return_conditional_losses_22199695

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ

X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡

ø
F__inference_dense_15_layer_call_and_return_conditional_losses_22199721

inputs1
matmul_readvariableop_resource:	x-
biasadd_readvariableop_resource:x
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


÷
F__inference_dense_16_layer_call_and_return_conditional_losses_22201096

inputs0
matmul_readvariableop_resource:xT-
biasadd_readvariableop_resource:T
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿx: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
Ã*
é
K__inference_sequential_16_layer_call_and_return_conditional_losses_22199999
sequential_15_input$
sequential_15_22199967:	,
conv2d_10_22199970: 
conv2d_10_22199972:,
conv2d_11_22199976: 
conv2d_11_22199978:$
dense_15_22199983:	x
dense_15_22199985:x#
dense_16_22199988:xT
dense_16_22199990:T#
dense_17_22199993:T

dense_17_22199995:

identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢%sequential_15/StatefulPartitionedCall
%sequential_15/StatefulPartitionedCallStatefulPartitionedCallsequential_15_inputsequential_15_22199967*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_15_layer_call_and_return_conditional_losses_22199322ª
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall.sequential_15/StatefulPartitionedCall:output:0conv2d_10_22199970conv2d_10_22199972*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_22199677þ
$average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_22199641©
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_10/PartitionedCall:output:0conv2d_11_22199976conv2d_11_22199978*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_22199695þ
$average_pooling2d_11/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_22199653ä
flatten_5/PartitionedCallPartitionedCall-average_pooling2d_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_22199708
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_15_22199983dense_15_22199985*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_22199721
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_22199988dense_16_22199990*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_22199738
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_22199993dense_17_22199995*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_22199754x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall&^sequential_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2N
%sequential_15/StatefulPartitionedCall%sequential_15/StatefulPartitionedCall:d `
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
-
_user_specified_namesequential_15_input
Ô$
×
!__inference__traced_save_22201604
file_prefix/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop:
6savev2_random_crop_5_cond_statevar_read_readvariableop	Q
Msavev2_sequential_16_sequential_15_random_flip_5_statevar_read_readvariableop	
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ø
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
value÷BôB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B õ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop6savev2_random_crop_5_cond_statevar_read_readvariableopMsavev2_sequential_16_sequential_15_random_flip_5_statevar_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2		
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*
_input_shapess
q: :::::	x:x:xT:T:T
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
:	x: 
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
Æ

+__inference_dense_17_layer_call_fn_22201105

inputs
unknown:T

	unknown_0:

identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_22199754o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿT: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs


G__inference_conv2d_10_layer_call_and_return_conditional_losses_22201005

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
É	
÷
F__inference_dense_17_layer_call_and_return_conditional_losses_22199754

inputs0
matmul_readvariableop_resource:T
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿT: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs
ë}
Æ
 random_crop_5_cond_true_22200628I
;random_crop_5_cond_stateful_uniform_rngreadandskip_resource:	,
(random_crop_5_cond_add_random_crop_5_sub0
,random_crop_5_cond_add_1_random_crop_5_sub_18
4random_crop_5_cond_crop_to_bounding_box_shape_inputs
random_crop_5_cond_identity¢5random_crop_5/cond/crop_to_bounding_box/Assert/Assert¢7random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert¢7random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert¢7random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert¢2random_crop_5/cond/stateful_uniform/RngReadAndSkips
)random_crop_5/cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
'random_crop_5/cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : m
'random_crop_5/cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :ÿÿÿÿs
)random_crop_5/cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¹
(random_crop_5/cond/stateful_uniform/ProdProd2random_crop_5/cond/stateful_uniform/shape:output:02random_crop_5/cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: l
*random_crop_5/cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
*random_crop_5/cond/stateful_uniform/Cast_1Cast1random_crop_5/cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
2random_crop_5/cond/stateful_uniform/RngReadAndSkipRngReadAndSkip;random_crop_5_cond_stateful_uniform_rngreadandskip_resource3random_crop_5/cond/stateful_uniform/Cast/x:output:0.random_crop_5/cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:
7random_crop_5/cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9random_crop_5/cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9random_crop_5/cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1random_crop_5/cond/stateful_uniform/strided_sliceStridedSlice:random_crop_5/cond/stateful_uniform/RngReadAndSkip:value:0@random_crop_5/cond/stateful_uniform/strided_slice/stack:output:0Brandom_crop_5/cond/stateful_uniform/strided_slice/stack_1:output:0Brandom_crop_5/cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask£
+random_crop_5/cond/stateful_uniform/BitcastBitcast:random_crop_5/cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
9random_crop_5/cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
;random_crop_5/cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;random_crop_5/cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3random_crop_5/cond/stateful_uniform/strided_slice_1StridedSlice:random_crop_5/cond/stateful_uniform/RngReadAndSkip:value:0Brandom_crop_5/cond/stateful_uniform/strided_slice_1/stack:output:0Drandom_crop_5/cond/stateful_uniform/strided_slice_1/stack_1:output:0Drandom_crop_5/cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:§
-random_crop_5/cond/stateful_uniform/Bitcast_1Bitcast<random_crop_5/cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0i
'random_crop_5/cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :£
#random_crop_5/cond/stateful_uniformStatelessRandomUniformIntV22random_crop_5/cond/stateful_uniform/shape:output:06random_crop_5/cond/stateful_uniform/Bitcast_1:output:04random_crop_5/cond/stateful_uniform/Bitcast:output:00random_crop_5/cond/stateful_uniform/alg:output:00random_crop_5/cond/stateful_uniform/min:output:00random_crop_5/cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0p
&random_crop_5/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(random_crop_5/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(random_crop_5/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
 random_crop_5/cond/strided_sliceStridedSlice,random_crop_5/cond/stateful_uniform:output:0/random_crop_5/cond/strided_slice/stack:output:01random_crop_5/cond/strided_slice/stack_1:output:01random_crop_5/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
random_crop_5/cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :
random_crop_5/cond/addAddV2(random_crop_5_cond_add_random_crop_5_sub!random_crop_5/cond/add/y:output:0*
T0*
_output_shapes
: 
random_crop_5/cond/modFloorMod)random_crop_5/cond/strided_slice:output:0random_crop_5/cond/add:z:0*
T0*
_output_shapes
: r
(random_crop_5/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*random_crop_5/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*random_crop_5/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ã
"random_crop_5/cond/strided_slice_1StridedSlice,random_crop_5/cond/stateful_uniform:output:01random_crop_5/cond/strided_slice_1/stack:output:03random_crop_5/cond/strided_slice_1/stack_1:output:03random_crop_5/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
random_crop_5/cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
random_crop_5/cond/add_1AddV2,random_crop_5_cond_add_1_random_crop_5_sub_1#random_crop_5/cond/add_1/y:output:0*
T0*
_output_shapes
: 
random_crop_5/cond/mod_1FloorMod+random_crop_5/cond/strided_slice_1:output:0random_crop_5/cond/add_1:z:0*
T0*
_output_shapes
: 
-random_crop_5/cond/crop_to_bounding_box/ShapeShape4random_crop_5_cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:£
/random_crop_5/cond/crop_to_bounding_box/unstackUnpack6random_crop_5/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numx
6random_crop_5/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : Ä
4random_crop_5/cond/crop_to_bounding_box/GreaterEqualGreaterEqualrandom_crop_5/cond/mod_1:z:0?random_crop_5/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: 
4random_crop_5/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.
<random_crop_5/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.â
5random_crop_5/cond/crop_to_bounding_box/Assert/AssertAssert8random_crop_5/cond/crop_to_bounding_box/GreaterEqual:z:0Erandom_crop_5/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 z
8random_crop_5/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : Æ
6random_crop_5/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualrandom_crop_5/cond/mod:z:0Arandom_crop_5/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: 
6random_crop_5/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.
>random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0. 
7random_crop_5/cond/crop_to_bounding_box/Assert_1/AssertAssert:random_crop_5/cond/crop_to_bounding_box/GreaterEqual_1:z:0Grandom_crop_5/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:06^random_crop_5/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 o
-random_crop_5/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B : «
+random_crop_5/cond/crop_to_bounding_box/addAddV26random_crop_5/cond/crop_to_bounding_box/add/x:output:0random_crop_5/cond/mod_1:z:0*
T0*
_output_shapes
: u
3random_crop_5/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B : Î
1random_crop_5/cond/crop_to_bounding_box/LessEqual	LessEqual/random_crop_5/cond/crop_to_bounding_box/add:z:0<random_crop_5/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: 
6random_crop_5/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset. 
>random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.
7random_crop_5/cond/crop_to_bounding_box/Assert_2/AssertAssert5random_crop_5/cond/crop_to_bounding_box/LessEqual:z:0Grandom_crop_5/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:08^random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 q
/random_crop_5/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : ­
-random_crop_5/cond/crop_to_bounding_box/add_1AddV28random_crop_5/cond/crop_to_bounding_box/add_1/x:output:0random_crop_5/cond/mod:z:0*
T0*
_output_shapes
: w
5random_crop_5/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : Ô
3random_crop_5/cond/crop_to_bounding_box/LessEqual_1	LessEqual1random_crop_5/cond/crop_to_bounding_box/add_1:z:0>random_crop_5/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: 
6random_crop_5/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.¡
>random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.
7random_crop_5/cond/crop_to_bounding_box/Assert_3/AssertAssert7random_crop_5/cond/crop_to_bounding_box/LessEqual_1:z:0Grandom_crop_5/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:08^random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ·
:random_crop_5/cond/crop_to_bounding_box/control_dependencyIdentity4random_crop_5_cond_crop_to_bounding_box_shape_inputs6^random_crop_5/cond/crop_to_bounding_box/Assert/Assert8^random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert8^random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert8^random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  q
/random_crop_5/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : q
/random_crop_5/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 
-random_crop_5/cond/crop_to_bounding_box/stackPack8random_crop_5/cond/crop_to_bounding_box/stack/0:output:0random_crop_5/cond/mod:z:0random_crop_5/cond/mod_1:z:08random_crop_5/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:¢
/random_crop_5/cond/crop_to_bounding_box/Shape_1ShapeCrandom_crop_5/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:
;random_crop_5/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=random_crop_5/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=random_crop_5/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5random_crop_5/cond/crop_to_bounding_box/strided_sliceStridedSlice8random_crop_5/cond/crop_to_bounding_box/Shape_1:output:0Drandom_crop_5/cond/crop_to_bounding_box/strided_slice/stack:output:0Frandom_crop_5/cond/crop_to_bounding_box/strided_slice/stack_1:output:0Frandom_crop_5/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¢
/random_crop_5/cond/crop_to_bounding_box/Shape_2ShapeCrandom_crop_5/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:
=random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:£
7random_crop_5/cond/crop_to_bounding_box/strided_slice_1StridedSlice8random_crop_5/cond/crop_to_bounding_box/Shape_2:output:0Frandom_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack:output:0Hrandom_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0Hrandom_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1random_crop_5/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : s
1random_crop_5/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : ß
/random_crop_5/cond/crop_to_bounding_box/stack_1Pack>random_crop_5/cond/crop_to_bounding_box/strided_slice:output:0:random_crop_5/cond/crop_to_bounding_box/stack_1/1:output:0:random_crop_5/cond/crop_to_bounding_box/stack_1/2:output:0@random_crop_5/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:´
-random_crop_5/cond/crop_to_bounding_box/SliceSliceCrandom_crop_5/cond/crop_to_bounding_box/control_dependency:output:06random_crop_5/cond/crop_to_bounding_box/stack:output:08random_crop_5/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ³
random_crop_5/cond/IdentityIdentity6random_crop_5/cond/crop_to_bounding_box/Slice:output:0^random_crop_5/cond/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ô
random_crop_5/cond/NoOpNoOp6^random_crop_5/cond/crop_to_bounding_box/Assert/Assert8^random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert8^random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert8^random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert3^random_crop_5/cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "C
random_crop_5_cond_identity$random_crop_5/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :ÿÿÿÿÿÿÿÿÿ  2n
5random_crop_5/cond/crop_to_bounding_box/Assert/Assert5random_crop_5/cond/crop_to_bounding_box/Assert/Assert2r
7random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert7random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert2r
7random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert7random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert2r
7random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert7random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert2h
2random_crop_5/cond/stateful_uniform/RngReadAndSkip2random_crop_5/cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  


÷
F__inference_dense_16_layer_call_and_return_conditional_losses_22199738

inputs0
matmul_readvariableop_resource:xT-
biasadd_readvariableop_resource:T
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿx: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
Ý*

K__inference_sequential_16_layer_call_and_return_conditional_losses_22199908

inputs$
sequential_15_22199874:	$
sequential_15_22199876:	,
conv2d_10_22199879: 
conv2d_10_22199881:,
conv2d_11_22199885: 
conv2d_11_22199887:$
dense_15_22199892:	x
dense_15_22199894:x#
dense_16_22199897:xT
dense_16_22199899:T#
dense_17_22199902:T

dense_17_22199904:

identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢%sequential_15/StatefulPartitionedCall
%sequential_15/StatefulPartitionedCallStatefulPartitionedCallinputssequential_15_22199874sequential_15_22199876*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_15_layer_call_and_return_conditional_losses_22199598ª
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall.sequential_15/StatefulPartitionedCall:output:0conv2d_10_22199879conv2d_10_22199881*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_22199677þ
$average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_22199641©
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_10/PartitionedCall:output:0conv2d_11_22199885conv2d_11_22199887*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_22199695þ
$average_pooling2d_11/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_22199653ä
flatten_5/PartitionedCallPartitionedCall-average_pooling2d_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_22199708
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_15_22199892dense_15_22199894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_22199721
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_22199897dense_16_22199899*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_22199738
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_22199902dense_17_22199904*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_22199754x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall&^sequential_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2N
%sequential_15/StatefulPartitionedCall%sequential_15/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
*
Ü
K__inference_sequential_16_layer_call_and_return_conditional_losses_22199761

inputs$
sequential_15_22199663:	,
conv2d_10_22199678: 
conv2d_10_22199680:,
conv2d_11_22199696: 
conv2d_11_22199698:$
dense_15_22199722:	x
dense_15_22199724:x#
dense_16_22199739:xT
dense_16_22199741:T#
dense_17_22199755:T

dense_17_22199757:

identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢%sequential_15/StatefulPartitionedCallõ
%sequential_15/StatefulPartitionedCallStatefulPartitionedCallinputssequential_15_22199663*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_15_layer_call_and_return_conditional_losses_22199322ª
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall.sequential_15/StatefulPartitionedCall:output:0conv2d_10_22199678conv2d_10_22199680*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_22199677þ
$average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_22199641©
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_10/PartitionedCall:output:0conv2d_11_22199696conv2d_11_22199698*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_22199695þ
$average_pooling2d_11/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_22199653ä
flatten_5/PartitionedCallPartitionedCall-average_pooling2d_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_22199708
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_15_22199722dense_15_22199724*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_22199721
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_22199739dense_16_22199741*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_22199738
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_22199755dense_17_22199757*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_22199754x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall&^sequential_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2N
%sequential_15/StatefulPartitionedCall%sequential_15/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
£Z
ì
K__inference_sequential_16_layer_call_and_return_conditional_losses_22200296

inputs6
(sequential_15_random_crop_5_cond_input_0:	B
(conv2d_10_conv2d_readvariableop_resource:7
)conv2d_10_biasadd_readvariableop_resource:B
(conv2d_11_conv2d_readvariableop_resource:7
)conv2d_11_biasadd_readvariableop_resource::
'dense_15_matmul_readvariableop_resource:	x6
(dense_15_biasadd_readvariableop_resource:x9
'dense_16_matmul_readvariableop_resource:xT6
(dense_16_biasadd_readvariableop_resource:T9
'dense_17_matmul_readvariableop_resource:T
6
(dense_17_biasadd_readvariableop_resource:

identity¢ conv2d_10/BiasAdd/ReadVariableOp¢conv2d_10/Conv2D/ReadVariableOp¢ conv2d_11/BiasAdd/ReadVariableOp¢conv2d_11/Conv2D/ReadVariableOp¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¢dense_16/BiasAdd/ReadVariableOp¢dense_16/MatMul/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢dense_17/MatMul/ReadVariableOp¢ sequential_15/random_crop_5/condW
!sequential_15/random_crop_5/ShapeShapeinputs*
T0*
_output_shapes
:
/sequential_15/random_crop_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
1sequential_15/random_crop_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ{
1sequential_15/random_crop_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)sequential_15/random_crop_5/strided_sliceStridedSlice*sequential_15/random_crop_5/Shape:output:08sequential_15/random_crop_5/strided_slice/stack:output:0:sequential_15/random_crop_5/strided_slice/stack_1:output:0:sequential_15/random_crop_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!sequential_15/random_crop_5/sub/yConst*
_output_shapes
: *
dtype0*
value	B : §
sequential_15/random_crop_5/subSub2sequential_15/random_crop_5/strided_slice:output:0*sequential_15/random_crop_5/sub/y:output:0*
T0*
_output_shapes
: 
1sequential_15/random_crop_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
3sequential_15/random_crop_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ}
3sequential_15/random_crop_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+sequential_15/random_crop_5/strided_slice_1StridedSlice*sequential_15/random_crop_5/Shape:output:0:sequential_15/random_crop_5/strided_slice_1/stack:output:0<sequential_15/random_crop_5/strided_slice_1/stack_1:output:0<sequential_15/random_crop_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sequential_15/random_crop_5/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B : ­
!sequential_15/random_crop_5/sub_1Sub4sequential_15/random_crop_5/strided_slice_1:output:0,sequential_15/random_crop_5/sub_1/y:output:0*
T0*
_output_shapes
: l
*sequential_15/random_crop_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ³
(sequential_15/random_crop_5/GreaterEqualGreaterEqual#sequential_15/random_crop_5/sub:z:03sequential_15/random_crop_5/GreaterEqual/y:output:0*
T0*
_output_shapes
: n
,sequential_15/random_crop_5/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ¹
*sequential_15/random_crop_5/GreaterEqual_1GreaterEqual%sequential_15/random_crop_5/sub_1:z:05sequential_15/random_crop_5/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: k
)sequential_15/random_crop_5/Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z ï
'sequential_15/random_crop_5/Rank/packedPack2sequential_15/random_crop_5/Rank/packed/0:output:0,sequential_15/random_crop_5/GreaterEqual:z:0.sequential_15/random_crop_5/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:b
 sequential_15/random_crop_5/RankConst*
_output_shapes
: *
dtype0*
value	B :i
'sequential_15/random_crop_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : i
'sequential_15/random_crop_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Õ
!sequential_15/random_crop_5/rangeRange0sequential_15/random_crop_5/range/start:output:0)sequential_15/random_crop_5/Rank:output:00sequential_15/random_crop_5/range/delta:output:0*
_output_shapes
:i
'sequential_15/random_crop_5/All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z ë
%sequential_15/random_crop_5/All/inputPack0sequential_15/random_crop_5/All/input/0:output:0,sequential_15/random_crop_5/GreaterEqual:z:0.sequential_15/random_crop_5/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:
sequential_15/random_crop_5/AllAll.sequential_15/random_crop_5/All/input:output:0*sequential_15/random_crop_5/range:output:0*
_output_shapes
: ¦
 sequential_15/random_crop_5/condIf(sequential_15/random_crop_5/All:output:0(sequential_15_random_crop_5_cond_input_0#sequential_15/random_crop_5/sub:z:0%sequential_15/random_crop_5/sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *B
else_branch3R1
/sequential_15_random_crop_5_cond_false_22200122*.
output_shapes
:ÿÿÿÿÿÿÿÿÿ  *A
then_branch2R0
.sequential_15_random_crop_5_cond_true_22200121
)sequential_15/random_crop_5/cond/IdentityIdentity)sequential_15/random_crop_5/cond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ú
conv2d_10/Conv2DConv2D2sequential_15/random_crop_5/cond/Identity:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
average_pooling2d_10/AvgPoolAvgPoolconv2d_10/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Í
conv2d_11/Conv2DConv2D%average_pooling2d_10/AvgPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
paddingVALID*
strides

 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

l
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

»
average_pooling2d_11/AvgPoolAvgPoolconv2d_11/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
`
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  
flatten_5/ReshapeReshape%average_pooling2d_11/AvgPool:output:0flatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	x*
dtype0
dense_15/MatMulMatMulflatten_5/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxb
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype0
dense_16/MatMulMatMuldense_15/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype0
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTb
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype0
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
IdentityIdentitydense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¼
NoOpNoOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp!^sequential_15/random_crop_5/cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2D
 sequential_15/random_crop_5/cond sequential_15/random_crop_5/cond:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

¹
0__inference_sequential_16_layer_call_fn_22200092

inputs
unknown:	
	unknown_0:	#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:	x
	unknown_6:x
	unknown_7:xT
	unknown_8:T
	unknown_9:T


unknown_10:

identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_16_layer_call_and_return_conditional_losses_22199908o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
¿g

cond_true_22199174;
-cond_stateful_uniform_rngreadandskip_resource:	
cond_add_sub
cond_add_1_sub_1*
&cond_crop_to_bounding_box_shape_inputs
cond_identity¢'cond/crop_to_bounding_box/Assert/Assert¢)cond/crop_to_bounding_box/Assert_1/Assert¢)cond/crop_to_bounding_box/Assert_2/Assert¢)cond/crop_to_bounding_box/Assert_3/Assert¢$cond/stateful_uniform/RngReadAndSkipe
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
valueB :ÿÿÿÿe
cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
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
: Ê
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
valueB:Å
#cond/stateful_uniform/strided_sliceStridedSlice,cond/stateful_uniform/RngReadAndSkip:value:02cond/stateful_uniform/strided_slice/stack:output:04cond/stateful_uniform/strided_slice/stack_1:output:04cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
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
valueB:»
%cond/stateful_uniform/strided_slice_1StridedSlice,cond/stateful_uniform/RngReadAndSkip:value:04cond/stateful_uniform/strided_slice_1/stack:output:06cond/stateful_uniform/strided_slice_1/stack_1:output:06cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
cond/stateful_uniform/Bitcast_1Bitcast.cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0[
cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :Á
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
valueB:õ
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
valueB:ý
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
:
!cond/crop_to_bounding_box/unstackUnpack(cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numj
(cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 
&cond/crop_to_bounding_box/GreaterEqualGreaterEqualcond/mod_1:z:01cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: 
&cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.
.cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.¸
'cond/crop_to_bounding_box/Assert/AssertAssert*cond/crop_to_bounding_box/GreaterEqual:z:07cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 l
*cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : 
(cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcond/mod:z:03cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: 
(cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.
0cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.è
)cond/crop_to_bounding_box/Assert_1/AssertAssert,cond/crop_to_bounding_box/GreaterEqual_1:z:09cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0(^cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 a
cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B : 
cond/crop_to_bounding_box/addAddV2(cond/crop_to_bounding_box/add/x:output:0cond/mod_1:z:0*
T0*
_output_shapes
: g
%cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ¤
#cond/crop_to_bounding_box/LessEqual	LessEqual!cond/crop_to_bounding_box/add:z:0.cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: 
(cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.
0cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.å
)cond/crop_to_bounding_box/Assert_2/AssertAssert'cond/crop_to_bounding_box/LessEqual:z:09cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 c
!cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : 
cond/crop_to_bounding_box/add_1AddV2*cond/crop_to_bounding_box/add_1/x:output:0cond/mod:z:0*
T0*
_output_shapes
: i
'cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ª
%cond/crop_to_bounding_box/LessEqual_1	LessEqual#cond/crop_to_bounding_box/add_1:z:00cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: 
(cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.
0cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.ç
)cond/crop_to_bounding_box/Assert_3/AssertAssert)cond/crop_to_bounding_box/LessEqual_1:z:09cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ã
,cond/crop_to_bounding_box/control_dependencyIdentity&cond_crop_to_bounding_box_shape_inputs(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  c
!cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : c
!cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : Ë
cond/crop_to_bounding_box/stackPack*cond/crop_to_bounding_box/stack/0:output:0cond/mod:z:0cond/mod_1:z:0*cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:
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
valueB:Õ
'cond/crop_to_bounding_box/strided_sliceStridedSlice*cond/crop_to_bounding_box/Shape_1:output:06cond/crop_to_bounding_box/strided_slice/stack:output:08cond/crop_to_bounding_box/strided_slice/stack_1:output:08cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
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
valueB:Ý
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
value	B : 
!cond/crop_to_bounding_box/stack_1Pack0cond/crop_to_bounding_box/strided_slice:output:0,cond/crop_to_bounding_box/stack_1/1:output:0,cond/crop_to_bounding_box/stack_1/2:output:02cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:ü
cond/crop_to_bounding_box/SliceSlice5cond/crop_to_bounding_box/control_dependency:output:0(cond/crop_to_bounding_box/stack:output:0*cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
cond/IdentityIdentity(cond/crop_to_bounding_box/Slice:output:0
^cond/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   
	cond/NoOpNoOp(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert%^cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :ÿÿÿÿÿÿÿÿÿ  2R
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
:ÿÿÿÿÿÿÿÿÿ  
4
í
$__inference__traced_restore_22201650
file_prefix;
!assignvariableop_conv2d_10_kernel:/
!assignvariableop_1_conv2d_10_bias:=
#assignvariableop_2_conv2d_11_kernel:/
!assignvariableop_3_conv2d_11_bias:5
"assignvariableop_4_dense_15_kernel:	x.
 assignvariableop_5_dense_15_bias:x4
"assignvariableop_6_dense_16_kernel:xT.
 assignvariableop_7_dense_16_bias:T4
"assignvariableop_8_dense_17_kernel:T
.
 assignvariableop_9_dense_17_bias:
=
/assignvariableop_10_random_crop_5_cond_statevar:	T
Fassignvariableop_11_sequential_16_sequential_15_random_flip_5_statevar:	
identity_13¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Û
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
value÷BôB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B ß
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_10_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_11_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_11_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_15_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_15_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_16_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_16_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_17_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_17_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
: 
AssignVariableOp_10AssignVariableOp/assignvariableop_10_random_crop_5_cond_statevarIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:·
AssignVariableOp_11AssignVariableOpFassignvariableop_11_sequential_16_sequential_15_random_flip_5_statevarIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ×
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: Ä
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
·

K__inference_random_crop_5_layer_call_and_return_conditional_losses_22201461

inputs
cond_input_0:	
identity¢cond;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿh
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
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
þÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
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
: â
condIfAll:output:0cond_input_0sub:z:0	sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *&
else_branchR
cond_false_22201325*.
output_shapes
:ÿÿÿÿÿÿÿÿÿ  *%
then_branchR
cond_true_22201324b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  M
NoOpNoOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  : 2
condcond:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
­'

cond_false_22201325
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
ýÿÿÿÿÿÿÿÿm
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿd
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
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
þÿÿÿÿÿÿÿÿo
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿf
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
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
value	B : 

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
ÿÿÿÿÿÿÿÿÿY
cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
cond/stack_1Packcond/stack_1/0:output:0cond/Minimum:z:0cond/Minimum_1:z:0cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:¢

cond/SliceSlicecond_shape_inputscond/stack:output:0cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ  ÿÿÿÿÿÿÿÿÿa
cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        °
cond/resize/ResizeBilinearResizeBilinearcond/Slice:output:0cond/resize/size:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
half_pixel_centers(
cond/IdentityIdentity+cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :ÿÿÿÿÿÿÿÿÿ  :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
Ë2
Ù
!random_crop_5_cond_false_22200629"
random_crop_5_cond_placeholder$
 random_crop_5_cond_placeholder_1$
 random_crop_5_cond_placeholder_2#
random_crop_5_cond_shape_inputs
random_crop_5_cond_identityg
random_crop_5/cond/ShapeShaperandom_crop_5_cond_shape_inputs*
T0*
_output_shapes
:y
&random_crop_5/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
(random_crop_5/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿr
(random_crop_5/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 random_crop_5/cond/strided_sliceStridedSlice!random_crop_5/cond/Shape:output:0/random_crop_5/cond/strided_slice/stack:output:01random_crop_5/cond/strided_slice/stack_1:output:01random_crop_5/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
(random_crop_5/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ}
*random_crop_5/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿt
*random_crop_5/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"random_crop_5/cond/strided_slice_1StridedSlice!random_crop_5/cond/Shape:output:01random_crop_5/cond/strided_slice_1/stack:output:03random_crop_5/cond/strided_slice_1/stack_1:output:03random_crop_5/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
random_crop_5/cond/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 
random_crop_5/cond/mulMul+random_crop_5/cond/strided_slice_1:output:0!random_crop_5/cond/mul/y:output:0*
T0*
_output_shapes
: k
random_crop_5/cond/CastCastrandom_crop_5/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: a
random_crop_5/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B
random_crop_5/cond/truedivRealDivrandom_crop_5/cond/Cast:y:0%random_crop_5/cond/truediv/y:output:0*
T0*
_output_shapes
: q
random_crop_5/cond/Cast_1Castrandom_crop_5/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: \
random_crop_5/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B : 
random_crop_5/cond/mul_1Mul)random_crop_5/cond/strided_slice:output:0#random_crop_5/cond/mul_1/y:output:0*
T0*
_output_shapes
: o
random_crop_5/cond/Cast_2Castrandom_crop_5/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: c
random_crop_5/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B
random_crop_5/cond/truediv_1RealDivrandom_crop_5/cond/Cast_2:y:0'random_crop_5/cond/truediv_1/y:output:0*
T0*
_output_shapes
: s
random_crop_5/cond/Cast_3Cast random_crop_5/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 
random_crop_5/cond/MinimumMinimum)random_crop_5/cond/strided_slice:output:0random_crop_5/cond/Cast_1:y:0*
T0*
_output_shapes
: 
random_crop_5/cond/Minimum_1Minimum+random_crop_5/cond/strided_slice_1:output:0random_crop_5/cond/Cast_3:y:0*
T0*
_output_shapes
: 
random_crop_5/cond/subSub)random_crop_5/cond/strided_slice:output:0random_crop_5/cond/Minimum:z:0*
T0*
_output_shapes
: m
random_crop_5/cond/Cast_4Castrandom_crop_5/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: c
random_crop_5/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
random_crop_5/cond/truediv_2RealDivrandom_crop_5/cond/Cast_4:y:0'random_crop_5/cond/truediv_2/y:output:0*
T0*
_output_shapes
: s
random_crop_5/cond/Cast_5Cast random_crop_5/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: 
random_crop_5/cond/sub_1Sub+random_crop_5/cond/strided_slice_1:output:0 random_crop_5/cond/Minimum_1:z:0*
T0*
_output_shapes
: o
random_crop_5/cond/Cast_6Castrandom_crop_5/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: c
random_crop_5/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
random_crop_5/cond/truediv_3RealDivrandom_crop_5/cond/Cast_6:y:0'random_crop_5/cond/truediv_3/y:output:0*
T0*
_output_shapes
: s
random_crop_5/cond/Cast_7Cast random_crop_5/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: \
random_crop_5/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : \
random_crop_5/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : Ö
random_crop_5/cond/stackPack#random_crop_5/cond/stack/0:output:0random_crop_5/cond/Cast_5:y:0random_crop_5/cond/Cast_7:y:0#random_crop_5/cond/stack/3:output:0*
N*
T0*
_output_shapes
:g
random_crop_5/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿg
random_crop_5/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿà
random_crop_5/cond/stack_1Pack%random_crop_5/cond/stack_1/0:output:0random_crop_5/cond/Minimum:z:0 random_crop_5/cond/Minimum_1:z:0%random_crop_5/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:Ú
random_crop_5/cond/SliceSlicerandom_crop_5_cond_shape_inputs!random_crop_5/cond/stack:output:0#random_crop_5/cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ  ÿÿÿÿÿÿÿÿÿo
random_crop_5/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        Ú
(random_crop_5/cond/resize/ResizeBilinearResizeBilinear!random_crop_5/cond/Slice:output:0'random_crop_5/cond/resize/size:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
half_pixel_centers(
random_crop_5/cond/IdentityIdentity9random_crop_5/cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "C
random_crop_5_cond_identity$random_crop_5/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :ÿÿÿÿÿÿÿÿÿ  :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
î=
­
/sequential_15_random_crop_5_cond_false_222003260
,sequential_15_random_crop_5_cond_placeholder2
.sequential_15_random_crop_5_cond_placeholder_12
.sequential_15_random_crop_5_cond_placeholder_21
-sequential_15_random_crop_5_cond_shape_inputs-
)sequential_15_random_crop_5_cond_identity
&sequential_15/random_crop_5/cond/ShapeShape-sequential_15_random_crop_5_cond_shape_inputs*
T0*
_output_shapes
:
4sequential_15/random_crop_5/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
6sequential_15/random_crop_5/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
6sequential_15/random_crop_5/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.sequential_15/random_crop_5/cond/strided_sliceStridedSlice/sequential_15/random_crop_5/cond/Shape:output:0=sequential_15/random_crop_5/cond/strided_slice/stack:output:0?sequential_15/random_crop_5/cond/strided_slice/stack_1:output:0?sequential_15/random_crop_5/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
6sequential_15/random_crop_5/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
8sequential_15/random_crop_5/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
8sequential_15/random_crop_5/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
0sequential_15/random_crop_5/cond/strided_slice_1StridedSlice/sequential_15/random_crop_5/cond/Shape:output:0?sequential_15/random_crop_5/cond/strided_slice_1/stack:output:0Asequential_15/random_crop_5/cond/strided_slice_1/stack_1:output:0Asequential_15/random_crop_5/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&sequential_15/random_crop_5/cond/mul/yConst*
_output_shapes
: *
dtype0*
value	B : ¸
$sequential_15/random_crop_5/cond/mulMul9sequential_15/random_crop_5/cond/strided_slice_1:output:0/sequential_15/random_crop_5/cond/mul/y:output:0*
T0*
_output_shapes
: 
%sequential_15/random_crop_5/cond/CastCast(sequential_15/random_crop_5/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: o
*sequential_15/random_crop_5/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B´
(sequential_15/random_crop_5/cond/truedivRealDiv)sequential_15/random_crop_5/cond/Cast:y:03sequential_15/random_crop_5/cond/truediv/y:output:0*
T0*
_output_shapes
: 
'sequential_15/random_crop_5/cond/Cast_1Cast,sequential_15/random_crop_5/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: j
(sequential_15/random_crop_5/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B : º
&sequential_15/random_crop_5/cond/mul_1Mul7sequential_15/random_crop_5/cond/strided_slice:output:01sequential_15/random_crop_5/cond/mul_1/y:output:0*
T0*
_output_shapes
: 
'sequential_15/random_crop_5/cond/Cast_2Cast*sequential_15/random_crop_5/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: q
,sequential_15/random_crop_5/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Bº
*sequential_15/random_crop_5/cond/truediv_1RealDiv+sequential_15/random_crop_5/cond/Cast_2:y:05sequential_15/random_crop_5/cond/truediv_1/y:output:0*
T0*
_output_shapes
: 
'sequential_15/random_crop_5/cond/Cast_3Cast.sequential_15/random_crop_5/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: º
(sequential_15/random_crop_5/cond/MinimumMinimum7sequential_15/random_crop_5/cond/strided_slice:output:0+sequential_15/random_crop_5/cond/Cast_1:y:0*
T0*
_output_shapes
: ¾
*sequential_15/random_crop_5/cond/Minimum_1Minimum9sequential_15/random_crop_5/cond/strided_slice_1:output:0+sequential_15/random_crop_5/cond/Cast_3:y:0*
T0*
_output_shapes
: ³
$sequential_15/random_crop_5/cond/subSub7sequential_15/random_crop_5/cond/strided_slice:output:0,sequential_15/random_crop_5/cond/Minimum:z:0*
T0*
_output_shapes
: 
'sequential_15/random_crop_5/cond/Cast_4Cast(sequential_15/random_crop_5/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: q
,sequential_15/random_crop_5/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @º
*sequential_15/random_crop_5/cond/truediv_2RealDiv+sequential_15/random_crop_5/cond/Cast_4:y:05sequential_15/random_crop_5/cond/truediv_2/y:output:0*
T0*
_output_shapes
: 
'sequential_15/random_crop_5/cond/Cast_5Cast.sequential_15/random_crop_5/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ¹
&sequential_15/random_crop_5/cond/sub_1Sub9sequential_15/random_crop_5/cond/strided_slice_1:output:0.sequential_15/random_crop_5/cond/Minimum_1:z:0*
T0*
_output_shapes
: 
'sequential_15/random_crop_5/cond/Cast_6Cast*sequential_15/random_crop_5/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: q
,sequential_15/random_crop_5/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @º
*sequential_15/random_crop_5/cond/truediv_3RealDiv+sequential_15/random_crop_5/cond/Cast_6:y:05sequential_15/random_crop_5/cond/truediv_3/y:output:0*
T0*
_output_shapes
: 
'sequential_15/random_crop_5/cond/Cast_7Cast.sequential_15/random_crop_5/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: j
(sequential_15/random_crop_5/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : j
(sequential_15/random_crop_5/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 
&sequential_15/random_crop_5/cond/stackPack1sequential_15/random_crop_5/cond/stack/0:output:0+sequential_15/random_crop_5/cond/Cast_5:y:0+sequential_15/random_crop_5/cond/Cast_7:y:01sequential_15/random_crop_5/cond/stack/3:output:0*
N*
T0*
_output_shapes
:u
*sequential_15/random_crop_5/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿu
*sequential_15/random_crop_5/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
(sequential_15/random_crop_5/cond/stack_1Pack3sequential_15/random_crop_5/cond/stack_1/0:output:0,sequential_15/random_crop_5/cond/Minimum:z:0.sequential_15/random_crop_5/cond/Minimum_1:z:03sequential_15/random_crop_5/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:
&sequential_15/random_crop_5/cond/SliceSlice-sequential_15_random_crop_5_cond_shape_inputs/sequential_15/random_crop_5/cond/stack:output:01sequential_15/random_crop_5/cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ  ÿÿÿÿÿÿÿÿÿ}
,sequential_15/random_crop_5/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        
6sequential_15/random_crop_5/cond/resize/ResizeBilinearResizeBilinear/sequential_15/random_crop_5/cond/Slice:output:05sequential_15/random_crop_5/cond/resize/size:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
half_pixel_centers(¸
)sequential_15/random_crop_5/cond/IdentityIdentityGsequential_15/random_crop_5/cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "_
)sequential_15_random_crop_5_cond_identity2sequential_15/random_crop_5/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :ÿÿÿÿÿÿÿÿÿ  :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
£
n
R__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_22201045

inputs
identity«
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë}
Æ
 random_crop_5_cond_true_22200794I
;random_crop_5_cond_stateful_uniform_rngreadandskip_resource:	,
(random_crop_5_cond_add_random_crop_5_sub0
,random_crop_5_cond_add_1_random_crop_5_sub_18
4random_crop_5_cond_crop_to_bounding_box_shape_inputs
random_crop_5_cond_identity¢5random_crop_5/cond/crop_to_bounding_box/Assert/Assert¢7random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert¢7random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert¢7random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert¢2random_crop_5/cond/stateful_uniform/RngReadAndSkips
)random_crop_5/cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
'random_crop_5/cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : m
'random_crop_5/cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :ÿÿÿÿs
)random_crop_5/cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¹
(random_crop_5/cond/stateful_uniform/ProdProd2random_crop_5/cond/stateful_uniform/shape:output:02random_crop_5/cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: l
*random_crop_5/cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
*random_crop_5/cond/stateful_uniform/Cast_1Cast1random_crop_5/cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
2random_crop_5/cond/stateful_uniform/RngReadAndSkipRngReadAndSkip;random_crop_5_cond_stateful_uniform_rngreadandskip_resource3random_crop_5/cond/stateful_uniform/Cast/x:output:0.random_crop_5/cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:
7random_crop_5/cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9random_crop_5/cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9random_crop_5/cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1random_crop_5/cond/stateful_uniform/strided_sliceStridedSlice:random_crop_5/cond/stateful_uniform/RngReadAndSkip:value:0@random_crop_5/cond/stateful_uniform/strided_slice/stack:output:0Brandom_crop_5/cond/stateful_uniform/strided_slice/stack_1:output:0Brandom_crop_5/cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask£
+random_crop_5/cond/stateful_uniform/BitcastBitcast:random_crop_5/cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
9random_crop_5/cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
;random_crop_5/cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;random_crop_5/cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3random_crop_5/cond/stateful_uniform/strided_slice_1StridedSlice:random_crop_5/cond/stateful_uniform/RngReadAndSkip:value:0Brandom_crop_5/cond/stateful_uniform/strided_slice_1/stack:output:0Drandom_crop_5/cond/stateful_uniform/strided_slice_1/stack_1:output:0Drandom_crop_5/cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:§
-random_crop_5/cond/stateful_uniform/Bitcast_1Bitcast<random_crop_5/cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0i
'random_crop_5/cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :£
#random_crop_5/cond/stateful_uniformStatelessRandomUniformIntV22random_crop_5/cond/stateful_uniform/shape:output:06random_crop_5/cond/stateful_uniform/Bitcast_1:output:04random_crop_5/cond/stateful_uniform/Bitcast:output:00random_crop_5/cond/stateful_uniform/alg:output:00random_crop_5/cond/stateful_uniform/min:output:00random_crop_5/cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0p
&random_crop_5/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(random_crop_5/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(random_crop_5/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
 random_crop_5/cond/strided_sliceStridedSlice,random_crop_5/cond/stateful_uniform:output:0/random_crop_5/cond/strided_slice/stack:output:01random_crop_5/cond/strided_slice/stack_1:output:01random_crop_5/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
random_crop_5/cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :
random_crop_5/cond/addAddV2(random_crop_5_cond_add_random_crop_5_sub!random_crop_5/cond/add/y:output:0*
T0*
_output_shapes
: 
random_crop_5/cond/modFloorMod)random_crop_5/cond/strided_slice:output:0random_crop_5/cond/add:z:0*
T0*
_output_shapes
: r
(random_crop_5/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*random_crop_5/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*random_crop_5/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ã
"random_crop_5/cond/strided_slice_1StridedSlice,random_crop_5/cond/stateful_uniform:output:01random_crop_5/cond/strided_slice_1/stack:output:03random_crop_5/cond/strided_slice_1/stack_1:output:03random_crop_5/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
random_crop_5/cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
random_crop_5/cond/add_1AddV2,random_crop_5_cond_add_1_random_crop_5_sub_1#random_crop_5/cond/add_1/y:output:0*
T0*
_output_shapes
: 
random_crop_5/cond/mod_1FloorMod+random_crop_5/cond/strided_slice_1:output:0random_crop_5/cond/add_1:z:0*
T0*
_output_shapes
: 
-random_crop_5/cond/crop_to_bounding_box/ShapeShape4random_crop_5_cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:£
/random_crop_5/cond/crop_to_bounding_box/unstackUnpack6random_crop_5/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numx
6random_crop_5/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : Ä
4random_crop_5/cond/crop_to_bounding_box/GreaterEqualGreaterEqualrandom_crop_5/cond/mod_1:z:0?random_crop_5/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: 
4random_crop_5/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.
<random_crop_5/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.â
5random_crop_5/cond/crop_to_bounding_box/Assert/AssertAssert8random_crop_5/cond/crop_to_bounding_box/GreaterEqual:z:0Erandom_crop_5/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 z
8random_crop_5/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : Æ
6random_crop_5/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualrandom_crop_5/cond/mod:z:0Arandom_crop_5/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: 
6random_crop_5/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.
>random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0. 
7random_crop_5/cond/crop_to_bounding_box/Assert_1/AssertAssert:random_crop_5/cond/crop_to_bounding_box/GreaterEqual_1:z:0Grandom_crop_5/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:06^random_crop_5/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 o
-random_crop_5/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B : «
+random_crop_5/cond/crop_to_bounding_box/addAddV26random_crop_5/cond/crop_to_bounding_box/add/x:output:0random_crop_5/cond/mod_1:z:0*
T0*
_output_shapes
: u
3random_crop_5/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B : Î
1random_crop_5/cond/crop_to_bounding_box/LessEqual	LessEqual/random_crop_5/cond/crop_to_bounding_box/add:z:0<random_crop_5/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: 
6random_crop_5/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset. 
>random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.
7random_crop_5/cond/crop_to_bounding_box/Assert_2/AssertAssert5random_crop_5/cond/crop_to_bounding_box/LessEqual:z:0Grandom_crop_5/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:08^random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 q
/random_crop_5/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : ­
-random_crop_5/cond/crop_to_bounding_box/add_1AddV28random_crop_5/cond/crop_to_bounding_box/add_1/x:output:0random_crop_5/cond/mod:z:0*
T0*
_output_shapes
: w
5random_crop_5/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : Ô
3random_crop_5/cond/crop_to_bounding_box/LessEqual_1	LessEqual1random_crop_5/cond/crop_to_bounding_box/add_1:z:0>random_crop_5/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: 
6random_crop_5/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.¡
>random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.
7random_crop_5/cond/crop_to_bounding_box/Assert_3/AssertAssert7random_crop_5/cond/crop_to_bounding_box/LessEqual_1:z:0Grandom_crop_5/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:08^random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ·
:random_crop_5/cond/crop_to_bounding_box/control_dependencyIdentity4random_crop_5_cond_crop_to_bounding_box_shape_inputs6^random_crop_5/cond/crop_to_bounding_box/Assert/Assert8^random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert8^random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert8^random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  q
/random_crop_5/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : q
/random_crop_5/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 
-random_crop_5/cond/crop_to_bounding_box/stackPack8random_crop_5/cond/crop_to_bounding_box/stack/0:output:0random_crop_5/cond/mod:z:0random_crop_5/cond/mod_1:z:08random_crop_5/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:¢
/random_crop_5/cond/crop_to_bounding_box/Shape_1ShapeCrandom_crop_5/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:
;random_crop_5/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=random_crop_5/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=random_crop_5/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5random_crop_5/cond/crop_to_bounding_box/strided_sliceStridedSlice8random_crop_5/cond/crop_to_bounding_box/Shape_1:output:0Drandom_crop_5/cond/crop_to_bounding_box/strided_slice/stack:output:0Frandom_crop_5/cond/crop_to_bounding_box/strided_slice/stack_1:output:0Frandom_crop_5/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¢
/random_crop_5/cond/crop_to_bounding_box/Shape_2ShapeCrandom_crop_5/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:
=random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:£
7random_crop_5/cond/crop_to_bounding_box/strided_slice_1StridedSlice8random_crop_5/cond/crop_to_bounding_box/Shape_2:output:0Frandom_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack:output:0Hrandom_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0Hrandom_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1random_crop_5/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : s
1random_crop_5/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : ß
/random_crop_5/cond/crop_to_bounding_box/stack_1Pack>random_crop_5/cond/crop_to_bounding_box/strided_slice:output:0:random_crop_5/cond/crop_to_bounding_box/stack_1/1:output:0:random_crop_5/cond/crop_to_bounding_box/stack_1/2:output:0@random_crop_5/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:´
-random_crop_5/cond/crop_to_bounding_box/SliceSliceCrandom_crop_5/cond/crop_to_bounding_box/control_dependency:output:06random_crop_5/cond/crop_to_bounding_box/stack:output:08random_crop_5/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ³
random_crop_5/cond/IdentityIdentity6random_crop_5/cond/crop_to_bounding_box/Slice:output:0^random_crop_5/cond/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ô
random_crop_5/cond/NoOpNoOp6^random_crop_5/cond/crop_to_bounding_box/Assert/Assert8^random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert8^random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert8^random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert3^random_crop_5/cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "C
random_crop_5_cond_identity$random_crop_5/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :ÿÿÿÿÿÿÿÿÿ  2n
5random_crop_5/cond/crop_to_bounding_box/Assert/Assert5random_crop_5/cond/crop_to_bounding_box/Assert/Assert2r
7random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert7random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert2r
7random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert7random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert2r
7random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert7random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert2h
2random_crop_5/cond/stateful_uniform/RngReadAndSkip2random_crop_5/cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
µ
H
,__inference_flatten_5_layer_call_fn_22201050

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_22199708a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ü
.sequential_15_random_crop_5_cond_true_22200121W
Isequential_15_random_crop_5_cond_stateful_uniform_rngreadandskip_resource:	H
Dsequential_15_random_crop_5_cond_add_sequential_15_random_crop_5_subL
Hsequential_15_random_crop_5_cond_add_1_sequential_15_random_crop_5_sub_1F
Bsequential_15_random_crop_5_cond_crop_to_bounding_box_shape_inputs-
)sequential_15_random_crop_5_cond_identity¢Csequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/Assert¢Esequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert¢Esequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert¢Esequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert¢@sequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkip
7sequential_15/random_crop_5/cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:w
5sequential_15/random_crop_5/cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : {
5sequential_15/random_crop_5/cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :ÿÿÿÿ
7sequential_15/random_crop_5/cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ã
6sequential_15/random_crop_5/cond/stateful_uniform/ProdProd@sequential_15/random_crop_5/cond/stateful_uniform/shape:output:0@sequential_15/random_crop_5/cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: z
8sequential_15/random_crop_5/cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :±
8sequential_15/random_crop_5/cond/stateful_uniform/Cast_1Cast?sequential_15/random_crop_5/cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: º
@sequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkipRngReadAndSkipIsequential_15_random_crop_5_cond_stateful_uniform_rngreadandskip_resourceAsequential_15/random_crop_5/cond/stateful_uniform/Cast/x:output:0<sequential_15/random_crop_5/cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:
Esequential_15/random_crop_5/cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Gsequential_15/random_crop_5/cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Gsequential_15/random_crop_5/cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
?sequential_15/random_crop_5/cond/stateful_uniform/strided_sliceStridedSliceHsequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkip:value:0Nsequential_15/random_crop_5/cond/stateful_uniform/strided_slice/stack:output:0Psequential_15/random_crop_5/cond/stateful_uniform/strided_slice/stack_1:output:0Psequential_15/random_crop_5/cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask¿
9sequential_15/random_crop_5/cond/stateful_uniform/BitcastBitcastHsequential_15/random_crop_5/cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Gsequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Isequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Isequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
Asequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1StridedSliceHsequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkip:value:0Psequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1/stack:output:0Rsequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1/stack_1:output:0Rsequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ã
;sequential_15/random_crop_5/cond/stateful_uniform/Bitcast_1BitcastJsequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0w
5sequential_15/random_crop_5/cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :
1sequential_15/random_crop_5/cond/stateful_uniformStatelessRandomUniformIntV2@sequential_15/random_crop_5/cond/stateful_uniform/shape:output:0Dsequential_15/random_crop_5/cond/stateful_uniform/Bitcast_1:output:0Bsequential_15/random_crop_5/cond/stateful_uniform/Bitcast:output:0>sequential_15/random_crop_5/cond/stateful_uniform/alg:output:0>sequential_15/random_crop_5/cond/stateful_uniform/min:output:0>sequential_15/random_crop_5/cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0~
4sequential_15/random_crop_5/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6sequential_15/random_crop_5/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6sequential_15/random_crop_5/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
.sequential_15/random_crop_5/cond/strided_sliceStridedSlice:sequential_15/random_crop_5/cond/stateful_uniform:output:0=sequential_15/random_crop_5/cond/strided_slice/stack:output:0?sequential_15/random_crop_5/cond/strided_slice/stack_1:output:0?sequential_15/random_crop_5/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&sequential_15/random_crop_5/cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :Å
$sequential_15/random_crop_5/cond/addAddV2Dsequential_15_random_crop_5_cond_add_sequential_15_random_crop_5_sub/sequential_15/random_crop_5/cond/add/y:output:0*
T0*
_output_shapes
: ´
$sequential_15/random_crop_5/cond/modFloorMod7sequential_15/random_crop_5/cond/strided_slice:output:0(sequential_15/random_crop_5/cond/add:z:0*
T0*
_output_shapes
: 
6sequential_15/random_crop_5/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
8sequential_15/random_crop_5/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8sequential_15/random_crop_5/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0sequential_15/random_crop_5/cond/strided_slice_1StridedSlice:sequential_15/random_crop_5/cond/stateful_uniform:output:0?sequential_15/random_crop_5/cond/strided_slice_1/stack:output:0Asequential_15/random_crop_5/cond/strided_slice_1/stack_1:output:0Asequential_15/random_crop_5/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_15/random_crop_5/cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Í
&sequential_15/random_crop_5/cond/add_1AddV2Hsequential_15_random_crop_5_cond_add_1_sequential_15_random_crop_5_sub_11sequential_15/random_crop_5/cond/add_1/y:output:0*
T0*
_output_shapes
: º
&sequential_15/random_crop_5/cond/mod_1FloorMod9sequential_15/random_crop_5/cond/strided_slice_1:output:0*sequential_15/random_crop_5/cond/add_1:z:0*
T0*
_output_shapes
: ­
;sequential_15/random_crop_5/cond/crop_to_bounding_box/ShapeShapeBsequential_15_random_crop_5_cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:¿
=sequential_15/random_crop_5/cond/crop_to_bounding_box/unstackUnpackDsequential_15/random_crop_5/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
num
Dsequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : î
Bsequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqualGreaterEqual*sequential_15/random_crop_5/cond/mod_1:z:0Msequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: 
Bsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.¥
Jsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.
Csequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/AssertAssertFsequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual:z:0Ssequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 
Fsequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ð
Dsequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqual(sequential_15/random_crop_5/cond/mod:z:0Osequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
:  
Dsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.¨
Lsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.Ø
Esequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/AssertAssertHsequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual_1:z:0Usequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0D^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 }
;sequential_15/random_crop_5/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B : Õ
9sequential_15/random_crop_5/cond/crop_to_bounding_box/addAddV2Dsequential_15/random_crop_5/cond/crop_to_bounding_box/add/x:output:0*sequential_15/random_crop_5/cond/mod_1:z:0*
T0*
_output_shapes
: 
Asequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ø
?sequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual	LessEqual=sequential_15/random_crop_5/cond/crop_to_bounding_box/add:z:0Jsequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ¦
Dsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.®
Lsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.Õ
Esequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/AssertAssertCsequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual:z:0Usequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0F^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 
=sequential_15/random_crop_5/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : ×
;sequential_15/random_crop_5/cond/crop_to_bounding_box/add_1AddV2Fsequential_15/random_crop_5/cond/crop_to_bounding_box/add_1/x:output:0(sequential_15/random_crop_5/cond/mod:z:0*
T0*
_output_shapes
: 
Csequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : þ
Asequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual_1	LessEqual?sequential_15/random_crop_5/cond/crop_to_bounding_box/add_1:z:0Lsequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: §
Dsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.¯
Lsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.×
Esequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/AssertAssertEsequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual_1:z:0Usequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0F^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 
Hsequential_15/random_crop_5/cond/crop_to_bounding_box/control_dependencyIdentityBsequential_15_random_crop_5_cond_crop_to_bounding_box_shape_inputsD^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/AssertF^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/AssertF^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/AssertF^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
=sequential_15/random_crop_5/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : 
=sequential_15/random_crop_5/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ×
;sequential_15/random_crop_5/cond/crop_to_bounding_box/stackPackFsequential_15/random_crop_5/cond/crop_to_bounding_box/stack/0:output:0(sequential_15/random_crop_5/cond/mod:z:0*sequential_15/random_crop_5/cond/mod_1:z:0Fsequential_15/random_crop_5/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:¾
=sequential_15/random_crop_5/cond/crop_to_bounding_box/Shape_1ShapeQsequential_15/random_crop_5/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:
Isequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ksequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ksequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:á
Csequential_15/random_crop_5/cond/crop_to_bounding_box/strided_sliceStridedSliceFsequential_15/random_crop_5/cond/crop_to_bounding_box/Shape_1:output:0Rsequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice/stack:output:0Tsequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice/stack_1:output:0Tsequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¾
=sequential_15/random_crop_5/cond/crop_to_bounding_box/Shape_2ShapeQsequential_15/random_crop_5/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:
Ksequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Msequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Msequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
Esequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1StridedSliceFsequential_15/random_crop_5/cond/crop_to_bounding_box/Shape_2:output:0Tsequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack:output:0Vsequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0Vsequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?sequential_15/random_crop_5/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
?sequential_15/random_crop_5/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : ¥
=sequential_15/random_crop_5/cond/crop_to_bounding_box/stack_1PackLsequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice:output:0Hsequential_15/random_crop_5/cond/crop_to_bounding_box/stack_1/1:output:0Hsequential_15/random_crop_5/cond/crop_to_bounding_box/stack_1/2:output:0Nsequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:ì
;sequential_15/random_crop_5/cond/crop_to_bounding_box/SliceSliceQsequential_15/random_crop_5/cond/crop_to_bounding_box/control_dependency:output:0Dsequential_15/random_crop_5/cond/crop_to_bounding_box/stack:output:0Fsequential_15/random_crop_5/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  Ý
)sequential_15/random_crop_5/cond/IdentityIdentityDsequential_15/random_crop_5/cond/crop_to_bounding_box/Slice:output:0&^sequential_15/random_crop_5/cond/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  È
%sequential_15/random_crop_5/cond/NoOpNoOpD^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/AssertF^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/AssertF^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/AssertF^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/AssertA^sequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "_
)sequential_15_random_crop_5_cond_identity2sequential_15/random_crop_5/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :ÿÿÿÿÿÿÿÿÿ  2
Csequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/AssertCsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/Assert2
Esequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/AssertEsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert2
Esequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/AssertEsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert2
Esequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/AssertEsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert2
@sequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkip@sequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
îª
¿
<sequential_16_sequential_15_random_crop_5_cond_true_22198963e
Wsequential_16_sequential_15_random_crop_5_cond_stateful_uniform_rngreadandskip_resource:	d
`sequential_16_sequential_15_random_crop_5_cond_add_sequential_16_sequential_15_random_crop_5_subh
dsequential_16_sequential_15_random_crop_5_cond_add_1_sequential_16_sequential_15_random_crop_5_sub_1a
]sequential_16_sequential_15_random_crop_5_cond_crop_to_bounding_box_shape_sequential_15_input;
7sequential_16_sequential_15_random_crop_5_cond_identity¢Qsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/Assert¢Ssequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert¢Ssequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert¢Ssequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert¢Nsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkip
Esequential_16/sequential_15/random_crop_5/cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:
Csequential_16/sequential_15/random_crop_5/cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : 
Csequential_16/sequential_15/random_crop_5/cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :ÿÿÿÿ
Esequential_16/sequential_15/random_crop_5/cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Dsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/ProdProdNsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/shape:output:0Nsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 
Fsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Í
Fsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/Cast_1CastMsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ò
Nsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkipRngReadAndSkipWsequential_16_sequential_15_random_crop_5_cond_stateful_uniform_rngreadandskip_resourceOsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/Cast/x:output:0Jsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:
Ssequential_16/sequential_15/random_crop_5/cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Usequential_16/sequential_15/random_crop_5/cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Usequential_16/sequential_15/random_crop_5/cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Msequential_16/sequential_15/random_crop_5/cond/stateful_uniform/strided_sliceStridedSliceVsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkip:value:0\sequential_16/sequential_15/random_crop_5/cond/stateful_uniform/strided_slice/stack:output:0^sequential_16/sequential_15/random_crop_5/cond/stateful_uniform/strided_slice/stack_1:output:0^sequential_16/sequential_15/random_crop_5/cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskÛ
Gsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/BitcastBitcastVsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Usequential_16/sequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:¡
Wsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:¡
Wsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Osequential_16/sequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1StridedSliceVsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkip:value:0^sequential_16/sequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1/stack:output:0`sequential_16/sequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1/stack_1:output:0`sequential_16/sequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:ß
Isequential_16/sequential_15/random_crop_5/cond/stateful_uniform/Bitcast_1BitcastXsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
Csequential_16/sequential_15/random_crop_5/cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :ç
?sequential_16/sequential_15/random_crop_5/cond/stateful_uniformStatelessRandomUniformIntV2Nsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/shape:output:0Rsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/Bitcast_1:output:0Psequential_16/sequential_15/random_crop_5/cond/stateful_uniform/Bitcast:output:0Lsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/alg:output:0Lsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/min:output:0Lsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0
Bsequential_16/sequential_15/random_crop_5/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Dsequential_16/sequential_15/random_crop_5/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Dsequential_16/sequential_15/random_crop_5/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
<sequential_16/sequential_15/random_crop_5/cond/strided_sliceStridedSliceHsequential_16/sequential_15/random_crop_5/cond/stateful_uniform:output:0Ksequential_16/sequential_15/random_crop_5/cond/strided_slice/stack:output:0Msequential_16/sequential_15/random_crop_5/cond/strided_slice/stack_1:output:0Msequential_16/sequential_15/random_crop_5/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4sequential_16/sequential_15/random_crop_5/cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :ý
2sequential_16/sequential_15/random_crop_5/cond/addAddV2`sequential_16_sequential_15_random_crop_5_cond_add_sequential_16_sequential_15_random_crop_5_sub=sequential_16/sequential_15/random_crop_5/cond/add/y:output:0*
T0*
_output_shapes
: Þ
2sequential_16/sequential_15/random_crop_5/cond/modFloorModEsequential_16/sequential_15/random_crop_5/cond/strided_slice:output:06sequential_16/sequential_15/random_crop_5/cond/add:z:0*
T0*
_output_shapes
: 
Dsequential_16/sequential_15/random_crop_5/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Fsequential_16/sequential_15/random_crop_5/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Fsequential_16/sequential_15/random_crop_5/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ï
>sequential_16/sequential_15/random_crop_5/cond/strided_slice_1StridedSliceHsequential_16/sequential_15/random_crop_5/cond/stateful_uniform:output:0Msequential_16/sequential_15/random_crop_5/cond/strided_slice_1/stack:output:0Osequential_16/sequential_15/random_crop_5/cond/strided_slice_1/stack_1:output:0Osequential_16/sequential_15/random_crop_5/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
6sequential_16/sequential_15/random_crop_5/cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
4sequential_16/sequential_15/random_crop_5/cond/add_1AddV2dsequential_16_sequential_15_random_crop_5_cond_add_1_sequential_16_sequential_15_random_crop_5_sub_1?sequential_16/sequential_15/random_crop_5/cond/add_1/y:output:0*
T0*
_output_shapes
: ä
4sequential_16/sequential_15/random_crop_5/cond/mod_1FloorModGsequential_16/sequential_15/random_crop_5/cond/strided_slice_1:output:08sequential_16/sequential_15/random_crop_5/cond/add_1:z:0*
T0*
_output_shapes
: Ö
Isequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/ShapeShape]sequential_16_sequential_15_random_crop_5_cond_crop_to_bounding_box_shape_sequential_15_input*
T0*
_output_shapes
:Û
Ksequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/unstackUnpackRsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
num
Rsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 
Psequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqualGreaterEqual8sequential_16/sequential_15/random_crop_5/cond/mod_1:z:0[sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: «
Psequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.³
Xsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.¶
Qsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/AssertAssertTsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual:z:0asequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 
Tsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : 
Rsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqual6sequential_16/sequential_15/random_crop_5/cond/mod:z:0]sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ®
Rsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.¶
Zsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.
Ssequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/AssertAssertVsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual_1:z:0csequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0R^sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 
Isequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B : ÿ
Gsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/addAddV2Rsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/add/x:output:08sequential_16/sequential_15/random_crop_5/cond/mod_1:z:0*
T0*
_output_shapes
: 
Osequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ¢
Msequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual	LessEqualKsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/add:z:0Xsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ´
Rsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.¼
Zsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.
Ssequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/AssertAssertQsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual:z:0csequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0T^sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 
Ksequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : 
Isequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/add_1AddV2Tsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/add_1/x:output:06sequential_16/sequential_15/random_crop_5/cond/mod:z:0*
T0*
_output_shapes
: 
Qsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ¨
Osequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual_1	LessEqualMsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/add_1:z:0Zsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: µ
Rsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.½
Zsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.
Ssequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/AssertAssertSsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual_1:z:0csequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0T^sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ù
Vsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/control_dependencyIdentity]sequential_16_sequential_15_random_crop_5_cond_crop_to_bounding_box_shape_sequential_15_inputR^sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/AssertT^sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/AssertT^sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/AssertT^sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert*
T0*&
_class
loc:@sequential_15_input*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
Ksequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : 
Ksequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 
Isequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/stackPackTsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/stack/0:output:06sequential_16/sequential_15/random_crop_5/cond/mod:z:08sequential_16/sequential_15/random_crop_5/cond/mod_1:z:0Tsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:Ú
Ksequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Shape_1Shape_sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:¡
Wsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: £
Ysequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:£
Ysequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:§
Qsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/strided_sliceStridedSliceTsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Shape_1:output:0`sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice/stack:output:0bsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice/stack_1:output:0bsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÚ
Ksequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Shape_2Shape_sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:£
Ysequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:¥
[sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:¥
[sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
Ssequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1StridedSliceTsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Shape_2:output:0bsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack:output:0dsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0dsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Msequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Msequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : ë
Ksequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/stack_1PackZsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice:output:0Vsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/stack_1/1:output:0Vsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/stack_1/2:output:0\sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:¤
Isequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/SliceSlice_sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/control_dependency:output:0Rsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/stack:output:0Tsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
7sequential_16/sequential_15/random_crop_5/cond/IdentityIdentityRsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Slice:output:04^sequential_16/sequential_15/random_crop_5/cond/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
3sequential_16/sequential_15/random_crop_5/cond/NoOpNoOpR^sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/AssertT^sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/AssertT^sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/AssertT^sequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/AssertO^sequential_16/sequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "{
7sequential_16_sequential_15_random_crop_5_cond_identity@sequential_16/sequential_15/random_crop_5/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :ÿÿÿÿÿÿÿÿÿ  2¦
Qsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/AssertQsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/Assert2ª
Ssequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/AssertSsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert2ª
Ssequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/AssertSsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert2ª
Ssequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/AssertSsequential_16/sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert2 
Nsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkipNsequential_16/sequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
¬I

=sequential_16_sequential_15_random_crop_5_cond_false_22198964>
:sequential_16_sequential_15_random_crop_5_cond_placeholder@
<sequential_16_sequential_15_random_crop_5_cond_placeholder_1@
<sequential_16_sequential_15_random_crop_5_cond_placeholder_2L
Hsequential_16_sequential_15_random_crop_5_cond_shape_sequential_15_input;
7sequential_16_sequential_15_random_crop_5_cond_identity¬
4sequential_16/sequential_15/random_crop_5/cond/ShapeShapeHsequential_16_sequential_15_random_crop_5_cond_shape_sequential_15_input*
T0*
_output_shapes
:
Bsequential_16/sequential_15/random_crop_5/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
Dsequential_16/sequential_15/random_crop_5/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
Dsequential_16/sequential_15/random_crop_5/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¼
<sequential_16/sequential_15/random_crop_5/cond/strided_sliceStridedSlice=sequential_16/sequential_15/random_crop_5/cond/Shape:output:0Ksequential_16/sequential_15/random_crop_5/cond/strided_slice/stack:output:0Msequential_16/sequential_15/random_crop_5/cond/strided_slice/stack_1:output:0Msequential_16/sequential_15/random_crop_5/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Dsequential_16/sequential_15/random_crop_5/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
Fsequential_16/sequential_15/random_crop_5/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
Fsequential_16/sequential_15/random_crop_5/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
>sequential_16/sequential_15/random_crop_5/cond/strided_slice_1StridedSlice=sequential_16/sequential_15/random_crop_5/cond/Shape:output:0Msequential_16/sequential_15/random_crop_5/cond/strided_slice_1/stack:output:0Osequential_16/sequential_15/random_crop_5/cond/strided_slice_1/stack_1:output:0Osequential_16/sequential_15/random_crop_5/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4sequential_16/sequential_15/random_crop_5/cond/mul/yConst*
_output_shapes
: *
dtype0*
value	B : â
2sequential_16/sequential_15/random_crop_5/cond/mulMulGsequential_16/sequential_15/random_crop_5/cond/strided_slice_1:output:0=sequential_16/sequential_15/random_crop_5/cond/mul/y:output:0*
T0*
_output_shapes
: £
3sequential_16/sequential_15/random_crop_5/cond/CastCast6sequential_16/sequential_15/random_crop_5/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: }
8sequential_16/sequential_15/random_crop_5/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   BÞ
6sequential_16/sequential_15/random_crop_5/cond/truedivRealDiv7sequential_16/sequential_15/random_crop_5/cond/Cast:y:0Asequential_16/sequential_15/random_crop_5/cond/truediv/y:output:0*
T0*
_output_shapes
: ©
5sequential_16/sequential_15/random_crop_5/cond/Cast_1Cast:sequential_16/sequential_15/random_crop_5/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: x
6sequential_16/sequential_15/random_crop_5/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B : ä
4sequential_16/sequential_15/random_crop_5/cond/mul_1MulEsequential_16/sequential_15/random_crop_5/cond/strided_slice:output:0?sequential_16/sequential_15/random_crop_5/cond/mul_1/y:output:0*
T0*
_output_shapes
: §
5sequential_16/sequential_15/random_crop_5/cond/Cast_2Cast8sequential_16/sequential_15/random_crop_5/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 
:sequential_16/sequential_15/random_crop_5/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Bä
8sequential_16/sequential_15/random_crop_5/cond/truediv_1RealDiv9sequential_16/sequential_15/random_crop_5/cond/Cast_2:y:0Csequential_16/sequential_15/random_crop_5/cond/truediv_1/y:output:0*
T0*
_output_shapes
: «
5sequential_16/sequential_15/random_crop_5/cond/Cast_3Cast<sequential_16/sequential_15/random_crop_5/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ä
6sequential_16/sequential_15/random_crop_5/cond/MinimumMinimumEsequential_16/sequential_15/random_crop_5/cond/strided_slice:output:09sequential_16/sequential_15/random_crop_5/cond/Cast_1:y:0*
T0*
_output_shapes
: è
8sequential_16/sequential_15/random_crop_5/cond/Minimum_1MinimumGsequential_16/sequential_15/random_crop_5/cond/strided_slice_1:output:09sequential_16/sequential_15/random_crop_5/cond/Cast_3:y:0*
T0*
_output_shapes
: Ý
2sequential_16/sequential_15/random_crop_5/cond/subSubEsequential_16/sequential_15/random_crop_5/cond/strided_slice:output:0:sequential_16/sequential_15/random_crop_5/cond/Minimum:z:0*
T0*
_output_shapes
: ¥
5sequential_16/sequential_15/random_crop_5/cond/Cast_4Cast6sequential_16/sequential_15/random_crop_5/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: 
:sequential_16/sequential_15/random_crop_5/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ä
8sequential_16/sequential_15/random_crop_5/cond/truediv_2RealDiv9sequential_16/sequential_15/random_crop_5/cond/Cast_4:y:0Csequential_16/sequential_15/random_crop_5/cond/truediv_2/y:output:0*
T0*
_output_shapes
: «
5sequential_16/sequential_15/random_crop_5/cond/Cast_5Cast<sequential_16/sequential_15/random_crop_5/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ã
4sequential_16/sequential_15/random_crop_5/cond/sub_1SubGsequential_16/sequential_15/random_crop_5/cond/strided_slice_1:output:0<sequential_16/sequential_15/random_crop_5/cond/Minimum_1:z:0*
T0*
_output_shapes
: §
5sequential_16/sequential_15/random_crop_5/cond/Cast_6Cast8sequential_16/sequential_15/random_crop_5/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 
:sequential_16/sequential_15/random_crop_5/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ä
8sequential_16/sequential_15/random_crop_5/cond/truediv_3RealDiv9sequential_16/sequential_15/random_crop_5/cond/Cast_6:y:0Csequential_16/sequential_15/random_crop_5/cond/truediv_3/y:output:0*
T0*
_output_shapes
: «
5sequential_16/sequential_15/random_crop_5/cond/Cast_7Cast<sequential_16/sequential_15/random_crop_5/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: x
6sequential_16/sequential_15/random_crop_5/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : x
6sequential_16/sequential_15/random_crop_5/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : â
4sequential_16/sequential_15/random_crop_5/cond/stackPack?sequential_16/sequential_15/random_crop_5/cond/stack/0:output:09sequential_16/sequential_15/random_crop_5/cond/Cast_5:y:09sequential_16/sequential_15/random_crop_5/cond/Cast_7:y:0?sequential_16/sequential_15/random_crop_5/cond/stack/3:output:0*
N*
T0*
_output_shapes
:
8sequential_16/sequential_15/random_crop_5/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
8sequential_16/sequential_15/random_crop_5/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿì
6sequential_16/sequential_15/random_crop_5/cond/stack_1PackAsequential_16/sequential_15/random_crop_5/cond/stack_1/0:output:0:sequential_16/sequential_15/random_crop_5/cond/Minimum:z:0<sequential_16/sequential_15/random_crop_5/cond/Minimum_1:z:0Asequential_16/sequential_15/random_crop_5/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:×
4sequential_16/sequential_15/random_crop_5/cond/SliceSliceHsequential_16_sequential_15_random_crop_5_cond_shape_sequential_15_input=sequential_16/sequential_15/random_crop_5/cond/stack:output:0?sequential_16/sequential_15/random_crop_5/cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ  ÿÿÿÿÿÿÿÿÿ
:sequential_16/sequential_15/random_crop_5/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        ®
Dsequential_16/sequential_15/random_crop_5/cond/resize/ResizeBilinearResizeBilinear=sequential_16/sequential_15/random_crop_5/cond/Slice:output:0Csequential_16/sequential_15/random_crop_5/cond/resize/size:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
half_pixel_centers(Ô
7sequential_16/sequential_15/random_crop_5/cond/IdentityIdentityUsequential_16/sequential_15/random_crop_5/cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "{
7sequential_16_sequential_15_random_crop_5_cond_identity@sequential_16/sequential_15/random_crop_5/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :ÿÿÿÿÿÿÿÿÿ  :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
¿g

cond_true_22199437;
-cond_stateful_uniform_rngreadandskip_resource:	
cond_add_sub
cond_add_1_sub_1*
&cond_crop_to_bounding_box_shape_inputs
cond_identity¢'cond/crop_to_bounding_box/Assert/Assert¢)cond/crop_to_bounding_box/Assert_1/Assert¢)cond/crop_to_bounding_box/Assert_2/Assert¢)cond/crop_to_bounding_box/Assert_3/Assert¢$cond/stateful_uniform/RngReadAndSkipe
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
valueB :ÿÿÿÿe
cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
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
: Ê
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
valueB:Å
#cond/stateful_uniform/strided_sliceStridedSlice,cond/stateful_uniform/RngReadAndSkip:value:02cond/stateful_uniform/strided_slice/stack:output:04cond/stateful_uniform/strided_slice/stack_1:output:04cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
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
valueB:»
%cond/stateful_uniform/strided_slice_1StridedSlice,cond/stateful_uniform/RngReadAndSkip:value:04cond/stateful_uniform/strided_slice_1/stack:output:06cond/stateful_uniform/strided_slice_1/stack_1:output:06cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
cond/stateful_uniform/Bitcast_1Bitcast.cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0[
cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :Á
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
valueB:õ
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
valueB:ý
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
:
!cond/crop_to_bounding_box/unstackUnpack(cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numj
(cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 
&cond/crop_to_bounding_box/GreaterEqualGreaterEqualcond/mod_1:z:01cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: 
&cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.
.cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.¸
'cond/crop_to_bounding_box/Assert/AssertAssert*cond/crop_to_bounding_box/GreaterEqual:z:07cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 l
*cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : 
(cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcond/mod:z:03cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: 
(cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.
0cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.è
)cond/crop_to_bounding_box/Assert_1/AssertAssert,cond/crop_to_bounding_box/GreaterEqual_1:z:09cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0(^cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 a
cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B : 
cond/crop_to_bounding_box/addAddV2(cond/crop_to_bounding_box/add/x:output:0cond/mod_1:z:0*
T0*
_output_shapes
: g
%cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ¤
#cond/crop_to_bounding_box/LessEqual	LessEqual!cond/crop_to_bounding_box/add:z:0.cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: 
(cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.
0cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.å
)cond/crop_to_bounding_box/Assert_2/AssertAssert'cond/crop_to_bounding_box/LessEqual:z:09cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 c
!cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : 
cond/crop_to_bounding_box/add_1AddV2*cond/crop_to_bounding_box/add_1/x:output:0cond/mod:z:0*
T0*
_output_shapes
: i
'cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ª
%cond/crop_to_bounding_box/LessEqual_1	LessEqual#cond/crop_to_bounding_box/add_1:z:00cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: 
(cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.
0cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.ç
)cond/crop_to_bounding_box/Assert_3/AssertAssert)cond/crop_to_bounding_box/LessEqual_1:z:09cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ã
,cond/crop_to_bounding_box/control_dependencyIdentity&cond_crop_to_bounding_box_shape_inputs(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  c
!cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : c
!cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : Ë
cond/crop_to_bounding_box/stackPack*cond/crop_to_bounding_box/stack/0:output:0cond/mod:z:0cond/mod_1:z:0*cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:
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
valueB:Õ
'cond/crop_to_bounding_box/strided_sliceStridedSlice*cond/crop_to_bounding_box/Shape_1:output:06cond/crop_to_bounding_box/strided_slice/stack:output:08cond/crop_to_bounding_box/strided_slice/stack_1:output:08cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
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
valueB:Ý
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
value	B : 
!cond/crop_to_bounding_box/stack_1Pack0cond/crop_to_bounding_box/strided_slice:output:0,cond/crop_to_bounding_box/stack_1/1:output:0,cond/crop_to_bounding_box/stack_1/2:output:02cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:ü
cond/crop_to_bounding_box/SliceSlice5cond/crop_to_bounding_box/control_dependency:output:0(cond/crop_to_bounding_box/stack:output:0*cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
cond/IdentityIdentity(cond/crop_to_bounding_box/Slice:output:0
^cond/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   
	cond/NoOpNoOp(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert%^cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :ÿÿÿÿÿÿÿÿÿ  2R
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
:ÿÿÿÿÿÿÿÿÿ  
¿g

cond_true_22201324;
-cond_stateful_uniform_rngreadandskip_resource:	
cond_add_sub
cond_add_1_sub_1*
&cond_crop_to_bounding_box_shape_inputs
cond_identity¢'cond/crop_to_bounding_box/Assert/Assert¢)cond/crop_to_bounding_box/Assert_1/Assert¢)cond/crop_to_bounding_box/Assert_2/Assert¢)cond/crop_to_bounding_box/Assert_3/Assert¢$cond/stateful_uniform/RngReadAndSkipe
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
valueB :ÿÿÿÿe
cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
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
: Ê
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
valueB:Å
#cond/stateful_uniform/strided_sliceStridedSlice,cond/stateful_uniform/RngReadAndSkip:value:02cond/stateful_uniform/strided_slice/stack:output:04cond/stateful_uniform/strided_slice/stack_1:output:04cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
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
valueB:»
%cond/stateful_uniform/strided_slice_1StridedSlice,cond/stateful_uniform/RngReadAndSkip:value:04cond/stateful_uniform/strided_slice_1/stack:output:06cond/stateful_uniform/strided_slice_1/stack_1:output:06cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
cond/stateful_uniform/Bitcast_1Bitcast.cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0[
cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :Á
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
valueB:õ
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
valueB:ý
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
:
!cond/crop_to_bounding_box/unstackUnpack(cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numj
(cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 
&cond/crop_to_bounding_box/GreaterEqualGreaterEqualcond/mod_1:z:01cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: 
&cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.
.cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.¸
'cond/crop_to_bounding_box/Assert/AssertAssert*cond/crop_to_bounding_box/GreaterEqual:z:07cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 l
*cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : 
(cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcond/mod:z:03cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: 
(cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.
0cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.è
)cond/crop_to_bounding_box/Assert_1/AssertAssert,cond/crop_to_bounding_box/GreaterEqual_1:z:09cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0(^cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 a
cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B : 
cond/crop_to_bounding_box/addAddV2(cond/crop_to_bounding_box/add/x:output:0cond/mod_1:z:0*
T0*
_output_shapes
: g
%cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ¤
#cond/crop_to_bounding_box/LessEqual	LessEqual!cond/crop_to_bounding_box/add:z:0.cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: 
(cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.
0cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.å
)cond/crop_to_bounding_box/Assert_2/AssertAssert'cond/crop_to_bounding_box/LessEqual:z:09cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 c
!cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : 
cond/crop_to_bounding_box/add_1AddV2*cond/crop_to_bounding_box/add_1/x:output:0cond/mod:z:0*
T0*
_output_shapes
: i
'cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ª
%cond/crop_to_bounding_box/LessEqual_1	LessEqual#cond/crop_to_bounding_box/add_1:z:00cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: 
(cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.
0cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.ç
)cond/crop_to_bounding_box/Assert_3/AssertAssert)cond/crop_to_bounding_box/LessEqual_1:z:09cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ã
,cond/crop_to_bounding_box/control_dependencyIdentity&cond_crop_to_bounding_box_shape_inputs(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  c
!cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : c
!cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : Ë
cond/crop_to_bounding_box/stackPack*cond/crop_to_bounding_box/stack/0:output:0cond/mod:z:0cond/mod_1:z:0*cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:
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
valueB:Õ
'cond/crop_to_bounding_box/strided_sliceStridedSlice*cond/crop_to_bounding_box/Shape_1:output:06cond/crop_to_bounding_box/strided_slice/stack:output:08cond/crop_to_bounding_box/strided_slice/stack_1:output:08cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
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
valueB:Ý
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
value	B : 
!cond/crop_to_bounding_box/stack_1Pack0cond/crop_to_bounding_box/strided_slice:output:0,cond/crop_to_bounding_box/stack_1/1:output:0,cond/crop_to_bounding_box/stack_1/2:output:02cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:ü
cond/crop_to_bounding_box/SliceSlice5cond/crop_to_bounding_box/control_dependency:output:0(cond/crop_to_bounding_box/stack:output:0*cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
cond/IdentityIdentity(cond/crop_to_bounding_box/Slice:output:0
^cond/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   
	cond/NoOpNoOp(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert%^cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :ÿÿÿÿÿÿÿÿÿ  2R
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
:ÿÿÿÿÿÿÿÿÿ  
¡

ø
F__inference_dense_15_layer_call_and_return_conditional_losses_22201076

inputs1
matmul_readvariableop_resource:	x-
biasadd_readvariableop_resource:x
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É	
÷
F__inference_dense_17_layer_call_and_return_conditional_losses_22201115

inputs0
matmul_readvariableop_resource:T
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿT: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs
Æ

+__inference_dense_16_layer_call_fn_22201085

inputs
unknown:xT
	unknown_0:T
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_22199738o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿx: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
Á

0__inference_sequential_15_layer_call_fn_22200590

inputs
unknown:	
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_15_layer_call_and_return_conditional_losses_22199322w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ë
L
0__inference_random_flip_5_layer_call_fn_22201466

inputs
identity¾
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_random_flip_5_layer_call_and_return_conditional_losses_22199319h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ð
¡
,__inference_conv2d_10_layer_call_fn_22200994

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_22199677w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
·

K__inference_random_crop_5_layer_call_and_return_conditional_losses_22201295

inputs
cond_input_0:	
identity¢cond;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿh
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
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
þÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
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
: â
condIfAll:output:0cond_input_0sub:z:0	sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *&
else_branchR
cond_false_22201159*.
output_shapes
:ÿÿÿÿÿÿÿÿÿ  *%
then_branchR
cond_true_22201158b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  M
NoOpNoOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  : 2
condcond:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Á

0__inference_random_crop_5_layer_call_fn_22201129

inputs
unknown:	
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_random_crop_5_layer_call_and_return_conditional_losses_22199574w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
·

K__inference_random_crop_5_layer_call_and_return_conditional_losses_22199311

inputs
cond_input_0:	
identity¢cond;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿh
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
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
þÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
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
: â
condIfAll:output:0cond_input_0sub:z:0	sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *&
else_branchR
cond_false_22199175*.
output_shapes
:ÿÿÿÿÿÿÿÿÿ  *%
then_branchR
cond_true_22199174b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  M
NoOpNoOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  : 2
condcond:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
+

K__inference_sequential_16_layer_call_and_return_conditional_losses_22200036
sequential_15_input$
sequential_15_22200002:	$
sequential_15_22200004:	,
conv2d_10_22200007: 
conv2d_10_22200009:,
conv2d_11_22200013: 
conv2d_11_22200015:$
dense_15_22200020:	x
dense_15_22200022:x#
dense_16_22200025:xT
dense_16_22200027:T#
dense_17_22200030:T

dense_17_22200032:

identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢%sequential_15/StatefulPartitionedCall
%sequential_15/StatefulPartitionedCallStatefulPartitionedCallsequential_15_inputsequential_15_22200002sequential_15_22200004*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_15_layer_call_and_return_conditional_losses_22199598ª
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall.sequential_15/StatefulPartitionedCall:output:0conv2d_10_22200007conv2d_10_22200009*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_22199677þ
$average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_22199641©
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_10/PartitionedCall:output:0conv2d_11_22200013conv2d_11_22200015*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_22199695þ
$average_pooling2d_11/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_22199653ä
flatten_5/PartitionedCallPartitionedCall-average_pooling2d_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_22199708
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_15_22200020dense_15_22200022*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_22199721
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_22200025dense_16_22200027*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_22199738
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_22200030dense_17_22200032*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_22199754x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall&^sequential_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2N
%sequential_15/StatefulPartitionedCall%sequential_15/StatefulPartitionedCall:d `
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
-
_user_specified_namesequential_15_input
Ý

¢
&__inference_signature_wrapper_22200583
sequential_15_input
unknown:	#
	unknown_0:
	unknown_1:#
	unknown_2:
	unknown_3:
	unknown_4:	x
	unknown_5:x
	unknown_6:xT
	unknown_7:T
	unknown_8:T

	unknown_9:

identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallsequential_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_22199138o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
-
_user_specified_namesequential_15_input
Ãn

#__inference__wrapped_model_22199138
sequential_15_inputD
6sequential_16_sequential_15_random_crop_5_cond_input_0:	P
6sequential_16_conv2d_10_conv2d_readvariableop_resource:E
7sequential_16_conv2d_10_biasadd_readvariableop_resource:P
6sequential_16_conv2d_11_conv2d_readvariableop_resource:E
7sequential_16_conv2d_11_biasadd_readvariableop_resource:H
5sequential_16_dense_15_matmul_readvariableop_resource:	xD
6sequential_16_dense_15_biasadd_readvariableop_resource:xG
5sequential_16_dense_16_matmul_readvariableop_resource:xTD
6sequential_16_dense_16_biasadd_readvariableop_resource:TG
5sequential_16_dense_17_matmul_readvariableop_resource:T
D
6sequential_16_dense_17_biasadd_readvariableop_resource:

identity¢.sequential_16/conv2d_10/BiasAdd/ReadVariableOp¢-sequential_16/conv2d_10/Conv2D/ReadVariableOp¢.sequential_16/conv2d_11/BiasAdd/ReadVariableOp¢-sequential_16/conv2d_11/Conv2D/ReadVariableOp¢-sequential_16/dense_15/BiasAdd/ReadVariableOp¢,sequential_16/dense_15/MatMul/ReadVariableOp¢-sequential_16/dense_16/BiasAdd/ReadVariableOp¢,sequential_16/dense_16/MatMul/ReadVariableOp¢-sequential_16/dense_17/BiasAdd/ReadVariableOp¢,sequential_16/dense_17/MatMul/ReadVariableOp¢.sequential_16/sequential_15/random_crop_5/condr
/sequential_16/sequential_15/random_crop_5/ShapeShapesequential_15_input*
T0*
_output_shapes
:
=sequential_16/sequential_15/random_crop_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
?sequential_16/sequential_15/random_crop_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
?sequential_16/sequential_15/random_crop_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:£
7sequential_16/sequential_15/random_crop_5/strided_sliceStridedSlice8sequential_16/sequential_15/random_crop_5/Shape:output:0Fsequential_16/sequential_15/random_crop_5/strided_slice/stack:output:0Hsequential_16/sequential_15/random_crop_5/strided_slice/stack_1:output:0Hsequential_16/sequential_15/random_crop_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/sequential_16/sequential_15/random_crop_5/sub/yConst*
_output_shapes
: *
dtype0*
value	B : Ñ
-sequential_16/sequential_15/random_crop_5/subSub@sequential_16/sequential_15/random_crop_5/strided_slice:output:08sequential_16/sequential_15/random_crop_5/sub/y:output:0*
T0*
_output_shapes
: 
?sequential_16/sequential_15/random_crop_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
Asequential_16/sequential_15/random_crop_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
Asequential_16/sequential_15/random_crop_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
9sequential_16/sequential_15/random_crop_5/strided_slice_1StridedSlice8sequential_16/sequential_15/random_crop_5/Shape:output:0Hsequential_16/sequential_15/random_crop_5/strided_slice_1/stack:output:0Jsequential_16/sequential_15/random_crop_5/strided_slice_1/stack_1:output:0Jsequential_16/sequential_15/random_crop_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1sequential_16/sequential_15/random_crop_5/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B : ×
/sequential_16/sequential_15/random_crop_5/sub_1SubBsequential_16/sequential_15/random_crop_5/strided_slice_1:output:0:sequential_16/sequential_15/random_crop_5/sub_1/y:output:0*
T0*
_output_shapes
: z
8sequential_16/sequential_15/random_crop_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : Ý
6sequential_16/sequential_15/random_crop_5/GreaterEqualGreaterEqual1sequential_16/sequential_15/random_crop_5/sub:z:0Asequential_16/sequential_15/random_crop_5/GreaterEqual/y:output:0*
T0*
_output_shapes
: |
:sequential_16/sequential_15/random_crop_5/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ã
8sequential_16/sequential_15/random_crop_5/GreaterEqual_1GreaterEqual3sequential_16/sequential_15/random_crop_5/sub_1:z:0Csequential_16/sequential_15/random_crop_5/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: y
7sequential_16/sequential_15/random_crop_5/Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z §
5sequential_16/sequential_15/random_crop_5/Rank/packedPack@sequential_16/sequential_15/random_crop_5/Rank/packed/0:output:0:sequential_16/sequential_15/random_crop_5/GreaterEqual:z:0<sequential_16/sequential_15/random_crop_5/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:p
.sequential_16/sequential_15/random_crop_5/RankConst*
_output_shapes
: *
dtype0*
value	B :w
5sequential_16/sequential_15/random_crop_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : w
5sequential_16/sequential_15/random_crop_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
/sequential_16/sequential_15/random_crop_5/rangeRange>sequential_16/sequential_15/random_crop_5/range/start:output:07sequential_16/sequential_15/random_crop_5/Rank:output:0>sequential_16/sequential_15/random_crop_5/range/delta:output:0*
_output_shapes
:w
5sequential_16/sequential_15/random_crop_5/All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z £
3sequential_16/sequential_15/random_crop_5/All/inputPack>sequential_16/sequential_15/random_crop_5/All/input/0:output:0:sequential_16/sequential_15/random_crop_5/GreaterEqual:z:0<sequential_16/sequential_15/random_crop_5/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:Ä
-sequential_16/sequential_15/random_crop_5/AllAll<sequential_16/sequential_15/random_crop_5/All/input:output:08sequential_16/sequential_15/random_crop_5/range:output:0*
_output_shapes
: 
.sequential_16/sequential_15/random_crop_5/condIf6sequential_16/sequential_15/random_crop_5/All:output:06sequential_16_sequential_15_random_crop_5_cond_input_01sequential_16/sequential_15/random_crop_5/sub:z:03sequential_16/sequential_15/random_crop_5/sub_1:z:0sequential_15_input*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *P
else_branchAR?
=sequential_16_sequential_15_random_crop_5_cond_false_22198964*.
output_shapes
:ÿÿÿÿÿÿÿÿÿ  *O
then_branch@R>
<sequential_16_sequential_15_random_crop_5_cond_true_22198963¶
7sequential_16/sequential_15/random_crop_5/cond/IdentityIdentity7sequential_16/sequential_15/random_crop_5/cond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¬
-sequential_16/conv2d_10/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
sequential_16/conv2d_10/Conv2DConv2D@sequential_16/sequential_15/random_crop_5/cond/Identity:output:05sequential_16/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
¢
.sequential_16/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Å
sequential_16/conv2d_10/BiasAddBiasAdd'sequential_16/conv2d_10/Conv2D:output:06sequential_16/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_16/conv2d_10/ReluRelu(sequential_16/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
*sequential_16/average_pooling2d_10/AvgPoolAvgPool*sequential_16/conv2d_10/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¬
-sequential_16/conv2d_11/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0÷
sequential_16/conv2d_11/Conv2DConv2D3sequential_16/average_pooling2d_10/AvgPool:output:05sequential_16/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
paddingVALID*
strides
¢
.sequential_16/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Å
sequential_16/conv2d_11/BiasAddBiasAdd'sequential_16/conv2d_11/Conv2D:output:06sequential_16/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


sequential_16/conv2d_11/ReluRelu(sequential_16/conv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

×
*sequential_16/average_pooling2d_11/AvgPoolAvgPool*sequential_16/conv2d_11/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
n
sequential_16/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  º
sequential_16/flatten_5/ReshapeReshape3sequential_16/average_pooling2d_11/AvgPool:output:0&sequential_16/flatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
,sequential_16/dense_15/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_15_matmul_readvariableop_resource*
_output_shapes
:	x*
dtype0¹
sequential_16/dense_15/MatMulMatMul(sequential_16/flatten_5/Reshape:output:04sequential_16/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 
-sequential_16/dense_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_15_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0»
sequential_16/dense_15/BiasAddBiasAdd'sequential_16/dense_15/MatMul:product:05sequential_16/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx~
sequential_16/dense_15/ReluRelu'sequential_16/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx¢
,sequential_16/dense_16/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_16_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype0º
sequential_16/dense_16/MatMulMatMul)sequential_16/dense_15/Relu:activations:04sequential_16/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT 
-sequential_16/dense_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_16_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype0»
sequential_16/dense_16/BiasAddBiasAdd'sequential_16/dense_16/MatMul:product:05sequential_16/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT~
sequential_16/dense_16/ReluRelu'sequential_16/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT¢
,sequential_16/dense_17/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_17_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype0º
sequential_16/dense_17/MatMulMatMul)sequential_16/dense_16/Relu:activations:04sequential_16/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
-sequential_16/dense_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_17_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0»
sequential_16/dense_17/BiasAddBiasAdd'sequential_16/dense_17/MatMul:product:05sequential_16/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
v
IdentityIdentity'sequential_16/dense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ö
NoOpNoOp/^sequential_16/conv2d_10/BiasAdd/ReadVariableOp.^sequential_16/conv2d_10/Conv2D/ReadVariableOp/^sequential_16/conv2d_11/BiasAdd/ReadVariableOp.^sequential_16/conv2d_11/Conv2D/ReadVariableOp.^sequential_16/dense_15/BiasAdd/ReadVariableOp-^sequential_16/dense_15/MatMul/ReadVariableOp.^sequential_16/dense_16/BiasAdd/ReadVariableOp-^sequential_16/dense_16/MatMul/ReadVariableOp.^sequential_16/dense_17/BiasAdd/ReadVariableOp-^sequential_16/dense_17/MatMul/ReadVariableOp/^sequential_16/sequential_15/random_crop_5/cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : 2`
.sequential_16/conv2d_10/BiasAdd/ReadVariableOp.sequential_16/conv2d_10/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_10/Conv2D/ReadVariableOp-sequential_16/conv2d_10/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_11/BiasAdd/ReadVariableOp.sequential_16/conv2d_11/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_11/Conv2D/ReadVariableOp-sequential_16/conv2d_11/Conv2D/ReadVariableOp2^
-sequential_16/dense_15/BiasAdd/ReadVariableOp-sequential_16/dense_15/BiasAdd/ReadVariableOp2\
,sequential_16/dense_15/MatMul/ReadVariableOp,sequential_16/dense_15/MatMul/ReadVariableOp2^
-sequential_16/dense_16/BiasAdd/ReadVariableOp-sequential_16/dense_16/BiasAdd/ReadVariableOp2\
,sequential_16/dense_16/MatMul/ReadVariableOp,sequential_16/dense_16/MatMul/ReadVariableOp2^
-sequential_16/dense_17/BiasAdd/ReadVariableOp-sequential_16/dense_17/BiasAdd/ReadVariableOp2\
,sequential_16/dense_17/MatMul/ReadVariableOp,sequential_16/dense_17/MatMul/ReadVariableOp2`
.sequential_16/sequential_15/random_crop_5/cond.sequential_16/sequential_15/random_crop_5/cond:d `
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
-
_user_specified_namesequential_15_input
õ#
©
K__inference_sequential_15_layer_call_and_return_conditional_losses_22200765

inputs(
random_crop_5_cond_input_0:	
identity¢random_crop_5/condI
random_crop_5/ShapeShapeinputs*
T0*
_output_shapes
:t
!random_crop_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿv
#random_crop_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿm
#random_crop_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
random_crop_5/strided_sliceStridedSlicerandom_crop_5/Shape:output:0*random_crop_5/strided_slice/stack:output:0,random_crop_5/strided_slice/stack_1:output:0,random_crop_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
random_crop_5/sub/yConst*
_output_shapes
: *
dtype0*
value	B : }
random_crop_5/subSub$random_crop_5/strided_slice:output:0random_crop_5/sub/y:output:0*
T0*
_output_shapes
: v
#random_crop_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿx
%random_crop_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿo
%random_crop_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
random_crop_5/strided_slice_1StridedSlicerandom_crop_5/Shape:output:0,random_crop_5/strided_slice_1/stack:output:0.random_crop_5/strided_slice_1/stack_1:output:0.random_crop_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_crop_5/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B : 
random_crop_5/sub_1Sub&random_crop_5/strided_slice_1:output:0random_crop_5/sub_1/y:output:0*
T0*
_output_shapes
: ^
random_crop_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 
random_crop_5/GreaterEqualGreaterEqualrandom_crop_5/sub:z:0%random_crop_5/GreaterEqual/y:output:0*
T0*
_output_shapes
: `
random_crop_5/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : 
random_crop_5/GreaterEqual_1GreaterEqualrandom_crop_5/sub_1:z:0'random_crop_5/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ]
random_crop_5/Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z ·
random_crop_5/Rank/packedPack$random_crop_5/Rank/packed/0:output:0random_crop_5/GreaterEqual:z:0 random_crop_5/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:T
random_crop_5/RankConst*
_output_shapes
: *
dtype0*
value	B :[
random_crop_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : [
random_crop_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
random_crop_5/rangeRange"random_crop_5/range/start:output:0random_crop_5/Rank:output:0"random_crop_5/range/delta:output:0*
_output_shapes
:[
random_crop_5/All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z ³
random_crop_5/All/inputPack"random_crop_5/All/input/0:output:0random_crop_5/GreaterEqual:z:0 random_crop_5/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:p
random_crop_5/AllAll random_crop_5/All/input:output:0random_crop_5/range:output:0*
_output_shapes
: Ä
random_crop_5/condIfrandom_crop_5/All:output:0random_crop_5_cond_input_0random_crop_5/sub:z:0random_crop_5/sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *4
else_branch%R#
!random_crop_5_cond_false_22200629*.
output_shapes
:ÿÿÿÿÿÿÿÿÿ  *3
then_branch$R"
 random_crop_5_cond_true_22200628~
random_crop_5/cond/IdentityIdentityrandom_crop_5/cond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  {
IdentityIdentity$random_crop_5/cond/Identity:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  [
NoOpNoOp^random_crop_5/cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  : 2(
random_crop_5/condrandom_crop_5/cond:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
¸
Æ
0__inference_sequential_16_layer_call_fn_22199964
sequential_15_input
unknown:	
	unknown_0:	#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:	x
	unknown_6:x
	unknown_7:xT
	unknown_8:T
	unknown_9:T


unknown_10:

identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallsequential_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_16_layer_call_and_return_conditional_losses_22199908o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
-
_user_specified_namesequential_15_input


G__inference_conv2d_10_layer_call_and_return_conditional_losses_22199677

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ë2
Ù
!random_crop_5_cond_false_22200795"
random_crop_5_cond_placeholder$
 random_crop_5_cond_placeholder_1$
 random_crop_5_cond_placeholder_2#
random_crop_5_cond_shape_inputs
random_crop_5_cond_identityg
random_crop_5/cond/ShapeShaperandom_crop_5_cond_shape_inputs*
T0*
_output_shapes
:y
&random_crop_5/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
(random_crop_5/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿr
(random_crop_5/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 random_crop_5/cond/strided_sliceStridedSlice!random_crop_5/cond/Shape:output:0/random_crop_5/cond/strided_slice/stack:output:01random_crop_5/cond/strided_slice/stack_1:output:01random_crop_5/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
(random_crop_5/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ}
*random_crop_5/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿt
*random_crop_5/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"random_crop_5/cond/strided_slice_1StridedSlice!random_crop_5/cond/Shape:output:01random_crop_5/cond/strided_slice_1/stack:output:03random_crop_5/cond/strided_slice_1/stack_1:output:03random_crop_5/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
random_crop_5/cond/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 
random_crop_5/cond/mulMul+random_crop_5/cond/strided_slice_1:output:0!random_crop_5/cond/mul/y:output:0*
T0*
_output_shapes
: k
random_crop_5/cond/CastCastrandom_crop_5/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: a
random_crop_5/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B
random_crop_5/cond/truedivRealDivrandom_crop_5/cond/Cast:y:0%random_crop_5/cond/truediv/y:output:0*
T0*
_output_shapes
: q
random_crop_5/cond/Cast_1Castrandom_crop_5/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: \
random_crop_5/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B : 
random_crop_5/cond/mul_1Mul)random_crop_5/cond/strided_slice:output:0#random_crop_5/cond/mul_1/y:output:0*
T0*
_output_shapes
: o
random_crop_5/cond/Cast_2Castrandom_crop_5/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: c
random_crop_5/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B
random_crop_5/cond/truediv_1RealDivrandom_crop_5/cond/Cast_2:y:0'random_crop_5/cond/truediv_1/y:output:0*
T0*
_output_shapes
: s
random_crop_5/cond/Cast_3Cast random_crop_5/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 
random_crop_5/cond/MinimumMinimum)random_crop_5/cond/strided_slice:output:0random_crop_5/cond/Cast_1:y:0*
T0*
_output_shapes
: 
random_crop_5/cond/Minimum_1Minimum+random_crop_5/cond/strided_slice_1:output:0random_crop_5/cond/Cast_3:y:0*
T0*
_output_shapes
: 
random_crop_5/cond/subSub)random_crop_5/cond/strided_slice:output:0random_crop_5/cond/Minimum:z:0*
T0*
_output_shapes
: m
random_crop_5/cond/Cast_4Castrandom_crop_5/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: c
random_crop_5/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
random_crop_5/cond/truediv_2RealDivrandom_crop_5/cond/Cast_4:y:0'random_crop_5/cond/truediv_2/y:output:0*
T0*
_output_shapes
: s
random_crop_5/cond/Cast_5Cast random_crop_5/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: 
random_crop_5/cond/sub_1Sub+random_crop_5/cond/strided_slice_1:output:0 random_crop_5/cond/Minimum_1:z:0*
T0*
_output_shapes
: o
random_crop_5/cond/Cast_6Castrandom_crop_5/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: c
random_crop_5/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
random_crop_5/cond/truediv_3RealDivrandom_crop_5/cond/Cast_6:y:0'random_crop_5/cond/truediv_3/y:output:0*
T0*
_output_shapes
: s
random_crop_5/cond/Cast_7Cast random_crop_5/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: \
random_crop_5/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : \
random_crop_5/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : Ö
random_crop_5/cond/stackPack#random_crop_5/cond/stack/0:output:0random_crop_5/cond/Cast_5:y:0random_crop_5/cond/Cast_7:y:0#random_crop_5/cond/stack/3:output:0*
N*
T0*
_output_shapes
:g
random_crop_5/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿg
random_crop_5/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿà
random_crop_5/cond/stack_1Pack%random_crop_5/cond/stack_1/0:output:0random_crop_5/cond/Minimum:z:0 random_crop_5/cond/Minimum_1:z:0%random_crop_5/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:Ú
random_crop_5/cond/SliceSlicerandom_crop_5_cond_shape_inputs!random_crop_5/cond/stack:output:0#random_crop_5/cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ  ÿÿÿÿÿÿÿÿÿo
random_crop_5/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        Ú
(random_crop_5/cond/resize/ResizeBilinearResizeBilinear!random_crop_5/cond/Slice:output:0'random_crop_5/cond/resize/size:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
half_pixel_centers(
random_crop_5/cond/IdentityIdentity9random_crop_5/cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "C
random_crop_5_cond_identity$random_crop_5/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :ÿÿÿÿÿÿÿÿÿ  :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
£
n
R__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_22201015

inputs
identity«
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸

K__inference_sequential_15_layer_call_and_return_conditional_losses_22199598

inputs$
random_crop_5_22199591:	$
random_flip_5_22199594:	
identity¢%random_crop_5/StatefulPartitionedCall¢%random_flip_5/StatefulPartitionedCallõ
%random_crop_5/StatefulPartitionedCallStatefulPartitionedCallinputsrandom_crop_5_22199591*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_random_crop_5_layer_call_and_return_conditional_losses_22199574
%random_flip_5/StatefulPartitionedCallStatefulPartitionedCall.random_crop_5/StatefulPartitionedCall:output:0random_flip_5_22199594*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_random_flip_5_layer_call_and_return_conditional_losses_22199393
IdentityIdentity.random_flip_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
NoOpNoOp&^random_crop_5/StatefulPartitionedCall&^random_flip_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 2N
%random_crop_5/StatefulPartitionedCall%random_crop_5/StatefulPartitionedCall2N
%random_flip_5/StatefulPartitionedCall%random_flip_5/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
õz
±
K__inference_sequential_15_layer_call_and_return_conditional_losses_22200985

inputs(
random_crop_5_cond_input_0:	M
?random_flip_5_stateful_uniform_full_int_rngreadandskip_resource:	
identity¢random_crop_5/cond¢6random_flip_5/stateful_uniform_full_int/RngReadAndSkipI
random_crop_5/ShapeShapeinputs*
T0*
_output_shapes
:t
!random_crop_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿv
#random_crop_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿm
#random_crop_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
random_crop_5/strided_sliceStridedSlicerandom_crop_5/Shape:output:0*random_crop_5/strided_slice/stack:output:0,random_crop_5/strided_slice/stack_1:output:0,random_crop_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
random_crop_5/sub/yConst*
_output_shapes
: *
dtype0*
value	B : }
random_crop_5/subSub$random_crop_5/strided_slice:output:0random_crop_5/sub/y:output:0*
T0*
_output_shapes
: v
#random_crop_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿx
%random_crop_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿo
%random_crop_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
random_crop_5/strided_slice_1StridedSlicerandom_crop_5/Shape:output:0,random_crop_5/strided_slice_1/stack:output:0.random_crop_5/strided_slice_1/stack_1:output:0.random_crop_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_crop_5/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B : 
random_crop_5/sub_1Sub&random_crop_5/strided_slice_1:output:0random_crop_5/sub_1/y:output:0*
T0*
_output_shapes
: ^
random_crop_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 
random_crop_5/GreaterEqualGreaterEqualrandom_crop_5/sub:z:0%random_crop_5/GreaterEqual/y:output:0*
T0*
_output_shapes
: `
random_crop_5/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : 
random_crop_5/GreaterEqual_1GreaterEqualrandom_crop_5/sub_1:z:0'random_crop_5/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ]
random_crop_5/Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z·
random_crop_5/Rank/packedPack$random_crop_5/Rank/packed/0:output:0random_crop_5/GreaterEqual:z:0 random_crop_5/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:T
random_crop_5/RankConst*
_output_shapes
: *
dtype0*
value	B :[
random_crop_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : [
random_crop_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
random_crop_5/rangeRange"random_crop_5/range/start:output:0random_crop_5/Rank:output:0"random_crop_5/range/delta:output:0*
_output_shapes
:[
random_crop_5/All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z³
random_crop_5/All/inputPack"random_crop_5/All/input/0:output:0random_crop_5/GreaterEqual:z:0 random_crop_5/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:p
random_crop_5/AllAll random_crop_5/All/input:output:0random_crop_5/range:output:0*
_output_shapes
: Ä
random_crop_5/condIfrandom_crop_5/All:output:0random_crop_5_cond_input_0random_crop_5/sub:z:0random_crop_5/sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *4
else_branch%R#
!random_crop_5_cond_false_22200795*.
output_shapes
:ÿÿÿÿÿÿÿÿÿ  *3
then_branch$R"
 random_crop_5_cond_true_22200794~
random_crop_5/cond/IdentityIdentityrandom_crop_5/cond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  w
-random_flip_5/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:w
-random_flip_5/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: Å
,random_flip_5/stateful_uniform_full_int/ProdProd6random_flip_5/stateful_uniform_full_int/shape:output:06random_flip_5/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: p
.random_flip_5/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
.random_flip_5/stateful_uniform_full_int/Cast_1Cast5random_flip_5/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
6random_flip_5/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip?random_flip_5_stateful_uniform_full_int_rngreadandskip_resource7random_flip_5/stateful_uniform_full_int/Cast/x:output:02random_flip_5/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
;random_flip_5/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=random_flip_5/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=random_flip_5/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5random_flip_5/stateful_uniform_full_int/strided_sliceStridedSlice>random_flip_5/stateful_uniform_full_int/RngReadAndSkip:value:0Drandom_flip_5/stateful_uniform_full_int/strided_slice/stack:output:0Frandom_flip_5/stateful_uniform_full_int/strided_slice/stack_1:output:0Frandom_flip_5/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask«
/random_flip_5/stateful_uniform_full_int/BitcastBitcast>random_flip_5/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
=random_flip_5/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?random_flip_5/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?random_flip_5/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
7random_flip_5/stateful_uniform_full_int/strided_slice_1StridedSlice>random_flip_5/stateful_uniform_full_int/RngReadAndSkip:value:0Frandom_flip_5/stateful_uniform_full_int/strided_slice_1/stack:output:0Hrandom_flip_5/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Hrandom_flip_5/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:¯
1random_flip_5/stateful_uniform_full_int/Bitcast_1Bitcast@random_flip_5/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0m
+random_flip_5/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :×
'random_flip_5/stateful_uniform_full_intStatelessRandomUniformFullIntV26random_flip_5/stateful_uniform_full_int/shape:output:0:random_flip_5/stateful_uniform_full_int/Bitcast_1:output:08random_flip_5/stateful_uniform_full_int/Bitcast:output:04random_flip_5/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	b
random_flip_5/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R ¢
random_flip_5/stackPack0random_flip_5/stateful_uniform_full_int:output:0!random_flip_5/zeros_like:output:0*
N*
T0	*
_output_shapes

:r
!random_flip_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        t
#random_flip_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#random_flip_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ½
random_flip_5/strided_sliceStridedSlicerandom_flip_5/stack:output:0*random_flip_5/strided_slice/stack:output:0,random_flip_5/strided_slice/stack_1:output:0,random_flip_5/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskÝ
Arandom_flip_5/stateless_random_flip_left_right/control_dependencyIdentity$random_crop_5/cond/Identity:output:0*
T0*.
_class$
" loc:@random_crop_5/cond/Identity*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ®
4random_flip_5/stateless_random_flip_left_right/ShapeShapeJrandom_flip_5/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:
Brandom_flip_5/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Drandom_flip_5/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Drandom_flip_5/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¼
<random_flip_5/stateless_random_flip_left_right/strided_sliceStridedSlice=random_flip_5/stateless_random_flip_left_right/Shape:output:0Krandom_flip_5/stateless_random_flip_left_right/strided_slice/stack:output:0Mrandom_flip_5/stateless_random_flip_left_right/strided_slice/stack_1:output:0Mrandom_flip_5/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÊ
Mrandom_flip_5/stateless_random_flip_left_right/stateless_random_uniform/shapePackErandom_flip_5/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:
Krandom_flip_5/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
Krandom_flip_5/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ì
drandom_flip_5/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter$random_flip_5/strided_slice:output:0* 
_output_shapes
::¦
drandom_flip_5/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Ä
`random_flip_5/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Vrandom_flip_5/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0jrandom_flip_5/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0nrandom_flip_5/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0mrandom_flip_5/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Krandom_flip_5/stateless_random_flip_left_right/stateless_random_uniform/subSubTrandom_flip_5/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Trandom_flip_5/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ¼
Krandom_flip_5/stateless_random_flip_left_right/stateless_random_uniform/mulMulirandom_flip_5/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Orandom_flip_5/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
Grandom_flip_5/stateless_random_flip_left_right/stateless_random_uniformAddV2Orandom_flip_5/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Trandom_flip_5/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>random_flip_5/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
>random_flip_5/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
>random_flip_5/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
<random_flip_5/stateless_random_flip_left_right/Reshape/shapePackErandom_flip_5/stateless_random_flip_left_right/strided_slice:output:0Grandom_flip_5/stateless_random_flip_left_right/Reshape/shape/1:output:0Grandom_flip_5/stateless_random_flip_left_right/Reshape/shape/2:output:0Grandom_flip_5/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
6random_flip_5/stateless_random_flip_left_right/ReshapeReshapeKrandom_flip_5/stateless_random_flip_left_right/stateless_random_uniform:z:0Erandom_flip_5/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
4random_flip_5/stateless_random_flip_left_right/RoundRound?random_flip_5/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=random_flip_5/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
8random_flip_5/stateless_random_flip_left_right/ReverseV2	ReverseV2Jrandom_flip_5/stateless_random_flip_left_right/control_dependency:output:0Frandom_flip_5/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ð
2random_flip_5/stateless_random_flip_left_right/mulMul8random_flip_5/stateless_random_flip_left_right/Round:y:0Arandom_flip_5/stateless_random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  y
4random_flip_5/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ì
2random_flip_5/stateless_random_flip_left_right/subSub=random_flip_5/stateless_random_flip_left_right/sub/x:output:08random_flip_5/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿù
4random_flip_5/stateless_random_flip_left_right/mul_1Mul6random_flip_5/stateless_random_flip_left_right/sub:z:0Jrandom_flip_5/stateless_random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ç
2random_flip_5/stateless_random_flip_left_right/addAddV26random_flip_5/stateless_random_flip_left_right/mul:z:08random_flip_5/stateless_random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
IdentityIdentity6random_flip_5/stateless_random_flip_left_right/add:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
NoOpNoOp^random_crop_5/cond7^random_flip_5/stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 2(
random_crop_5/condrandom_crop_5/cond2p
6random_flip_5/stateful_uniform_full_int/RngReadAndSkip6random_flip_5/stateful_uniform_full_int/RngReadAndSkip:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ð
¡
,__inference_conv2d_11_layer_call_fn_22201024

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_22199695w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
c
G__inference_flatten_5_layer_call_and_return_conditional_losses_22201056

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
S
7__inference_average_pooling2d_11_layer_call_fn_22201040

inputs
identityà
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_22199653
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É

+__inference_dense_15_layer_call_fn_22201065

inputs
unknown:	x
	unknown_0:x
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_22199721o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿g

cond_true_22201158;
-cond_stateful_uniform_rngreadandskip_resource:	
cond_add_sub
cond_add_1_sub_1*
&cond_crop_to_bounding_box_shape_inputs
cond_identity¢'cond/crop_to_bounding_box/Assert/Assert¢)cond/crop_to_bounding_box/Assert_1/Assert¢)cond/crop_to_bounding_box/Assert_2/Assert¢)cond/crop_to_bounding_box/Assert_3/Assert¢$cond/stateful_uniform/RngReadAndSkipe
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
valueB :ÿÿÿÿe
cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
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
: Ê
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
valueB:Å
#cond/stateful_uniform/strided_sliceStridedSlice,cond/stateful_uniform/RngReadAndSkip:value:02cond/stateful_uniform/strided_slice/stack:output:04cond/stateful_uniform/strided_slice/stack_1:output:04cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
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
valueB:»
%cond/stateful_uniform/strided_slice_1StridedSlice,cond/stateful_uniform/RngReadAndSkip:value:04cond/stateful_uniform/strided_slice_1/stack:output:06cond/stateful_uniform/strided_slice_1/stack_1:output:06cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
cond/stateful_uniform/Bitcast_1Bitcast.cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0[
cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :Á
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
valueB:õ
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
valueB:ý
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
:
!cond/crop_to_bounding_box/unstackUnpack(cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numj
(cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 
&cond/crop_to_bounding_box/GreaterEqualGreaterEqualcond/mod_1:z:01cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: 
&cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.
.cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.¸
'cond/crop_to_bounding_box/Assert/AssertAssert*cond/crop_to_bounding_box/GreaterEqual:z:07cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 l
*cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : 
(cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcond/mod:z:03cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: 
(cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.
0cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.è
)cond/crop_to_bounding_box/Assert_1/AssertAssert,cond/crop_to_bounding_box/GreaterEqual_1:z:09cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0(^cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 a
cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B : 
cond/crop_to_bounding_box/addAddV2(cond/crop_to_bounding_box/add/x:output:0cond/mod_1:z:0*
T0*
_output_shapes
: g
%cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ¤
#cond/crop_to_bounding_box/LessEqual	LessEqual!cond/crop_to_bounding_box/add:z:0.cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: 
(cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.
0cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.å
)cond/crop_to_bounding_box/Assert_2/AssertAssert'cond/crop_to_bounding_box/LessEqual:z:09cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 c
!cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : 
cond/crop_to_bounding_box/add_1AddV2*cond/crop_to_bounding_box/add_1/x:output:0cond/mod:z:0*
T0*
_output_shapes
: i
'cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ª
%cond/crop_to_bounding_box/LessEqual_1	LessEqual#cond/crop_to_bounding_box/add_1:z:00cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: 
(cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.
0cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.ç
)cond/crop_to_bounding_box/Assert_3/AssertAssert)cond/crop_to_bounding_box/LessEqual_1:z:09cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ã
,cond/crop_to_bounding_box/control_dependencyIdentity&cond_crop_to_bounding_box_shape_inputs(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  c
!cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : c
!cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : Ë
cond/crop_to_bounding_box/stackPack*cond/crop_to_bounding_box/stack/0:output:0cond/mod:z:0cond/mod_1:z:0*cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:
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
valueB:Õ
'cond/crop_to_bounding_box/strided_sliceStridedSlice*cond/crop_to_bounding_box/Shape_1:output:06cond/crop_to_bounding_box/strided_slice/stack:output:08cond/crop_to_bounding_box/strided_slice/stack_1:output:08cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
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
valueB:Ý
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
value	B : 
!cond/crop_to_bounding_box/stack_1Pack0cond/crop_to_bounding_box/strided_slice:output:0,cond/crop_to_bounding_box/stack_1/1:output:0,cond/crop_to_bounding_box/stack_1/2:output:02cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:ü
cond/crop_to_bounding_box/SliceSlice5cond/crop_to_bounding_box/control_dependency:output:0(cond/crop_to_bounding_box/stack:output:0*cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
cond/IdentityIdentity(cond/crop_to_bounding_box/Slice:output:0
^cond/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   
	cond/NoOpNoOp(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert%^cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :ÿÿÿÿÿÿÿÿÿ  2R
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
:ÿÿÿÿÿÿÿÿÿ  
Æ
S
7__inference_average_pooling2d_10_layer_call_fn_22201010

inputs
identityà
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_22199641
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß

K__inference_sequential_15_layer_call_and_return_conditional_losses_22199632
random_crop_5_input$
random_crop_5_22199625:	$
random_flip_5_22199628:	
identity¢%random_crop_5/StatefulPartitionedCall¢%random_flip_5/StatefulPartitionedCall
%random_crop_5/StatefulPartitionedCallStatefulPartitionedCallrandom_crop_5_inputrandom_crop_5_22199625*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_random_crop_5_layer_call_and_return_conditional_losses_22199574
%random_flip_5/StatefulPartitionedCallStatefulPartitionedCall.random_crop_5/StatefulPartitionedCall:output:0random_flip_5_22199628*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_random_flip_5_layer_call_and_return_conditional_losses_22199393
IdentityIdentity.random_flip_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
NoOpNoOp&^random_crop_5/StatefulPartitionedCall&^random_flip_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 2N
%random_crop_5/StatefulPartitionedCall%random_crop_5/StatefulPartitionedCall2N
%random_flip_5/StatefulPartitionedCall%random_flip_5/StatefulPartitionedCall:d `
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
-
_user_specified_namerandom_crop_5_input
è


0__inference_sequential_16_layer_call_fn_22200063

inputs
unknown:	#
	unknown_0:
	unknown_1:#
	unknown_2:
	unknown_3:
	unknown_4:	x
	unknown_5:x
	unknown_6:xT
	unknown_7:T
	unknown_8:T

	unknown_9:

identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_16_layer_call_and_return_conditional_losses_22199761o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ä
Å
K__inference_sequential_15_layer_call_and_return_conditional_losses_22199622
random_crop_5_input$
random_crop_5_22199617:	
identity¢%random_crop_5/StatefulPartitionedCall
%random_crop_5/StatefulPartitionedCallStatefulPartitionedCallrandom_crop_5_inputrandom_crop_5_22199617*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_random_crop_5_layer_call_and_return_conditional_losses_22199311ô
random_flip_5/PartitionedCallPartitionedCall.random_crop_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_random_flip_5_layer_call_and_return_conditional_losses_22199319}
IdentityIdentity&random_flip_5/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  n
NoOpNoOp&^random_crop_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  : 2N
%random_crop_5/StatefulPartitionedCall%random_crop_5/StatefulPartitionedCall:d `
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
-
_user_specified_namerandom_crop_5_input
Á

0__inference_random_crop_5_layer_call_fn_22201122

inputs
unknown:	
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_random_crop_5_layer_call_and_return_conditional_losses_22199311w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ÏM
Ö
K__inference_random_flip_5_layer_call_and_return_conditional_losses_22199393

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Ú
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
valueB:Ù
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
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
valueB:Ï
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :
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
valueB"      ÷
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:~
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask®
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :þ
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0_stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿû
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
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
value	B :Î
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:å
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:é
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  Æ
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  k
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Â
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ½
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  q
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  : 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

¬
0__inference_sequential_16_layer_call_fn_22199786
sequential_15_input
unknown:	#
	unknown_0:
	unknown_1:#
	unknown_2:
	unknown_3:
	unknown_4:	x
	unknown_5:x
	unknown_6:xT
	unknown_7:T
	unknown_8:T

	unknown_9:

identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallsequential_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_16_layer_call_and_return_conditional_losses_22199761o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
-
_user_specified_namesequential_15_input
­'

cond_false_22199175
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
ýÿÿÿÿÿÿÿÿm
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿd
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
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
þÿÿÿÿÿÿÿÿo
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿf
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
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
value	B : 

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
ÿÿÿÿÿÿÿÿÿY
cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
cond/stack_1Packcond/stack_1/0:output:0cond/Minimum:z:0cond/Minimum_1:z:0cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:¢

cond/SliceSlicecond_shape_inputscond/stack:output:0cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ  ÿÿÿÿÿÿÿÿÿa
cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        °
cond/resize/ResizeBilinearResizeBilinearcond/Slice:output:0cond/resize/size:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
half_pixel_centers(
cond/IdentityIdentity+cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :ÿÿÿÿÿÿÿÿÿ  :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
è

0__inference_sequential_15_layer_call_fn_22199327
random_crop_5_input
unknown:	
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallrandom_crop_5_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_15_layer_call_and_return_conditional_losses_22199322w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
-
_user_specified_namerandom_crop_5_input
­'

cond_false_22201159
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
ýÿÿÿÿÿÿÿÿm
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿd
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
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
þÿÿÿÿÿÿÿÿo
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿf
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
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
value	B : 

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
ÿÿÿÿÿÿÿÿÿY
cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
cond/stack_1Packcond/stack_1/0:output:0cond/Minimum:z:0cond/Minimum_1:z:0cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:¢

cond/SliceSlicecond_shape_inputscond/stack:output:0cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ  ÿÿÿÿÿÿÿÿÿa
cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        °
cond/resize/ResizeBilinearResizeBilinearcond/Slice:output:0cond/resize/size:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
half_pixel_centers(
cond/IdentityIdentity+cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :ÿÿÿÿÿÿÿÿÿ  :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
Á

0__inference_random_flip_5_layer_call_fn_22201473

inputs
unknown:	
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_random_flip_5_layer_call_and_return_conditional_losses_22199393w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

g
K__inference_random_flip_5_layer_call_and_return_conditional_losses_22201477

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
£
n
R__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_22199641

inputs
identity«
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ü
.sequential_15_random_crop_5_cond_true_22200325W
Isequential_15_random_crop_5_cond_stateful_uniform_rngreadandskip_resource:	H
Dsequential_15_random_crop_5_cond_add_sequential_15_random_crop_5_subL
Hsequential_15_random_crop_5_cond_add_1_sequential_15_random_crop_5_sub_1F
Bsequential_15_random_crop_5_cond_crop_to_bounding_box_shape_inputs-
)sequential_15_random_crop_5_cond_identity¢Csequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/Assert¢Esequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert¢Esequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert¢Esequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert¢@sequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkip
7sequential_15/random_crop_5/cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:w
5sequential_15/random_crop_5/cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : {
5sequential_15/random_crop_5/cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :ÿÿÿÿ
7sequential_15/random_crop_5/cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ã
6sequential_15/random_crop_5/cond/stateful_uniform/ProdProd@sequential_15/random_crop_5/cond/stateful_uniform/shape:output:0@sequential_15/random_crop_5/cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: z
8sequential_15/random_crop_5/cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :±
8sequential_15/random_crop_5/cond/stateful_uniform/Cast_1Cast?sequential_15/random_crop_5/cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: º
@sequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkipRngReadAndSkipIsequential_15_random_crop_5_cond_stateful_uniform_rngreadandskip_resourceAsequential_15/random_crop_5/cond/stateful_uniform/Cast/x:output:0<sequential_15/random_crop_5/cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:
Esequential_15/random_crop_5/cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Gsequential_15/random_crop_5/cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Gsequential_15/random_crop_5/cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
?sequential_15/random_crop_5/cond/stateful_uniform/strided_sliceStridedSliceHsequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkip:value:0Nsequential_15/random_crop_5/cond/stateful_uniform/strided_slice/stack:output:0Psequential_15/random_crop_5/cond/stateful_uniform/strided_slice/stack_1:output:0Psequential_15/random_crop_5/cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask¿
9sequential_15/random_crop_5/cond/stateful_uniform/BitcastBitcastHsequential_15/random_crop_5/cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Gsequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Isequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Isequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
Asequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1StridedSliceHsequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkip:value:0Psequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1/stack:output:0Rsequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1/stack_1:output:0Rsequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ã
;sequential_15/random_crop_5/cond/stateful_uniform/Bitcast_1BitcastJsequential_15/random_crop_5/cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0w
5sequential_15/random_crop_5/cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :
1sequential_15/random_crop_5/cond/stateful_uniformStatelessRandomUniformIntV2@sequential_15/random_crop_5/cond/stateful_uniform/shape:output:0Dsequential_15/random_crop_5/cond/stateful_uniform/Bitcast_1:output:0Bsequential_15/random_crop_5/cond/stateful_uniform/Bitcast:output:0>sequential_15/random_crop_5/cond/stateful_uniform/alg:output:0>sequential_15/random_crop_5/cond/stateful_uniform/min:output:0>sequential_15/random_crop_5/cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0~
4sequential_15/random_crop_5/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6sequential_15/random_crop_5/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6sequential_15/random_crop_5/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
.sequential_15/random_crop_5/cond/strided_sliceStridedSlice:sequential_15/random_crop_5/cond/stateful_uniform:output:0=sequential_15/random_crop_5/cond/strided_slice/stack:output:0?sequential_15/random_crop_5/cond/strided_slice/stack_1:output:0?sequential_15/random_crop_5/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&sequential_15/random_crop_5/cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :Å
$sequential_15/random_crop_5/cond/addAddV2Dsequential_15_random_crop_5_cond_add_sequential_15_random_crop_5_sub/sequential_15/random_crop_5/cond/add/y:output:0*
T0*
_output_shapes
: ´
$sequential_15/random_crop_5/cond/modFloorMod7sequential_15/random_crop_5/cond/strided_slice:output:0(sequential_15/random_crop_5/cond/add:z:0*
T0*
_output_shapes
: 
6sequential_15/random_crop_5/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
8sequential_15/random_crop_5/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8sequential_15/random_crop_5/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0sequential_15/random_crop_5/cond/strided_slice_1StridedSlice:sequential_15/random_crop_5/cond/stateful_uniform:output:0?sequential_15/random_crop_5/cond/strided_slice_1/stack:output:0Asequential_15/random_crop_5/cond/strided_slice_1/stack_1:output:0Asequential_15/random_crop_5/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_15/random_crop_5/cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Í
&sequential_15/random_crop_5/cond/add_1AddV2Hsequential_15_random_crop_5_cond_add_1_sequential_15_random_crop_5_sub_11sequential_15/random_crop_5/cond/add_1/y:output:0*
T0*
_output_shapes
: º
&sequential_15/random_crop_5/cond/mod_1FloorMod9sequential_15/random_crop_5/cond/strided_slice_1:output:0*sequential_15/random_crop_5/cond/add_1:z:0*
T0*
_output_shapes
: ­
;sequential_15/random_crop_5/cond/crop_to_bounding_box/ShapeShapeBsequential_15_random_crop_5_cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:¿
=sequential_15/random_crop_5/cond/crop_to_bounding_box/unstackUnpackDsequential_15/random_crop_5/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
num
Dsequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : î
Bsequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqualGreaterEqual*sequential_15/random_crop_5/cond/mod_1:z:0Msequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: 
Bsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.¥
Jsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.
Csequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/AssertAssertFsequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual:z:0Ssequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 
Fsequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ð
Dsequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqual(sequential_15/random_crop_5/cond/mod:z:0Osequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
:  
Dsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.¨
Lsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.Ø
Esequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/AssertAssertHsequential_15/random_crop_5/cond/crop_to_bounding_box/GreaterEqual_1:z:0Usequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0D^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 }
;sequential_15/random_crop_5/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B : Õ
9sequential_15/random_crop_5/cond/crop_to_bounding_box/addAddV2Dsequential_15/random_crop_5/cond/crop_to_bounding_box/add/x:output:0*sequential_15/random_crop_5/cond/mod_1:z:0*
T0*
_output_shapes
: 
Asequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ø
?sequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual	LessEqual=sequential_15/random_crop_5/cond/crop_to_bounding_box/add:z:0Jsequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ¦
Dsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.®
Lsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.Õ
Esequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/AssertAssertCsequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual:z:0Usequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0F^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 
=sequential_15/random_crop_5/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B : ×
;sequential_15/random_crop_5/cond/crop_to_bounding_box/add_1AddV2Fsequential_15/random_crop_5/cond/crop_to_bounding_box/add_1/x:output:0(sequential_15/random_crop_5/cond/mod:z:0*
T0*
_output_shapes
: 
Csequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : þ
Asequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual_1	LessEqual?sequential_15/random_crop_5/cond/crop_to_bounding_box/add_1:z:0Lsequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: §
Dsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.¯
Lsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.×
Esequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/AssertAssertEsequential_15/random_crop_5/cond/crop_to_bounding_box/LessEqual_1:z:0Usequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0F^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 
Hsequential_15/random_crop_5/cond/crop_to_bounding_box/control_dependencyIdentityBsequential_15_random_crop_5_cond_crop_to_bounding_box_shape_inputsD^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/AssertF^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/AssertF^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/AssertF^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
=sequential_15/random_crop_5/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : 
=sequential_15/random_crop_5/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ×
;sequential_15/random_crop_5/cond/crop_to_bounding_box/stackPackFsequential_15/random_crop_5/cond/crop_to_bounding_box/stack/0:output:0(sequential_15/random_crop_5/cond/mod:z:0*sequential_15/random_crop_5/cond/mod_1:z:0Fsequential_15/random_crop_5/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:¾
=sequential_15/random_crop_5/cond/crop_to_bounding_box/Shape_1ShapeQsequential_15/random_crop_5/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:
Isequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ksequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ksequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:á
Csequential_15/random_crop_5/cond/crop_to_bounding_box/strided_sliceStridedSliceFsequential_15/random_crop_5/cond/crop_to_bounding_box/Shape_1:output:0Rsequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice/stack:output:0Tsequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice/stack_1:output:0Tsequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¾
=sequential_15/random_crop_5/cond/crop_to_bounding_box/Shape_2ShapeQsequential_15/random_crop_5/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:
Ksequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Msequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Msequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
Esequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1StridedSliceFsequential_15/random_crop_5/cond/crop_to_bounding_box/Shape_2:output:0Tsequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack:output:0Vsequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0Vsequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?sequential_15/random_crop_5/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
?sequential_15/random_crop_5/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : ¥
=sequential_15/random_crop_5/cond/crop_to_bounding_box/stack_1PackLsequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice:output:0Hsequential_15/random_crop_5/cond/crop_to_bounding_box/stack_1/1:output:0Hsequential_15/random_crop_5/cond/crop_to_bounding_box/stack_1/2:output:0Nsequential_15/random_crop_5/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:ì
;sequential_15/random_crop_5/cond/crop_to_bounding_box/SliceSliceQsequential_15/random_crop_5/cond/crop_to_bounding_box/control_dependency:output:0Dsequential_15/random_crop_5/cond/crop_to_bounding_box/stack:output:0Fsequential_15/random_crop_5/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  Ý
)sequential_15/random_crop_5/cond/IdentityIdentityDsequential_15/random_crop_5/cond/crop_to_bounding_box/Slice:output:0&^sequential_15/random_crop_5/cond/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  È
%sequential_15/random_crop_5/cond/NoOpNoOpD^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/AssertF^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/AssertF^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/AssertF^sequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/AssertA^sequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "_
)sequential_15_random_crop_5_cond_identity2sequential_15/random_crop_5/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :ÿÿÿÿÿÿÿÿÿ  2
Csequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/AssertCsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert/Assert2
Esequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/AssertEsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_1/Assert2
Esequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/AssertEsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_2/Assert2
Esequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/AssertEsequential_15/random_crop_5/cond/crop_to_bounding_box/Assert_3/Assert2
@sequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkip@sequential_15/random_crop_5/cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
É
c
G__inference_flatten_5_layer_call_and_return_conditional_losses_22199708

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è

0__inference_sequential_15_layer_call_fn_22200599

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_15_layer_call_and_return_conditional_losses_22199598w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs


G__inference_conv2d_11_layer_call_and_return_conditional_losses_22201035

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ

X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÏM
Ö
K__inference_random_flip_5_layer_call_and_return_conditional_losses_22201535

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Ú
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
valueB:Ù
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
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
valueB:Ï
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :
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
valueB"      ÷
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:~
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask®
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :þ
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0_stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿû
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
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
value	B :Î
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:å
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:é
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  Æ
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  k
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Â
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ½
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  q
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  : 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
­'

cond_false_22199438
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
ýÿÿÿÿÿÿÿÿm
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿd
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
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
þÿÿÿÿÿÿÿÿo
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿf
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
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
value	B : 

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
ÿÿÿÿÿÿÿÿÿY
cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
cond/stack_1Packcond/stack_1/0:output:0cond/Minimum:z:0cond/Minimum_1:z:0cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:¢

cond/SliceSlicecond_shape_inputscond/stack:output:0cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ  ÿÿÿÿÿÿÿÿÿa
cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        °
cond/resize/ResizeBilinearResizeBilinearcond/Slice:output:0cond/resize/size:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
half_pixel_centers(
cond/IdentityIdentity+cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : : :ÿÿÿÿÿÿÿÿÿ  :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ë
serving_default·
[
sequential_15_inputD
%serving_default_sequential_15_input:0ÿÿÿÿÿÿÿÿÿ  <
dense_170
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:ÙÅ
Î
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
Ä
layer-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
»

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
»

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_layer
»

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
Ê
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
2
0__inference_sequential_16_layer_call_fn_22199786
0__inference_sequential_16_layer_call_fn_22200063
0__inference_sequential_16_layer_call_fn_22200092
0__inference_sequential_16_layer_call_fn_22199964À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
K__inference_sequential_16_layer_call_and_return_conditional_losses_22200296
K__inference_sequential_16_layer_call_and_return_conditional_losses_22200554
K__inference_sequential_16_layer_call_and_return_conditional_losses_22199999
K__inference_sequential_16_layer_call_and_return_conditional_losses_22200036À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÚB×
#__inference__wrapped_model_22199138sequential_15_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
Yserving_default"
signature_map
¼
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^_random_generator
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
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
­
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
2
0__inference_sequential_15_layer_call_fn_22199327
0__inference_sequential_15_layer_call_fn_22200590
0__inference_sequential_15_layer_call_fn_22200599
0__inference_sequential_15_layer_call_fn_22199614À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
K__inference_sequential_15_layer_call_and_return_conditional_losses_22200765
K__inference_sequential_15_layer_call_and_return_conditional_losses_22200985
K__inference_sequential_15_layer_call_and_return_conditional_losses_22199622
K__inference_sequential_15_layer_call_and_return_conditional_losses_22199632À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
*:(2conv2d_10/kernel
:2conv2d_10/bias
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
­
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
Ö2Ó
,__inference_conv2d_10_layer_call_fn_22200994¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_conv2d_10_layer_call_and_return_conditional_losses_22201005¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
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
á2Þ
7__inference_average_pooling2d_10_layer_call_fn_22201010¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ü2ù
R__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_22201015¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
*:(2conv2d_11/kernel
:2conv2d_11/bias
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
­
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
Ö2Ó
,__inference_conv2d_11_layer_call_fn_22201024¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_conv2d_11_layer_call_and_return_conditional_losses_22201035¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
®
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
á2Þ
7__inference_average_pooling2d_11_layer_call_fn_22201040¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ü2ù
R__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_22201045¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_flatten_5_layer_call_fn_22201050¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_flatten_5_layer_call_and_return_conditional_losses_22201056¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": 	x2dense_15/kernel
:x2dense_15/bias
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
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_15_layer_call_fn_22201065¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_15_layer_call_and_return_conditional_losses_22201076¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:xT2dense_16/kernel
:T2dense_16/bias
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
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_16_layer_call_fn_22201085¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_16_layer_call_and_return_conditional_losses_22201096¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:T
2dense_17/kernel
:
2dense_17/bias
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
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_17_layer_call_fn_22201105¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_17_layer_call_and_return_conditional_losses_22201115¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÙBÖ
&__inference_signature_wrapper_22200583sequential_15_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Z	variables
[trainable_variables
\regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
/

_generator"
_generic_user_object
2
0__inference_random_crop_5_layer_call_fn_22201122
0__inference_random_crop_5_layer_call_fn_22201129´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
K__inference_random_crop_5_layer_call_and_return_conditional_losses_22201295
K__inference_random_crop_5_layer_call_and_return_conditional_losses_22201461´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
a	variables
btrainable_variables
cregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
/
 
_generator"
_generic_user_object
2
0__inference_random_flip_5_layer_call_fn_22201466
0__inference_random_flip_5_layer_call_fn_22201473´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
K__inference_random_flip_5_layer_call_and_return_conditional_losses_22201477
K__inference_random_flip_5_layer_call_and_return_conditional_losses_22201535´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
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
¡
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
¢
_state_var"
_generic_user_object
':%	2random_crop_5/cond/StateVar
>:<	22sequential_16/sequential_15/random_flip_5/StateVar±
#__inference__wrapped_model_22199138¡()<=DELMD¢A
:¢7
52
sequential_15_inputÿÿÿÿÿÿÿÿÿ  
ª "3ª0
.
dense_17"
dense_17ÿÿÿÿÿÿÿÿÿ
õ
R__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_22201015R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Í
7__inference_average_pooling2d_10_layer_call_fn_22201010R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿõ
R__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_22201045R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Í
7__inference_average_pooling2d_11_layer_call_fn_22201040R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ·
G__inference_conv2d_10_layer_call_and_return_conditional_losses_22201005l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
,__inference_conv2d_10_layer_call_fn_22200994_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ·
G__inference_conv2d_11_layer_call_and_return_conditional_losses_22201035l()7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ


 
,__inference_conv2d_11_layer_call_fn_22201024_()7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ

§
F__inference_dense_15_layer_call_and_return_conditional_losses_22201076]<=0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿx
 
+__inference_dense_15_layer_call_fn_22201065P<=0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿx¦
F__inference_dense_16_layer_call_and_return_conditional_losses_22201096\DE/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿx
ª "%¢"

0ÿÿÿÿÿÿÿÿÿT
 ~
+__inference_dense_16_layer_call_fn_22201085ODE/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿx
ª "ÿÿÿÿÿÿÿÿÿT¦
F__inference_dense_17_layer_call_and_return_conditional_losses_22201115\LM/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿT
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ~
+__inference_dense_17_layer_call_fn_22201105OLM/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿT
ª "ÿÿÿÿÿÿÿÿÿ
¬
G__inference_flatten_5_layer_call_and_return_conditional_losses_22201056a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_flatten_5_layer_call_fn_22201050T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¿
K__inference_random_crop_5_layer_call_and_return_conditional_losses_22201295p¡;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ  
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 ¿
K__inference_random_crop_5_layer_call_and_return_conditional_losses_22201461p¡;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ  
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
0__inference_random_crop_5_layer_call_fn_22201122c¡;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ  
p 
ª " ÿÿÿÿÿÿÿÿÿ  
0__inference_random_crop_5_layer_call_fn_22201129c¡;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ  
p
ª " ÿÿÿÿÿÿÿÿÿ  »
K__inference_random_flip_5_layer_call_and_return_conditional_losses_22201477l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ  
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 ¿
K__inference_random_flip_5_layer_call_and_return_conditional_losses_22201535p¢;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ  
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
0__inference_random_flip_5_layer_call_fn_22201466_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ  
p 
ª " ÿÿÿÿÿÿÿÿÿ  
0__inference_random_flip_5_layer_call_fn_22201473c¢;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ  
p
ª " ÿÿÿÿÿÿÿÿÿ  Ñ
K__inference_sequential_15_layer_call_and_return_conditional_losses_22199622¡L¢I
B¢?
52
random_crop_5_inputÿÿÿÿÿÿÿÿÿ  
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 Ó
K__inference_sequential_15_layer_call_and_return_conditional_losses_22199632¡¢L¢I
B¢?
52
random_crop_5_inputÿÿÿÿÿÿÿÿÿ  
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 Ã
K__inference_sequential_15_layer_call_and_return_conditional_losses_22200765t¡?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 Å
K__inference_sequential_15_layer_call_and_return_conditional_losses_22200985v¡¢?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 ¨
0__inference_sequential_15_layer_call_fn_22199327t¡L¢I
B¢?
52
random_crop_5_inputÿÿÿÿÿÿÿÿÿ  
p 

 
ª " ÿÿÿÿÿÿÿÿÿ  ª
0__inference_sequential_15_layer_call_fn_22199614v¡¢L¢I
B¢?
52
random_crop_5_inputÿÿÿÿÿÿÿÿÿ  
p

 
ª " ÿÿÿÿÿÿÿÿÿ  
0__inference_sequential_15_layer_call_fn_22200590g¡?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª " ÿÿÿÿÿÿÿÿÿ  
0__inference_sequential_15_layer_call_fn_22200599i¡¢?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª " ÿÿÿÿÿÿÿÿÿ  Ó
K__inference_sequential_16_layer_call_and_return_conditional_losses_22199999¡()<=DELML¢I
B¢?
52
sequential_15_inputÿÿÿÿÿÿÿÿÿ  
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Õ
K__inference_sequential_16_layer_call_and_return_conditional_losses_22200036¡¢()<=DELML¢I
B¢?
52
sequential_15_inputÿÿÿÿÿÿÿÿÿ  
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Å
K__inference_sequential_16_layer_call_and_return_conditional_losses_22200296v¡()<=DELM?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ç
K__inference_sequential_16_layer_call_and_return_conditional_losses_22200554x¡¢()<=DELM?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ª
0__inference_sequential_16_layer_call_fn_22199786v¡()<=DELML¢I
B¢?
52
sequential_15_inputÿÿÿÿÿÿÿÿÿ  
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
¬
0__inference_sequential_16_layer_call_fn_22199964x¡¢()<=DELML¢I
B¢?
52
sequential_15_inputÿÿÿÿÿÿÿÿÿ  
p

 
ª "ÿÿÿÿÿÿÿÿÿ

0__inference_sequential_16_layer_call_fn_22200063i¡()<=DELM?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "ÿÿÿÿÿÿÿÿÿ

0__inference_sequential_16_layer_call_fn_22200092k¡¢()<=DELM?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "ÿÿÿÿÿÿÿÿÿ
Ë
&__inference_signature_wrapper_22200583 ¡()<=DELM[¢X
¢ 
QªN
L
sequential_15_input52
sequential_15_inputÿÿÿÿÿÿÿÿÿ  "3ª0
.
dense_17"
dense_17ÿÿÿÿÿÿÿÿÿ
