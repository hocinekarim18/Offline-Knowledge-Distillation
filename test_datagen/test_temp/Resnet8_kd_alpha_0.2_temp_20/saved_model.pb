×"
Î
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
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
ú
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Çü

conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_27/kernel
}
$conv2d_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_27/kernel*&
_output_shapes
:*
dtype0
t
conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_27/bias
m
"conv2d_27/bias/Read/ReadVariableOpReadVariableOpconv2d_27/bias*
_output_shapes
:*
dtype0

batch_normalization_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_21/gamma

0batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_21/gamma*
_output_shapes
:*
dtype0

batch_normalization_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_21/beta

/batch_normalization_21/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_21/beta*
_output_shapes
:*
dtype0

"batch_normalization_21/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_21/moving_mean

6batch_normalization_21/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_21/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_21/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_21/moving_variance

:batch_normalization_21/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_21/moving_variance*
_output_shapes
:*
dtype0

conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_28/kernel
}
$conv2d_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_28/kernel*&
_output_shapes
:*
dtype0
t
conv2d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_28/bias
m
"conv2d_28/bias/Read/ReadVariableOpReadVariableOpconv2d_28/bias*
_output_shapes
:*
dtype0

batch_normalization_22/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_22/gamma

0batch_normalization_22/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_22/gamma*
_output_shapes
:*
dtype0

batch_normalization_22/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_22/beta

/batch_normalization_22/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_22/beta*
_output_shapes
:*
dtype0

"batch_normalization_22/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_22/moving_mean

6batch_normalization_22/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_22/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_22/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_22/moving_variance

:batch_normalization_22/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_22/moving_variance*
_output_shapes
:*
dtype0

conv2d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_29/kernel
}
$conv2d_29/kernel/Read/ReadVariableOpReadVariableOpconv2d_29/kernel*&
_output_shapes
:*
dtype0
t
conv2d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_29/bias
m
"conv2d_29/bias/Read/ReadVariableOpReadVariableOpconv2d_29/bias*
_output_shapes
:*
dtype0

batch_normalization_23/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_23/gamma

0batch_normalization_23/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_23/gamma*
_output_shapes
:*
dtype0

batch_normalization_23/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_23/beta

/batch_normalization_23/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_23/beta*
_output_shapes
:*
dtype0

"batch_normalization_23/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_23/moving_mean

6batch_normalization_23/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_23/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_23/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_23/moving_variance

:batch_normalization_23/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_23/moving_variance*
_output_shapes
:*
dtype0

conv2d_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_30/kernel
}
$conv2d_30/kernel/Read/ReadVariableOpReadVariableOpconv2d_30/kernel*&
_output_shapes
: *
dtype0
t
conv2d_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_30/bias
m
"conv2d_30/bias/Read/ReadVariableOpReadVariableOpconv2d_30/bias*
_output_shapes
: *
dtype0

batch_normalization_24/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_24/gamma

0batch_normalization_24/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_24/gamma*
_output_shapes
: *
dtype0

batch_normalization_24/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_24/beta

/batch_normalization_24/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_24/beta*
_output_shapes
: *
dtype0

"batch_normalization_24/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_24/moving_mean

6batch_normalization_24/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_24/moving_mean*
_output_shapes
: *
dtype0
¤
&batch_normalization_24/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_24/moving_variance

:batch_normalization_24/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_24/moving_variance*
_output_shapes
: *
dtype0

conv2d_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_31/kernel
}
$conv2d_31/kernel/Read/ReadVariableOpReadVariableOpconv2d_31/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_31/bias
m
"conv2d_31/bias/Read/ReadVariableOpReadVariableOpconv2d_31/bias*
_output_shapes
: *
dtype0

conv2d_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_32/kernel
}
$conv2d_32/kernel/Read/ReadVariableOpReadVariableOpconv2d_32/kernel*&
_output_shapes
: *
dtype0
t
conv2d_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_32/bias
m
"conv2d_32/bias/Read/ReadVariableOpReadVariableOpconv2d_32/bias*
_output_shapes
: *
dtype0

batch_normalization_25/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_25/gamma

0batch_normalization_25/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_25/gamma*
_output_shapes
: *
dtype0

batch_normalization_25/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_25/beta

/batch_normalization_25/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_25/beta*
_output_shapes
: *
dtype0

"batch_normalization_25/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_25/moving_mean

6batch_normalization_25/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_25/moving_mean*
_output_shapes
: *
dtype0
¤
&batch_normalization_25/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_25/moving_variance

:batch_normalization_25/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_25/moving_variance*
_output_shapes
: *
dtype0

conv2d_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_33/kernel
}
$conv2d_33/kernel/Read/ReadVariableOpReadVariableOpconv2d_33/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_33/bias
m
"conv2d_33/bias/Read/ReadVariableOpReadVariableOpconv2d_33/bias*
_output_shapes
:@*
dtype0

batch_normalization_26/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_26/gamma

0batch_normalization_26/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_26/gamma*
_output_shapes
:@*
dtype0

batch_normalization_26/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_26/beta

/batch_normalization_26/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_26/beta*
_output_shapes
:@*
dtype0

"batch_normalization_26/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_26/moving_mean

6batch_normalization_26/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_26/moving_mean*
_output_shapes
:@*
dtype0
¤
&batch_normalization_26/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_26/moving_variance

:batch_normalization_26/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_26/moving_variance*
_output_shapes
:@*
dtype0

conv2d_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_34/kernel
}
$conv2d_34/kernel/Read/ReadVariableOpReadVariableOpconv2d_34/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_34/bias
m
"conv2d_34/bias/Read/ReadVariableOpReadVariableOpconv2d_34/bias*
_output_shapes
:@*
dtype0

conv2d_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_35/kernel
}
$conv2d_35/kernel/Read/ReadVariableOpReadVariableOpconv2d_35/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_35/bias
m
"conv2d_35/bias/Read/ReadVariableOpReadVariableOpconv2d_35/bias*
_output_shapes
:@*
dtype0

batch_normalization_27/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_27/gamma

0batch_normalization_27/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_27/gamma*
_output_shapes
:@*
dtype0

batch_normalization_27/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_27/beta

/batch_normalization_27/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_27/beta*
_output_shapes
:@*
dtype0

"batch_normalization_27/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_27/moving_mean

6batch_normalization_27/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_27/moving_mean*
_output_shapes
:@*
dtype0
¤
&batch_normalization_27/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_27/moving_variance

:batch_normalization_27/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_27/moving_variance*
_output_shapes
:@*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:@
*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
á¶
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¶
value¶B¶ B¶

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer-17
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer-21
layer_with_weights-13
layer-22
layer_with_weights-14
layer-23
layer_with_weights-15
layer-24
layer-25
layer-26
layer-27
layer-28
layer_with_weights-16
layer-29
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%_default_save_signature
&
signatures*
* 
¦

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
Õ
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses*

:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses* 
¦

@kernel
Abias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses*
Õ
Haxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses*

S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses* 
¦

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses*
Õ
aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses*

l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses* 

r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses* 
¦

xkernel
ybias
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses*
à
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses*
à
	¡axis

¢gamma
	£beta
¤moving_mean
¥moving_variance
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
ª__call__
+«&call_and_return_all_conditional_losses*

¬	variables
­trainable_variables
®regularization_losses
¯	keras_api
°__call__
+±&call_and_return_all_conditional_losses* 

²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses* 
®
¸kernel
	¹bias
º	variables
»trainable_variables
¼regularization_losses
½	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses*
à
	Àaxis

Ágamma
	Âbeta
Ãmoving_mean
Ämoving_variance
Å	variables
Ætrainable_variables
Çregularization_losses
È	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses*

Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses* 
®
Ñkernel
	Òbias
Ó	variables
Ôtrainable_variables
Õregularization_losses
Ö	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses*
®
Ùkernel
	Úbias
Û	variables
Ütrainable_variables
Ýregularization_losses
Þ	keras_api
ß__call__
+à&call_and_return_all_conditional_losses*
à
	áaxis

âgamma
	ãbeta
ämoving_mean
åmoving_variance
æ	variables
çtrainable_variables
èregularization_losses
é	keras_api
ê__call__
+ë&call_and_return_all_conditional_losses*

ì	variables
ítrainable_variables
îregularization_losses
ï	keras_api
ð__call__
+ñ&call_and_return_all_conditional_losses* 

ò	variables
ótrainable_variables
ôregularization_losses
õ	keras_api
ö__call__
+÷&call_and_return_all_conditional_losses* 

ø	variables
ùtrainable_variables
úregularization_losses
û	keras_api
ü__call__
+ý&call_and_return_all_conditional_losses* 

þ	variables
ÿtrainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

'0
(1
02
13
24
35
@6
A7
I8
J9
K10
L11
Y12
Z13
b14
c15
d16
e17
x18
y19
20
21
22
23
24
25
26
27
¢28
£29
¤30
¥31
¸32
¹33
Á34
Â35
Ã36
Ä37
Ñ38
Ò39
Ù40
Ú41
â42
ã43
ä44
å45
46
47*

'0
(1
02
13
@4
A5
I6
J7
Y8
Z9
b10
c11
x12
y13
14
15
16
17
18
19
¢20
£21
¸22
¹23
Á24
Â25
Ñ26
Ò27
Ù28
Ú29
â30
ã31
32
33*
J
0
1
2
3
4
5
6
7
8* 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
%_default_save_signature
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*
* 
* 
* 

serving_default* 
`Z
VARIABLE_VALUEconv2d_27/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_27/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

'0
(1*

'0
(1*


0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_21/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_21/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_21/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_21/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
00
11
22
33*

00
11*
* 

 non_trainable_variables
¡layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEconv2d_28/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_28/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

@0
A1*

@0
A1*


0* 

ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_22/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_22/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_22/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_22/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
I0
J1
K2
L3*

I0
J1*
* 

¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEconv2d_29/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_29/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

Y0
Z1*

Y0
Z1*


0* 

¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_23/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_23/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_23/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_23/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
b0
c1
d2
e3*

b0
c1*
* 

¾non_trainable_variables
¿layers
Àmetrics
 Álayer_regularization_losses
Âlayer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEconv2d_30/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_30/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

x0
y1*

x0
y1*


0* 

Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_24/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_24/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_24/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_24/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

0
1*
* 

Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEconv2d_31/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_31/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*


0* 

Ünon_trainable_variables
Ýlayers
Þmetrics
 ßlayer_regularization_losses
àlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
`Z
VARIABLE_VALUEconv2d_32/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_32/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*


0* 

ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_25/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_25/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_25/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_25/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
¢0
£1
¤2
¥3*

¢0
£1*
* 

ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
¦	variables
§trainable_variables
¨regularization_losses
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
¬	variables
­trainable_variables
®regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEconv2d_33/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_33/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

¸0
¹1*

¸0
¹1*


0* 

õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
º	variables
»trainable_variables
¼regularization_losses
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_26/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_26/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_26/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_26/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
Á0
Â1
Ã2
Ä3*

Á0
Â1*
* 

únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
Å	variables
Ætrainable_variables
Çregularization_losses
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ë	variables
Ìtrainable_variables
Íregularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEconv2d_34/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_34/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*

Ñ0
Ò1*

Ñ0
Ò1*


0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ó	variables
Ôtrainable_variables
Õregularization_losses
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses*
* 
* 
a[
VARIABLE_VALUEconv2d_35/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_35/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*

Ù0
Ú1*

Ù0
Ú1*


0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Û	variables
Ütrainable_variables
Ýregularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_27/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_27/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_27/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_27/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
â0
ã1
ä2
å3*

â0
ã1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
æ	variables
çtrainable_variables
èregularization_losses
ê__call__
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ì	variables
ítrainable_variables
îregularization_losses
ð__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ò	variables
ótrainable_variables
ôregularization_losses
ö__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
ø	variables
ùtrainable_variables
úregularization_losses
ü__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
þ	variables
ÿtrainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_3/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_3/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
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
r
20
31
K2
L3
d4
e5
6
7
¤8
¥9
Ã10
Ä11
ä12
å13*
ê
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29*
* 
* 
* 
* 
* 
* 
* 


0* 
* 

20
31*
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


0* 
* 

K0
L1*
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


0* 
* 

d0
e1*
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


0* 
* 

0
1*
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


0* 
* 
* 
* 
* 


0* 
* 

¤0
¥1*
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


0* 
* 

Ã0
Ä1*
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


0* 
* 
* 
* 
* 


0* 
* 

ä0
å1*
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

serving_default_input_4Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ  

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4conv2d_27/kernelconv2d_27/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_varianceconv2d_28/kernelconv2d_28/biasbatch_normalization_22/gammabatch_normalization_22/beta"batch_normalization_22/moving_mean&batch_normalization_22/moving_varianceconv2d_29/kernelconv2d_29/biasbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_varianceconv2d_30/kernelconv2d_30/biasbatch_normalization_24/gammabatch_normalization_24/beta"batch_normalization_24/moving_mean&batch_normalization_24/moving_varianceconv2d_31/kernelconv2d_31/biasconv2d_32/kernelconv2d_32/biasbatch_normalization_25/gammabatch_normalization_25/beta"batch_normalization_25/moving_mean&batch_normalization_25/moving_varianceconv2d_33/kernelconv2d_33/biasbatch_normalization_26/gammabatch_normalization_26/beta"batch_normalization_26/moving_mean&batch_normalization_26/moving_varianceconv2d_34/kernelconv2d_34/biasconv2d_35/kernelconv2d_35/biasbatch_normalization_27/gammabatch_normalization_27/beta"batch_normalization_27/moving_mean&batch_normalization_27/moving_variancedense_3/kerneldense_3/bias*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_13429529
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_27/kernel/Read/ReadVariableOp"conv2d_27/bias/Read/ReadVariableOp0batch_normalization_21/gamma/Read/ReadVariableOp/batch_normalization_21/beta/Read/ReadVariableOp6batch_normalization_21/moving_mean/Read/ReadVariableOp:batch_normalization_21/moving_variance/Read/ReadVariableOp$conv2d_28/kernel/Read/ReadVariableOp"conv2d_28/bias/Read/ReadVariableOp0batch_normalization_22/gamma/Read/ReadVariableOp/batch_normalization_22/beta/Read/ReadVariableOp6batch_normalization_22/moving_mean/Read/ReadVariableOp:batch_normalization_22/moving_variance/Read/ReadVariableOp$conv2d_29/kernel/Read/ReadVariableOp"conv2d_29/bias/Read/ReadVariableOp0batch_normalization_23/gamma/Read/ReadVariableOp/batch_normalization_23/beta/Read/ReadVariableOp6batch_normalization_23/moving_mean/Read/ReadVariableOp:batch_normalization_23/moving_variance/Read/ReadVariableOp$conv2d_30/kernel/Read/ReadVariableOp"conv2d_30/bias/Read/ReadVariableOp0batch_normalization_24/gamma/Read/ReadVariableOp/batch_normalization_24/beta/Read/ReadVariableOp6batch_normalization_24/moving_mean/Read/ReadVariableOp:batch_normalization_24/moving_variance/Read/ReadVariableOp$conv2d_31/kernel/Read/ReadVariableOp"conv2d_31/bias/Read/ReadVariableOp$conv2d_32/kernel/Read/ReadVariableOp"conv2d_32/bias/Read/ReadVariableOp0batch_normalization_25/gamma/Read/ReadVariableOp/batch_normalization_25/beta/Read/ReadVariableOp6batch_normalization_25/moving_mean/Read/ReadVariableOp:batch_normalization_25/moving_variance/Read/ReadVariableOp$conv2d_33/kernel/Read/ReadVariableOp"conv2d_33/bias/Read/ReadVariableOp0batch_normalization_26/gamma/Read/ReadVariableOp/batch_normalization_26/beta/Read/ReadVariableOp6batch_normalization_26/moving_mean/Read/ReadVariableOp:batch_normalization_26/moving_variance/Read/ReadVariableOp$conv2d_34/kernel/Read/ReadVariableOp"conv2d_34/bias/Read/ReadVariableOp$conv2d_35/kernel/Read/ReadVariableOp"conv2d_35/bias/Read/ReadVariableOp0batch_normalization_27/gamma/Read/ReadVariableOp/batch_normalization_27/beta/Read/ReadVariableOp6batch_normalization_27/moving_mean/Read/ReadVariableOp:batch_normalization_27/moving_variance/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpConst*=
Tin6
422*
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
!__inference__traced_save_13430654
É
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_27/kernelconv2d_27/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_varianceconv2d_28/kernelconv2d_28/biasbatch_normalization_22/gammabatch_normalization_22/beta"batch_normalization_22/moving_mean&batch_normalization_22/moving_varianceconv2d_29/kernelconv2d_29/biasbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_varianceconv2d_30/kernelconv2d_30/biasbatch_normalization_24/gammabatch_normalization_24/beta"batch_normalization_24/moving_mean&batch_normalization_24/moving_varianceconv2d_31/kernelconv2d_31/biasconv2d_32/kernelconv2d_32/biasbatch_normalization_25/gammabatch_normalization_25/beta"batch_normalization_25/moving_mean&batch_normalization_25/moving_varianceconv2d_33/kernelconv2d_33/biasbatch_normalization_26/gammabatch_normalization_26/beta"batch_normalization_26/moving_mean&batch_normalization_26/moving_varianceconv2d_34/kernelconv2d_34/biasconv2d_35/kernelconv2d_35/biasbatch_normalization_27/gammabatch_normalization_27/beta"batch_normalization_27/moving_mean&batch_normalization_27/moving_variancedense_3/kerneldense_3/bias*<
Tin5
321*
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
$__inference__traced_restore_13430808á­
ï
g
K__inference_activation_21_layer_call_and_return_conditional_losses_13427214

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ý
Ã
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_13429828

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
¡
,__inference_conv2d_35_layer_call_fn_13430248

inputs!
unknown: @
	unknown_0:@
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_35_layer_call_and_return_conditional_losses_13427482w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
â
µ
G__inference_conv2d_33_layer_call_and_return_conditional_losses_13427422

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_33/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_33/kernel/Regularizer/SquareSquare:conv2d_33/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_33/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_33/kernel/Regularizer/SumSum'conv2d_33/kernel/Regularizer/Square:y:0+conv2d_33/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_33/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_33/kernel/Regularizer/mulMul+conv2d_33/kernel/Regularizer/mul/x:output:0)conv2d_33/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_33/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_33/kernel/Regularizer/Square/ReadVariableOp2conv2d_33/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
â
µ
G__inference_conv2d_33_layer_call_and_return_conditional_losses_13430130

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_33/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_33/kernel/Regularizer/SquareSquare:conv2d_33/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_33/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_33/kernel/Regularizer/SumSum'conv2d_33/kernel/Regularizer/Square:y:0+conv2d_33/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_33/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_33/kernel/Regularizer/mulMul+conv2d_33/kernel/Regularizer/mul/x:output:0)conv2d_33/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_33/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_33/kernel/Regularizer/Square/ReadVariableOp2conv2d_33/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ð
¡
,__inference_conv2d_30_layer_call_fn_13429865

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_30_layer_call_and_return_conditional_losses_13427316w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ý
Ã
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_13427148

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_13429604

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
L
0__inference_activation_24_layer_call_fn_13429948

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
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_24_layer_call_and_return_conditional_losses_13427336h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
§
.
E__inference_model_3_layer_call_and_return_conditional_losses_13429197

inputsB
(conv2d_27_conv2d_readvariableop_resource:7
)conv2d_27_biasadd_readvariableop_resource:<
.batch_normalization_21_readvariableop_resource:>
0batch_normalization_21_readvariableop_1_resource:M
?batch_normalization_21_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_28_conv2d_readvariableop_resource:7
)conv2d_28_biasadd_readvariableop_resource:<
.batch_normalization_22_readvariableop_resource:>
0batch_normalization_22_readvariableop_1_resource:M
?batch_normalization_22_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_29_conv2d_readvariableop_resource:7
)conv2d_29_biasadd_readvariableop_resource:<
.batch_normalization_23_readvariableop_resource:>
0batch_normalization_23_readvariableop_1_resource:M
?batch_normalization_23_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_30_conv2d_readvariableop_resource: 7
)conv2d_30_biasadd_readvariableop_resource: <
.batch_normalization_24_readvariableop_resource: >
0batch_normalization_24_readvariableop_1_resource: M
?batch_normalization_24_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_31_conv2d_readvariableop_resource:  7
)conv2d_31_biasadd_readvariableop_resource: B
(conv2d_32_conv2d_readvariableop_resource: 7
)conv2d_32_biasadd_readvariableop_resource: <
.batch_normalization_25_readvariableop_resource: >
0batch_normalization_25_readvariableop_1_resource: M
?batch_normalization_25_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_33_conv2d_readvariableop_resource: @7
)conv2d_33_biasadd_readvariableop_resource:@<
.batch_normalization_26_readvariableop_resource:@>
0batch_normalization_26_readvariableop_1_resource:@M
?batch_normalization_26_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:@B
(conv2d_34_conv2d_readvariableop_resource:@@7
)conv2d_34_biasadd_readvariableop_resource:@B
(conv2d_35_conv2d_readvariableop_resource: @7
)conv2d_35_biasadd_readvariableop_resource:@<
.batch_normalization_27_readvariableop_resource:@>
0batch_normalization_27_readvariableop_1_resource:@M
?batch_normalization_27_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:@8
&dense_3_matmul_readvariableop_resource:@
5
'dense_3_biasadd_readvariableop_resource:

identity¢6batch_normalization_21/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_21/ReadVariableOp¢'batch_normalization_21/ReadVariableOp_1¢6batch_normalization_22/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_22/ReadVariableOp¢'batch_normalization_22/ReadVariableOp_1¢6batch_normalization_23/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_23/ReadVariableOp¢'batch_normalization_23/ReadVariableOp_1¢6batch_normalization_24/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_24/ReadVariableOp¢'batch_normalization_24/ReadVariableOp_1¢6batch_normalization_25/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_25/ReadVariableOp¢'batch_normalization_25/ReadVariableOp_1¢6batch_normalization_26/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_26/ReadVariableOp¢'batch_normalization_26/ReadVariableOp_1¢6batch_normalization_27/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_27/ReadVariableOp¢'batch_normalization_27/ReadVariableOp_1¢ conv2d_27/BiasAdd/ReadVariableOp¢conv2d_27/Conv2D/ReadVariableOp¢2conv2d_27/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_28/BiasAdd/ReadVariableOp¢conv2d_28/Conv2D/ReadVariableOp¢2conv2d_28/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_29/BiasAdd/ReadVariableOp¢conv2d_29/Conv2D/ReadVariableOp¢2conv2d_29/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_30/BiasAdd/ReadVariableOp¢conv2d_30/Conv2D/ReadVariableOp¢2conv2d_30/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_31/BiasAdd/ReadVariableOp¢conv2d_31/Conv2D/ReadVariableOp¢2conv2d_31/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_32/BiasAdd/ReadVariableOp¢conv2d_32/Conv2D/ReadVariableOp¢2conv2d_32/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_33/BiasAdd/ReadVariableOp¢conv2d_33/Conv2D/ReadVariableOp¢2conv2d_33/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_34/BiasAdd/ReadVariableOp¢conv2d_34/Conv2D/ReadVariableOp¢2conv2d_34/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_35/BiasAdd/ReadVariableOp¢conv2d_35/Conv2D/ReadVariableOp¢2conv2d_35/kernel/Regularizer/Square/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0­
conv2d_27/Conv2DConv2Dinputs'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes
:*
dtype0²
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¶
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0½
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3conv2d_27/BiasAdd:output:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
is_training( 
activation_21/ReluRelu+batch_normalization_21/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ç
conv2d_28/Conv2DConv2D activation_21/Relu:activations:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
%batch_normalization_22/ReadVariableOpReadVariableOp.batch_normalization_22_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_22/ReadVariableOp_1ReadVariableOp0batch_normalization_22_readvariableop_1_resource*
_output_shapes
:*
dtype0²
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¶
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0½
'batch_normalization_22/FusedBatchNormV3FusedBatchNormV3conv2d_28/BiasAdd:output:0-batch_normalization_22/ReadVariableOp:value:0/batch_normalization_22/ReadVariableOp_1:value:0>batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
is_training( 
activation_22/ReluRelu+batch_normalization_22/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ç
conv2d_29/Conv2DConv2D activation_22/Relu:activations:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes
:*
dtype0²
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¶
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0½
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3conv2d_29/BiasAdd:output:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
is_training( 
	add_9/addAddV2 activation_21/Relu:activations:0+batch_normalization_23/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  c
activation_23/ReluReluadd_9/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ç
conv2d_30/Conv2DConv2D activation_23/Relu:activations:0'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%batch_normalization_24/ReadVariableOpReadVariableOp.batch_normalization_24_readvariableop_resource*
_output_shapes
: *
dtype0
'batch_normalization_24/ReadVariableOp_1ReadVariableOp0batch_normalization_24_readvariableop_1_resource*
_output_shapes
: *
dtype0²
6batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¶
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0½
'batch_normalization_24/FusedBatchNormV3FusedBatchNormV3conv2d_30/BiasAdd:output:0-batch_normalization_24/ReadVariableOp:value:0/batch_normalization_24/ReadVariableOp_1:value:0>batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 
activation_24/ReluRelu+batch_normalization_24/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Ç
conv2d_31/Conv2DConv2D activation_24/Relu:activations:0'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ç
conv2d_32/Conv2DConv2D activation_23/Relu:activations:0'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%batch_normalization_25/ReadVariableOpReadVariableOp.batch_normalization_25_readvariableop_resource*
_output_shapes
: *
dtype0
'batch_normalization_25/ReadVariableOp_1ReadVariableOp0batch_normalization_25_readvariableop_1_resource*
_output_shapes
: *
dtype0²
6batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¶
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0½
'batch_normalization_25/FusedBatchNormV3FusedBatchNormV3conv2d_31/BiasAdd:output:0-batch_normalization_25/ReadVariableOp:value:0/batch_normalization_25/ReadVariableOp_1:value:0>batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 

add_10/addAddV2conv2d_32/BiasAdd:output:0+batch_normalization_25/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
activation_25/ReluReluadd_10/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ç
conv2d_33/Conv2DConv2D activation_25/Relu:activations:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%batch_normalization_26/ReadVariableOpReadVariableOp.batch_normalization_26_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_26/ReadVariableOp_1ReadVariableOp0batch_normalization_26_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0½
'batch_normalization_26/FusedBatchNormV3FusedBatchNormV3conv2d_33/BiasAdd:output:0-batch_normalization_26/ReadVariableOp:value:0/batch_normalization_26/ReadVariableOp_1:value:0>batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 
activation_26/ReluRelu+batch_normalization_26/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ç
conv2d_34/Conv2DConv2D activation_26/Relu:activations:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ç
conv2d_35/Conv2DConv2D activation_25/Relu:activations:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0½
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3conv2d_34/BiasAdd:output:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 

add_11/addAddV2conv2d_35/BiasAdd:output:0+batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
activation_27/ReluReluadd_11/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
average_pooling2d_3/AvgPoolAvgPool activation_27/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
flatten_3/ReshapeReshape$average_pooling2d_3/AvgPool:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype0
dense_3/MatMulMatMulflatten_3/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
£
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_29/kernel/Regularizer/SquareSquare:conv2d_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_29/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_29/kernel/Regularizer/SumSum'conv2d_29/kernel/Regularizer/Square:y:0+conv2d_29/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_29/kernel/Regularizer/mulMul+conv2d_29/kernel/Regularizer/mul/x:output:0)conv2d_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_30/kernel/Regularizer/SquareSquare:conv2d_30/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_30/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_30/kernel/Regularizer/SumSum'conv2d_30/kernel/Regularizer/Square:y:0+conv2d_30/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_30/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_30/kernel/Regularizer/mulMul+conv2d_30/kernel/Regularizer/mul/x:output:0)conv2d_30/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0
#conv2d_31/kernel/Regularizer/SquareSquare:conv2d_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_31/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_31/kernel/Regularizer/SumSum'conv2d_31/kernel/Regularizer/Square:y:0+conv2d_31/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_31/kernel/Regularizer/mulMul+conv2d_31/kernel/Regularizer/mul/x:output:0)conv2d_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_32/kernel/Regularizer/SquareSquare:conv2d_32/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_32/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_32/kernel/Regularizer/SumSum'conv2d_32/kernel/Regularizer/Square:y:0+conv2d_32/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_32/kernel/Regularizer/mulMul+conv2d_32/kernel/Regularizer/mul/x:output:0)conv2d_32/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_33/kernel/Regularizer/SquareSquare:conv2d_33/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_33/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_33/kernel/Regularizer/SumSum'conv2d_33/kernel/Regularizer/Square:y:0+conv2d_33/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_33/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_33/kernel/Regularizer/mulMul+conv2d_33/kernel/Regularizer/mul/x:output:0)conv2d_33/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
#conv2d_34/kernel/Regularizer/SquareSquare:conv2d_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@{
"conv2d_34/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_34/kernel/Regularizer/SumSum'conv2d_34/kernel/Regularizer/Square:y:0+conv2d_34/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_34/kernel/Regularizer/mulMul+conv2d_34/kernel/Regularizer/mul/x:output:0)conv2d_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
»
NoOpNoOp7^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_17^batch_normalization_22/FusedBatchNormV3/ReadVariableOp9^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_22/ReadVariableOp(^batch_normalization_22/ReadVariableOp_17^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_17^batch_normalization_24/FusedBatchNormV3/ReadVariableOp9^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_24/ReadVariableOp(^batch_normalization_24/ReadVariableOp_17^batch_normalization_25/FusedBatchNormV3/ReadVariableOp9^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_25/ReadVariableOp(^batch_normalization_25/ReadVariableOp_17^batch_normalization_26/FusedBatchNormV3/ReadVariableOp9^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_26/ReadVariableOp(^batch_normalization_26/ReadVariableOp_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_1!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp3^conv2d_27/kernel/Regularizer/Square/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp3^conv2d_28/kernel/Regularizer/Square/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp3^conv2d_29/kernel/Regularizer/Square/ReadVariableOp!^conv2d_30/BiasAdd/ReadVariableOp ^conv2d_30/Conv2D/ReadVariableOp3^conv2d_30/kernel/Regularizer/Square/ReadVariableOp!^conv2d_31/BiasAdd/ReadVariableOp ^conv2d_31/Conv2D/ReadVariableOp3^conv2d_31/kernel/Regularizer/Square/ReadVariableOp!^conv2d_32/BiasAdd/ReadVariableOp ^conv2d_32/Conv2D/ReadVariableOp3^conv2d_32/kernel/Regularizer/Square/ReadVariableOp!^conv2d_33/BiasAdd/ReadVariableOp ^conv2d_33/Conv2D/ReadVariableOp3^conv2d_33/kernel/Regularizer/Square/ReadVariableOp!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp3^conv2d_34/kernel/Regularizer/Square/ReadVariableOp!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp3^conv2d_35/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12p
6batch_normalization_22/FusedBatchNormV3/ReadVariableOp6batch_normalization_22/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_18batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_22/ReadVariableOp%batch_normalization_22/ReadVariableOp2R
'batch_normalization_22/ReadVariableOp_1'batch_normalization_22/ReadVariableOp_12p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12p
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp6batch_normalization_24/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_18batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_24/ReadVariableOp%batch_normalization_24/ReadVariableOp2R
'batch_normalization_24/ReadVariableOp_1'batch_normalization_24/ReadVariableOp_12p
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp6batch_normalization_25/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_18batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_25/ReadVariableOp%batch_normalization_25/ReadVariableOp2R
'batch_normalization_25/ReadVariableOp_1'batch_normalization_25/ReadVariableOp_12p
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp6batch_normalization_26/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_18batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_26/ReadVariableOp%batch_normalization_26/ReadVariableOp2R
'batch_normalization_26/ReadVariableOp_1'batch_normalization_26/ReadVariableOp_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2h
2conv2d_29/kernel/Regularizer/Square/ReadVariableOp2conv2d_29/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_30/BiasAdd/ReadVariableOp conv2d_30/BiasAdd/ReadVariableOp2B
conv2d_30/Conv2D/ReadVariableOpconv2d_30/Conv2D/ReadVariableOp2h
2conv2d_30/kernel/Regularizer/Square/ReadVariableOp2conv2d_30/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_31/BiasAdd/ReadVariableOp conv2d_31/BiasAdd/ReadVariableOp2B
conv2d_31/Conv2D/ReadVariableOpconv2d_31/Conv2D/ReadVariableOp2h
2conv2d_31/kernel/Regularizer/Square/ReadVariableOp2conv2d_31/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_32/BiasAdd/ReadVariableOp conv2d_32/BiasAdd/ReadVariableOp2B
conv2d_32/Conv2D/ReadVariableOpconv2d_32/Conv2D/ReadVariableOp2h
2conv2d_32/kernel/Regularizer/Square/ReadVariableOp2conv2d_32/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_33/BiasAdd/ReadVariableOp conv2d_33/BiasAdd/ReadVariableOp2B
conv2d_33/Conv2D/ReadVariableOpconv2d_33/Conv2D/ReadVariableOp2h
2conv2d_33/kernel/Regularizer/Square/ReadVariableOp2conv2d_33/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp2h
2conv2d_34/kernel/Regularizer/Square/ReadVariableOp2conv2d_34/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
â
µ
G__inference_conv2d_31_layer_call_and_return_conditional_losses_13429984

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_31/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0
#conv2d_31/kernel/Regularizer/SquareSquare:conv2d_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_31/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_31/kernel/Regularizer/SumSum'conv2d_31/kernel/Regularizer/Square:y:0+conv2d_31/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_31/kernel/Regularizer/mulMul+conv2d_31/kernel/Regularizer/mul/x:output:0)conv2d_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_31/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_31/kernel/Regularizer/Square/ReadVariableOp2conv2d_31/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¢
m
Q__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_13427168

inputs
identity«
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
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
­
¦
*__inference_model_3_layer_call_fn_13428968

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23:  

unknown_24: $

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: $

unknown_31: @

unknown_32:@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:@$

unknown_37:@@

unknown_38:@$

unknown_39: @

unknown_40:@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@


unknown_46:

identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*D
_read_only_resource_inputs&
$"	
!"#$'()*+,/0*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_model_3_layer_call_and_return_conditional_losses_13428146o
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
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ï
g
K__inference_activation_25_layer_call_and_return_conditional_losses_13427404

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ë
L
0__inference_activation_27_layer_call_fn_13430343

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
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_27_layer_call_and_return_conditional_losses_13427510h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_21_layer_call_fn_13429586

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_13426764
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
Ã
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_13429622

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
µ
G__inference_conv2d_29_layer_call_and_return_conditional_losses_13427270

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_29/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_29/kernel/Regularizer/SquareSquare:conv2d_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_29/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_29/kernel/Regularizer/SumSum'conv2d_29/kernel/Regularizer/Square:y:0+conv2d_29/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_29/kernel/Regularizer/mulMul+conv2d_29/kernel/Regularizer/mul/x:output:0)conv2d_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_29/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_29/kernel/Regularizer/Square/ReadVariableOp2conv2d_29/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_22_layer_call_fn_13429689

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_13426828
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_dense_3_layer_call_and_return_conditional_losses_13430388

inputs0
matmul_readvariableop_resource:@
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
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
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ð
¡
,__inference_conv2d_28_layer_call_fn_13429647

inputs!
unknown:
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
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_28_layer_call_and_return_conditional_losses_13427232w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ý
Ã
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_13430077

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ý
Ã
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_13427084

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
â
µ
G__inference_conv2d_32_layer_call_and_return_conditional_losses_13430015

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_32/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_32/kernel/Regularizer/SquareSquare:conv2d_32/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_32/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_32/kernel/Regularizer/SumSum'conv2d_32/kernel/Regularizer/Square:y:0+conv2d_32/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_32/kernel/Regularizer/mulMul+conv2d_32/kernel/Regularizer/mul/x:output:0)conv2d_32/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_32/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_32/kernel/Regularizer/Square/ReadVariableOp2conv2d_32/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ð
¡
,__inference_conv2d_33_layer_call_fn_13430114

inputs!
unknown: @
	unknown_0:@
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_33_layer_call_and_return_conditional_losses_13427422w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
â
µ
G__inference_conv2d_34_layer_call_and_return_conditional_losses_13430233

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_34/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
#conv2d_34/kernel/Regularizer/SquareSquare:conv2d_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@{
"conv2d_34/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_34/kernel/Regularizer/SumSum'conv2d_34/kernel/Regularizer/Square:y:0+conv2d_34/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_34/kernel/Regularizer/mulMul+conv2d_34/kernel/Regularizer/mul/x:output:0)conv2d_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_34/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_34/kernel/Regularizer/Square/ReadVariableOp2conv2d_34/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ë
½
__inference_loss_fn_7_13430476U
;conv2d_34_kernel_regularizer_square_readvariableop_resource:@@
identity¢2conv2d_34/kernel/Regularizer/Square/ReadVariableOp¶
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_34_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype0
#conv2d_34/kernel/Regularizer/SquareSquare:conv2d_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@{
"conv2d_34/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_34/kernel/Regularizer/SumSum'conv2d_34/kernel/Regularizer/Square:y:0+conv2d_34/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_34/kernel/Regularizer/mulMul+conv2d_34/kernel/Regularizer/mul/x:output:0)conv2d_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_34/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_34/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_34/kernel/Regularizer/Square/ReadVariableOp2conv2d_34/kernel/Regularizer/Square/ReadVariableOp
ï
g
K__inference_activation_22_layer_call_and_return_conditional_losses_13427252

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ë
L
0__inference_activation_25_layer_call_fn_13430094

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
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_25_layer_call_and_return_conditional_losses_13427404h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
â
µ
G__inference_conv2d_28_layer_call_and_return_conditional_losses_13427232

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_28/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_28/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_13426797

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó×
¿2
E__inference_model_3_layer_call_and_return_conditional_losses_13429426

inputsB
(conv2d_27_conv2d_readvariableop_resource:7
)conv2d_27_biasadd_readvariableop_resource:<
.batch_normalization_21_readvariableop_resource:>
0batch_normalization_21_readvariableop_1_resource:M
?batch_normalization_21_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_28_conv2d_readvariableop_resource:7
)conv2d_28_biasadd_readvariableop_resource:<
.batch_normalization_22_readvariableop_resource:>
0batch_normalization_22_readvariableop_1_resource:M
?batch_normalization_22_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_29_conv2d_readvariableop_resource:7
)conv2d_29_biasadd_readvariableop_resource:<
.batch_normalization_23_readvariableop_resource:>
0batch_normalization_23_readvariableop_1_resource:M
?batch_normalization_23_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_30_conv2d_readvariableop_resource: 7
)conv2d_30_biasadd_readvariableop_resource: <
.batch_normalization_24_readvariableop_resource: >
0batch_normalization_24_readvariableop_1_resource: M
?batch_normalization_24_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_31_conv2d_readvariableop_resource:  7
)conv2d_31_biasadd_readvariableop_resource: B
(conv2d_32_conv2d_readvariableop_resource: 7
)conv2d_32_biasadd_readvariableop_resource: <
.batch_normalization_25_readvariableop_resource: >
0batch_normalization_25_readvariableop_1_resource: M
?batch_normalization_25_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_33_conv2d_readvariableop_resource: @7
)conv2d_33_biasadd_readvariableop_resource:@<
.batch_normalization_26_readvariableop_resource:@>
0batch_normalization_26_readvariableop_1_resource:@M
?batch_normalization_26_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:@B
(conv2d_34_conv2d_readvariableop_resource:@@7
)conv2d_34_biasadd_readvariableop_resource:@B
(conv2d_35_conv2d_readvariableop_resource: @7
)conv2d_35_biasadd_readvariableop_resource:@<
.batch_normalization_27_readvariableop_resource:@>
0batch_normalization_27_readvariableop_1_resource:@M
?batch_normalization_27_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:@8
&dense_3_matmul_readvariableop_resource:@
5
'dense_3_biasadd_readvariableop_resource:

identity¢%batch_normalization_21/AssignNewValue¢'batch_normalization_21/AssignNewValue_1¢6batch_normalization_21/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_21/ReadVariableOp¢'batch_normalization_21/ReadVariableOp_1¢%batch_normalization_22/AssignNewValue¢'batch_normalization_22/AssignNewValue_1¢6batch_normalization_22/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_22/ReadVariableOp¢'batch_normalization_22/ReadVariableOp_1¢%batch_normalization_23/AssignNewValue¢'batch_normalization_23/AssignNewValue_1¢6batch_normalization_23/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_23/ReadVariableOp¢'batch_normalization_23/ReadVariableOp_1¢%batch_normalization_24/AssignNewValue¢'batch_normalization_24/AssignNewValue_1¢6batch_normalization_24/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_24/ReadVariableOp¢'batch_normalization_24/ReadVariableOp_1¢%batch_normalization_25/AssignNewValue¢'batch_normalization_25/AssignNewValue_1¢6batch_normalization_25/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_25/ReadVariableOp¢'batch_normalization_25/ReadVariableOp_1¢%batch_normalization_26/AssignNewValue¢'batch_normalization_26/AssignNewValue_1¢6batch_normalization_26/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_26/ReadVariableOp¢'batch_normalization_26/ReadVariableOp_1¢%batch_normalization_27/AssignNewValue¢'batch_normalization_27/AssignNewValue_1¢6batch_normalization_27/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_27/ReadVariableOp¢'batch_normalization_27/ReadVariableOp_1¢ conv2d_27/BiasAdd/ReadVariableOp¢conv2d_27/Conv2D/ReadVariableOp¢2conv2d_27/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_28/BiasAdd/ReadVariableOp¢conv2d_28/Conv2D/ReadVariableOp¢2conv2d_28/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_29/BiasAdd/ReadVariableOp¢conv2d_29/Conv2D/ReadVariableOp¢2conv2d_29/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_30/BiasAdd/ReadVariableOp¢conv2d_30/Conv2D/ReadVariableOp¢2conv2d_30/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_31/BiasAdd/ReadVariableOp¢conv2d_31/Conv2D/ReadVariableOp¢2conv2d_31/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_32/BiasAdd/ReadVariableOp¢conv2d_32/Conv2D/ReadVariableOp¢2conv2d_32/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_33/BiasAdd/ReadVariableOp¢conv2d_33/Conv2D/ReadVariableOp¢2conv2d_33/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_34/BiasAdd/ReadVariableOp¢conv2d_34/Conv2D/ReadVariableOp¢2conv2d_34/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_35/BiasAdd/ReadVariableOp¢conv2d_35/Conv2D/ReadVariableOp¢2conv2d_35/kernel/Regularizer/Square/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0­
conv2d_27/Conv2DConv2Dinputs'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes
:*
dtype0²
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¶
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ë
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3conv2d_27/BiasAdd:output:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_21/AssignNewValueAssignVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource4batch_normalization_21/FusedBatchNormV3:batch_mean:07^batch_normalization_21/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_21/AssignNewValue_1AssignVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_21/FusedBatchNormV3:batch_variance:09^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
activation_21/ReluRelu+batch_normalization_21/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ç
conv2d_28/Conv2DConv2D activation_21/Relu:activations:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
%batch_normalization_22/ReadVariableOpReadVariableOp.batch_normalization_22_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_22/ReadVariableOp_1ReadVariableOp0batch_normalization_22_readvariableop_1_resource*
_output_shapes
:*
dtype0²
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¶
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ë
'batch_normalization_22/FusedBatchNormV3FusedBatchNormV3conv2d_28/BiasAdd:output:0-batch_normalization_22/ReadVariableOp:value:0/batch_normalization_22/ReadVariableOp_1:value:0>batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_22/AssignNewValueAssignVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource4batch_normalization_22/FusedBatchNormV3:batch_mean:07^batch_normalization_22/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_22/AssignNewValue_1AssignVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_22/FusedBatchNormV3:batch_variance:09^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
activation_22/ReluRelu+batch_normalization_22/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ç
conv2d_29/Conv2DConv2D activation_22/Relu:activations:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes
:*
dtype0²
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¶
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ë
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3conv2d_29/BiasAdd:output:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_23/AssignNewValueAssignVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource4batch_normalization_23/FusedBatchNormV3:batch_mean:07^batch_normalization_23/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_23/AssignNewValue_1AssignVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_23/FusedBatchNormV3:batch_variance:09^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
	add_9/addAddV2 activation_21/Relu:activations:0+batch_normalization_23/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  c
activation_23/ReluReluadd_9/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ç
conv2d_30/Conv2DConv2D activation_23/Relu:activations:0'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%batch_normalization_24/ReadVariableOpReadVariableOp.batch_normalization_24_readvariableop_resource*
_output_shapes
: *
dtype0
'batch_normalization_24/ReadVariableOp_1ReadVariableOp0batch_normalization_24_readvariableop_1_resource*
_output_shapes
: *
dtype0²
6batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¶
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ë
'batch_normalization_24/FusedBatchNormV3FusedBatchNormV3conv2d_30/BiasAdd:output:0-batch_normalization_24/ReadVariableOp:value:0/batch_normalization_24/ReadVariableOp_1:value:0>batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_24/AssignNewValueAssignVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource4batch_normalization_24/FusedBatchNormV3:batch_mean:07^batch_normalization_24/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_24/AssignNewValue_1AssignVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_24/FusedBatchNormV3:batch_variance:09^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
activation_24/ReluRelu+batch_normalization_24/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Ç
conv2d_31/Conv2DConv2D activation_24/Relu:activations:0'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ç
conv2d_32/Conv2DConv2D activation_23/Relu:activations:0'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%batch_normalization_25/ReadVariableOpReadVariableOp.batch_normalization_25_readvariableop_resource*
_output_shapes
: *
dtype0
'batch_normalization_25/ReadVariableOp_1ReadVariableOp0batch_normalization_25_readvariableop_1_resource*
_output_shapes
: *
dtype0²
6batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¶
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ë
'batch_normalization_25/FusedBatchNormV3FusedBatchNormV3conv2d_31/BiasAdd:output:0-batch_normalization_25/ReadVariableOp:value:0/batch_normalization_25/ReadVariableOp_1:value:0>batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_25/AssignNewValueAssignVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource4batch_normalization_25/FusedBatchNormV3:batch_mean:07^batch_normalization_25/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_25/AssignNewValue_1AssignVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_25/FusedBatchNormV3:batch_variance:09^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0

add_10/addAddV2conv2d_32/BiasAdd:output:0+batch_normalization_25/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
activation_25/ReluReluadd_10/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ç
conv2d_33/Conv2DConv2D activation_25/Relu:activations:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%batch_normalization_26/ReadVariableOpReadVariableOp.batch_normalization_26_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_26/ReadVariableOp_1ReadVariableOp0batch_normalization_26_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ë
'batch_normalization_26/FusedBatchNormV3FusedBatchNormV3conv2d_33/BiasAdd:output:0-batch_normalization_26/ReadVariableOp:value:0/batch_normalization_26/ReadVariableOp_1:value:0>batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_26/AssignNewValueAssignVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource4batch_normalization_26/FusedBatchNormV3:batch_mean:07^batch_normalization_26/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_26/AssignNewValue_1AssignVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_26/FusedBatchNormV3:batch_variance:09^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
activation_26/ReluRelu+batch_normalization_26/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ç
conv2d_34/Conv2DConv2D activation_26/Relu:activations:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ç
conv2d_35/Conv2DConv2D activation_25/Relu:activations:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ë
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3conv2d_34/BiasAdd:output:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_27/AssignNewValueAssignVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource4batch_normalization_27/FusedBatchNormV3:batch_mean:07^batch_normalization_27/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_27/AssignNewValue_1AssignVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_27/FusedBatchNormV3:batch_variance:09^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0

add_11/addAddV2conv2d_35/BiasAdd:output:0+batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
activation_27/ReluReluadd_11/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
average_pooling2d_3/AvgPoolAvgPool activation_27/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
flatten_3/ReshapeReshape$average_pooling2d_3/AvgPool:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype0
dense_3/MatMulMatMulflatten_3/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
£
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_29/kernel/Regularizer/SquareSquare:conv2d_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_29/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_29/kernel/Regularizer/SumSum'conv2d_29/kernel/Regularizer/Square:y:0+conv2d_29/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_29/kernel/Regularizer/mulMul+conv2d_29/kernel/Regularizer/mul/x:output:0)conv2d_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_30/kernel/Regularizer/SquareSquare:conv2d_30/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_30/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_30/kernel/Regularizer/SumSum'conv2d_30/kernel/Regularizer/Square:y:0+conv2d_30/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_30/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_30/kernel/Regularizer/mulMul+conv2d_30/kernel/Regularizer/mul/x:output:0)conv2d_30/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0
#conv2d_31/kernel/Regularizer/SquareSquare:conv2d_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_31/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_31/kernel/Regularizer/SumSum'conv2d_31/kernel/Regularizer/Square:y:0+conv2d_31/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_31/kernel/Regularizer/mulMul+conv2d_31/kernel/Regularizer/mul/x:output:0)conv2d_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_32/kernel/Regularizer/SquareSquare:conv2d_32/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_32/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_32/kernel/Regularizer/SumSum'conv2d_32/kernel/Regularizer/Square:y:0+conv2d_32/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_32/kernel/Regularizer/mulMul+conv2d_32/kernel/Regularizer/mul/x:output:0)conv2d_32/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_33/kernel/Regularizer/SquareSquare:conv2d_33/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_33/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_33/kernel/Regularizer/SumSum'conv2d_33/kernel/Regularizer/Square:y:0+conv2d_33/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_33/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_33/kernel/Regularizer/mulMul+conv2d_33/kernel/Regularizer/mul/x:output:0)conv2d_33/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
#conv2d_34/kernel/Regularizer/SquareSquare:conv2d_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@{
"conv2d_34/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_34/kernel/Regularizer/SumSum'conv2d_34/kernel/Regularizer/Square:y:0+conv2d_34/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_34/kernel/Regularizer/mulMul+conv2d_34/kernel/Regularizer/mul/x:output:0)conv2d_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ù
NoOpNoOp&^batch_normalization_21/AssignNewValue(^batch_normalization_21/AssignNewValue_17^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_1&^batch_normalization_22/AssignNewValue(^batch_normalization_22/AssignNewValue_17^batch_normalization_22/FusedBatchNormV3/ReadVariableOp9^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_22/ReadVariableOp(^batch_normalization_22/ReadVariableOp_1&^batch_normalization_23/AssignNewValue(^batch_normalization_23/AssignNewValue_17^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_1&^batch_normalization_24/AssignNewValue(^batch_normalization_24/AssignNewValue_17^batch_normalization_24/FusedBatchNormV3/ReadVariableOp9^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_24/ReadVariableOp(^batch_normalization_24/ReadVariableOp_1&^batch_normalization_25/AssignNewValue(^batch_normalization_25/AssignNewValue_17^batch_normalization_25/FusedBatchNormV3/ReadVariableOp9^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_25/ReadVariableOp(^batch_normalization_25/ReadVariableOp_1&^batch_normalization_26/AssignNewValue(^batch_normalization_26/AssignNewValue_17^batch_normalization_26/FusedBatchNormV3/ReadVariableOp9^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_26/ReadVariableOp(^batch_normalization_26/ReadVariableOp_1&^batch_normalization_27/AssignNewValue(^batch_normalization_27/AssignNewValue_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_1!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp3^conv2d_27/kernel/Regularizer/Square/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp3^conv2d_28/kernel/Regularizer/Square/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp3^conv2d_29/kernel/Regularizer/Square/ReadVariableOp!^conv2d_30/BiasAdd/ReadVariableOp ^conv2d_30/Conv2D/ReadVariableOp3^conv2d_30/kernel/Regularizer/Square/ReadVariableOp!^conv2d_31/BiasAdd/ReadVariableOp ^conv2d_31/Conv2D/ReadVariableOp3^conv2d_31/kernel/Regularizer/Square/ReadVariableOp!^conv2d_32/BiasAdd/ReadVariableOp ^conv2d_32/Conv2D/ReadVariableOp3^conv2d_32/kernel/Regularizer/Square/ReadVariableOp!^conv2d_33/BiasAdd/ReadVariableOp ^conv2d_33/Conv2D/ReadVariableOp3^conv2d_33/kernel/Regularizer/Square/ReadVariableOp!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp3^conv2d_34/kernel/Regularizer/Square/ReadVariableOp!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp3^conv2d_35/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_21/AssignNewValue%batch_normalization_21/AssignNewValue2R
'batch_normalization_21/AssignNewValue_1'batch_normalization_21/AssignNewValue_12p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12N
%batch_normalization_22/AssignNewValue%batch_normalization_22/AssignNewValue2R
'batch_normalization_22/AssignNewValue_1'batch_normalization_22/AssignNewValue_12p
6batch_normalization_22/FusedBatchNormV3/ReadVariableOp6batch_normalization_22/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_18batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_22/ReadVariableOp%batch_normalization_22/ReadVariableOp2R
'batch_normalization_22/ReadVariableOp_1'batch_normalization_22/ReadVariableOp_12N
%batch_normalization_23/AssignNewValue%batch_normalization_23/AssignNewValue2R
'batch_normalization_23/AssignNewValue_1'batch_normalization_23/AssignNewValue_12p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12N
%batch_normalization_24/AssignNewValue%batch_normalization_24/AssignNewValue2R
'batch_normalization_24/AssignNewValue_1'batch_normalization_24/AssignNewValue_12p
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp6batch_normalization_24/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_18batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_24/ReadVariableOp%batch_normalization_24/ReadVariableOp2R
'batch_normalization_24/ReadVariableOp_1'batch_normalization_24/ReadVariableOp_12N
%batch_normalization_25/AssignNewValue%batch_normalization_25/AssignNewValue2R
'batch_normalization_25/AssignNewValue_1'batch_normalization_25/AssignNewValue_12p
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp6batch_normalization_25/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_18batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_25/ReadVariableOp%batch_normalization_25/ReadVariableOp2R
'batch_normalization_25/ReadVariableOp_1'batch_normalization_25/ReadVariableOp_12N
%batch_normalization_26/AssignNewValue%batch_normalization_26/AssignNewValue2R
'batch_normalization_26/AssignNewValue_1'batch_normalization_26/AssignNewValue_12p
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp6batch_normalization_26/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_18batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_26/ReadVariableOp%batch_normalization_26/ReadVariableOp2R
'batch_normalization_26/ReadVariableOp_1'batch_normalization_26/ReadVariableOp_12N
%batch_normalization_27/AssignNewValue%batch_normalization_27/AssignNewValue2R
'batch_normalization_27/AssignNewValue_1'batch_normalization_27/AssignNewValue_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2h
2conv2d_29/kernel/Regularizer/Square/ReadVariableOp2conv2d_29/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_30/BiasAdd/ReadVariableOp conv2d_30/BiasAdd/ReadVariableOp2B
conv2d_30/Conv2D/ReadVariableOpconv2d_30/Conv2D/ReadVariableOp2h
2conv2d_30/kernel/Regularizer/Square/ReadVariableOp2conv2d_30/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_31/BiasAdd/ReadVariableOp conv2d_31/BiasAdd/ReadVariableOp2B
conv2d_31/Conv2D/ReadVariableOpconv2d_31/Conv2D/ReadVariableOp2h
2conv2d_31/kernel/Regularizer/Square/ReadVariableOp2conv2d_31/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_32/BiasAdd/ReadVariableOp conv2d_32/BiasAdd/ReadVariableOp2B
conv2d_32/Conv2D/ReadVariableOpconv2d_32/Conv2D/ReadVariableOp2h
2conv2d_32/kernel/Regularizer/Square/ReadVariableOp2conv2d_32/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_33/BiasAdd/ReadVariableOp conv2d_33/BiasAdd/ReadVariableOp2B
conv2d_33/Conv2D/ReadVariableOpconv2d_33/Conv2D/ReadVariableOp2h
2conv2d_33/kernel/Regularizer/Square/ReadVariableOp2conv2d_33/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp2h
2conv2d_34/kernel/Regularizer/Square/ReadVariableOp2conv2d_34/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_13429707

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é
n
D__inference_add_11_layer_call_and_return_conditional_losses_13427503

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
°
§
*__inference_model_3_layer_call_fn_13428346
input_4!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23:  

unknown_24: $

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: $

unknown_31: @

unknown_32:@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:@$

unknown_37:@@

unknown_38:@$

unknown_39: @

unknown_40:@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@


unknown_46:

identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*D
_read_only_resource_inputs&
$"	
!"#$'()*+,/0*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_model_3_layer_call_and_return_conditional_losses_13428146o
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
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_4
	
Ô
9__inference_batch_normalization_25_layer_call_fn_13430041

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_13427020
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ç
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_13427519

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_23_layer_call_fn_13429792

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_13426892
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
µ
G__inference_conv2d_28_layer_call_and_return_conditional_losses_13429663

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_28/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_28/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ð
¡
,__inference_conv2d_29_layer_call_fn_13429750

inputs!
unknown:
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
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_29_layer_call_and_return_conditional_losses_13427270w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_26_layer_call_fn_13430156

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_13427084
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ë
½
__inference_loss_fn_6_13430465U
;conv2d_33_kernel_regularizer_square_readvariableop_resource: @
identity¢2conv2d_33/kernel/Regularizer/Square/ReadVariableOp¶
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_33_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_33/kernel/Regularizer/SquareSquare:conv2d_33/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_33/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_33/kernel/Regularizer/SumSum'conv2d_33/kernel/Regularizer/Square:y:0+conv2d_33/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_33/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_33/kernel/Regularizer/mulMul+conv2d_33/kernel/Regularizer/mul/x:output:0)conv2d_33/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_33/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_33/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_33/kernel/Regularizer/Square/ReadVariableOp2conv2d_33/kernel/Regularizer/Square/ReadVariableOp
ï
g
K__inference_activation_27_layer_call_and_return_conditional_losses_13430348

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ë
½
__inference_loss_fn_4_13430443U
;conv2d_31_kernel_regularizer_square_readvariableop_resource:  
identity¢2conv2d_31/kernel/Regularizer/Square/ReadVariableOp¶
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_31_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype0
#conv2d_31/kernel/Regularizer/SquareSquare:conv2d_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_31/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_31/kernel/Regularizer/SumSum'conv2d_31/kernel/Regularizer/Square:y:0+conv2d_31/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_31/kernel/Regularizer/mulMul+conv2d_31/kernel/Regularizer/mul/x:output:0)conv2d_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_31/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_31/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_31/kernel/Regularizer/Square/ReadVariableOp2conv2d_31/kernel/Regularizer/Square/ReadVariableOp
Ï

T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_13430174

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¾
§
*__inference_model_3_layer_call_fn_13427691
input_4!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23:  

unknown_24: $

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: $

unknown_31: @

unknown_32:@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:@$

unknown_37:@@

unknown_38:@$

unknown_39: @

unknown_40:@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@


unknown_46:

identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_model_3_layer_call_and_return_conditional_losses_13427592o
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
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_4
Ï

T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_13426733

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï
g
K__inference_activation_26_layer_call_and_return_conditional_losses_13427442

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ï
g
K__inference_activation_26_layer_call_and_return_conditional_losses_13430202

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
â
µ
G__inference_conv2d_34_layer_call_and_return_conditional_losses_13427460

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_34/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
#conv2d_34/kernel/Regularizer/SquareSquare:conv2d_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@{
"conv2d_34/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_34/kernel/Regularizer/SumSum'conv2d_34/kernel/Regularizer/Square:y:0+conv2d_34/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_34/kernel/Regularizer/mulMul+conv2d_34/kernel/Regularizer/mul/x:output:0)conv2d_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_34/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_34/kernel/Regularizer/Square/ReadVariableOp2conv2d_34/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ë
½
__inference_loss_fn_8_13430487U
;conv2d_35_kernel_regularizer_square_readvariableop_resource: @
identity¢2conv2d_35/kernel/Regularizer/Square/ReadVariableOp¶
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_35_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_35/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_35/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp
â
µ
G__inference_conv2d_30_layer_call_and_return_conditional_losses_13427316

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_30/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_30/kernel/Regularizer/SquareSquare:conv2d_30/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_30/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_30/kernel/Regularizer/SumSum'conv2d_30/kernel/Regularizer/Square:y:0+conv2d_30/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_30/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_30/kernel/Regularizer/mulMul+conv2d_30/kernel/Regularizer/mul/x:output:0)conv2d_30/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_30/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_30/kernel/Regularizer/Square/ReadVariableOp2conv2d_30/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ï
g
K__inference_activation_21_layer_call_and_return_conditional_losses_13429632

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ð
T
(__inference_add_9_layer_call_fn_13429834
inputs_0
inputs_1
identityÃ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_add_9_layer_call_and_return_conditional_losses_13427291h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ  :Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
inputs/1
ï
g
K__inference_activation_25_layer_call_and_return_conditional_losses_13430099

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¯Ø
·
E__inference_model_3_layer_call_and_return_conditional_losses_13428529
input_4,
conv2d_27_13428349: 
conv2d_27_13428351:-
batch_normalization_21_13428354:-
batch_normalization_21_13428356:-
batch_normalization_21_13428358:-
batch_normalization_21_13428360:,
conv2d_28_13428364: 
conv2d_28_13428366:-
batch_normalization_22_13428369:-
batch_normalization_22_13428371:-
batch_normalization_22_13428373:-
batch_normalization_22_13428375:,
conv2d_29_13428379: 
conv2d_29_13428381:-
batch_normalization_23_13428384:-
batch_normalization_23_13428386:-
batch_normalization_23_13428388:-
batch_normalization_23_13428390:,
conv2d_30_13428395:  
conv2d_30_13428397: -
batch_normalization_24_13428400: -
batch_normalization_24_13428402: -
batch_normalization_24_13428404: -
batch_normalization_24_13428406: ,
conv2d_31_13428410:   
conv2d_31_13428412: ,
conv2d_32_13428415:  
conv2d_32_13428417: -
batch_normalization_25_13428420: -
batch_normalization_25_13428422: -
batch_normalization_25_13428424: -
batch_normalization_25_13428426: ,
conv2d_33_13428431: @ 
conv2d_33_13428433:@-
batch_normalization_26_13428436:@-
batch_normalization_26_13428438:@-
batch_normalization_26_13428440:@-
batch_normalization_26_13428442:@,
conv2d_34_13428446:@@ 
conv2d_34_13428448:@,
conv2d_35_13428451: @ 
conv2d_35_13428453:@-
batch_normalization_27_13428456:@-
batch_normalization_27_13428458:@-
batch_normalization_27_13428460:@-
batch_normalization_27_13428462:@"
dense_3_13428469:@

dense_3_13428471:

identity¢.batch_normalization_21/StatefulPartitionedCall¢.batch_normalization_22/StatefulPartitionedCall¢.batch_normalization_23/StatefulPartitionedCall¢.batch_normalization_24/StatefulPartitionedCall¢.batch_normalization_25/StatefulPartitionedCall¢.batch_normalization_26/StatefulPartitionedCall¢.batch_normalization_27/StatefulPartitionedCall¢!conv2d_27/StatefulPartitionedCall¢2conv2d_27/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_28/StatefulPartitionedCall¢2conv2d_28/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_29/StatefulPartitionedCall¢2conv2d_29/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_30/StatefulPartitionedCall¢2conv2d_30/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_31/StatefulPartitionedCall¢2conv2d_31/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_32/StatefulPartitionedCall¢2conv2d_32/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_33/StatefulPartitionedCall¢2conv2d_33/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_34/StatefulPartitionedCall¢2conv2d_34/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_35/StatefulPartitionedCall¢2conv2d_35/kernel/Regularizer/Square/ReadVariableOp¢dense_3/StatefulPartitionedCall
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_27_13428349conv2d_27_13428351*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_27_layer_call_and_return_conditional_losses_13427194 
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0batch_normalization_21_13428354batch_normalization_21_13428356batch_normalization_21_13428358batch_normalization_21_13428360*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_13426733ý
activation_21/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_21_layer_call_and_return_conditional_losses_13427214¢
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall&activation_21/PartitionedCall:output:0conv2d_28_13428364conv2d_28_13428366*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_28_layer_call_and_return_conditional_losses_13427232 
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_22_13428369batch_normalization_22_13428371batch_normalization_22_13428373batch_normalization_22_13428375*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_13426797ý
activation_22/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_22_layer_call_and_return_conditional_losses_13427252¢
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall&activation_22/PartitionedCall:output:0conv2d_29_13428379conv2d_29_13428381*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_29_layer_call_and_return_conditional_losses_13427270 
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0batch_normalization_23_13428384batch_normalization_23_13428386batch_normalization_23_13428388batch_normalization_23_13428390*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_13426861
add_9/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:07batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_add_9_layer_call_and_return_conditional_losses_13427291ä
activation_23/PartitionedCallPartitionedCalladd_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_23_layer_call_and_return_conditional_losses_13427298¢
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0conv2d_30_13428395conv2d_30_13428397*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_30_layer_call_and_return_conditional_losses_13427316 
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0batch_normalization_24_13428400batch_normalization_24_13428402batch_normalization_24_13428404batch_normalization_24_13428406*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_13426925ý
activation_24/PartitionedCallPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_24_layer_call_and_return_conditional_losses_13427336¢
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall&activation_24/PartitionedCall:output:0conv2d_31_13428410conv2d_31_13428412*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_31_layer_call_and_return_conditional_losses_13427354¢
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0conv2d_32_13428415conv2d_32_13428417*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_32_layer_call_and_return_conditional_losses_13427376 
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0batch_normalization_25_13428420batch_normalization_25_13428422batch_normalization_25_13428424batch_normalization_25_13428426*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_13426989
add_10/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:07batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_add_10_layer_call_and_return_conditional_losses_13427397å
activation_25/PartitionedCallPartitionedCalladd_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_25_layer_call_and_return_conditional_losses_13427404¢
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall&activation_25/PartitionedCall:output:0conv2d_33_13428431conv2d_33_13428433*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_33_layer_call_and_return_conditional_losses_13427422 
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0batch_normalization_26_13428436batch_normalization_26_13428438batch_normalization_26_13428440batch_normalization_26_13428442*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_13427053ý
activation_26/PartitionedCallPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_26_layer_call_and_return_conditional_losses_13427442¢
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall&activation_26/PartitionedCall:output:0conv2d_34_13428446conv2d_34_13428448*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_34_layer_call_and_return_conditional_losses_13427460¢
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall&activation_25/PartitionedCall:output:0conv2d_35_13428451conv2d_35_13428453*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_35_layer_call_and_return_conditional_losses_13427482 
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0batch_normalization_27_13428456batch_normalization_27_13428458batch_normalization_27_13428460batch_normalization_27_13428462*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_13427117
add_11/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:07batch_normalization_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_add_11_layer_call_and_return_conditional_losses_13427503å
activation_27/PartitionedCallPartitionedCalladd_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_27_layer_call_and_return_conditional_losses_13427510ø
#average_pooling2d_3/PartitionedCallPartitionedCall&activation_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_13427168â
flatten_3/PartitionedCallPartitionedCall,average_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_13427519
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_13428469dense_3_13428471*
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
GPU 2J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_13427531
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_13428349*&
_output_shapes
:*
dtype0
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_28_13428364*&
_output_shapes
:*
dtype0
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_29_13428379*&
_output_shapes
:*
dtype0
#conv2d_29/kernel/Regularizer/SquareSquare:conv2d_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_29/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_29/kernel/Regularizer/SumSum'conv2d_29/kernel/Regularizer/Square:y:0+conv2d_29/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_29/kernel/Regularizer/mulMul+conv2d_29/kernel/Regularizer/mul/x:output:0)conv2d_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_30_13428395*&
_output_shapes
: *
dtype0
#conv2d_30/kernel/Regularizer/SquareSquare:conv2d_30/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_30/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_30/kernel/Regularizer/SumSum'conv2d_30/kernel/Regularizer/Square:y:0+conv2d_30/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_30/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_30/kernel/Regularizer/mulMul+conv2d_30/kernel/Regularizer/mul/x:output:0)conv2d_30/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_31_13428410*&
_output_shapes
:  *
dtype0
#conv2d_31/kernel/Regularizer/SquareSquare:conv2d_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_31/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_31/kernel/Regularizer/SumSum'conv2d_31/kernel/Regularizer/Square:y:0+conv2d_31/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_31/kernel/Regularizer/mulMul+conv2d_31/kernel/Regularizer/mul/x:output:0)conv2d_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_32_13428415*&
_output_shapes
: *
dtype0
#conv2d_32/kernel/Regularizer/SquareSquare:conv2d_32/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_32/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_32/kernel/Regularizer/SumSum'conv2d_32/kernel/Regularizer/Square:y:0+conv2d_32/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_32/kernel/Regularizer/mulMul+conv2d_32/kernel/Regularizer/mul/x:output:0)conv2d_32/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_33_13428431*&
_output_shapes
: @*
dtype0
#conv2d_33/kernel/Regularizer/SquareSquare:conv2d_33/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_33/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_33/kernel/Regularizer/SumSum'conv2d_33/kernel/Regularizer/Square:y:0+conv2d_33/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_33/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_33/kernel/Regularizer/mulMul+conv2d_33/kernel/Regularizer/mul/x:output:0)conv2d_33/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_34_13428446*&
_output_shapes
:@@*
dtype0
#conv2d_34/kernel/Regularizer/SquareSquare:conv2d_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@{
"conv2d_34/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_34/kernel/Regularizer/SumSum'conv2d_34/kernel/Regularizer/Square:y:0+conv2d_34/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_34/kernel/Regularizer/mulMul+conv2d_34/kernel/Regularizer/mul/x:output:0)conv2d_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_35_13428451*&
_output_shapes
: @*
dtype0
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
à	
NoOpNoOp/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall3^conv2d_27/kernel/Regularizer/Square/ReadVariableOp"^conv2d_28/StatefulPartitionedCall3^conv2d_28/kernel/Regularizer/Square/ReadVariableOp"^conv2d_29/StatefulPartitionedCall3^conv2d_29/kernel/Regularizer/Square/ReadVariableOp"^conv2d_30/StatefulPartitionedCall3^conv2d_30/kernel/Regularizer/Square/ReadVariableOp"^conv2d_31/StatefulPartitionedCall3^conv2d_31/kernel/Regularizer/Square/ReadVariableOp"^conv2d_32/StatefulPartitionedCall3^conv2d_32/kernel/Regularizer/Square/ReadVariableOp"^conv2d_33/StatefulPartitionedCall3^conv2d_33/kernel/Regularizer/Square/ReadVariableOp"^conv2d_34/StatefulPartitionedCall3^conv2d_34/kernel/Regularizer/Square/ReadVariableOp"^conv2d_35/StatefulPartitionedCall3^conv2d_35/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2h
2conv2d_29/kernel/Regularizer/Square/ReadVariableOp2conv2d_29/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2h
2conv2d_30/kernel/Regularizer/Square/ReadVariableOp2conv2d_30/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2h
2conv2d_31/kernel/Regularizer/Square/ReadVariableOp2conv2d_31/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2h
2conv2d_32/kernel/Regularizer/Square/ReadVariableOp2conv2d_32/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2h
2conv2d_33/kernel/Regularizer/Square/ReadVariableOp2conv2d_33/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2h
2conv2d_34/kernel/Regularizer/Square/ReadVariableOp2conv2d_34/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_4
Ï

T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_13429925

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_13430059

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
â
µ
G__inference_conv2d_29_layer_call_and_return_conditional_losses_13429766

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_29/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_29/kernel/Regularizer/SquareSquare:conv2d_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_29/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_29/kernel/Regularizer/SumSum'conv2d_29/kernel/Regularizer/Square:y:0+conv2d_29/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_29/kernel/Regularizer/mulMul+conv2d_29/kernel/Regularizer/mul/x:output:0)conv2d_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_29/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_29/kernel/Regularizer/Square/ReadVariableOp2conv2d_29/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
è
m
C__inference_add_9_layer_call_and_return_conditional_losses_13427291

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ  :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ç
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_13430369

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
â
µ
G__inference_conv2d_27_layer_call_and_return_conditional_losses_13429560

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_27/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_27/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_13427117

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ý
Ã
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_13430192

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ï
g
K__inference_activation_23_layer_call_and_return_conditional_losses_13427298

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_26_layer_call_fn_13430143

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_13427053
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ð
¡
,__inference_conv2d_32_layer_call_fn_13429999

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_32_layer_call_and_return_conditional_losses_13427376w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

£
&__inference_signature_wrapper_13429529
input_4!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23:  

unknown_24: $

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: $

unknown_31: @

unknown_32:@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:@$

unknown_37:@@

unknown_38:@$

unknown_39: @

unknown_40:@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@


unknown_46:

identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_13426711o
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
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_4
Ï

T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_13427053

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ë
½
__inference_loss_fn_3_13430432U
;conv2d_30_kernel_regularizer_square_readvariableop_resource: 
identity¢2conv2d_30/kernel/Regularizer/Square/ReadVariableOp¶
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_30_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_30/kernel/Regularizer/SquareSquare:conv2d_30/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_30/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_30/kernel/Regularizer/SumSum'conv2d_30/kernel/Regularizer/Square:y:0+conv2d_30/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_30/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_30/kernel/Regularizer/mulMul+conv2d_30/kernel/Regularizer/mul/x:output:0)conv2d_30/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_30/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_30/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_30/kernel/Regularizer/Square/ReadVariableOp2conv2d_30/kernel/Regularizer/Square/ReadVariableOp
ð
¡
,__inference_conv2d_31_layer_call_fn_13429968

inputs!
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_31_layer_call_and_return_conditional_losses_13427354w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_27_layer_call_fn_13430290

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_13427148
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¢
m
Q__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_13430358

inputs
identity«
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
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
»
¦
*__inference_model_3_layer_call_fn_13428867

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23:  

unknown_24: $

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: $

unknown_31: @

unknown_32:@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:@$

unknown_37:@@

unknown_38:@$

unknown_39: @

unknown_40:@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@


unknown_46:

identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_model_3_layer_call_and_return_conditional_losses_13427592o
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
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ë
½
__inference_loss_fn_1_13430410U
;conv2d_28_kernel_regularizer_square_readvariableop_resource:
identity¢2conv2d_28/kernel/Regularizer/Square/ReadVariableOp¶
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_28_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_28/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_28/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp
ïÀ
¤ 
$__inference__traced_restore_13430808
file_prefix;
!assignvariableop_conv2d_27_kernel:/
!assignvariableop_1_conv2d_27_bias:=
/assignvariableop_2_batch_normalization_21_gamma:<
.assignvariableop_3_batch_normalization_21_beta:C
5assignvariableop_4_batch_normalization_21_moving_mean:G
9assignvariableop_5_batch_normalization_21_moving_variance:=
#assignvariableop_6_conv2d_28_kernel:/
!assignvariableop_7_conv2d_28_bias:=
/assignvariableop_8_batch_normalization_22_gamma:<
.assignvariableop_9_batch_normalization_22_beta:D
6assignvariableop_10_batch_normalization_22_moving_mean:H
:assignvariableop_11_batch_normalization_22_moving_variance:>
$assignvariableop_12_conv2d_29_kernel:0
"assignvariableop_13_conv2d_29_bias:>
0assignvariableop_14_batch_normalization_23_gamma:=
/assignvariableop_15_batch_normalization_23_beta:D
6assignvariableop_16_batch_normalization_23_moving_mean:H
:assignvariableop_17_batch_normalization_23_moving_variance:>
$assignvariableop_18_conv2d_30_kernel: 0
"assignvariableop_19_conv2d_30_bias: >
0assignvariableop_20_batch_normalization_24_gamma: =
/assignvariableop_21_batch_normalization_24_beta: D
6assignvariableop_22_batch_normalization_24_moving_mean: H
:assignvariableop_23_batch_normalization_24_moving_variance: >
$assignvariableop_24_conv2d_31_kernel:  0
"assignvariableop_25_conv2d_31_bias: >
$assignvariableop_26_conv2d_32_kernel: 0
"assignvariableop_27_conv2d_32_bias: >
0assignvariableop_28_batch_normalization_25_gamma: =
/assignvariableop_29_batch_normalization_25_beta: D
6assignvariableop_30_batch_normalization_25_moving_mean: H
:assignvariableop_31_batch_normalization_25_moving_variance: >
$assignvariableop_32_conv2d_33_kernel: @0
"assignvariableop_33_conv2d_33_bias:@>
0assignvariableop_34_batch_normalization_26_gamma:@=
/assignvariableop_35_batch_normalization_26_beta:@D
6assignvariableop_36_batch_normalization_26_moving_mean:@H
:assignvariableop_37_batch_normalization_26_moving_variance:@>
$assignvariableop_38_conv2d_34_kernel:@@0
"assignvariableop_39_conv2d_34_bias:@>
$assignvariableop_40_conv2d_35_kernel: @0
"assignvariableop_41_conv2d_35_bias:@>
0assignvariableop_42_batch_normalization_27_gamma:@=
/assignvariableop_43_batch_normalization_27_beta:@D
6assignvariableop_44_batch_normalization_27_moving_mean:@H
:assignvariableop_45_batch_normalization_27_moving_variance:@4
"assignvariableop_46_dense_3_kernel:@
.
 assignvariableop_47_dense_3_bias:

identity_49¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ú
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*
valueöBó1B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÒ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ú
_output_shapesÇ
Ä:::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes5
321[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_27_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_27_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_21_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_21_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_21_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_21_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_28_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_28_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_22_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_22_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_22_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_22_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_29_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_29_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_23_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_23_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_23_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_23_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_30_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_30_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_24_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_24_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_24_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_24_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_31_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_31_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp$assignvariableop_26_conv2d_32_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp"assignvariableop_27_conv2d_32_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_28AssignVariableOp0assignvariableop_28_batch_normalization_25_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_29AssignVariableOp/assignvariableop_29_batch_normalization_25_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_30AssignVariableOp6assignvariableop_30_batch_normalization_25_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_31AssignVariableOp:assignvariableop_31_batch_normalization_25_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp$assignvariableop_32_conv2d_33_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp"assignvariableop_33_conv2d_33_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_34AssignVariableOp0assignvariableop_34_batch_normalization_26_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_35AssignVariableOp/assignvariableop_35_batch_normalization_26_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_36AssignVariableOp6assignvariableop_36_batch_normalization_26_moving_meanIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_37AssignVariableOp:assignvariableop_37_batch_normalization_26_moving_varianceIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp$assignvariableop_38_conv2d_34_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp"assignvariableop_39_conv2d_34_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp$assignvariableop_40_conv2d_35_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp"assignvariableop_41_conv2d_35_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_42AssignVariableOp0assignvariableop_42_batch_normalization_27_gammaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_43AssignVariableOp/assignvariableop_43_batch_normalization_27_betaIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_44AssignVariableOp6assignvariableop_44_batch_normalization_27_moving_meanIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_45AssignVariableOp:assignvariableop_45_batch_normalization_27_moving_varianceIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp"assignvariableop_46_dense_3_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp assignvariableop_47_dense_3_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ï
Identity_48Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_49IdentityIdentity_48:output:0^NoOp_1*
T0*
_output_shapes
: Ü
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_49Identity_49:output:0*u
_input_shapesd
b: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472(
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
Ý
Ã
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_13426956

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ë
½
__inference_loss_fn_0_13430399U
;conv2d_27_kernel_regularizer_square_readvariableop_resource:
identity¢2conv2d_27/kernel/Regularizer/Square/ReadVariableOp¶
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_27_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_27/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_27/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp
Ë
L
0__inference_activation_23_layer_call_fn_13429845

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
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_23_layer_call_and_return_conditional_losses_13427298h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ò
U
)__inference_add_10_layer_call_fn_13430083
inputs_0
inputs_1
identityÄ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_add_10_layer_call_and_return_conditional_losses_13427397h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/1
ï
g
K__inference_activation_22_layer_call_and_return_conditional_losses_13429735

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
â
µ
G__inference_conv2d_27_layer_call_and_return_conditional_losses_13427194

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_27/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_27/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
³
H
,__inference_flatten_3_layer_call_fn_13430363

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_13427519`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ë
½
__inference_loss_fn_5_13430454U
;conv2d_32_kernel_regularizer_square_readvariableop_resource: 
identity¢2conv2d_32/kernel/Regularizer/Square/ReadVariableOp¶
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_32_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_32/kernel/Regularizer/SquareSquare:conv2d_32/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_32/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_32/kernel/Regularizer/SumSum'conv2d_32/kernel/Regularizer/Square:y:0+conv2d_32/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_32/kernel/Regularizer/mulMul+conv2d_32/kernel/Regularizer/mul/x:output:0)conv2d_32/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_32/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_32/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_32/kernel/Regularizer/Square/ReadVariableOp2conv2d_32/kernel/Regularizer/Square/ReadVariableOp
ï
g
K__inference_activation_24_layer_call_and_return_conditional_losses_13427336

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_24_layer_call_fn_13429907

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_13426956
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ý
Ã
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_13426764

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_13429810

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
Ã
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_13429943

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ï
g
K__inference_activation_24_layer_call_and_return_conditional_losses_13429953

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
·b
¿
!__inference__traced_save_13430654
file_prefix/
+savev2_conv2d_27_kernel_read_readvariableop-
)savev2_conv2d_27_bias_read_readvariableop;
7savev2_batch_normalization_21_gamma_read_readvariableop:
6savev2_batch_normalization_21_beta_read_readvariableopA
=savev2_batch_normalization_21_moving_mean_read_readvariableopE
Asavev2_batch_normalization_21_moving_variance_read_readvariableop/
+savev2_conv2d_28_kernel_read_readvariableop-
)savev2_conv2d_28_bias_read_readvariableop;
7savev2_batch_normalization_22_gamma_read_readvariableop:
6savev2_batch_normalization_22_beta_read_readvariableopA
=savev2_batch_normalization_22_moving_mean_read_readvariableopE
Asavev2_batch_normalization_22_moving_variance_read_readvariableop/
+savev2_conv2d_29_kernel_read_readvariableop-
)savev2_conv2d_29_bias_read_readvariableop;
7savev2_batch_normalization_23_gamma_read_readvariableop:
6savev2_batch_normalization_23_beta_read_readvariableopA
=savev2_batch_normalization_23_moving_mean_read_readvariableopE
Asavev2_batch_normalization_23_moving_variance_read_readvariableop/
+savev2_conv2d_30_kernel_read_readvariableop-
)savev2_conv2d_30_bias_read_readvariableop;
7savev2_batch_normalization_24_gamma_read_readvariableop:
6savev2_batch_normalization_24_beta_read_readvariableopA
=savev2_batch_normalization_24_moving_mean_read_readvariableopE
Asavev2_batch_normalization_24_moving_variance_read_readvariableop/
+savev2_conv2d_31_kernel_read_readvariableop-
)savev2_conv2d_31_bias_read_readvariableop/
+savev2_conv2d_32_kernel_read_readvariableop-
)savev2_conv2d_32_bias_read_readvariableop;
7savev2_batch_normalization_25_gamma_read_readvariableop:
6savev2_batch_normalization_25_beta_read_readvariableopA
=savev2_batch_normalization_25_moving_mean_read_readvariableopE
Asavev2_batch_normalization_25_moving_variance_read_readvariableop/
+savev2_conv2d_33_kernel_read_readvariableop-
)savev2_conv2d_33_bias_read_readvariableop;
7savev2_batch_normalization_26_gamma_read_readvariableop:
6savev2_batch_normalization_26_beta_read_readvariableopA
=savev2_batch_normalization_26_moving_mean_read_readvariableopE
Asavev2_batch_normalization_26_moving_variance_read_readvariableop/
+savev2_conv2d_34_kernel_read_readvariableop-
)savev2_conv2d_34_bias_read_readvariableop/
+savev2_conv2d_35_kernel_read_readvariableop-
)savev2_conv2d_35_bias_read_readvariableop;
7savev2_batch_normalization_27_gamma_read_readvariableop:
6savev2_batch_normalization_27_beta_read_readvariableopA
=savev2_batch_normalization_27_moving_mean_read_readvariableopE
Asavev2_batch_normalization_27_moving_variance_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop
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
: ×
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*
valueöBó1B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÏ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ñ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_27_kernel_read_readvariableop)savev2_conv2d_27_bias_read_readvariableop7savev2_batch_normalization_21_gamma_read_readvariableop6savev2_batch_normalization_21_beta_read_readvariableop=savev2_batch_normalization_21_moving_mean_read_readvariableopAsavev2_batch_normalization_21_moving_variance_read_readvariableop+savev2_conv2d_28_kernel_read_readvariableop)savev2_conv2d_28_bias_read_readvariableop7savev2_batch_normalization_22_gamma_read_readvariableop6savev2_batch_normalization_22_beta_read_readvariableop=savev2_batch_normalization_22_moving_mean_read_readvariableopAsavev2_batch_normalization_22_moving_variance_read_readvariableop+savev2_conv2d_29_kernel_read_readvariableop)savev2_conv2d_29_bias_read_readvariableop7savev2_batch_normalization_23_gamma_read_readvariableop6savev2_batch_normalization_23_beta_read_readvariableop=savev2_batch_normalization_23_moving_mean_read_readvariableopAsavev2_batch_normalization_23_moving_variance_read_readvariableop+savev2_conv2d_30_kernel_read_readvariableop)savev2_conv2d_30_bias_read_readvariableop7savev2_batch_normalization_24_gamma_read_readvariableop6savev2_batch_normalization_24_beta_read_readvariableop=savev2_batch_normalization_24_moving_mean_read_readvariableopAsavev2_batch_normalization_24_moving_variance_read_readvariableop+savev2_conv2d_31_kernel_read_readvariableop)savev2_conv2d_31_bias_read_readvariableop+savev2_conv2d_32_kernel_read_readvariableop)savev2_conv2d_32_bias_read_readvariableop7savev2_batch_normalization_25_gamma_read_readvariableop6savev2_batch_normalization_25_beta_read_readvariableop=savev2_batch_normalization_25_moving_mean_read_readvariableopAsavev2_batch_normalization_25_moving_variance_read_readvariableop+savev2_conv2d_33_kernel_read_readvariableop)savev2_conv2d_33_bias_read_readvariableop7savev2_batch_normalization_26_gamma_read_readvariableop6savev2_batch_normalization_26_beta_read_readvariableop=savev2_batch_normalization_26_moving_mean_read_readvariableopAsavev2_batch_normalization_26_moving_variance_read_readvariableop+savev2_conv2d_34_kernel_read_readvariableop)savev2_conv2d_34_bias_read_readvariableop+savev2_conv2d_35_kernel_read_readvariableop)savev2_conv2d_35_bias_read_readvariableop7savev2_batch_normalization_27_gamma_read_readvariableop6savev2_batch_normalization_27_beta_read_readvariableop=savev2_batch_normalization_27_moving_mean_read_readvariableopAsavev2_batch_normalization_27_moving_variance_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes5
321
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

identity_1Identity_1:output:0*©
_input_shapes
: ::::::::::::::::::: : : : : : :  : : : : : : : : @:@:@:@:@:@:@@:@: @:@:@:@:@:@:@
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :  

_output_shapes
: :,!(
&
_output_shapes
: @: "

_output_shapes
:@: #

_output_shapes
:@: $

_output_shapes
:@: %

_output_shapes
:@: &

_output_shapes
:@:,'(
&
_output_shapes
:@@: (

_output_shapes
:@:,)(
&
_output_shapes
: @: *

_output_shapes
:@: +

_output_shapes
:@: ,

_output_shapes
:@: -

_output_shapes
:@: .

_output_shapes
:@:$/ 

_output_shapes

:@
: 0

_output_shapes
:
:1

_output_shapes
: 
â
µ
G__inference_conv2d_32_layer_call_and_return_conditional_losses_13427376

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_32/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_32/kernel/Regularizer/SquareSquare:conv2d_32/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_32/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_32/kernel/Regularizer/SumSum'conv2d_32/kernel/Regularizer/Square:y:0+conv2d_32/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_32/kernel/Regularizer/mulMul+conv2d_32/kernel/Regularizer/mul/x:output:0)conv2d_32/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_32/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_32/kernel/Regularizer/Square/ReadVariableOp2conv2d_32/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ð
o
C__inference_add_9_layer_call_and_return_conditional_losses_13429840
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ  :Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
inputs/1
	
Ô
9__inference_batch_normalization_22_layer_call_fn_13429676

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_13426797
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï
g
K__inference_activation_23_layer_call_and_return_conditional_losses_13429850

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_25_layer_call_fn_13430028

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_13426989
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
â
µ
G__inference_conv2d_31_layer_call_and_return_conditional_losses_13427354

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_31/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0
#conv2d_31/kernel/Regularizer/SquareSquare:conv2d_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_31/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_31/kernel/Regularizer/SumSum'conv2d_31/kernel/Regularizer/Square:y:0+conv2d_31/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_31/kernel/Regularizer/mulMul+conv2d_31/kernel/Regularizer/mul/x:output:0)conv2d_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_31/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_31/kernel/Regularizer/Square/ReadVariableOp2conv2d_31/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ð
¡
,__inference_conv2d_34_layer_call_fn_13430217

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_34_layer_call_and_return_conditional_losses_13427460w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
é
n
D__inference_add_10_layer_call_and_return_conditional_losses_13427397

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ò
U
)__inference_add_11_layer_call_fn_13430332
inputs_0
inputs_1
identityÄ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_add_11_layer_call_and_return_conditional_losses_13427503h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/1
Ä

*__inference_dense_3_layer_call_fn_13430378

inputs
unknown:@

	unknown_0:

identity¢StatefulPartitionedCallÚ
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
GPU 2J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_13427531o
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
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_13430308

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ý
Ã
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_13429725

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_13426861

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
½
__inference_loss_fn_2_13430421U
;conv2d_29_kernel_regularizer_square_readvariableop_resource:
identity¢2conv2d_29/kernel/Regularizer/Square/ReadVariableOp¶
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_29_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_29/kernel/Regularizer/SquareSquare:conv2d_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_29/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_29/kernel/Regularizer/SumSum'conv2d_29/kernel/Regularizer/Square:y:0+conv2d_29/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_29/kernel/Regularizer/mulMul+conv2d_29/kernel/Regularizer/mul/x:output:0)conv2d_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_29/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_29/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_29/kernel/Regularizer/Square/ReadVariableOp2conv2d_29/kernel/Regularizer/Square/ReadVariableOp
	
Ô
9__inference_batch_normalization_21_layer_call_fn_13429573

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_13426733
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
L
0__inference_activation_21_layer_call_fn_13429627

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
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_21_layer_call_and_return_conditional_losses_13427214h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ý
Ã
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_13427020

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
â
µ
G__inference_conv2d_35_layer_call_and_return_conditional_losses_13427482

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_35/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_35/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ð
¡
,__inference_conv2d_27_layer_call_fn_13429544

inputs!
unknown:
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
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_27_layer_call_and_return_conditional_losses_13427194w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
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
ñ
p
D__inference_add_10_layer_call_and_return_conditional_losses_13430089
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/1
Ë
L
0__inference_activation_22_layer_call_fn_13429730

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
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_22_layer_call_and_return_conditional_losses_13427252h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ý
Ã
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_13426892

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
µ
G__inference_conv2d_35_layer_call_and_return_conditional_losses_13430264

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_35/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_35/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ï
g
K__inference_activation_27_layer_call_and_return_conditional_losses_13427510

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ñ
p
D__inference_add_11_layer_call_and_return_conditional_losses_13430338
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/1
Ï

T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_13426989

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_13426925

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ø
¶
E__inference_model_3_layer_call_and_return_conditional_losses_13428146

inputs,
conv2d_27_13427966: 
conv2d_27_13427968:-
batch_normalization_21_13427971:-
batch_normalization_21_13427973:-
batch_normalization_21_13427975:-
batch_normalization_21_13427977:,
conv2d_28_13427981: 
conv2d_28_13427983:-
batch_normalization_22_13427986:-
batch_normalization_22_13427988:-
batch_normalization_22_13427990:-
batch_normalization_22_13427992:,
conv2d_29_13427996: 
conv2d_29_13427998:-
batch_normalization_23_13428001:-
batch_normalization_23_13428003:-
batch_normalization_23_13428005:-
batch_normalization_23_13428007:,
conv2d_30_13428012:  
conv2d_30_13428014: -
batch_normalization_24_13428017: -
batch_normalization_24_13428019: -
batch_normalization_24_13428021: -
batch_normalization_24_13428023: ,
conv2d_31_13428027:   
conv2d_31_13428029: ,
conv2d_32_13428032:  
conv2d_32_13428034: -
batch_normalization_25_13428037: -
batch_normalization_25_13428039: -
batch_normalization_25_13428041: -
batch_normalization_25_13428043: ,
conv2d_33_13428048: @ 
conv2d_33_13428050:@-
batch_normalization_26_13428053:@-
batch_normalization_26_13428055:@-
batch_normalization_26_13428057:@-
batch_normalization_26_13428059:@,
conv2d_34_13428063:@@ 
conv2d_34_13428065:@,
conv2d_35_13428068: @ 
conv2d_35_13428070:@-
batch_normalization_27_13428073:@-
batch_normalization_27_13428075:@-
batch_normalization_27_13428077:@-
batch_normalization_27_13428079:@"
dense_3_13428086:@

dense_3_13428088:

identity¢.batch_normalization_21/StatefulPartitionedCall¢.batch_normalization_22/StatefulPartitionedCall¢.batch_normalization_23/StatefulPartitionedCall¢.batch_normalization_24/StatefulPartitionedCall¢.batch_normalization_25/StatefulPartitionedCall¢.batch_normalization_26/StatefulPartitionedCall¢.batch_normalization_27/StatefulPartitionedCall¢!conv2d_27/StatefulPartitionedCall¢2conv2d_27/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_28/StatefulPartitionedCall¢2conv2d_28/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_29/StatefulPartitionedCall¢2conv2d_29/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_30/StatefulPartitionedCall¢2conv2d_30/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_31/StatefulPartitionedCall¢2conv2d_31/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_32/StatefulPartitionedCall¢2conv2d_32/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_33/StatefulPartitionedCall¢2conv2d_33/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_34/StatefulPartitionedCall¢2conv2d_34/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_35/StatefulPartitionedCall¢2conv2d_35/kernel/Regularizer/Square/ReadVariableOp¢dense_3/StatefulPartitionedCall
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_27_13427966conv2d_27_13427968*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_27_layer_call_and_return_conditional_losses_13427194
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0batch_normalization_21_13427971batch_normalization_21_13427973batch_normalization_21_13427975batch_normalization_21_13427977*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_13426764ý
activation_21/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_21_layer_call_and_return_conditional_losses_13427214¢
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall&activation_21/PartitionedCall:output:0conv2d_28_13427981conv2d_28_13427983*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_28_layer_call_and_return_conditional_losses_13427232
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_22_13427986batch_normalization_22_13427988batch_normalization_22_13427990batch_normalization_22_13427992*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_13426828ý
activation_22/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_22_layer_call_and_return_conditional_losses_13427252¢
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall&activation_22/PartitionedCall:output:0conv2d_29_13427996conv2d_29_13427998*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_29_layer_call_and_return_conditional_losses_13427270
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0batch_normalization_23_13428001batch_normalization_23_13428003batch_normalization_23_13428005batch_normalization_23_13428007*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_13426892
add_9/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:07batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_add_9_layer_call_and_return_conditional_losses_13427291ä
activation_23/PartitionedCallPartitionedCalladd_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_23_layer_call_and_return_conditional_losses_13427298¢
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0conv2d_30_13428012conv2d_30_13428014*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_30_layer_call_and_return_conditional_losses_13427316
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0batch_normalization_24_13428017batch_normalization_24_13428019batch_normalization_24_13428021batch_normalization_24_13428023*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_13426956ý
activation_24/PartitionedCallPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_24_layer_call_and_return_conditional_losses_13427336¢
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall&activation_24/PartitionedCall:output:0conv2d_31_13428027conv2d_31_13428029*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_31_layer_call_and_return_conditional_losses_13427354¢
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0conv2d_32_13428032conv2d_32_13428034*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_32_layer_call_and_return_conditional_losses_13427376
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0batch_normalization_25_13428037batch_normalization_25_13428039batch_normalization_25_13428041batch_normalization_25_13428043*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_13427020
add_10/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:07batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_add_10_layer_call_and_return_conditional_losses_13427397å
activation_25/PartitionedCallPartitionedCalladd_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_25_layer_call_and_return_conditional_losses_13427404¢
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall&activation_25/PartitionedCall:output:0conv2d_33_13428048conv2d_33_13428050*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_33_layer_call_and_return_conditional_losses_13427422
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0batch_normalization_26_13428053batch_normalization_26_13428055batch_normalization_26_13428057batch_normalization_26_13428059*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_13427084ý
activation_26/PartitionedCallPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_26_layer_call_and_return_conditional_losses_13427442¢
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall&activation_26/PartitionedCall:output:0conv2d_34_13428063conv2d_34_13428065*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_34_layer_call_and_return_conditional_losses_13427460¢
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall&activation_25/PartitionedCall:output:0conv2d_35_13428068conv2d_35_13428070*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_35_layer_call_and_return_conditional_losses_13427482
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0batch_normalization_27_13428073batch_normalization_27_13428075batch_normalization_27_13428077batch_normalization_27_13428079*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_13427148
add_11/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:07batch_normalization_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_add_11_layer_call_and_return_conditional_losses_13427503å
activation_27/PartitionedCallPartitionedCalladd_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_27_layer_call_and_return_conditional_losses_13427510ø
#average_pooling2d_3/PartitionedCallPartitionedCall&activation_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_13427168â
flatten_3/PartitionedCallPartitionedCall,average_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_13427519
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_13428086dense_3_13428088*
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
GPU 2J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_13427531
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_13427966*&
_output_shapes
:*
dtype0
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_28_13427981*&
_output_shapes
:*
dtype0
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_29_13427996*&
_output_shapes
:*
dtype0
#conv2d_29/kernel/Regularizer/SquareSquare:conv2d_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_29/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_29/kernel/Regularizer/SumSum'conv2d_29/kernel/Regularizer/Square:y:0+conv2d_29/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_29/kernel/Regularizer/mulMul+conv2d_29/kernel/Regularizer/mul/x:output:0)conv2d_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_30_13428012*&
_output_shapes
: *
dtype0
#conv2d_30/kernel/Regularizer/SquareSquare:conv2d_30/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_30/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_30/kernel/Regularizer/SumSum'conv2d_30/kernel/Regularizer/Square:y:0+conv2d_30/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_30/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_30/kernel/Regularizer/mulMul+conv2d_30/kernel/Regularizer/mul/x:output:0)conv2d_30/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_31_13428027*&
_output_shapes
:  *
dtype0
#conv2d_31/kernel/Regularizer/SquareSquare:conv2d_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_31/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_31/kernel/Regularizer/SumSum'conv2d_31/kernel/Regularizer/Square:y:0+conv2d_31/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_31/kernel/Regularizer/mulMul+conv2d_31/kernel/Regularizer/mul/x:output:0)conv2d_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_32_13428032*&
_output_shapes
: *
dtype0
#conv2d_32/kernel/Regularizer/SquareSquare:conv2d_32/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_32/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_32/kernel/Regularizer/SumSum'conv2d_32/kernel/Regularizer/Square:y:0+conv2d_32/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_32/kernel/Regularizer/mulMul+conv2d_32/kernel/Regularizer/mul/x:output:0)conv2d_32/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_33_13428048*&
_output_shapes
: @*
dtype0
#conv2d_33/kernel/Regularizer/SquareSquare:conv2d_33/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_33/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_33/kernel/Regularizer/SumSum'conv2d_33/kernel/Regularizer/Square:y:0+conv2d_33/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_33/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_33/kernel/Regularizer/mulMul+conv2d_33/kernel/Regularizer/mul/x:output:0)conv2d_33/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_34_13428063*&
_output_shapes
:@@*
dtype0
#conv2d_34/kernel/Regularizer/SquareSquare:conv2d_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@{
"conv2d_34/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_34/kernel/Regularizer/SumSum'conv2d_34/kernel/Regularizer/Square:y:0+conv2d_34/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_34/kernel/Regularizer/mulMul+conv2d_34/kernel/Regularizer/mul/x:output:0)conv2d_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_35_13428068*&
_output_shapes
: @*
dtype0
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
à	
NoOpNoOp/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall3^conv2d_27/kernel/Regularizer/Square/ReadVariableOp"^conv2d_28/StatefulPartitionedCall3^conv2d_28/kernel/Regularizer/Square/ReadVariableOp"^conv2d_29/StatefulPartitionedCall3^conv2d_29/kernel/Regularizer/Square/ReadVariableOp"^conv2d_30/StatefulPartitionedCall3^conv2d_30/kernel/Regularizer/Square/ReadVariableOp"^conv2d_31/StatefulPartitionedCall3^conv2d_31/kernel/Regularizer/Square/ReadVariableOp"^conv2d_32/StatefulPartitionedCall3^conv2d_32/kernel/Regularizer/Square/ReadVariableOp"^conv2d_33/StatefulPartitionedCall3^conv2d_33/kernel/Regularizer/Square/ReadVariableOp"^conv2d_34/StatefulPartitionedCall3^conv2d_34/kernel/Regularizer/Square/ReadVariableOp"^conv2d_35/StatefulPartitionedCall3^conv2d_35/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2h
2conv2d_29/kernel/Regularizer/Square/ReadVariableOp2conv2d_29/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2h
2conv2d_30/kernel/Regularizer/Square/ReadVariableOp2conv2d_30/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2h
2conv2d_31/kernel/Regularizer/Square/ReadVariableOp2conv2d_31/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2h
2conv2d_32/kernel/Regularizer/Square/ReadVariableOp2conv2d_32/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2h
2conv2d_33/kernel/Regularizer/Square/ReadVariableOp2conv2d_33/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2h
2conv2d_34/kernel/Regularizer/Square/ReadVariableOp2conv2d_34/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
¡Ø
·
E__inference_model_3_layer_call_and_return_conditional_losses_13428712
input_4,
conv2d_27_13428532: 
conv2d_27_13428534:-
batch_normalization_21_13428537:-
batch_normalization_21_13428539:-
batch_normalization_21_13428541:-
batch_normalization_21_13428543:,
conv2d_28_13428547: 
conv2d_28_13428549:-
batch_normalization_22_13428552:-
batch_normalization_22_13428554:-
batch_normalization_22_13428556:-
batch_normalization_22_13428558:,
conv2d_29_13428562: 
conv2d_29_13428564:-
batch_normalization_23_13428567:-
batch_normalization_23_13428569:-
batch_normalization_23_13428571:-
batch_normalization_23_13428573:,
conv2d_30_13428578:  
conv2d_30_13428580: -
batch_normalization_24_13428583: -
batch_normalization_24_13428585: -
batch_normalization_24_13428587: -
batch_normalization_24_13428589: ,
conv2d_31_13428593:   
conv2d_31_13428595: ,
conv2d_32_13428598:  
conv2d_32_13428600: -
batch_normalization_25_13428603: -
batch_normalization_25_13428605: -
batch_normalization_25_13428607: -
batch_normalization_25_13428609: ,
conv2d_33_13428614: @ 
conv2d_33_13428616:@-
batch_normalization_26_13428619:@-
batch_normalization_26_13428621:@-
batch_normalization_26_13428623:@-
batch_normalization_26_13428625:@,
conv2d_34_13428629:@@ 
conv2d_34_13428631:@,
conv2d_35_13428634: @ 
conv2d_35_13428636:@-
batch_normalization_27_13428639:@-
batch_normalization_27_13428641:@-
batch_normalization_27_13428643:@-
batch_normalization_27_13428645:@"
dense_3_13428652:@

dense_3_13428654:

identity¢.batch_normalization_21/StatefulPartitionedCall¢.batch_normalization_22/StatefulPartitionedCall¢.batch_normalization_23/StatefulPartitionedCall¢.batch_normalization_24/StatefulPartitionedCall¢.batch_normalization_25/StatefulPartitionedCall¢.batch_normalization_26/StatefulPartitionedCall¢.batch_normalization_27/StatefulPartitionedCall¢!conv2d_27/StatefulPartitionedCall¢2conv2d_27/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_28/StatefulPartitionedCall¢2conv2d_28/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_29/StatefulPartitionedCall¢2conv2d_29/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_30/StatefulPartitionedCall¢2conv2d_30/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_31/StatefulPartitionedCall¢2conv2d_31/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_32/StatefulPartitionedCall¢2conv2d_32/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_33/StatefulPartitionedCall¢2conv2d_33/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_34/StatefulPartitionedCall¢2conv2d_34/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_35/StatefulPartitionedCall¢2conv2d_35/kernel/Regularizer/Square/ReadVariableOp¢dense_3/StatefulPartitionedCall
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_27_13428532conv2d_27_13428534*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_27_layer_call_and_return_conditional_losses_13427194
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0batch_normalization_21_13428537batch_normalization_21_13428539batch_normalization_21_13428541batch_normalization_21_13428543*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_13426764ý
activation_21/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_21_layer_call_and_return_conditional_losses_13427214¢
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall&activation_21/PartitionedCall:output:0conv2d_28_13428547conv2d_28_13428549*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_28_layer_call_and_return_conditional_losses_13427232
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_22_13428552batch_normalization_22_13428554batch_normalization_22_13428556batch_normalization_22_13428558*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_13426828ý
activation_22/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_22_layer_call_and_return_conditional_losses_13427252¢
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall&activation_22/PartitionedCall:output:0conv2d_29_13428562conv2d_29_13428564*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_29_layer_call_and_return_conditional_losses_13427270
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0batch_normalization_23_13428567batch_normalization_23_13428569batch_normalization_23_13428571batch_normalization_23_13428573*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_13426892
add_9/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:07batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_add_9_layer_call_and_return_conditional_losses_13427291ä
activation_23/PartitionedCallPartitionedCalladd_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_23_layer_call_and_return_conditional_losses_13427298¢
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0conv2d_30_13428578conv2d_30_13428580*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_30_layer_call_and_return_conditional_losses_13427316
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0batch_normalization_24_13428583batch_normalization_24_13428585batch_normalization_24_13428587batch_normalization_24_13428589*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_13426956ý
activation_24/PartitionedCallPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_24_layer_call_and_return_conditional_losses_13427336¢
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall&activation_24/PartitionedCall:output:0conv2d_31_13428593conv2d_31_13428595*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_31_layer_call_and_return_conditional_losses_13427354¢
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0conv2d_32_13428598conv2d_32_13428600*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_32_layer_call_and_return_conditional_losses_13427376
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0batch_normalization_25_13428603batch_normalization_25_13428605batch_normalization_25_13428607batch_normalization_25_13428609*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_13427020
add_10/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:07batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_add_10_layer_call_and_return_conditional_losses_13427397å
activation_25/PartitionedCallPartitionedCalladd_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_25_layer_call_and_return_conditional_losses_13427404¢
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall&activation_25/PartitionedCall:output:0conv2d_33_13428614conv2d_33_13428616*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_33_layer_call_and_return_conditional_losses_13427422
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0batch_normalization_26_13428619batch_normalization_26_13428621batch_normalization_26_13428623batch_normalization_26_13428625*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_13427084ý
activation_26/PartitionedCallPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_26_layer_call_and_return_conditional_losses_13427442¢
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall&activation_26/PartitionedCall:output:0conv2d_34_13428629conv2d_34_13428631*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_34_layer_call_and_return_conditional_losses_13427460¢
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall&activation_25/PartitionedCall:output:0conv2d_35_13428634conv2d_35_13428636*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_35_layer_call_and_return_conditional_losses_13427482
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0batch_normalization_27_13428639batch_normalization_27_13428641batch_normalization_27_13428643batch_normalization_27_13428645*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_13427148
add_11/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:07batch_normalization_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_add_11_layer_call_and_return_conditional_losses_13427503å
activation_27/PartitionedCallPartitionedCalladd_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_27_layer_call_and_return_conditional_losses_13427510ø
#average_pooling2d_3/PartitionedCallPartitionedCall&activation_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_13427168â
flatten_3/PartitionedCallPartitionedCall,average_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_13427519
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_13428652dense_3_13428654*
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
GPU 2J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_13427531
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_13428532*&
_output_shapes
:*
dtype0
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_28_13428547*&
_output_shapes
:*
dtype0
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_29_13428562*&
_output_shapes
:*
dtype0
#conv2d_29/kernel/Regularizer/SquareSquare:conv2d_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_29/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_29/kernel/Regularizer/SumSum'conv2d_29/kernel/Regularizer/Square:y:0+conv2d_29/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_29/kernel/Regularizer/mulMul+conv2d_29/kernel/Regularizer/mul/x:output:0)conv2d_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_30_13428578*&
_output_shapes
: *
dtype0
#conv2d_30/kernel/Regularizer/SquareSquare:conv2d_30/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_30/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_30/kernel/Regularizer/SumSum'conv2d_30/kernel/Regularizer/Square:y:0+conv2d_30/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_30/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_30/kernel/Regularizer/mulMul+conv2d_30/kernel/Regularizer/mul/x:output:0)conv2d_30/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_31_13428593*&
_output_shapes
:  *
dtype0
#conv2d_31/kernel/Regularizer/SquareSquare:conv2d_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_31/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_31/kernel/Regularizer/SumSum'conv2d_31/kernel/Regularizer/Square:y:0+conv2d_31/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_31/kernel/Regularizer/mulMul+conv2d_31/kernel/Regularizer/mul/x:output:0)conv2d_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_32_13428598*&
_output_shapes
: *
dtype0
#conv2d_32/kernel/Regularizer/SquareSquare:conv2d_32/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_32/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_32/kernel/Regularizer/SumSum'conv2d_32/kernel/Regularizer/Square:y:0+conv2d_32/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_32/kernel/Regularizer/mulMul+conv2d_32/kernel/Regularizer/mul/x:output:0)conv2d_32/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_33_13428614*&
_output_shapes
: @*
dtype0
#conv2d_33/kernel/Regularizer/SquareSquare:conv2d_33/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_33/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_33/kernel/Regularizer/SumSum'conv2d_33/kernel/Regularizer/Square:y:0+conv2d_33/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_33/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_33/kernel/Regularizer/mulMul+conv2d_33/kernel/Regularizer/mul/x:output:0)conv2d_33/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_34_13428629*&
_output_shapes
:@@*
dtype0
#conv2d_34/kernel/Regularizer/SquareSquare:conv2d_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@{
"conv2d_34/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_34/kernel/Regularizer/SumSum'conv2d_34/kernel/Regularizer/Square:y:0+conv2d_34/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_34/kernel/Regularizer/mulMul+conv2d_34/kernel/Regularizer/mul/x:output:0)conv2d_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_35_13428634*&
_output_shapes
: @*
dtype0
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
à	
NoOpNoOp/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall3^conv2d_27/kernel/Regularizer/Square/ReadVariableOp"^conv2d_28/StatefulPartitionedCall3^conv2d_28/kernel/Regularizer/Square/ReadVariableOp"^conv2d_29/StatefulPartitionedCall3^conv2d_29/kernel/Regularizer/Square/ReadVariableOp"^conv2d_30/StatefulPartitionedCall3^conv2d_30/kernel/Regularizer/Square/ReadVariableOp"^conv2d_31/StatefulPartitionedCall3^conv2d_31/kernel/Regularizer/Square/ReadVariableOp"^conv2d_32/StatefulPartitionedCall3^conv2d_32/kernel/Regularizer/Square/ReadVariableOp"^conv2d_33/StatefulPartitionedCall3^conv2d_33/kernel/Regularizer/Square/ReadVariableOp"^conv2d_34/StatefulPartitionedCall3^conv2d_34/kernel/Regularizer/Square/ReadVariableOp"^conv2d_35/StatefulPartitionedCall3^conv2d_35/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2h
2conv2d_29/kernel/Regularizer/Square/ReadVariableOp2conv2d_29/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2h
2conv2d_30/kernel/Regularizer/Square/ReadVariableOp2conv2d_30/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2h
2conv2d_31/kernel/Regularizer/Square/ReadVariableOp2conv2d_31/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2h
2conv2d_32/kernel/Regularizer/Square/ReadVariableOp2conv2d_32/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2h
2conv2d_33/kernel/Regularizer/Square/ReadVariableOp2conv2d_33/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2h
2conv2d_34/kernel/Regularizer/Square/ReadVariableOp2conv2d_34/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_4
Ä
R
6__inference_average_pooling2d_3_layer_call_fn_13430353

inputs
identityß
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
GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_13427168
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
â
µ
G__inference_conv2d_30_layer_call_and_return_conditional_losses_13429881

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_30/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_30/kernel/Regularizer/SquareSquare:conv2d_30/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_30/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_30/kernel/Regularizer/SumSum'conv2d_30/kernel/Regularizer/Square:y:0+conv2d_30/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_30/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_30/kernel/Regularizer/mulMul+conv2d_30/kernel/Regularizer/mul/x:output:0)conv2d_30/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_30/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_30/kernel/Regularizer/Square/ReadVariableOp2conv2d_30/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_24_layer_call_fn_13429894

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_13426925
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¬Ø
¶
E__inference_model_3_layer_call_and_return_conditional_losses_13427592

inputs,
conv2d_27_13427195: 
conv2d_27_13427197:-
batch_normalization_21_13427200:-
batch_normalization_21_13427202:-
batch_normalization_21_13427204:-
batch_normalization_21_13427206:,
conv2d_28_13427233: 
conv2d_28_13427235:-
batch_normalization_22_13427238:-
batch_normalization_22_13427240:-
batch_normalization_22_13427242:-
batch_normalization_22_13427244:,
conv2d_29_13427271: 
conv2d_29_13427273:-
batch_normalization_23_13427276:-
batch_normalization_23_13427278:-
batch_normalization_23_13427280:-
batch_normalization_23_13427282:,
conv2d_30_13427317:  
conv2d_30_13427319: -
batch_normalization_24_13427322: -
batch_normalization_24_13427324: -
batch_normalization_24_13427326: -
batch_normalization_24_13427328: ,
conv2d_31_13427355:   
conv2d_31_13427357: ,
conv2d_32_13427377:  
conv2d_32_13427379: -
batch_normalization_25_13427382: -
batch_normalization_25_13427384: -
batch_normalization_25_13427386: -
batch_normalization_25_13427388: ,
conv2d_33_13427423: @ 
conv2d_33_13427425:@-
batch_normalization_26_13427428:@-
batch_normalization_26_13427430:@-
batch_normalization_26_13427432:@-
batch_normalization_26_13427434:@,
conv2d_34_13427461:@@ 
conv2d_34_13427463:@,
conv2d_35_13427483: @ 
conv2d_35_13427485:@-
batch_normalization_27_13427488:@-
batch_normalization_27_13427490:@-
batch_normalization_27_13427492:@-
batch_normalization_27_13427494:@"
dense_3_13427532:@

dense_3_13427534:

identity¢.batch_normalization_21/StatefulPartitionedCall¢.batch_normalization_22/StatefulPartitionedCall¢.batch_normalization_23/StatefulPartitionedCall¢.batch_normalization_24/StatefulPartitionedCall¢.batch_normalization_25/StatefulPartitionedCall¢.batch_normalization_26/StatefulPartitionedCall¢.batch_normalization_27/StatefulPartitionedCall¢!conv2d_27/StatefulPartitionedCall¢2conv2d_27/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_28/StatefulPartitionedCall¢2conv2d_28/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_29/StatefulPartitionedCall¢2conv2d_29/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_30/StatefulPartitionedCall¢2conv2d_30/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_31/StatefulPartitionedCall¢2conv2d_31/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_32/StatefulPartitionedCall¢2conv2d_32/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_33/StatefulPartitionedCall¢2conv2d_33/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_34/StatefulPartitionedCall¢2conv2d_34/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_35/StatefulPartitionedCall¢2conv2d_35/kernel/Regularizer/Square/ReadVariableOp¢dense_3/StatefulPartitionedCall
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_27_13427195conv2d_27_13427197*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_27_layer_call_and_return_conditional_losses_13427194 
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0batch_normalization_21_13427200batch_normalization_21_13427202batch_normalization_21_13427204batch_normalization_21_13427206*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_13426733ý
activation_21/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_21_layer_call_and_return_conditional_losses_13427214¢
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall&activation_21/PartitionedCall:output:0conv2d_28_13427233conv2d_28_13427235*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_28_layer_call_and_return_conditional_losses_13427232 
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_22_13427238batch_normalization_22_13427240batch_normalization_22_13427242batch_normalization_22_13427244*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_13426797ý
activation_22/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_22_layer_call_and_return_conditional_losses_13427252¢
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall&activation_22/PartitionedCall:output:0conv2d_29_13427271conv2d_29_13427273*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_29_layer_call_and_return_conditional_losses_13427270 
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0batch_normalization_23_13427276batch_normalization_23_13427278batch_normalization_23_13427280batch_normalization_23_13427282*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_13426861
add_9/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:07batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_add_9_layer_call_and_return_conditional_losses_13427291ä
activation_23/PartitionedCallPartitionedCalladd_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_23_layer_call_and_return_conditional_losses_13427298¢
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0conv2d_30_13427317conv2d_30_13427319*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_30_layer_call_and_return_conditional_losses_13427316 
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0batch_normalization_24_13427322batch_normalization_24_13427324batch_normalization_24_13427326batch_normalization_24_13427328*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_13426925ý
activation_24/PartitionedCallPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_24_layer_call_and_return_conditional_losses_13427336¢
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall&activation_24/PartitionedCall:output:0conv2d_31_13427355conv2d_31_13427357*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_31_layer_call_and_return_conditional_losses_13427354¢
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall&activation_23/PartitionedCall:output:0conv2d_32_13427377conv2d_32_13427379*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_32_layer_call_and_return_conditional_losses_13427376 
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0batch_normalization_25_13427382batch_normalization_25_13427384batch_normalization_25_13427386batch_normalization_25_13427388*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_13426989
add_10/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:07batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_add_10_layer_call_and_return_conditional_losses_13427397å
activation_25/PartitionedCallPartitionedCalladd_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_25_layer_call_and_return_conditional_losses_13427404¢
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall&activation_25/PartitionedCall:output:0conv2d_33_13427423conv2d_33_13427425*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_33_layer_call_and_return_conditional_losses_13427422 
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0batch_normalization_26_13427428batch_normalization_26_13427430batch_normalization_26_13427432batch_normalization_26_13427434*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_13427053ý
activation_26/PartitionedCallPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_26_layer_call_and_return_conditional_losses_13427442¢
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall&activation_26/PartitionedCall:output:0conv2d_34_13427461conv2d_34_13427463*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_34_layer_call_and_return_conditional_losses_13427460¢
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall&activation_25/PartitionedCall:output:0conv2d_35_13427483conv2d_35_13427485*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_35_layer_call_and_return_conditional_losses_13427482 
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0batch_normalization_27_13427488batch_normalization_27_13427490batch_normalization_27_13427492batch_normalization_27_13427494*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_13427117
add_11/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:07batch_normalization_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_add_11_layer_call_and_return_conditional_losses_13427503å
activation_27/PartitionedCallPartitionedCalladd_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_27_layer_call_and_return_conditional_losses_13427510ø
#average_pooling2d_3/PartitionedCallPartitionedCall&activation_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_13427168â
flatten_3/PartitionedCallPartitionedCall,average_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_13427519
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_13427532dense_3_13427534*
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
GPU 2J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_13427531
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_13427195*&
_output_shapes
:*
dtype0
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_28_13427233*&
_output_shapes
:*
dtype0
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_29_13427271*&
_output_shapes
:*
dtype0
#conv2d_29/kernel/Regularizer/SquareSquare:conv2d_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_29/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_29/kernel/Regularizer/SumSum'conv2d_29/kernel/Regularizer/Square:y:0+conv2d_29/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_29/kernel/Regularizer/mulMul+conv2d_29/kernel/Regularizer/mul/x:output:0)conv2d_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_30/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_30_13427317*&
_output_shapes
: *
dtype0
#conv2d_30/kernel/Regularizer/SquareSquare:conv2d_30/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_30/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_30/kernel/Regularizer/SumSum'conv2d_30/kernel/Regularizer/Square:y:0+conv2d_30/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_30/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_30/kernel/Regularizer/mulMul+conv2d_30/kernel/Regularizer/mul/x:output:0)conv2d_30/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_31_13427355*&
_output_shapes
:  *
dtype0
#conv2d_31/kernel/Regularizer/SquareSquare:conv2d_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_31/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_31/kernel/Regularizer/SumSum'conv2d_31/kernel/Regularizer/Square:y:0+conv2d_31/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_31/kernel/Regularizer/mulMul+conv2d_31/kernel/Regularizer/mul/x:output:0)conv2d_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_32/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_32_13427377*&
_output_shapes
: *
dtype0
#conv2d_32/kernel/Regularizer/SquareSquare:conv2d_32/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_32/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_32/kernel/Regularizer/SumSum'conv2d_32/kernel/Regularizer/Square:y:0+conv2d_32/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_32/kernel/Regularizer/mulMul+conv2d_32/kernel/Regularizer/mul/x:output:0)conv2d_32/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_33/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_33_13427423*&
_output_shapes
: @*
dtype0
#conv2d_33/kernel/Regularizer/SquareSquare:conv2d_33/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_33/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_33/kernel/Regularizer/SumSum'conv2d_33/kernel/Regularizer/Square:y:0+conv2d_33/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_33/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_33/kernel/Regularizer/mulMul+conv2d_33/kernel/Regularizer/mul/x:output:0)conv2d_33/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_34_13427461*&
_output_shapes
:@@*
dtype0
#conv2d_34/kernel/Regularizer/SquareSquare:conv2d_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@{
"conv2d_34/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_34/kernel/Regularizer/SumSum'conv2d_34/kernel/Regularizer/Square:y:0+conv2d_34/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_34/kernel/Regularizer/mulMul+conv2d_34/kernel/Regularizer/mul/x:output:0)conv2d_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_35_13427483*&
_output_shapes
: @*
dtype0
#conv2d_35/kernel/Regularizer/SquareSquare:conv2d_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_35/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_35/kernel/Regularizer/SumSum'conv2d_35/kernel/Regularizer/Square:y:0+conv2d_35/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_35/kernel/Regularizer/mulMul+conv2d_35/kernel/Regularizer/mul/x:output:0)conv2d_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
à	
NoOpNoOp/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall3^conv2d_27/kernel/Regularizer/Square/ReadVariableOp"^conv2d_28/StatefulPartitionedCall3^conv2d_28/kernel/Regularizer/Square/ReadVariableOp"^conv2d_29/StatefulPartitionedCall3^conv2d_29/kernel/Regularizer/Square/ReadVariableOp"^conv2d_30/StatefulPartitionedCall3^conv2d_30/kernel/Regularizer/Square/ReadVariableOp"^conv2d_31/StatefulPartitionedCall3^conv2d_31/kernel/Regularizer/Square/ReadVariableOp"^conv2d_32/StatefulPartitionedCall3^conv2d_32/kernel/Regularizer/Square/ReadVariableOp"^conv2d_33/StatefulPartitionedCall3^conv2d_33/kernel/Regularizer/Square/ReadVariableOp"^conv2d_34/StatefulPartitionedCall3^conv2d_34/kernel/Regularizer/Square/ReadVariableOp"^conv2d_35/StatefulPartitionedCall3^conv2d_35/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2h
2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2conv2d_27/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2h
2conv2d_29/kernel/Regularizer/Square/ReadVariableOp2conv2d_29/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2h
2conv2d_30/kernel/Regularizer/Square/ReadVariableOp2conv2d_30/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2h
2conv2d_31/kernel/Regularizer/Square/ReadVariableOp2conv2d_31/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2h
2conv2d_32/kernel/Regularizer/Square/ReadVariableOp2conv2d_32/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2h
2conv2d_33/kernel/Regularizer/Square/ReadVariableOp2conv2d_33/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2h
2conv2d_34/kernel/Regularizer/Square/ReadVariableOp2conv2d_34/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2h
2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2conv2d_35/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
È	
ö
E__inference_dense_3_layer_call_and_return_conditional_losses_13427531

inputs0
matmul_readvariableop_resource:@
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
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
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_27_layer_call_fn_13430277

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_13427117
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ý
Ã
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_13430326

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ý
Ã
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_13426828

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
L
0__inference_activation_26_layer_call_fn_13430197

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
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_activation_26_layer_call_and_return_conditional_losses_13427442h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_23_layer_call_fn_13429779

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_13426861
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿÷
0
#__inference__wrapped_model_13426711
input_4J
0model_3_conv2d_27_conv2d_readvariableop_resource:?
1model_3_conv2d_27_biasadd_readvariableop_resource:D
6model_3_batch_normalization_21_readvariableop_resource:F
8model_3_batch_normalization_21_readvariableop_1_resource:U
Gmodel_3_batch_normalization_21_fusedbatchnormv3_readvariableop_resource:W
Imodel_3_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:J
0model_3_conv2d_28_conv2d_readvariableop_resource:?
1model_3_conv2d_28_biasadd_readvariableop_resource:D
6model_3_batch_normalization_22_readvariableop_resource:F
8model_3_batch_normalization_22_readvariableop_1_resource:U
Gmodel_3_batch_normalization_22_fusedbatchnormv3_readvariableop_resource:W
Imodel_3_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource:J
0model_3_conv2d_29_conv2d_readvariableop_resource:?
1model_3_conv2d_29_biasadd_readvariableop_resource:D
6model_3_batch_normalization_23_readvariableop_resource:F
8model_3_batch_normalization_23_readvariableop_1_resource:U
Gmodel_3_batch_normalization_23_fusedbatchnormv3_readvariableop_resource:W
Imodel_3_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource:J
0model_3_conv2d_30_conv2d_readvariableop_resource: ?
1model_3_conv2d_30_biasadd_readvariableop_resource: D
6model_3_batch_normalization_24_readvariableop_resource: F
8model_3_batch_normalization_24_readvariableop_1_resource: U
Gmodel_3_batch_normalization_24_fusedbatchnormv3_readvariableop_resource: W
Imodel_3_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource: J
0model_3_conv2d_31_conv2d_readvariableop_resource:  ?
1model_3_conv2d_31_biasadd_readvariableop_resource: J
0model_3_conv2d_32_conv2d_readvariableop_resource: ?
1model_3_conv2d_32_biasadd_readvariableop_resource: D
6model_3_batch_normalization_25_readvariableop_resource: F
8model_3_batch_normalization_25_readvariableop_1_resource: U
Gmodel_3_batch_normalization_25_fusedbatchnormv3_readvariableop_resource: W
Imodel_3_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource: J
0model_3_conv2d_33_conv2d_readvariableop_resource: @?
1model_3_conv2d_33_biasadd_readvariableop_resource:@D
6model_3_batch_normalization_26_readvariableop_resource:@F
8model_3_batch_normalization_26_readvariableop_1_resource:@U
Gmodel_3_batch_normalization_26_fusedbatchnormv3_readvariableop_resource:@W
Imodel_3_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:@J
0model_3_conv2d_34_conv2d_readvariableop_resource:@@?
1model_3_conv2d_34_biasadd_readvariableop_resource:@J
0model_3_conv2d_35_conv2d_readvariableop_resource: @?
1model_3_conv2d_35_biasadd_readvariableop_resource:@D
6model_3_batch_normalization_27_readvariableop_resource:@F
8model_3_batch_normalization_27_readvariableop_1_resource:@U
Gmodel_3_batch_normalization_27_fusedbatchnormv3_readvariableop_resource:@W
Imodel_3_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:@@
.model_3_dense_3_matmul_readvariableop_resource:@
=
/model_3_dense_3_biasadd_readvariableop_resource:

identity¢>model_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp¢@model_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1¢-model_3/batch_normalization_21/ReadVariableOp¢/model_3/batch_normalization_21/ReadVariableOp_1¢>model_3/batch_normalization_22/FusedBatchNormV3/ReadVariableOp¢@model_3/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1¢-model_3/batch_normalization_22/ReadVariableOp¢/model_3/batch_normalization_22/ReadVariableOp_1¢>model_3/batch_normalization_23/FusedBatchNormV3/ReadVariableOp¢@model_3/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1¢-model_3/batch_normalization_23/ReadVariableOp¢/model_3/batch_normalization_23/ReadVariableOp_1¢>model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp¢@model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1¢-model_3/batch_normalization_24/ReadVariableOp¢/model_3/batch_normalization_24/ReadVariableOp_1¢>model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp¢@model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1¢-model_3/batch_normalization_25/ReadVariableOp¢/model_3/batch_normalization_25/ReadVariableOp_1¢>model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp¢@model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1¢-model_3/batch_normalization_26/ReadVariableOp¢/model_3/batch_normalization_26/ReadVariableOp_1¢>model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp¢@model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1¢-model_3/batch_normalization_27/ReadVariableOp¢/model_3/batch_normalization_27/ReadVariableOp_1¢(model_3/conv2d_27/BiasAdd/ReadVariableOp¢'model_3/conv2d_27/Conv2D/ReadVariableOp¢(model_3/conv2d_28/BiasAdd/ReadVariableOp¢'model_3/conv2d_28/Conv2D/ReadVariableOp¢(model_3/conv2d_29/BiasAdd/ReadVariableOp¢'model_3/conv2d_29/Conv2D/ReadVariableOp¢(model_3/conv2d_30/BiasAdd/ReadVariableOp¢'model_3/conv2d_30/Conv2D/ReadVariableOp¢(model_3/conv2d_31/BiasAdd/ReadVariableOp¢'model_3/conv2d_31/Conv2D/ReadVariableOp¢(model_3/conv2d_32/BiasAdd/ReadVariableOp¢'model_3/conv2d_32/Conv2D/ReadVariableOp¢(model_3/conv2d_33/BiasAdd/ReadVariableOp¢'model_3/conv2d_33/Conv2D/ReadVariableOp¢(model_3/conv2d_34/BiasAdd/ReadVariableOp¢'model_3/conv2d_34/Conv2D/ReadVariableOp¢(model_3/conv2d_35/BiasAdd/ReadVariableOp¢'model_3/conv2d_35/Conv2D/ReadVariableOp¢&model_3/dense_3/BiasAdd/ReadVariableOp¢%model_3/dense_3/MatMul/ReadVariableOp 
'model_3/conv2d_27/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¾
model_3/conv2d_27/Conv2DConv2Dinput_4/model_3/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

(model_3/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0³
model_3/conv2d_27/BiasAddBiasAdd!model_3/conv2d_27/Conv2D:output:00model_3/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   
-model_3/batch_normalization_21/ReadVariableOpReadVariableOp6model_3_batch_normalization_21_readvariableop_resource*
_output_shapes
:*
dtype0¤
/model_3/batch_normalization_21/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_21_readvariableop_1_resource*
_output_shapes
:*
dtype0Â
>model_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Æ
@model_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0í
/model_3/batch_normalization_21/FusedBatchNormV3FusedBatchNormV3"model_3/conv2d_27/BiasAdd:output:05model_3/batch_normalization_21/ReadVariableOp:value:07model_3/batch_normalization_21/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
is_training( 
model_3/activation_21/ReluRelu3model_3/batch_normalization_21/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   
'model_3/conv2d_28/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ß
model_3/conv2d_28/Conv2DConv2D(model_3/activation_21/Relu:activations:0/model_3/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

(model_3/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0³
model_3/conv2d_28/BiasAddBiasAdd!model_3/conv2d_28/Conv2D:output:00model_3/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   
-model_3/batch_normalization_22/ReadVariableOpReadVariableOp6model_3_batch_normalization_22_readvariableop_resource*
_output_shapes
:*
dtype0¤
/model_3/batch_normalization_22/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_22_readvariableop_1_resource*
_output_shapes
:*
dtype0Â
>model_3/batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Æ
@model_3/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0í
/model_3/batch_normalization_22/FusedBatchNormV3FusedBatchNormV3"model_3/conv2d_28/BiasAdd:output:05model_3/batch_normalization_22/ReadVariableOp:value:07model_3/batch_normalization_22/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
is_training( 
model_3/activation_22/ReluRelu3model_3/batch_normalization_22/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   
'model_3/conv2d_29/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ß
model_3/conv2d_29/Conv2DConv2D(model_3/activation_22/Relu:activations:0/model_3/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

(model_3/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0³
model_3/conv2d_29/BiasAddBiasAdd!model_3/conv2d_29/Conv2D:output:00model_3/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   
-model_3/batch_normalization_23/ReadVariableOpReadVariableOp6model_3_batch_normalization_23_readvariableop_resource*
_output_shapes
:*
dtype0¤
/model_3/batch_normalization_23/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_23_readvariableop_1_resource*
_output_shapes
:*
dtype0Â
>model_3/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Æ
@model_3/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0í
/model_3/batch_normalization_23/FusedBatchNormV3FusedBatchNormV3"model_3/conv2d_29/BiasAdd:output:05model_3/batch_normalization_23/ReadVariableOp:value:07model_3/batch_normalization_23/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
is_training( ³
model_3/add_9/addAddV2(model_3/activation_21/Relu:activations:03model_3/batch_normalization_23/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  s
model_3/activation_23/ReluRelumodel_3/add_9/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   
'model_3/conv2d_30/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ß
model_3/conv2d_30/Conv2DConv2D(model_3/activation_23/Relu:activations:0/model_3/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

(model_3/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_30_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0³
model_3/conv2d_30/BiasAddBiasAdd!model_3/conv2d_30/Conv2D:output:00model_3/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
-model_3/batch_normalization_24/ReadVariableOpReadVariableOp6model_3_batch_normalization_24_readvariableop_resource*
_output_shapes
: *
dtype0¤
/model_3/batch_normalization_24/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_24_readvariableop_1_resource*
_output_shapes
: *
dtype0Â
>model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Æ
@model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0í
/model_3/batch_normalization_24/FusedBatchNormV3FusedBatchNormV3"model_3/conv2d_30/BiasAdd:output:05model_3/batch_normalization_24/ReadVariableOp:value:07model_3/batch_normalization_24/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 
model_3/activation_24/ReluRelu3model_3/batch_normalization_24/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
'model_3/conv2d_31/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0ß
model_3/conv2d_31/Conv2DConv2D(model_3/activation_24/Relu:activations:0/model_3/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

(model_3/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0³
model_3/conv2d_31/BiasAddBiasAdd!model_3/conv2d_31/Conv2D:output:00model_3/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
'model_3/conv2d_32/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ß
model_3/conv2d_32/Conv2DConv2D(model_3/activation_23/Relu:activations:0/model_3/conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

(model_3/conv2d_32/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0³
model_3/conv2d_32/BiasAddBiasAdd!model_3/conv2d_32/Conv2D:output:00model_3/conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
-model_3/batch_normalization_25/ReadVariableOpReadVariableOp6model_3_batch_normalization_25_readvariableop_resource*
_output_shapes
: *
dtype0¤
/model_3/batch_normalization_25/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_25_readvariableop_1_resource*
_output_shapes
: *
dtype0Â
>model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Æ
@model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0í
/model_3/batch_normalization_25/FusedBatchNormV3FusedBatchNormV3"model_3/conv2d_31/BiasAdd:output:05model_3/batch_normalization_25/ReadVariableOp:value:07model_3/batch_normalization_25/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( ®
model_3/add_10/addAddV2"model_3/conv2d_32/BiasAdd:output:03model_3/batch_normalization_25/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
model_3/activation_25/ReluRelumodel_3/add_10/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
'model_3/conv2d_33/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0ß
model_3/conv2d_33/Conv2DConv2D(model_3/activation_25/Relu:activations:0/model_3/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

(model_3/conv2d_33/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
model_3/conv2d_33/BiasAddBiasAdd!model_3/conv2d_33/Conv2D:output:00model_3/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
-model_3/batch_normalization_26/ReadVariableOpReadVariableOp6model_3_batch_normalization_26_readvariableop_resource*
_output_shapes
:@*
dtype0¤
/model_3/batch_normalization_26/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_26_readvariableop_1_resource*
_output_shapes
:@*
dtype0Â
>model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Æ
@model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0í
/model_3/batch_normalization_26/FusedBatchNormV3FusedBatchNormV3"model_3/conv2d_33/BiasAdd:output:05model_3/batch_normalization_26/ReadVariableOp:value:07model_3/batch_normalization_26/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 
model_3/activation_26/ReluRelu3model_3/batch_normalization_26/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
'model_3/conv2d_34/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ß
model_3/conv2d_34/Conv2DConv2D(model_3/activation_26/Relu:activations:0/model_3/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

(model_3/conv2d_34/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
model_3/conv2d_34/BiasAddBiasAdd!model_3/conv2d_34/Conv2D:output:00model_3/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
'model_3/conv2d_35/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0ß
model_3/conv2d_35/Conv2DConv2D(model_3/activation_25/Relu:activations:0/model_3/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

(model_3/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
model_3/conv2d_35/BiasAddBiasAdd!model_3/conv2d_35/Conv2D:output:00model_3/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
-model_3/batch_normalization_27/ReadVariableOpReadVariableOp6model_3_batch_normalization_27_readvariableop_resource*
_output_shapes
:@*
dtype0¤
/model_3/batch_normalization_27/ReadVariableOp_1ReadVariableOp8model_3_batch_normalization_27_readvariableop_1_resource*
_output_shapes
:@*
dtype0Â
>model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_3_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Æ
@model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_3_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0í
/model_3/batch_normalization_27/FusedBatchNormV3FusedBatchNormV3"model_3/conv2d_34/BiasAdd:output:05model_3/batch_normalization_27/ReadVariableOp:value:07model_3/batch_normalization_27/ReadVariableOp_1:value:0Fmodel_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( ®
model_3/add_11/addAddV2"model_3/conv2d_35/BiasAdd:output:03model_3/batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
model_3/activation_27/ReluRelumodel_3/add_11/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
#model_3/average_pooling2d_3/AvgPoolAvgPool(model_3/activation_27/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
h
model_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¦
model_3/flatten_3/ReshapeReshape,model_3/average_pooling2d_3/AvgPool:output:0 model_3/flatten_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%model_3/dense_3/MatMul/ReadVariableOpReadVariableOp.model_3_dense_3_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype0¥
model_3/dense_3/MatMulMatMul"model_3/flatten_3/Reshape:output:0-model_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&model_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_3_dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0¦
model_3/dense_3/BiasAddBiasAdd model_3/dense_3/MatMul:product:0.model_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
IdentityIdentity model_3/dense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Þ
NoOpNoOp?^model_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_21/ReadVariableOp0^model_3/batch_normalization_21/ReadVariableOp_1?^model_3/batch_normalization_22/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_22/ReadVariableOp0^model_3/batch_normalization_22/ReadVariableOp_1?^model_3/batch_normalization_23/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_23/ReadVariableOp0^model_3/batch_normalization_23/ReadVariableOp_1?^model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_24/ReadVariableOp0^model_3/batch_normalization_24/ReadVariableOp_1?^model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_25/ReadVariableOp0^model_3/batch_normalization_25/ReadVariableOp_1?^model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_26/ReadVariableOp0^model_3/batch_normalization_26/ReadVariableOp_1?^model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOpA^model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1.^model_3/batch_normalization_27/ReadVariableOp0^model_3/batch_normalization_27/ReadVariableOp_1)^model_3/conv2d_27/BiasAdd/ReadVariableOp(^model_3/conv2d_27/Conv2D/ReadVariableOp)^model_3/conv2d_28/BiasAdd/ReadVariableOp(^model_3/conv2d_28/Conv2D/ReadVariableOp)^model_3/conv2d_29/BiasAdd/ReadVariableOp(^model_3/conv2d_29/Conv2D/ReadVariableOp)^model_3/conv2d_30/BiasAdd/ReadVariableOp(^model_3/conv2d_30/Conv2D/ReadVariableOp)^model_3/conv2d_31/BiasAdd/ReadVariableOp(^model_3/conv2d_31/Conv2D/ReadVariableOp)^model_3/conv2d_32/BiasAdd/ReadVariableOp(^model_3/conv2d_32/Conv2D/ReadVariableOp)^model_3/conv2d_33/BiasAdd/ReadVariableOp(^model_3/conv2d_33/Conv2D/ReadVariableOp)^model_3/conv2d_34/BiasAdd/ReadVariableOp(^model_3/conv2d_34/Conv2D/ReadVariableOp)^model_3/conv2d_35/BiasAdd/ReadVariableOp(^model_3/conv2d_35/Conv2D/ReadVariableOp'^model_3/dense_3/BiasAdd/ReadVariableOp&^model_3/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
>model_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp2
@model_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_21/ReadVariableOp-model_3/batch_normalization_21/ReadVariableOp2b
/model_3/batch_normalization_21/ReadVariableOp_1/model_3/batch_normalization_21/ReadVariableOp_12
>model_3/batch_normalization_22/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_22/FusedBatchNormV3/ReadVariableOp2
@model_3/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_22/ReadVariableOp-model_3/batch_normalization_22/ReadVariableOp2b
/model_3/batch_normalization_22/ReadVariableOp_1/model_3/batch_normalization_22/ReadVariableOp_12
>model_3/batch_normalization_23/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2
@model_3/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_23/ReadVariableOp-model_3/batch_normalization_23/ReadVariableOp2b
/model_3/batch_normalization_23/ReadVariableOp_1/model_3/batch_normalization_23/ReadVariableOp_12
>model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp2
@model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_24/ReadVariableOp-model_3/batch_normalization_24/ReadVariableOp2b
/model_3/batch_normalization_24/ReadVariableOp_1/model_3/batch_normalization_24/ReadVariableOp_12
>model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2
@model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_25/ReadVariableOp-model_3/batch_normalization_25/ReadVariableOp2b
/model_3/batch_normalization_25/ReadVariableOp_1/model_3/batch_normalization_25/ReadVariableOp_12
>model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2
@model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_26/ReadVariableOp-model_3/batch_normalization_26/ReadVariableOp2b
/model_3/batch_normalization_26/ReadVariableOp_1/model_3/batch_normalization_26/ReadVariableOp_12
>model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp>model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2
@model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1@model_3/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12^
-model_3/batch_normalization_27/ReadVariableOp-model_3/batch_normalization_27/ReadVariableOp2b
/model_3/batch_normalization_27/ReadVariableOp_1/model_3/batch_normalization_27/ReadVariableOp_12T
(model_3/conv2d_27/BiasAdd/ReadVariableOp(model_3/conv2d_27/BiasAdd/ReadVariableOp2R
'model_3/conv2d_27/Conv2D/ReadVariableOp'model_3/conv2d_27/Conv2D/ReadVariableOp2T
(model_3/conv2d_28/BiasAdd/ReadVariableOp(model_3/conv2d_28/BiasAdd/ReadVariableOp2R
'model_3/conv2d_28/Conv2D/ReadVariableOp'model_3/conv2d_28/Conv2D/ReadVariableOp2T
(model_3/conv2d_29/BiasAdd/ReadVariableOp(model_3/conv2d_29/BiasAdd/ReadVariableOp2R
'model_3/conv2d_29/Conv2D/ReadVariableOp'model_3/conv2d_29/Conv2D/ReadVariableOp2T
(model_3/conv2d_30/BiasAdd/ReadVariableOp(model_3/conv2d_30/BiasAdd/ReadVariableOp2R
'model_3/conv2d_30/Conv2D/ReadVariableOp'model_3/conv2d_30/Conv2D/ReadVariableOp2T
(model_3/conv2d_31/BiasAdd/ReadVariableOp(model_3/conv2d_31/BiasAdd/ReadVariableOp2R
'model_3/conv2d_31/Conv2D/ReadVariableOp'model_3/conv2d_31/Conv2D/ReadVariableOp2T
(model_3/conv2d_32/BiasAdd/ReadVariableOp(model_3/conv2d_32/BiasAdd/ReadVariableOp2R
'model_3/conv2d_32/Conv2D/ReadVariableOp'model_3/conv2d_32/Conv2D/ReadVariableOp2T
(model_3/conv2d_33/BiasAdd/ReadVariableOp(model_3/conv2d_33/BiasAdd/ReadVariableOp2R
'model_3/conv2d_33/Conv2D/ReadVariableOp'model_3/conv2d_33/Conv2D/ReadVariableOp2T
(model_3/conv2d_34/BiasAdd/ReadVariableOp(model_3/conv2d_34/BiasAdd/ReadVariableOp2R
'model_3/conv2d_34/Conv2D/ReadVariableOp'model_3/conv2d_34/Conv2D/ReadVariableOp2T
(model_3/conv2d_35/BiasAdd/ReadVariableOp(model_3/conv2d_35/BiasAdd/ReadVariableOp2R
'model_3/conv2d_35/Conv2D/ReadVariableOp'model_3/conv2d_35/Conv2D/ReadVariableOp2P
&model_3/dense_3/BiasAdd/ReadVariableOp&model_3/dense_3/BiasAdd/ReadVariableOp2N
%model_3/dense_3/MatMul/ReadVariableOp%model_3/dense_3/MatMul/ReadVariableOp:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_4"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*²
serving_default
C
input_48
serving_default_input_4:0ÿÿÿÿÿÿÿÿÿ  ;
dense_30
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:Óï
¯
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer-17
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer-21
layer_with_weights-13
layer-22
layer_with_weights-14
layer-23
layer_with_weights-15
layer-24
layer-25
layer-26
layer-27
layer-28
layer_with_weights-16
layer-29
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%_default_save_signature
&
signatures"
_tf_keras_network
"
_tf_keras_input_layer
»

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
»

@kernel
Abias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
Haxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layer
»

xkernel
ybias
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	¡axis

¢gamma
	£beta
¤moving_mean
¥moving_variance
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
ª__call__
+«&call_and_return_all_conditional_losses"
_tf_keras_layer
«
¬	variables
­trainable_variables
®regularization_losses
¯	keras_api
°__call__
+±&call_and_return_all_conditional_losses"
_tf_keras_layer
«
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
¸kernel
	¹bias
º	variables
»trainable_variables
¼regularization_losses
½	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	Àaxis

Ágamma
	Âbeta
Ãmoving_mean
Ämoving_variance
Å	variables
Ætrainable_variables
Çregularization_losses
È	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Ñkernel
	Òbias
Ó	variables
Ôtrainable_variables
Õregularization_losses
Ö	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Ùkernel
	Úbias
Û	variables
Ütrainable_variables
Ýregularization_losses
Þ	keras_api
ß__call__
+à&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	áaxis

âgamma
	ãbeta
ämoving_mean
åmoving_variance
æ	variables
çtrainable_variables
èregularization_losses
é	keras_api
ê__call__
+ë&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ì	variables
ítrainable_variables
îregularization_losses
ï	keras_api
ð__call__
+ñ&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ò	variables
ótrainable_variables
ôregularization_losses
õ	keras_api
ö__call__
+÷&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ø	variables
ùtrainable_variables
úregularization_losses
û	keras_api
ü__call__
+ý&call_and_return_all_conditional_losses"
_tf_keras_layer
«
þ	variables
ÿtrainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
²
'0
(1
02
13
24
35
@6
A7
I8
J9
K10
L11
Y12
Z13
b14
c15
d16
e17
x18
y19
20
21
22
23
24
25
26
27
¢28
£29
¤30
¥31
¸32
¹33
Á34
Â35
Ã36
Ä37
Ñ38
Ò39
Ù40
Ú41
â42
ã43
ä44
å45
46
47"
trackable_list_wrapper
º
'0
(1
02
13
@4
A5
I6
J7
Y8
Z9
b10
c11
x12
y13
14
15
16
17
18
19
¢20
£21
¸22
¹23
Á24
Â25
Ñ26
Ò27
Ù28
Ú29
â30
ã31
32
33"
trackable_list_wrapper
h
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
Ï
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
%_default_save_signature
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
ö2ó
*__inference_model_3_layer_call_fn_13427691
*__inference_model_3_layer_call_fn_13428867
*__inference_model_3_layer_call_fn_13428968
*__inference_model_3_layer_call_fn_13428346À
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
â2ß
E__inference_model_3_layer_call_and_return_conditional_losses_13429197
E__inference_model_3_layer_call_and_return_conditional_losses_13429426
E__inference_model_3_layer_call_and_return_conditional_losses_13428529
E__inference_model_3_layer_call_and_return_conditional_losses_13428712À
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
ÎBË
#__inference__wrapped_model_13426711input_4"
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
-
serving_default"
signature_map
*:(2conv2d_27/kernel
:2conv2d_27/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_conv2d_27_layer_call_fn_13429544¢
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
G__inference_conv2d_27_layer_call_and_return_conditional_losses_13429560¢
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
*:(2batch_normalization_21/gamma
):'2batch_normalization_21/beta
2:0 (2"batch_normalization_21/moving_mean
6:4 (2&batch_normalization_21/moving_variance
<
00
11
22
33"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
²
 non_trainable_variables
¡layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_21_layer_call_fn_13429573
9__inference_batch_normalization_21_layer_call_fn_13429586´
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
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_13429604
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_13429622´
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
p 

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
¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_activation_21_layer_call_fn_13429627¢
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
õ2ò
K__inference_activation_21_layer_call_and_return_conditional_losses_13429632¢
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
*:(2conv2d_28/kernel
:2conv2d_28/bias
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
²
ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_conv2d_28_layer_call_fn_13429647¢
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
G__inference_conv2d_28_layer_call_and_return_conditional_losses_13429663¢
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
*:(2batch_normalization_22/gamma
):'2batch_normalization_22/beta
2:0 (2"batch_normalization_22/moving_mean
6:4 (2&batch_normalization_22/moving_variance
<
I0
J1
K2
L3"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_22_layer_call_fn_13429676
9__inference_batch_normalization_22_layer_call_fn_13429689´
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
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_13429707
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_13429725´
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
p 

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
´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_activation_22_layer_call_fn_13429730¢
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
õ2ò
K__inference_activation_22_layer_call_and_return_conditional_losses_13429735¢
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
*:(2conv2d_29/kernel
:2conv2d_29/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
²
¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_conv2d_29_layer_call_fn_13429750¢
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
G__inference_conv2d_29_layer_call_and_return_conditional_losses_13429766¢
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
*:(2batch_normalization_23/gamma
):'2batch_normalization_23/beta
2:0 (2"batch_normalization_23/moving_mean
6:4 (2&batch_normalization_23/moving_variance
<
b0
c1
d2
e3"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¾non_trainable_variables
¿layers
Àmetrics
 Álayer_regularization_losses
Âlayer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_23_layer_call_fn_13429779
9__inference_batch_normalization_23_layer_call_fn_13429792´
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
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_13429810
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_13429828´
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
p 

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
Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_add_9_layer_call_fn_13429834¢
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
í2ê
C__inference_add_9_layer_call_and_return_conditional_losses_13429840¢
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
Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_activation_23_layer_call_fn_13429845¢
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
õ2ò
K__inference_activation_23_layer_call_and_return_conditional_losses_13429850¢
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
*:( 2conv2d_30/kernel
: 2conv2d_30/bias
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
²
Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_conv2d_30_layer_call_fn_13429865¢
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
G__inference_conv2d_30_layer_call_and_return_conditional_losses_13429881¢
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
*:( 2batch_normalization_24/gamma
):' 2batch_normalization_24/beta
2:0  (2"batch_normalization_24/moving_mean
6:4  (2&batch_normalization_24/moving_variance
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_24_layer_call_fn_13429894
9__inference_batch_normalization_24_layer_call_fn_13429907´
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
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_13429925
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_13429943´
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
p 

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
¸
×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_activation_24_layer_call_fn_13429948¢
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
õ2ò
K__inference_activation_24_layer_call_and_return_conditional_losses_13429953¢
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
*:(  2conv2d_31/kernel
: 2conv2d_31/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
¸
Ünon_trainable_variables
Ýlayers
Þmetrics
 ßlayer_regularization_losses
àlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_conv2d_31_layer_call_fn_13429968¢
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
G__inference_conv2d_31_layer_call_and_return_conditional_losses_13429984¢
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
*:( 2conv2d_32/kernel
: 2conv2d_32/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
¸
ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_conv2d_32_layer_call_fn_13429999¢
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
G__inference_conv2d_32_layer_call_and_return_conditional_losses_13430015¢
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
*:( 2batch_normalization_25/gamma
):' 2batch_normalization_25/beta
2:0  (2"batch_normalization_25/moving_mean
6:4  (2&batch_normalization_25/moving_variance
@
¢0
£1
¤2
¥3"
trackable_list_wrapper
0
¢0
£1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
¦	variables
§trainable_variables
¨regularization_losses
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_25_layer_call_fn_13430028
9__inference_batch_normalization_25_layer_call_fn_13430041´
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
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_13430059
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_13430077´
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
p 

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
¸
ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
¬	variables
­trainable_variables
®regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_add_10_layer_call_fn_13430083¢
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
î2ë
D__inference_add_10_layer_call_and_return_conditional_losses_13430089¢
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
¸
ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_activation_25_layer_call_fn_13430094¢
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
õ2ò
K__inference_activation_25_layer_call_and_return_conditional_losses_13430099¢
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
*:( @2conv2d_33/kernel
:@2conv2d_33/bias
0
¸0
¹1"
trackable_list_wrapper
0
¸0
¹1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
¸
õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
º	variables
»trainable_variables
¼regularization_losses
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_conv2d_33_layer_call_fn_13430114¢
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
G__inference_conv2d_33_layer_call_and_return_conditional_losses_13430130¢
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
*:(@2batch_normalization_26/gamma
):'@2batch_normalization_26/beta
2:0@ (2"batch_normalization_26/moving_mean
6:4@ (2&batch_normalization_26/moving_variance
@
Á0
Â1
Ã2
Ä3"
trackable_list_wrapper
0
Á0
Â1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
Å	variables
Ætrainable_variables
Çregularization_losses
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_26_layer_call_fn_13430143
9__inference_batch_normalization_26_layer_call_fn_13430156´
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
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_13430174
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_13430192´
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
p 

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
¸
ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ë	variables
Ìtrainable_variables
Íregularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_activation_26_layer_call_fn_13430197¢
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
õ2ò
K__inference_activation_26_layer_call_and_return_conditional_losses_13430202¢
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
*:(@@2conv2d_34/kernel
:@2conv2d_34/bias
0
Ñ0
Ò1"
trackable_list_wrapper
0
Ñ0
Ò1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ó	variables
Ôtrainable_variables
Õregularization_losses
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_conv2d_34_layer_call_fn_13430217¢
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
G__inference_conv2d_34_layer_call_and_return_conditional_losses_13430233¢
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
*:( @2conv2d_35/kernel
:@2conv2d_35/bias
0
Ù0
Ú1"
trackable_list_wrapper
0
Ù0
Ú1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Û	variables
Ütrainable_variables
Ýregularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_conv2d_35_layer_call_fn_13430248¢
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
G__inference_conv2d_35_layer_call_and_return_conditional_losses_13430264¢
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
*:(@2batch_normalization_27/gamma
):'@2batch_normalization_27/beta
2:0@ (2"batch_normalization_27/moving_mean
6:4@ (2&batch_normalization_27/moving_variance
@
â0
ã1
ä2
å3"
trackable_list_wrapper
0
â0
ã1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
æ	variables
çtrainable_variables
èregularization_losses
ê__call__
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_27_layer_call_fn_13430277
9__inference_batch_normalization_27_layer_call_fn_13430290´
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
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_13430308
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_13430326´
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
p 

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
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ì	variables
ítrainable_variables
îregularization_losses
ð__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_add_11_layer_call_fn_13430332¢
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
î2ë
D__inference_add_11_layer_call_and_return_conditional_losses_13430338¢
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
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ò	variables
ótrainable_variables
ôregularization_losses
ö__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_activation_27_layer_call_fn_13430343¢
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
õ2ò
K__inference_activation_27_layer_call_and_return_conditional_losses_13430348¢
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
¸
non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
ø	variables
ùtrainable_variables
úregularization_losses
ü__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
à2Ý
6__inference_average_pooling2d_3_layer_call_fn_13430353¢
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
û2ø
Q__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_13430358¢
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
¸
¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
þ	variables
ÿtrainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_flatten_3_layer_call_fn_13430363¢
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
G__inference_flatten_3_layer_call_and_return_conditional_losses_13430369¢
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
 :@
2dense_3/kernel
:
2dense_3/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_3_layer_call_fn_13430378¢
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
ï2ì
E__inference_dense_3_layer_call_and_return_conditional_losses_13430388¢
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
µ2²
__inference_loss_fn_0_13430399
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_1_13430410
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_2_13430421
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_3_13430432
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_4_13430443
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_5_13430454
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_6_13430465
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_7_13430476
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
µ2²
__inference_loss_fn_8_13430487
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 

20
31
K2
L3
d4
e5
6
7
¤8
¥9
Ã10
Ä11
ä12
å13"
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÍBÊ
&__inference_signature_wrapper_13429529input_4"
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
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
20
31"
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
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
K0
L1"
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
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
d0
e1"
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
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
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
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
¤0
¥1"
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
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ã0
Ä1"
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
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
ä0
å1"
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
trackable_dict_wrapperã
#__inference__wrapped_model_13426711»L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå8¢5
.¢+
)&
input_4ÿÿÿÿÿÿÿÿÿ  
ª "1ª.
,
dense_3!
dense_3ÿÿÿÿÿÿÿÿÿ
·
K__inference_activation_21_layer_call_and_return_conditional_losses_13429632h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
0__inference_activation_21_layer_call_fn_13429627[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ·
K__inference_activation_22_layer_call_and_return_conditional_losses_13429735h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
0__inference_activation_22_layer_call_fn_13429730[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ·
K__inference_activation_23_layer_call_and_return_conditional_losses_13429850h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
0__inference_activation_23_layer_call_fn_13429845[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ·
K__inference_activation_24_layer_call_and_return_conditional_losses_13429953h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
0__inference_activation_24_layer_call_fn_13429948[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ ·
K__inference_activation_25_layer_call_and_return_conditional_losses_13430099h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
0__inference_activation_25_layer_call_fn_13430094[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ ·
K__inference_activation_26_layer_call_and_return_conditional_losses_13430202h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
0__inference_activation_26_layer_call_fn_13430197[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@·
K__inference_activation_27_layer_call_and_return_conditional_losses_13430348h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
0__inference_activation_27_layer_call_fn_13430343[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@ä
D__inference_add_10_layer_call_and_return_conditional_losses_13430089j¢g
`¢]
[X
*'
inputs/0ÿÿÿÿÿÿÿÿÿ 
*'
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 ¼
)__inference_add_10_layer_call_fn_13430083j¢g
`¢]
[X
*'
inputs/0ÿÿÿÿÿÿÿÿÿ 
*'
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ ä
D__inference_add_11_layer_call_and_return_conditional_losses_13430338j¢g
`¢]
[X
*'
inputs/0ÿÿÿÿÿÿÿÿÿ@
*'
inputs/1ÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 ¼
)__inference_add_11_layer_call_fn_13430332j¢g
`¢]
[X
*'
inputs/0ÿÿÿÿÿÿÿÿÿ@
*'
inputs/1ÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@ã
C__inference_add_9_layer_call_and_return_conditional_losses_13429840j¢g
`¢]
[X
*'
inputs/0ÿÿÿÿÿÿÿÿÿ  
*'
inputs/1ÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 »
(__inference_add_9_layer_call_fn_13429834j¢g
`¢]
[X
*'
inputs/0ÿÿÿÿÿÿÿÿÿ  
*'
inputs/1ÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ô
Q__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_13430358R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ì
6__inference_average_pooling2d_3_layer_call_fn_13430353R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_134296040123M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ï
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_134296220123M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
9__inference_batch_normalization_21_layer_call_fn_134295730123M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÇ
9__inference_batch_normalization_21_layer_call_fn_134295860123M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_13429707IJKLM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ï
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_13429725IJKLM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
9__inference_batch_normalization_22_layer_call_fn_13429676IJKLM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÇ
9__inference_batch_normalization_22_layer_call_fn_13429689IJKLM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_13429810bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ï
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_13429828bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
9__inference_batch_normalization_23_layer_call_fn_13429779bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÇ
9__inference_batch_normalization_23_layer_call_fn_13429792bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿó
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_13429925M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ó
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_13429943M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ë
9__inference_batch_normalization_24_layer_call_fn_13429894M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ë
9__inference_batch_normalization_24_layer_call_fn_13429907M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ó
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_13430059¢£¤¥M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ó
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_13430077¢£¤¥M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ë
9__inference_batch_normalization_25_layer_call_fn_13430028¢£¤¥M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ë
9__inference_batch_normalization_25_layer_call_fn_13430041¢£¤¥M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ó
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_13430174ÁÂÃÄM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ó
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_13430192ÁÂÃÄM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ë
9__inference_batch_normalization_26_layer_call_fn_13430143ÁÂÃÄM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ë
9__inference_batch_normalization_26_layer_call_fn_13430156ÁÂÃÄM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ó
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_13430308âãäåM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ó
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_13430326âãäåM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ë
9__inference_batch_normalization_27_layer_call_fn_13430277âãäåM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ë
9__inference_batch_normalization_27_layer_call_fn_13430290âãäåM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@·
G__inference_conv2d_27_layer_call_and_return_conditional_losses_13429560l'(7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
,__inference_conv2d_27_layer_call_fn_13429544_'(7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ·
G__inference_conv2d_28_layer_call_and_return_conditional_losses_13429663l@A7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
,__inference_conv2d_28_layer_call_fn_13429647_@A7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ·
G__inference_conv2d_29_layer_call_and_return_conditional_losses_13429766lYZ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
,__inference_conv2d_29_layer_call_fn_13429750_YZ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ·
G__inference_conv2d_30_layer_call_and_return_conditional_losses_13429881lxy7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_conv2d_30_layer_call_fn_13429865_xy7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ ¹
G__inference_conv2d_31_layer_call_and_return_conditional_losses_13429984n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_conv2d_31_layer_call_fn_13429968a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ ¹
G__inference_conv2d_32_layer_call_and_return_conditional_losses_13430015n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_conv2d_32_layer_call_fn_13429999a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ ¹
G__inference_conv2d_33_layer_call_and_return_conditional_losses_13430130n¸¹7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_conv2d_33_layer_call_fn_13430114a¸¹7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ@¹
G__inference_conv2d_34_layer_call_and_return_conditional_losses_13430233nÑÒ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_conv2d_34_layer_call_fn_13430217aÑÒ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@¹
G__inference_conv2d_35_layer_call_and_return_conditional_losses_13430264nÙÚ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_conv2d_35_layer_call_fn_13430248aÙÚ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ@§
E__inference_dense_3_layer_call_and_return_conditional_losses_13430388^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
*__inference_dense_3_layer_call_fn_13430378Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ
«
G__inference_flatten_3_layer_call_and_return_conditional_losses_13430369`7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_flatten_3_layer_call_fn_13430363S7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@=
__inference_loss_fn_0_13430399'¢

¢ 
ª " =
__inference_loss_fn_1_13430410@¢

¢ 
ª " =
__inference_loss_fn_2_13430421Y¢

¢ 
ª " =
__inference_loss_fn_3_13430432x¢

¢ 
ª " >
__inference_loss_fn_4_13430443¢

¢ 
ª " >
__inference_loss_fn_5_13430454¢

¢ 
ª " >
__inference_loss_fn_6_13430465¸¢

¢ 
ª " >
__inference_loss_fn_7_13430476Ñ¢

¢ 
ª " >
__inference_loss_fn_8_13430487Ù¢

¢ 
ª " 
E__inference_model_3_layer_call_and_return_conditional_losses_13428529·L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå@¢=
6¢3
)&
input_4ÿÿÿÿÿÿÿÿÿ  
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
E__inference_model_3_layer_call_and_return_conditional_losses_13428712·L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå@¢=
6¢3
)&
input_4ÿÿÿÿÿÿÿÿÿ  
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
E__inference_model_3_layer_call_and_return_conditional_losses_13429197¶L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå?¢<
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
 
E__inference_model_3_layer_call_and_return_conditional_losses_13429426¶L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå?¢<
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
 Ù
*__inference_model_3_layer_call_fn_13427691ªL'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå@¢=
6¢3
)&
input_4ÿÿÿÿÿÿÿÿÿ  
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
Ù
*__inference_model_3_layer_call_fn_13428346ªL'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå@¢=
6¢3
)&
input_4ÿÿÿÿÿÿÿÿÿ  
p

 
ª "ÿÿÿÿÿÿÿÿÿ
Ø
*__inference_model_3_layer_call_fn_13428867©L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
Ø
*__inference_model_3_layer_call_fn_13428968©L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "ÿÿÿÿÿÿÿÿÿ
ñ
&__inference_signature_wrapper_13429529ÆL'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäåC¢@
¢ 
9ª6
4
input_4)&
input_4ÿÿÿÿÿÿÿÿÿ  "1ª.
,
dense_3!
dense_3ÿÿÿÿÿÿÿÿÿ
