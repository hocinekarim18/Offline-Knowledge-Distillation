ñ"
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Ýü

conv2d_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_54/kernel
}
$conv2d_54/kernel/Read/ReadVariableOpReadVariableOpconv2d_54/kernel*&
_output_shapes
:*
dtype0
t
conv2d_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_54/bias
m
"conv2d_54/bias/Read/ReadVariableOpReadVariableOpconv2d_54/bias*
_output_shapes
:*
dtype0

batch_normalization_42/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_42/gamma

0batch_normalization_42/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_42/gamma*
_output_shapes
:*
dtype0

batch_normalization_42/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_42/beta

/batch_normalization_42/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_42/beta*
_output_shapes
:*
dtype0

"batch_normalization_42/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_42/moving_mean

6batch_normalization_42/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_42/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_42/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_42/moving_variance

:batch_normalization_42/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_42/moving_variance*
_output_shapes
:*
dtype0

conv2d_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_55/kernel
}
$conv2d_55/kernel/Read/ReadVariableOpReadVariableOpconv2d_55/kernel*&
_output_shapes
:*
dtype0
t
conv2d_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_55/bias
m
"conv2d_55/bias/Read/ReadVariableOpReadVariableOpconv2d_55/bias*
_output_shapes
:*
dtype0

batch_normalization_43/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_43/gamma

0batch_normalization_43/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_43/gamma*
_output_shapes
:*
dtype0

batch_normalization_43/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_43/beta

/batch_normalization_43/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_43/beta*
_output_shapes
:*
dtype0

"batch_normalization_43/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_43/moving_mean

6batch_normalization_43/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_43/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_43/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_43/moving_variance

:batch_normalization_43/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_43/moving_variance*
_output_shapes
:*
dtype0

conv2d_56/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_56/kernel
}
$conv2d_56/kernel/Read/ReadVariableOpReadVariableOpconv2d_56/kernel*&
_output_shapes
:*
dtype0
t
conv2d_56/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_56/bias
m
"conv2d_56/bias/Read/ReadVariableOpReadVariableOpconv2d_56/bias*
_output_shapes
:*
dtype0

batch_normalization_44/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_44/gamma

0batch_normalization_44/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_44/gamma*
_output_shapes
:*
dtype0

batch_normalization_44/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_44/beta

/batch_normalization_44/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_44/beta*
_output_shapes
:*
dtype0

"batch_normalization_44/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_44/moving_mean

6batch_normalization_44/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_44/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_44/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_44/moving_variance

:batch_normalization_44/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_44/moving_variance*
_output_shapes
:*
dtype0

conv2d_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_57/kernel
}
$conv2d_57/kernel/Read/ReadVariableOpReadVariableOpconv2d_57/kernel*&
_output_shapes
: *
dtype0
t
conv2d_57/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_57/bias
m
"conv2d_57/bias/Read/ReadVariableOpReadVariableOpconv2d_57/bias*
_output_shapes
: *
dtype0

batch_normalization_45/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_45/gamma

0batch_normalization_45/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_45/gamma*
_output_shapes
: *
dtype0

batch_normalization_45/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_45/beta

/batch_normalization_45/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_45/beta*
_output_shapes
: *
dtype0

"batch_normalization_45/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_45/moving_mean

6batch_normalization_45/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_45/moving_mean*
_output_shapes
: *
dtype0
¤
&batch_normalization_45/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_45/moving_variance

:batch_normalization_45/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_45/moving_variance*
_output_shapes
: *
dtype0

conv2d_58/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_58/kernel
}
$conv2d_58/kernel/Read/ReadVariableOpReadVariableOpconv2d_58/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_58/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_58/bias
m
"conv2d_58/bias/Read/ReadVariableOpReadVariableOpconv2d_58/bias*
_output_shapes
: *
dtype0

conv2d_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_59/kernel
}
$conv2d_59/kernel/Read/ReadVariableOpReadVariableOpconv2d_59/kernel*&
_output_shapes
: *
dtype0
t
conv2d_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_59/bias
m
"conv2d_59/bias/Read/ReadVariableOpReadVariableOpconv2d_59/bias*
_output_shapes
: *
dtype0

batch_normalization_46/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_46/gamma

0batch_normalization_46/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_46/gamma*
_output_shapes
: *
dtype0

batch_normalization_46/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_46/beta

/batch_normalization_46/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_46/beta*
_output_shapes
: *
dtype0

"batch_normalization_46/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_46/moving_mean

6batch_normalization_46/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_46/moving_mean*
_output_shapes
: *
dtype0
¤
&batch_normalization_46/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_46/moving_variance

:batch_normalization_46/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_46/moving_variance*
_output_shapes
: *
dtype0

conv2d_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_60/kernel
}
$conv2d_60/kernel/Read/ReadVariableOpReadVariableOpconv2d_60/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_60/bias
m
"conv2d_60/bias/Read/ReadVariableOpReadVariableOpconv2d_60/bias*
_output_shapes
:@*
dtype0

batch_normalization_47/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_47/gamma

0batch_normalization_47/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_47/gamma*
_output_shapes
:@*
dtype0

batch_normalization_47/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_47/beta

/batch_normalization_47/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_47/beta*
_output_shapes
:@*
dtype0

"batch_normalization_47/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_47/moving_mean

6batch_normalization_47/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_47/moving_mean*
_output_shapes
:@*
dtype0
¤
&batch_normalization_47/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_47/moving_variance

:batch_normalization_47/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_47/moving_variance*
_output_shapes
:@*
dtype0

conv2d_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_61/kernel
}
$conv2d_61/kernel/Read/ReadVariableOpReadVariableOpconv2d_61/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_61/bias
m
"conv2d_61/bias/Read/ReadVariableOpReadVariableOpconv2d_61/bias*
_output_shapes
:@*
dtype0

conv2d_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_62/kernel
}
$conv2d_62/kernel/Read/ReadVariableOpReadVariableOpconv2d_62/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_62/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_62/bias
m
"conv2d_62/bias/Read/ReadVariableOpReadVariableOpconv2d_62/bias*
_output_shapes
:@*
dtype0

batch_normalization_48/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_48/gamma

0batch_normalization_48/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_48/gamma*
_output_shapes
:@*
dtype0

batch_normalization_48/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_48/beta

/batch_normalization_48/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_48/beta*
_output_shapes
:@*
dtype0

"batch_normalization_48/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_48/moving_mean

6batch_normalization_48/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_48/moving_mean*
_output_shapes
:@*
dtype0
¤
&batch_normalization_48/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_48/moving_variance

:batch_normalization_48/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_48/moving_variance*
_output_shapes
:@*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:@
*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
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
VARIABLE_VALUEconv2d_54/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_54/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_42/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_42/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_42/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_42/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
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
VARIABLE_VALUEconv2d_55/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_55/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_43/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_43/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_43/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_43/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
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
VARIABLE_VALUEconv2d_56/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_56/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_44/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_44/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_44/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_44/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
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
VARIABLE_VALUEconv2d_57/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_57/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_45/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_45/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_45/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_45/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_58/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_58/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_59/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_59/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_46/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_46/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_46/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_46/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_60/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_60/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_47/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_47/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_47/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_47/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_61/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_61/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_62/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_62/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_48/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_48/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_48/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_48/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_6/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_6/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
serving_default_input_7Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ  

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_7conv2d_54/kernelconv2d_54/biasbatch_normalization_42/gammabatch_normalization_42/beta"batch_normalization_42/moving_mean&batch_normalization_42/moving_varianceconv2d_55/kernelconv2d_55/biasbatch_normalization_43/gammabatch_normalization_43/beta"batch_normalization_43/moving_mean&batch_normalization_43/moving_varianceconv2d_56/kernelconv2d_56/biasbatch_normalization_44/gammabatch_normalization_44/beta"batch_normalization_44/moving_mean&batch_normalization_44/moving_varianceconv2d_57/kernelconv2d_57/biasbatch_normalization_45/gammabatch_normalization_45/beta"batch_normalization_45/moving_mean&batch_normalization_45/moving_varianceconv2d_58/kernelconv2d_58/biasconv2d_59/kernelconv2d_59/biasbatch_normalization_46/gammabatch_normalization_46/beta"batch_normalization_46/moving_mean&batch_normalization_46/moving_varianceconv2d_60/kernelconv2d_60/biasbatch_normalization_47/gammabatch_normalization_47/beta"batch_normalization_47/moving_mean&batch_normalization_47/moving_varianceconv2d_61/kernelconv2d_61/biasconv2d_62/kernelconv2d_62/biasbatch_normalization_48/gammabatch_normalization_48/beta"batch_normalization_48/moving_mean&batch_normalization_48/moving_variancedense_6/kerneldense_6/bias*<
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
&__inference_signature_wrapper_23490312
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_54/kernel/Read/ReadVariableOp"conv2d_54/bias/Read/ReadVariableOp0batch_normalization_42/gamma/Read/ReadVariableOp/batch_normalization_42/beta/Read/ReadVariableOp6batch_normalization_42/moving_mean/Read/ReadVariableOp:batch_normalization_42/moving_variance/Read/ReadVariableOp$conv2d_55/kernel/Read/ReadVariableOp"conv2d_55/bias/Read/ReadVariableOp0batch_normalization_43/gamma/Read/ReadVariableOp/batch_normalization_43/beta/Read/ReadVariableOp6batch_normalization_43/moving_mean/Read/ReadVariableOp:batch_normalization_43/moving_variance/Read/ReadVariableOp$conv2d_56/kernel/Read/ReadVariableOp"conv2d_56/bias/Read/ReadVariableOp0batch_normalization_44/gamma/Read/ReadVariableOp/batch_normalization_44/beta/Read/ReadVariableOp6batch_normalization_44/moving_mean/Read/ReadVariableOp:batch_normalization_44/moving_variance/Read/ReadVariableOp$conv2d_57/kernel/Read/ReadVariableOp"conv2d_57/bias/Read/ReadVariableOp0batch_normalization_45/gamma/Read/ReadVariableOp/batch_normalization_45/beta/Read/ReadVariableOp6batch_normalization_45/moving_mean/Read/ReadVariableOp:batch_normalization_45/moving_variance/Read/ReadVariableOp$conv2d_58/kernel/Read/ReadVariableOp"conv2d_58/bias/Read/ReadVariableOp$conv2d_59/kernel/Read/ReadVariableOp"conv2d_59/bias/Read/ReadVariableOp0batch_normalization_46/gamma/Read/ReadVariableOp/batch_normalization_46/beta/Read/ReadVariableOp6batch_normalization_46/moving_mean/Read/ReadVariableOp:batch_normalization_46/moving_variance/Read/ReadVariableOp$conv2d_60/kernel/Read/ReadVariableOp"conv2d_60/bias/Read/ReadVariableOp0batch_normalization_47/gamma/Read/ReadVariableOp/batch_normalization_47/beta/Read/ReadVariableOp6batch_normalization_47/moving_mean/Read/ReadVariableOp:batch_normalization_47/moving_variance/Read/ReadVariableOp$conv2d_61/kernel/Read/ReadVariableOp"conv2d_61/bias/Read/ReadVariableOp$conv2d_62/kernel/Read/ReadVariableOp"conv2d_62/bias/Read/ReadVariableOp0batch_normalization_48/gamma/Read/ReadVariableOp/batch_normalization_48/beta/Read/ReadVariableOp6batch_normalization_48/moving_mean/Read/ReadVariableOp:batch_normalization_48/moving_variance/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOpConst*=
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
!__inference__traced_save_23491437
É
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_54/kernelconv2d_54/biasbatch_normalization_42/gammabatch_normalization_42/beta"batch_normalization_42/moving_mean&batch_normalization_42/moving_varianceconv2d_55/kernelconv2d_55/biasbatch_normalization_43/gammabatch_normalization_43/beta"batch_normalization_43/moving_mean&batch_normalization_43/moving_varianceconv2d_56/kernelconv2d_56/biasbatch_normalization_44/gammabatch_normalization_44/beta"batch_normalization_44/moving_mean&batch_normalization_44/moving_varianceconv2d_57/kernelconv2d_57/biasbatch_normalization_45/gammabatch_normalization_45/beta"batch_normalization_45/moving_mean&batch_normalization_45/moving_varianceconv2d_58/kernelconv2d_58/biasconv2d_59/kernelconv2d_59/biasbatch_normalization_46/gammabatch_normalization_46/beta"batch_normalization_46/moving_mean&batch_normalization_46/moving_varianceconv2d_60/kernelconv2d_60/biasbatch_normalization_47/gammabatch_normalization_47/beta"batch_normalization_47/moving_mean&batch_normalization_47/moving_varianceconv2d_61/kernelconv2d_61/biasconv2d_62/kernelconv2d_62/biasbatch_normalization_48/gammabatch_normalization_48/beta"batch_normalization_48/moving_mean&batch_normalization_48/moving_variancedense_6/kerneldense_6/bias*<
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
$__inference__traced_restore_23491591÷­
â
µ
G__inference_conv2d_60_layer_call_and_return_conditional_losses_23490913

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_60/kernel/Regularizer/Square/ReadVariableOp|
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
2conv2d_60/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_60/kernel/Regularizer/SquareSquare:conv2d_60/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_60/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_60/kernel/Regularizer/SumSum'conv2d_60/kernel/Regularizer/Square:y:0+conv2d_60/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_60/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_60/kernel/Regularizer/mulMul+conv2d_60/kernel/Regularizer/mul/x:output:0)conv2d_60/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_60/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_60/kernel/Regularizer/Square/ReadVariableOp2conv2d_60/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23487772

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
G__inference_conv2d_62_layer_call_and_return_conditional_losses_23491047

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_62/kernel/Regularizer/Square/ReadVariableOp|
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
2conv2d_62/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_62/kernel/Regularizer/SquareSquare:conv2d_62/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_62/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_62/kernel/Regularizer/SumSum'conv2d_62/kernel/Regularizer/Square:y:0+conv2d_62/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_62/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_62/kernel/Regularizer/mulMul+conv2d_62/kernel/Regularizer/mul/x:output:0)conv2d_62/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_62/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_62/kernel/Regularizer/Square/ReadVariableOp2conv2d_62/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ý
Ã
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23487867

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
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23487611

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
ï
g
K__inference_activation_45_layer_call_and_return_conditional_losses_23490736

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
ð
¡
,__inference_conv2d_55_layer_call_fn_23490430

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
G__inference_conv2d_55_layer_call_and_return_conditional_losses_23488015w
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
Ï

T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23490842

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
»
¦
*__inference_model_6_layer_call_fn_23489650

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
E__inference_model_6_layer_call_and_return_conditional_losses_23488375o
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
§
.
E__inference_model_6_layer_call_and_return_conditional_losses_23489980

inputsB
(conv2d_54_conv2d_readvariableop_resource:7
)conv2d_54_biasadd_readvariableop_resource:<
.batch_normalization_42_readvariableop_resource:>
0batch_normalization_42_readvariableop_1_resource:M
?batch_normalization_42_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_42_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_55_conv2d_readvariableop_resource:7
)conv2d_55_biasadd_readvariableop_resource:<
.batch_normalization_43_readvariableop_resource:>
0batch_normalization_43_readvariableop_1_resource:M
?batch_normalization_43_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_56_conv2d_readvariableop_resource:7
)conv2d_56_biasadd_readvariableop_resource:<
.batch_normalization_44_readvariableop_resource:>
0batch_normalization_44_readvariableop_1_resource:M
?batch_normalization_44_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_57_conv2d_readvariableop_resource: 7
)conv2d_57_biasadd_readvariableop_resource: <
.batch_normalization_45_readvariableop_resource: >
0batch_normalization_45_readvariableop_1_resource: M
?batch_normalization_45_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_58_conv2d_readvariableop_resource:  7
)conv2d_58_biasadd_readvariableop_resource: B
(conv2d_59_conv2d_readvariableop_resource: 7
)conv2d_59_biasadd_readvariableop_resource: <
.batch_normalization_46_readvariableop_resource: >
0batch_normalization_46_readvariableop_1_resource: M
?batch_normalization_46_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_60_conv2d_readvariableop_resource: @7
)conv2d_60_biasadd_readvariableop_resource:@<
.batch_normalization_47_readvariableop_resource:@>
0batch_normalization_47_readvariableop_1_resource:@M
?batch_normalization_47_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource:@B
(conv2d_61_conv2d_readvariableop_resource:@@7
)conv2d_61_biasadd_readvariableop_resource:@B
(conv2d_62_conv2d_readvariableop_resource: @7
)conv2d_62_biasadd_readvariableop_resource:@<
.batch_normalization_48_readvariableop_resource:@>
0batch_normalization_48_readvariableop_1_resource:@M
?batch_normalization_48_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource:@8
&dense_6_matmul_readvariableop_resource:@
5
'dense_6_biasadd_readvariableop_resource:

identity¢6batch_normalization_42/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_42/ReadVariableOp¢'batch_normalization_42/ReadVariableOp_1¢6batch_normalization_43/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_43/ReadVariableOp¢'batch_normalization_43/ReadVariableOp_1¢6batch_normalization_44/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_44/ReadVariableOp¢'batch_normalization_44/ReadVariableOp_1¢6batch_normalization_45/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_45/ReadVariableOp¢'batch_normalization_45/ReadVariableOp_1¢6batch_normalization_46/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_46/ReadVariableOp¢'batch_normalization_46/ReadVariableOp_1¢6batch_normalization_47/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_47/ReadVariableOp¢'batch_normalization_47/ReadVariableOp_1¢6batch_normalization_48/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_48/ReadVariableOp¢'batch_normalization_48/ReadVariableOp_1¢ conv2d_54/BiasAdd/ReadVariableOp¢conv2d_54/Conv2D/ReadVariableOp¢2conv2d_54/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_55/BiasAdd/ReadVariableOp¢conv2d_55/Conv2D/ReadVariableOp¢2conv2d_55/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_56/BiasAdd/ReadVariableOp¢conv2d_56/Conv2D/ReadVariableOp¢2conv2d_56/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_57/BiasAdd/ReadVariableOp¢conv2d_57/Conv2D/ReadVariableOp¢2conv2d_57/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_58/BiasAdd/ReadVariableOp¢conv2d_58/Conv2D/ReadVariableOp¢2conv2d_58/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_59/BiasAdd/ReadVariableOp¢conv2d_59/Conv2D/ReadVariableOp¢2conv2d_59/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_60/BiasAdd/ReadVariableOp¢conv2d_60/Conv2D/ReadVariableOp¢2conv2d_60/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_61/BiasAdd/ReadVariableOp¢conv2d_61/Conv2D/ReadVariableOp¢2conv2d_61/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_62/BiasAdd/ReadVariableOp¢conv2d_62/Conv2D/ReadVariableOp¢2conv2d_62/kernel/Regularizer/Square/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp
conv2d_54/Conv2D/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0­
conv2d_54/Conv2DConv2Dinputs'conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

 conv2d_54/BiasAdd/ReadVariableOpReadVariableOp)conv2d_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_54/BiasAddBiasAddconv2d_54/Conv2D:output:0(conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
%batch_normalization_42/ReadVariableOpReadVariableOp.batch_normalization_42_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_42/ReadVariableOp_1ReadVariableOp0batch_normalization_42_readvariableop_1_resource*
_output_shapes
:*
dtype0²
6batch_normalization_42/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_42_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¶
8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_42_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0½
'batch_normalization_42/FusedBatchNormV3FusedBatchNormV3conv2d_54/BiasAdd:output:0-batch_normalization_42/ReadVariableOp:value:0/batch_normalization_42/ReadVariableOp_1:value:0>batch_normalization_42/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
is_training( 
activation_42/ReluRelu+batch_normalization_42/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
conv2d_55/Conv2D/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ç
conv2d_55/Conv2DConv2D activation_42/Relu:activations:0'conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

 conv2d_55/BiasAdd/ReadVariableOpReadVariableOp)conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_55/BiasAddBiasAddconv2d_55/Conv2D:output:0(conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
%batch_normalization_43/ReadVariableOpReadVariableOp.batch_normalization_43_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_43/ReadVariableOp_1ReadVariableOp0batch_normalization_43_readvariableop_1_resource*
_output_shapes
:*
dtype0²
6batch_normalization_43/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_43_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¶
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0½
'batch_normalization_43/FusedBatchNormV3FusedBatchNormV3conv2d_55/BiasAdd:output:0-batch_normalization_43/ReadVariableOp:value:0/batch_normalization_43/ReadVariableOp_1:value:0>batch_normalization_43/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
is_training( 
activation_43/ReluRelu+batch_normalization_43/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
conv2d_56/Conv2D/ReadVariableOpReadVariableOp(conv2d_56_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ç
conv2d_56/Conv2DConv2D activation_43/Relu:activations:0'conv2d_56/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

 conv2d_56/BiasAdd/ReadVariableOpReadVariableOp)conv2d_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_56/BiasAddBiasAddconv2d_56/Conv2D:output:0(conv2d_56/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
%batch_normalization_44/ReadVariableOpReadVariableOp.batch_normalization_44_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_44/ReadVariableOp_1ReadVariableOp0batch_normalization_44_readvariableop_1_resource*
_output_shapes
:*
dtype0²
6batch_normalization_44/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_44_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¶
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0½
'batch_normalization_44/FusedBatchNormV3FusedBatchNormV3conv2d_56/BiasAdd:output:0-batch_normalization_44/ReadVariableOp:value:0/batch_normalization_44/ReadVariableOp_1:value:0>batch_normalization_44/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
is_training( 

add_18/addAddV2 activation_42/Relu:activations:0+batch_normalization_44/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  d
activation_44/ReluReluadd_18/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
conv2d_57/Conv2D/ReadVariableOpReadVariableOp(conv2d_57_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ç
conv2d_57/Conv2DConv2D activation_44/Relu:activations:0'conv2d_57/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

 conv2d_57/BiasAdd/ReadVariableOpReadVariableOp)conv2d_57_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_57/BiasAddBiasAddconv2d_57/Conv2D:output:0(conv2d_57/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%batch_normalization_45/ReadVariableOpReadVariableOp.batch_normalization_45_readvariableop_resource*
_output_shapes
: *
dtype0
'batch_normalization_45/ReadVariableOp_1ReadVariableOp0batch_normalization_45_readvariableop_1_resource*
_output_shapes
: *
dtype0²
6batch_normalization_45/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_45_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¶
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0½
'batch_normalization_45/FusedBatchNormV3FusedBatchNormV3conv2d_57/BiasAdd:output:0-batch_normalization_45/ReadVariableOp:value:0/batch_normalization_45/ReadVariableOp_1:value:0>batch_normalization_45/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 
activation_45/ReluRelu+batch_normalization_45/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_58/Conv2D/ReadVariableOpReadVariableOp(conv2d_58_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Ç
conv2d_58/Conv2DConv2D activation_45/Relu:activations:0'conv2d_58/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

 conv2d_58/BiasAdd/ReadVariableOpReadVariableOp)conv2d_58_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_58/BiasAddBiasAddconv2d_58/Conv2D:output:0(conv2d_58/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_59/Conv2D/ReadVariableOpReadVariableOp(conv2d_59_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ç
conv2d_59/Conv2DConv2D activation_44/Relu:activations:0'conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

 conv2d_59/BiasAdd/ReadVariableOpReadVariableOp)conv2d_59_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_59/BiasAddBiasAddconv2d_59/Conv2D:output:0(conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%batch_normalization_46/ReadVariableOpReadVariableOp.batch_normalization_46_readvariableop_resource*
_output_shapes
: *
dtype0
'batch_normalization_46/ReadVariableOp_1ReadVariableOp0batch_normalization_46_readvariableop_1_resource*
_output_shapes
: *
dtype0²
6batch_normalization_46/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_46_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¶
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0½
'batch_normalization_46/FusedBatchNormV3FusedBatchNormV3conv2d_58/BiasAdd:output:0-batch_normalization_46/ReadVariableOp:value:0/batch_normalization_46/ReadVariableOp_1:value:0>batch_normalization_46/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 

add_19/addAddV2conv2d_59/BiasAdd:output:0+batch_normalization_46/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
activation_46/ReluReluadd_19/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ç
conv2d_60/Conv2DConv2D activation_46/Relu:activations:0'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%batch_normalization_47/ReadVariableOpReadVariableOp.batch_normalization_47_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_47/ReadVariableOp_1ReadVariableOp0batch_normalization_47_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0½
'batch_normalization_47/FusedBatchNormV3FusedBatchNormV3conv2d_60/BiasAdd:output:0-batch_normalization_47/ReadVariableOp:value:0/batch_normalization_47/ReadVariableOp_1:value:0>batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 
activation_47/ReluRelu+batch_normalization_47/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_61/Conv2D/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ç
conv2d_61/Conv2DConv2D activation_47/Relu:activations:0'conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

 conv2d_61/BiasAdd/ReadVariableOpReadVariableOp)conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_61/BiasAddBiasAddconv2d_61/Conv2D:output:0(conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ç
conv2d_62/Conv2DConv2D activation_46/Relu:activations:0'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%batch_normalization_48/ReadVariableOpReadVariableOp.batch_normalization_48_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_48/ReadVariableOp_1ReadVariableOp0batch_normalization_48_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0½
'batch_normalization_48/FusedBatchNormV3FusedBatchNormV3conv2d_61/BiasAdd:output:0-batch_normalization_48/ReadVariableOp:value:0/batch_normalization_48/ReadVariableOp_1:value:0>batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 

add_20/addAddV2conv2d_62/BiasAdd:output:0+batch_normalization_48/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
activation_48/ReluReluadd_20/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
average_pooling2d_6/AvgPoolAvgPool activation_48/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
flatten_6/ReshapeReshape$average_pooling2d_6/AvgPool:output:0flatten_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype0
dense_6/MatMulMatMulflatten_6/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
£
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_54/kernel/Regularizer/SquareSquare:conv2d_54/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_54/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_54/kernel/Regularizer/SumSum'conv2d_54/kernel/Regularizer/Square:y:0+conv2d_54/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_54/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_54/kernel/Regularizer/mulMul+conv2d_54/kernel/Regularizer/mul/x:output:0)conv2d_54/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_55/kernel/Regularizer/SquareSquare:conv2d_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_55/kernel/Regularizer/SumSum'conv2d_55/kernel/Regularizer/Square:y:0+conv2d_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_55/kernel/Regularizer/mulMul+conv2d_55/kernel/Regularizer/mul/x:output:0)conv2d_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_56_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_56/kernel/Regularizer/SquareSquare:conv2d_56/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_56/kernel/Regularizer/SumSum'conv2d_56/kernel/Regularizer/Square:y:0+conv2d_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_56/kernel/Regularizer/mulMul+conv2d_56/kernel/Regularizer/mul/x:output:0)conv2d_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_57_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_57/kernel/Regularizer/SquareSquare:conv2d_57/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_57/kernel/Regularizer/SumSum'conv2d_57/kernel/Regularizer/Square:y:0+conv2d_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_57/kernel/Regularizer/mulMul+conv2d_57/kernel/Regularizer/mul/x:output:0)conv2d_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_58_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0
#conv2d_58/kernel/Regularizer/SquareSquare:conv2d_58/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_58/kernel/Regularizer/SumSum'conv2d_58/kernel/Regularizer/Square:y:0+conv2d_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_58/kernel/Regularizer/mulMul+conv2d_58/kernel/Regularizer/mul/x:output:0)conv2d_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_59_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_59/kernel/Regularizer/SquareSquare:conv2d_59/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_59/kernel/Regularizer/SumSum'conv2d_59/kernel/Regularizer/Square:y:0+conv2d_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_59/kernel/Regularizer/mulMul+conv2d_59/kernel/Regularizer/mul/x:output:0)conv2d_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_60/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_60/kernel/Regularizer/SquareSquare:conv2d_60/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_60/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_60/kernel/Regularizer/SumSum'conv2d_60/kernel/Regularizer/Square:y:0+conv2d_60/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_60/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_60/kernel/Regularizer/mulMul+conv2d_60/kernel/Regularizer/mul/x:output:0)conv2d_60/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_61/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
#conv2d_61/kernel/Regularizer/SquareSquare:conv2d_61/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@{
"conv2d_61/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_61/kernel/Regularizer/SumSum'conv2d_61/kernel/Regularizer/Square:y:0+conv2d_61/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_61/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_61/kernel/Regularizer/mulMul+conv2d_61/kernel/Regularizer/mul/x:output:0)conv2d_61/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_62/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_62/kernel/Regularizer/SquareSquare:conv2d_62/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_62/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_62/kernel/Regularizer/SumSum'conv2d_62/kernel/Regularizer/Square:y:0+conv2d_62/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_62/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_62/kernel/Regularizer/mulMul+conv2d_62/kernel/Regularizer/mul/x:output:0)conv2d_62/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
»
NoOpNoOp7^batch_normalization_42/FusedBatchNormV3/ReadVariableOp9^batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_42/ReadVariableOp(^batch_normalization_42/ReadVariableOp_17^batch_normalization_43/FusedBatchNormV3/ReadVariableOp9^batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_43/ReadVariableOp(^batch_normalization_43/ReadVariableOp_17^batch_normalization_44/FusedBatchNormV3/ReadVariableOp9^batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_44/ReadVariableOp(^batch_normalization_44/ReadVariableOp_17^batch_normalization_45/FusedBatchNormV3/ReadVariableOp9^batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_45/ReadVariableOp(^batch_normalization_45/ReadVariableOp_17^batch_normalization_46/FusedBatchNormV3/ReadVariableOp9^batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_46/ReadVariableOp(^batch_normalization_46/ReadVariableOp_17^batch_normalization_47/FusedBatchNormV3/ReadVariableOp9^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_47/ReadVariableOp(^batch_normalization_47/ReadVariableOp_17^batch_normalization_48/FusedBatchNormV3/ReadVariableOp9^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_48/ReadVariableOp(^batch_normalization_48/ReadVariableOp_1!^conv2d_54/BiasAdd/ReadVariableOp ^conv2d_54/Conv2D/ReadVariableOp3^conv2d_54/kernel/Regularizer/Square/ReadVariableOp!^conv2d_55/BiasAdd/ReadVariableOp ^conv2d_55/Conv2D/ReadVariableOp3^conv2d_55/kernel/Regularizer/Square/ReadVariableOp!^conv2d_56/BiasAdd/ReadVariableOp ^conv2d_56/Conv2D/ReadVariableOp3^conv2d_56/kernel/Regularizer/Square/ReadVariableOp!^conv2d_57/BiasAdd/ReadVariableOp ^conv2d_57/Conv2D/ReadVariableOp3^conv2d_57/kernel/Regularizer/Square/ReadVariableOp!^conv2d_58/BiasAdd/ReadVariableOp ^conv2d_58/Conv2D/ReadVariableOp3^conv2d_58/kernel/Regularizer/Square/ReadVariableOp!^conv2d_59/BiasAdd/ReadVariableOp ^conv2d_59/Conv2D/ReadVariableOp3^conv2d_59/kernel/Regularizer/Square/ReadVariableOp!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp3^conv2d_60/kernel/Regularizer/Square/ReadVariableOp!^conv2d_61/BiasAdd/ReadVariableOp ^conv2d_61/Conv2D/ReadVariableOp3^conv2d_61/kernel/Regularizer/Square/ReadVariableOp!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp3^conv2d_62/kernel/Regularizer/Square/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_42/FusedBatchNormV3/ReadVariableOp6batch_normalization_42/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_18batch_normalization_42/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_42/ReadVariableOp%batch_normalization_42/ReadVariableOp2R
'batch_normalization_42/ReadVariableOp_1'batch_normalization_42/ReadVariableOp_12p
6batch_normalization_43/FusedBatchNormV3/ReadVariableOp6batch_normalization_43/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_18batch_normalization_43/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_43/ReadVariableOp%batch_normalization_43/ReadVariableOp2R
'batch_normalization_43/ReadVariableOp_1'batch_normalization_43/ReadVariableOp_12p
6batch_normalization_44/FusedBatchNormV3/ReadVariableOp6batch_normalization_44/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_18batch_normalization_44/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_44/ReadVariableOp%batch_normalization_44/ReadVariableOp2R
'batch_normalization_44/ReadVariableOp_1'batch_normalization_44/ReadVariableOp_12p
6batch_normalization_45/FusedBatchNormV3/ReadVariableOp6batch_normalization_45/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_18batch_normalization_45/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_45/ReadVariableOp%batch_normalization_45/ReadVariableOp2R
'batch_normalization_45/ReadVariableOp_1'batch_normalization_45/ReadVariableOp_12p
6batch_normalization_46/FusedBatchNormV3/ReadVariableOp6batch_normalization_46/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_18batch_normalization_46/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_46/ReadVariableOp%batch_normalization_46/ReadVariableOp2R
'batch_normalization_46/ReadVariableOp_1'batch_normalization_46/ReadVariableOp_12p
6batch_normalization_47/FusedBatchNormV3/ReadVariableOp6batch_normalization_47/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_18batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_47/ReadVariableOp%batch_normalization_47/ReadVariableOp2R
'batch_normalization_47/ReadVariableOp_1'batch_normalization_47/ReadVariableOp_12p
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp6batch_normalization_48/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_18batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_48/ReadVariableOp%batch_normalization_48/ReadVariableOp2R
'batch_normalization_48/ReadVariableOp_1'batch_normalization_48/ReadVariableOp_12D
 conv2d_54/BiasAdd/ReadVariableOp conv2d_54/BiasAdd/ReadVariableOp2B
conv2d_54/Conv2D/ReadVariableOpconv2d_54/Conv2D/ReadVariableOp2h
2conv2d_54/kernel/Regularizer/Square/ReadVariableOp2conv2d_54/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_55/BiasAdd/ReadVariableOp conv2d_55/BiasAdd/ReadVariableOp2B
conv2d_55/Conv2D/ReadVariableOpconv2d_55/Conv2D/ReadVariableOp2h
2conv2d_55/kernel/Regularizer/Square/ReadVariableOp2conv2d_55/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_56/BiasAdd/ReadVariableOp conv2d_56/BiasAdd/ReadVariableOp2B
conv2d_56/Conv2D/ReadVariableOpconv2d_56/Conv2D/ReadVariableOp2h
2conv2d_56/kernel/Regularizer/Square/ReadVariableOp2conv2d_56/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_57/BiasAdd/ReadVariableOp conv2d_57/BiasAdd/ReadVariableOp2B
conv2d_57/Conv2D/ReadVariableOpconv2d_57/Conv2D/ReadVariableOp2h
2conv2d_57/kernel/Regularizer/Square/ReadVariableOp2conv2d_57/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_58/BiasAdd/ReadVariableOp conv2d_58/BiasAdd/ReadVariableOp2B
conv2d_58/Conv2D/ReadVariableOpconv2d_58/Conv2D/ReadVariableOp2h
2conv2d_58/kernel/Regularizer/Square/ReadVariableOp2conv2d_58/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_59/BiasAdd/ReadVariableOp conv2d_59/BiasAdd/ReadVariableOp2B
conv2d_59/Conv2D/ReadVariableOpconv2d_59/Conv2D/ReadVariableOp2h
2conv2d_59/kernel/Regularizer/Square/ReadVariableOp2conv2d_59/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp2h
2conv2d_60/kernel/Regularizer/Square/ReadVariableOp2conv2d_60/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_61/BiasAdd/ReadVariableOp conv2d_61/BiasAdd/ReadVariableOp2B
conv2d_61/Conv2D/ReadVariableOpconv2d_61/Conv2D/ReadVariableOp2h
2conv2d_61/kernel/Regularizer/Square/ReadVariableOp2conv2d_61/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp2h
2conv2d_62/kernel/Regularizer/Square/ReadVariableOp2conv2d_62/kernel/Regularizer/Square/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_48_layer_call_fn_23491060

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
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23487900
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
ï
g
K__inference_activation_48_layer_call_and_return_conditional_losses_23491131

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
Ä
R
6__inference_average_pooling2d_6_layer_call_fn_23491136

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
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_23487951
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
Ý
Ã
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23490975

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
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23487644

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
Ï

T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23490957

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
Ä

*__inference_dense_6_layer_call_fn_23491161

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
E__inference_dense_6_layer_call_and_return_conditional_losses_23488314o
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
ï
g
K__inference_activation_43_layer_call_and_return_conditional_losses_23488035

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
9__inference_batch_normalization_42_layer_call_fn_23490356

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
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23487516
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
G__inference_conv2d_59_layer_call_and_return_conditional_losses_23490798

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_59/kernel/Regularizer/Square/ReadVariableOp|
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
2conv2d_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_59/kernel/Regularizer/SquareSquare:conv2d_59/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_59/kernel/Regularizer/SumSum'conv2d_59/kernel/Regularizer/Square:y:0+conv2d_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_59/kernel/Regularizer/mulMul+conv2d_59/kernel/Regularizer/mul/x:output:0)conv2d_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_59/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_59/kernel/Regularizer/Square/ReadVariableOp2conv2d_59/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ý
Ã
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23490611

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
G__inference_conv2d_61_layer_call_and_return_conditional_losses_23491016

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_61/kernel/Regularizer/Square/ReadVariableOp|
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
2conv2d_61/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
#conv2d_61/kernel/Regularizer/SquareSquare:conv2d_61/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@{
"conv2d_61/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_61/kernel/Regularizer/SumSum'conv2d_61/kernel/Regularizer/Square:y:0+conv2d_61/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_61/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_61/kernel/Regularizer/mulMul+conv2d_61/kernel/Regularizer/mul/x:output:0)conv2d_61/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_61/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_61/kernel/Regularizer/Square/ReadVariableOp2conv2d_61/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_45_layer_call_fn_23490677

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
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23487708
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
¯Ø
¶
E__inference_model_6_layer_call_and_return_conditional_losses_23488375

inputs,
conv2d_54_23487978: 
conv2d_54_23487980:-
batch_normalization_42_23487983:-
batch_normalization_42_23487985:-
batch_normalization_42_23487987:-
batch_normalization_42_23487989:,
conv2d_55_23488016: 
conv2d_55_23488018:-
batch_normalization_43_23488021:-
batch_normalization_43_23488023:-
batch_normalization_43_23488025:-
batch_normalization_43_23488027:,
conv2d_56_23488054: 
conv2d_56_23488056:-
batch_normalization_44_23488059:-
batch_normalization_44_23488061:-
batch_normalization_44_23488063:-
batch_normalization_44_23488065:,
conv2d_57_23488100:  
conv2d_57_23488102: -
batch_normalization_45_23488105: -
batch_normalization_45_23488107: -
batch_normalization_45_23488109: -
batch_normalization_45_23488111: ,
conv2d_58_23488138:   
conv2d_58_23488140: ,
conv2d_59_23488160:  
conv2d_59_23488162: -
batch_normalization_46_23488165: -
batch_normalization_46_23488167: -
batch_normalization_46_23488169: -
batch_normalization_46_23488171: ,
conv2d_60_23488206: @ 
conv2d_60_23488208:@-
batch_normalization_47_23488211:@-
batch_normalization_47_23488213:@-
batch_normalization_47_23488215:@-
batch_normalization_47_23488217:@,
conv2d_61_23488244:@@ 
conv2d_61_23488246:@,
conv2d_62_23488266: @ 
conv2d_62_23488268:@-
batch_normalization_48_23488271:@-
batch_normalization_48_23488273:@-
batch_normalization_48_23488275:@-
batch_normalization_48_23488277:@"
dense_6_23488315:@

dense_6_23488317:

identity¢.batch_normalization_42/StatefulPartitionedCall¢.batch_normalization_43/StatefulPartitionedCall¢.batch_normalization_44/StatefulPartitionedCall¢.batch_normalization_45/StatefulPartitionedCall¢.batch_normalization_46/StatefulPartitionedCall¢.batch_normalization_47/StatefulPartitionedCall¢.batch_normalization_48/StatefulPartitionedCall¢!conv2d_54/StatefulPartitionedCall¢2conv2d_54/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_55/StatefulPartitionedCall¢2conv2d_55/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_56/StatefulPartitionedCall¢2conv2d_56/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_57/StatefulPartitionedCall¢2conv2d_57/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_58/StatefulPartitionedCall¢2conv2d_58/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_59/StatefulPartitionedCall¢2conv2d_59/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_60/StatefulPartitionedCall¢2conv2d_60/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_61/StatefulPartitionedCall¢2conv2d_61/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_62/StatefulPartitionedCall¢2conv2d_62/kernel/Regularizer/Square/ReadVariableOp¢dense_6/StatefulPartitionedCall
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_54_23487978conv2d_54_23487980*
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
G__inference_conv2d_54_layer_call_and_return_conditional_losses_23487977 
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_42_23487983batch_normalization_42_23487985batch_normalization_42_23487987batch_normalization_42_23487989*
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
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23487516ý
activation_42/PartitionedCallPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0*
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
K__inference_activation_42_layer_call_and_return_conditional_losses_23487997¢
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall&activation_42/PartitionedCall:output:0conv2d_55_23488016conv2d_55_23488018*
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
G__inference_conv2d_55_layer_call_and_return_conditional_losses_23488015 
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0batch_normalization_43_23488021batch_normalization_43_23488023batch_normalization_43_23488025batch_normalization_43_23488027*
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
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23487580ý
activation_43/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
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
K__inference_activation_43_layer_call_and_return_conditional_losses_23488035¢
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall&activation_43/PartitionedCall:output:0conv2d_56_23488054conv2d_56_23488056*
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
G__inference_conv2d_56_layer_call_and_return_conditional_losses_23488053 
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_56/StatefulPartitionedCall:output:0batch_normalization_44_23488059batch_normalization_44_23488061batch_normalization_44_23488063batch_normalization_44_23488065*
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
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23487644
add_18/PartitionedCallPartitionedCall&activation_42/PartitionedCall:output:07batch_normalization_44/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *M
fHRF
D__inference_add_18_layer_call_and_return_conditional_losses_23488074å
activation_44/PartitionedCallPartitionedCalladd_18/PartitionedCall:output:0*
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
K__inference_activation_44_layer_call_and_return_conditional_losses_23488081¢
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCall&activation_44/PartitionedCall:output:0conv2d_57_23488100conv2d_57_23488102*
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
G__inference_conv2d_57_layer_call_and_return_conditional_losses_23488099 
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0batch_normalization_45_23488105batch_normalization_45_23488107batch_normalization_45_23488109batch_normalization_45_23488111*
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
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23487708ý
activation_45/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
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
K__inference_activation_45_layer_call_and_return_conditional_losses_23488119¢
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall&activation_45/PartitionedCall:output:0conv2d_58_23488138conv2d_58_23488140*
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
G__inference_conv2d_58_layer_call_and_return_conditional_losses_23488137¢
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall&activation_44/PartitionedCall:output:0conv2d_59_23488160conv2d_59_23488162*
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
G__inference_conv2d_59_layer_call_and_return_conditional_losses_23488159 
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0batch_normalization_46_23488165batch_normalization_46_23488167batch_normalization_46_23488169batch_normalization_46_23488171*
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
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23487772
add_19/PartitionedCallPartitionedCall*conv2d_59/StatefulPartitionedCall:output:07batch_normalization_46/StatefulPartitionedCall:output:0*
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
D__inference_add_19_layer_call_and_return_conditional_losses_23488180å
activation_46/PartitionedCallPartitionedCalladd_19/PartitionedCall:output:0*
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
K__inference_activation_46_layer_call_and_return_conditional_losses_23488187¢
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall&activation_46/PartitionedCall:output:0conv2d_60_23488206conv2d_60_23488208*
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
G__inference_conv2d_60_layer_call_and_return_conditional_losses_23488205 
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0batch_normalization_47_23488211batch_normalization_47_23488213batch_normalization_47_23488215batch_normalization_47_23488217*
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
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23487836ý
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
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
K__inference_activation_47_layer_call_and_return_conditional_losses_23488225¢
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall&activation_47/PartitionedCall:output:0conv2d_61_23488244conv2d_61_23488246*
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
G__inference_conv2d_61_layer_call_and_return_conditional_losses_23488243¢
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall&activation_46/PartitionedCall:output:0conv2d_62_23488266conv2d_62_23488268*
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
G__inference_conv2d_62_layer_call_and_return_conditional_losses_23488265 
.batch_normalization_48/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0batch_normalization_48_23488271batch_normalization_48_23488273batch_normalization_48_23488275batch_normalization_48_23488277*
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
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23487900
add_20/PartitionedCallPartitionedCall*conv2d_62/StatefulPartitionedCall:output:07batch_normalization_48/StatefulPartitionedCall:output:0*
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
D__inference_add_20_layer_call_and_return_conditional_losses_23488286å
activation_48/PartitionedCallPartitionedCalladd_20/PartitionedCall:output:0*
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
K__inference_activation_48_layer_call_and_return_conditional_losses_23488293ø
#average_pooling2d_6/PartitionedCallPartitionedCall&activation_48/PartitionedCall:output:0*
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
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_23487951â
flatten_6/PartitionedCallPartitionedCall,average_pooling2d_6/PartitionedCall:output:0*
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
G__inference_flatten_6_layer_call_and_return_conditional_losses_23488302
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_6_23488315dense_6_23488317*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_23488314
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_54_23487978*&
_output_shapes
:*
dtype0
#conv2d_54/kernel/Regularizer/SquareSquare:conv2d_54/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_54/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_54/kernel/Regularizer/SumSum'conv2d_54/kernel/Regularizer/Square:y:0+conv2d_54/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_54/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_54/kernel/Regularizer/mulMul+conv2d_54/kernel/Regularizer/mul/x:output:0)conv2d_54/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_55_23488016*&
_output_shapes
:*
dtype0
#conv2d_55/kernel/Regularizer/SquareSquare:conv2d_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_55/kernel/Regularizer/SumSum'conv2d_55/kernel/Regularizer/Square:y:0+conv2d_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_55/kernel/Regularizer/mulMul+conv2d_55/kernel/Regularizer/mul/x:output:0)conv2d_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_56_23488054*&
_output_shapes
:*
dtype0
#conv2d_56/kernel/Regularizer/SquareSquare:conv2d_56/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_56/kernel/Regularizer/SumSum'conv2d_56/kernel/Regularizer/Square:y:0+conv2d_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_56/kernel/Regularizer/mulMul+conv2d_56/kernel/Regularizer/mul/x:output:0)conv2d_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_57_23488100*&
_output_shapes
: *
dtype0
#conv2d_57/kernel/Regularizer/SquareSquare:conv2d_57/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_57/kernel/Regularizer/SumSum'conv2d_57/kernel/Regularizer/Square:y:0+conv2d_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_57/kernel/Regularizer/mulMul+conv2d_57/kernel/Regularizer/mul/x:output:0)conv2d_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_58_23488138*&
_output_shapes
:  *
dtype0
#conv2d_58/kernel/Regularizer/SquareSquare:conv2d_58/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_58/kernel/Regularizer/SumSum'conv2d_58/kernel/Regularizer/Square:y:0+conv2d_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_58/kernel/Regularizer/mulMul+conv2d_58/kernel/Regularizer/mul/x:output:0)conv2d_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_59_23488160*&
_output_shapes
: *
dtype0
#conv2d_59/kernel/Regularizer/SquareSquare:conv2d_59/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_59/kernel/Regularizer/SumSum'conv2d_59/kernel/Regularizer/Square:y:0+conv2d_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_59/kernel/Regularizer/mulMul+conv2d_59/kernel/Regularizer/mul/x:output:0)conv2d_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_60/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_60_23488206*&
_output_shapes
: @*
dtype0
#conv2d_60/kernel/Regularizer/SquareSquare:conv2d_60/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_60/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_60/kernel/Regularizer/SumSum'conv2d_60/kernel/Regularizer/Square:y:0+conv2d_60/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_60/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_60/kernel/Regularizer/mulMul+conv2d_60/kernel/Regularizer/mul/x:output:0)conv2d_60/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_61/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_61_23488244*&
_output_shapes
:@@*
dtype0
#conv2d_61/kernel/Regularizer/SquareSquare:conv2d_61/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@{
"conv2d_61/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_61/kernel/Regularizer/SumSum'conv2d_61/kernel/Regularizer/Square:y:0+conv2d_61/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_61/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_61/kernel/Regularizer/mulMul+conv2d_61/kernel/Regularizer/mul/x:output:0)conv2d_61/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_62/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_62_23488266*&
_output_shapes
: @*
dtype0
#conv2d_62/kernel/Regularizer/SquareSquare:conv2d_62/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_62/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_62/kernel/Regularizer/SumSum'conv2d_62/kernel/Regularizer/Square:y:0+conv2d_62/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_62/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_62/kernel/Regularizer/mulMul+conv2d_62/kernel/Regularizer/mul/x:output:0)conv2d_62/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
à	
NoOpNoOp/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_46/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_48/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall3^conv2d_54/kernel/Regularizer/Square/ReadVariableOp"^conv2d_55/StatefulPartitionedCall3^conv2d_55/kernel/Regularizer/Square/ReadVariableOp"^conv2d_56/StatefulPartitionedCall3^conv2d_56/kernel/Regularizer/Square/ReadVariableOp"^conv2d_57/StatefulPartitionedCall3^conv2d_57/kernel/Regularizer/Square/ReadVariableOp"^conv2d_58/StatefulPartitionedCall3^conv2d_58/kernel/Regularizer/Square/ReadVariableOp"^conv2d_59/StatefulPartitionedCall3^conv2d_59/kernel/Regularizer/Square/ReadVariableOp"^conv2d_60/StatefulPartitionedCall3^conv2d_60/kernel/Regularizer/Square/ReadVariableOp"^conv2d_61/StatefulPartitionedCall3^conv2d_61/kernel/Regularizer/Square/ReadVariableOp"^conv2d_62/StatefulPartitionedCall3^conv2d_62/kernel/Regularizer/Square/ReadVariableOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_46/StatefulPartitionedCall.batch_normalization_46/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_48/StatefulPartitionedCall.batch_normalization_48/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2h
2conv2d_54/kernel/Regularizer/Square/ReadVariableOp2conv2d_54/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2h
2conv2d_55/kernel/Regularizer/Square/ReadVariableOp2conv2d_55/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_56/StatefulPartitionedCall!conv2d_56/StatefulPartitionedCall2h
2conv2d_56/kernel/Regularizer/Square/ReadVariableOp2conv2d_56/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_57/StatefulPartitionedCall!conv2d_57/StatefulPartitionedCall2h
2conv2d_57/kernel/Regularizer/Square/ReadVariableOp2conv2d_57/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2h
2conv2d_58/kernel/Regularizer/Square/ReadVariableOp2conv2d_58/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2h
2conv2d_59/kernel/Regularizer/Square/ReadVariableOp2conv2d_59/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2h
2conv2d_60/kernel/Regularizer/Square/ReadVariableOp2conv2d_60/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2h
2conv2d_61/kernel/Regularizer/Square/ReadVariableOp2conv2d_61/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2h
2conv2d_62/kernel/Regularizer/Square/ReadVariableOp2conv2d_62/kernel/Regularizer/Square/ReadVariableOp2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
°
§
*__inference_model_6_layer_call_fn_23489129
input_7!
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
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
E__inference_model_6_layer_call_and_return_conditional_losses_23488929o
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
_user_specified_name	input_7
ï
g
K__inference_activation_48_layer_call_and_return_conditional_losses_23488293

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
ð
¡
,__inference_conv2d_61_layer_call_fn_23491000

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
G__inference_conv2d_61_layer_call_and_return_conditional_losses_23488243w
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
ïÀ
¤ 
$__inference__traced_restore_23491591
file_prefix;
!assignvariableop_conv2d_54_kernel:/
!assignvariableop_1_conv2d_54_bias:=
/assignvariableop_2_batch_normalization_42_gamma:<
.assignvariableop_3_batch_normalization_42_beta:C
5assignvariableop_4_batch_normalization_42_moving_mean:G
9assignvariableop_5_batch_normalization_42_moving_variance:=
#assignvariableop_6_conv2d_55_kernel:/
!assignvariableop_7_conv2d_55_bias:=
/assignvariableop_8_batch_normalization_43_gamma:<
.assignvariableop_9_batch_normalization_43_beta:D
6assignvariableop_10_batch_normalization_43_moving_mean:H
:assignvariableop_11_batch_normalization_43_moving_variance:>
$assignvariableop_12_conv2d_56_kernel:0
"assignvariableop_13_conv2d_56_bias:>
0assignvariableop_14_batch_normalization_44_gamma:=
/assignvariableop_15_batch_normalization_44_beta:D
6assignvariableop_16_batch_normalization_44_moving_mean:H
:assignvariableop_17_batch_normalization_44_moving_variance:>
$assignvariableop_18_conv2d_57_kernel: 0
"assignvariableop_19_conv2d_57_bias: >
0assignvariableop_20_batch_normalization_45_gamma: =
/assignvariableop_21_batch_normalization_45_beta: D
6assignvariableop_22_batch_normalization_45_moving_mean: H
:assignvariableop_23_batch_normalization_45_moving_variance: >
$assignvariableop_24_conv2d_58_kernel:  0
"assignvariableop_25_conv2d_58_bias: >
$assignvariableop_26_conv2d_59_kernel: 0
"assignvariableop_27_conv2d_59_bias: >
0assignvariableop_28_batch_normalization_46_gamma: =
/assignvariableop_29_batch_normalization_46_beta: D
6assignvariableop_30_batch_normalization_46_moving_mean: H
:assignvariableop_31_batch_normalization_46_moving_variance: >
$assignvariableop_32_conv2d_60_kernel: @0
"assignvariableop_33_conv2d_60_bias:@>
0assignvariableop_34_batch_normalization_47_gamma:@=
/assignvariableop_35_batch_normalization_47_beta:@D
6assignvariableop_36_batch_normalization_47_moving_mean:@H
:assignvariableop_37_batch_normalization_47_moving_variance:@>
$assignvariableop_38_conv2d_61_kernel:@@0
"assignvariableop_39_conv2d_61_bias:@>
$assignvariableop_40_conv2d_62_kernel: @0
"assignvariableop_41_conv2d_62_bias:@>
0assignvariableop_42_batch_normalization_48_gamma:@=
/assignvariableop_43_batch_normalization_48_beta:@D
6assignvariableop_44_batch_normalization_48_moving_mean:@H
:assignvariableop_45_batch_normalization_48_moving_variance:@4
"assignvariableop_46_dense_6_kernel:@
.
 assignvariableop_47_dense_6_bias:
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
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_54_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_54_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_42_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_42_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_42_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_42_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_55_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_55_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_43_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_43_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_43_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_43_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_56_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_56_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_44_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_44_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_44_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_44_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_57_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_57_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_45_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_45_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_45_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_45_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_58_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_58_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp$assignvariableop_26_conv2d_59_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp"assignvariableop_27_conv2d_59_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_28AssignVariableOp0assignvariableop_28_batch_normalization_46_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_29AssignVariableOp/assignvariableop_29_batch_normalization_46_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_30AssignVariableOp6assignvariableop_30_batch_normalization_46_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_31AssignVariableOp:assignvariableop_31_batch_normalization_46_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp$assignvariableop_32_conv2d_60_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp"assignvariableop_33_conv2d_60_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_34AssignVariableOp0assignvariableop_34_batch_normalization_47_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_35AssignVariableOp/assignvariableop_35_batch_normalization_47_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_36AssignVariableOp6assignvariableop_36_batch_normalization_47_moving_meanIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_37AssignVariableOp:assignvariableop_37_batch_normalization_47_moving_varianceIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp$assignvariableop_38_conv2d_61_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp"assignvariableop_39_conv2d_61_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp$assignvariableop_40_conv2d_62_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp"assignvariableop_41_conv2d_62_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_42AssignVariableOp0assignvariableop_42_batch_normalization_48_gammaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_43AssignVariableOp/assignvariableop_43_batch_normalization_48_betaIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_44AssignVariableOp6assignvariableop_44_batch_normalization_48_moving_meanIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_45AssignVariableOp:assignvariableop_45_batch_normalization_48_moving_varianceIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp"assignvariableop_46_dense_6_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp assignvariableop_47_dense_6_biasIdentity_47:output:0"/device:CPU:0*
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
ð
¡
,__inference_conv2d_59_layer_call_fn_23490782

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
G__inference_conv2d_59_layer_call_and_return_conditional_losses_23488159w
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
ë
½
__inference_loss_fn_5_23491237U
;conv2d_59_kernel_regularizer_square_readvariableop_resource: 
identity¢2conv2d_59/kernel/Regularizer/Square/ReadVariableOp¶
2conv2d_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_59_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_59/kernel/Regularizer/SquareSquare:conv2d_59/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_59/kernel/Regularizer/SumSum'conv2d_59/kernel/Regularizer/Square:y:0+conv2d_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_59/kernel/Regularizer/mulMul+conv2d_59/kernel/Regularizer/mul/x:output:0)conv2d_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_59/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_59/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_59/kernel/Regularizer/Square/ReadVariableOp2conv2d_59/kernel/Regularizer/Square/ReadVariableOp
â
µ
G__inference_conv2d_57_layer_call_and_return_conditional_losses_23490664

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_57/kernel/Regularizer/Square/ReadVariableOp|
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
2conv2d_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_57/kernel/Regularizer/SquareSquare:conv2d_57/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_57/kernel/Regularizer/SumSum'conv2d_57/kernel/Regularizer/Square:y:0+conv2d_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_57/kernel/Regularizer/mulMul+conv2d_57/kernel/Regularizer/mul/x:output:0)conv2d_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_57/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_57/kernel/Regularizer/Square/ReadVariableOp2conv2d_57/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ñ
p
D__inference_add_19_layer_call_and_return_conditional_losses_23490872
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
é
n
D__inference_add_18_layer_call_and_return_conditional_losses_23488074

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
ï
g
K__inference_activation_47_layer_call_and_return_conditional_losses_23488225

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
Ë
L
0__inference_activation_43_layer_call_fn_23490513

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
K__inference_activation_43_layer_call_and_return_conditional_losses_23488035h
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
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23487803

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
ð
¡
,__inference_conv2d_60_layer_call_fn_23490897

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
G__inference_conv2d_60_layer_call_and_return_conditional_losses_23488205w
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
ï
g
K__inference_activation_44_layer_call_and_return_conditional_losses_23490633

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
9__inference_batch_normalization_47_layer_call_fn_23490926

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
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23487836
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
¤Ø
·
E__inference_model_6_layer_call_and_return_conditional_losses_23489495
input_7,
conv2d_54_23489315: 
conv2d_54_23489317:-
batch_normalization_42_23489320:-
batch_normalization_42_23489322:-
batch_normalization_42_23489324:-
batch_normalization_42_23489326:,
conv2d_55_23489330: 
conv2d_55_23489332:-
batch_normalization_43_23489335:-
batch_normalization_43_23489337:-
batch_normalization_43_23489339:-
batch_normalization_43_23489341:,
conv2d_56_23489345: 
conv2d_56_23489347:-
batch_normalization_44_23489350:-
batch_normalization_44_23489352:-
batch_normalization_44_23489354:-
batch_normalization_44_23489356:,
conv2d_57_23489361:  
conv2d_57_23489363: -
batch_normalization_45_23489366: -
batch_normalization_45_23489368: -
batch_normalization_45_23489370: -
batch_normalization_45_23489372: ,
conv2d_58_23489376:   
conv2d_58_23489378: ,
conv2d_59_23489381:  
conv2d_59_23489383: -
batch_normalization_46_23489386: -
batch_normalization_46_23489388: -
batch_normalization_46_23489390: -
batch_normalization_46_23489392: ,
conv2d_60_23489397: @ 
conv2d_60_23489399:@-
batch_normalization_47_23489402:@-
batch_normalization_47_23489404:@-
batch_normalization_47_23489406:@-
batch_normalization_47_23489408:@,
conv2d_61_23489412:@@ 
conv2d_61_23489414:@,
conv2d_62_23489417: @ 
conv2d_62_23489419:@-
batch_normalization_48_23489422:@-
batch_normalization_48_23489424:@-
batch_normalization_48_23489426:@-
batch_normalization_48_23489428:@"
dense_6_23489435:@

dense_6_23489437:

identity¢.batch_normalization_42/StatefulPartitionedCall¢.batch_normalization_43/StatefulPartitionedCall¢.batch_normalization_44/StatefulPartitionedCall¢.batch_normalization_45/StatefulPartitionedCall¢.batch_normalization_46/StatefulPartitionedCall¢.batch_normalization_47/StatefulPartitionedCall¢.batch_normalization_48/StatefulPartitionedCall¢!conv2d_54/StatefulPartitionedCall¢2conv2d_54/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_55/StatefulPartitionedCall¢2conv2d_55/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_56/StatefulPartitionedCall¢2conv2d_56/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_57/StatefulPartitionedCall¢2conv2d_57/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_58/StatefulPartitionedCall¢2conv2d_58/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_59/StatefulPartitionedCall¢2conv2d_59/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_60/StatefulPartitionedCall¢2conv2d_60/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_61/StatefulPartitionedCall¢2conv2d_61/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_62/StatefulPartitionedCall¢2conv2d_62/kernel/Regularizer/Square/ReadVariableOp¢dense_6/StatefulPartitionedCall
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCallinput_7conv2d_54_23489315conv2d_54_23489317*
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
G__inference_conv2d_54_layer_call_and_return_conditional_losses_23487977
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_42_23489320batch_normalization_42_23489322batch_normalization_42_23489324batch_normalization_42_23489326*
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
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23487547ý
activation_42/PartitionedCallPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0*
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
K__inference_activation_42_layer_call_and_return_conditional_losses_23487997¢
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall&activation_42/PartitionedCall:output:0conv2d_55_23489330conv2d_55_23489332*
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
G__inference_conv2d_55_layer_call_and_return_conditional_losses_23488015
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0batch_normalization_43_23489335batch_normalization_43_23489337batch_normalization_43_23489339batch_normalization_43_23489341*
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
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23487611ý
activation_43/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
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
K__inference_activation_43_layer_call_and_return_conditional_losses_23488035¢
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall&activation_43/PartitionedCall:output:0conv2d_56_23489345conv2d_56_23489347*
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
G__inference_conv2d_56_layer_call_and_return_conditional_losses_23488053
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_56/StatefulPartitionedCall:output:0batch_normalization_44_23489350batch_normalization_44_23489352batch_normalization_44_23489354batch_normalization_44_23489356*
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
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23487675
add_18/PartitionedCallPartitionedCall&activation_42/PartitionedCall:output:07batch_normalization_44/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *M
fHRF
D__inference_add_18_layer_call_and_return_conditional_losses_23488074å
activation_44/PartitionedCallPartitionedCalladd_18/PartitionedCall:output:0*
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
K__inference_activation_44_layer_call_and_return_conditional_losses_23488081¢
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCall&activation_44/PartitionedCall:output:0conv2d_57_23489361conv2d_57_23489363*
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
G__inference_conv2d_57_layer_call_and_return_conditional_losses_23488099
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0batch_normalization_45_23489366batch_normalization_45_23489368batch_normalization_45_23489370batch_normalization_45_23489372*
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
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23487739ý
activation_45/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
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
K__inference_activation_45_layer_call_and_return_conditional_losses_23488119¢
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall&activation_45/PartitionedCall:output:0conv2d_58_23489376conv2d_58_23489378*
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
G__inference_conv2d_58_layer_call_and_return_conditional_losses_23488137¢
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall&activation_44/PartitionedCall:output:0conv2d_59_23489381conv2d_59_23489383*
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
G__inference_conv2d_59_layer_call_and_return_conditional_losses_23488159
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0batch_normalization_46_23489386batch_normalization_46_23489388batch_normalization_46_23489390batch_normalization_46_23489392*
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
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23487803
add_19/PartitionedCallPartitionedCall*conv2d_59/StatefulPartitionedCall:output:07batch_normalization_46/StatefulPartitionedCall:output:0*
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
D__inference_add_19_layer_call_and_return_conditional_losses_23488180å
activation_46/PartitionedCallPartitionedCalladd_19/PartitionedCall:output:0*
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
K__inference_activation_46_layer_call_and_return_conditional_losses_23488187¢
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall&activation_46/PartitionedCall:output:0conv2d_60_23489397conv2d_60_23489399*
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
G__inference_conv2d_60_layer_call_and_return_conditional_losses_23488205
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0batch_normalization_47_23489402batch_normalization_47_23489404batch_normalization_47_23489406batch_normalization_47_23489408*
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
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23487867ý
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
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
K__inference_activation_47_layer_call_and_return_conditional_losses_23488225¢
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall&activation_47/PartitionedCall:output:0conv2d_61_23489412conv2d_61_23489414*
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
G__inference_conv2d_61_layer_call_and_return_conditional_losses_23488243¢
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall&activation_46/PartitionedCall:output:0conv2d_62_23489417conv2d_62_23489419*
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
G__inference_conv2d_62_layer_call_and_return_conditional_losses_23488265
.batch_normalization_48/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0batch_normalization_48_23489422batch_normalization_48_23489424batch_normalization_48_23489426batch_normalization_48_23489428*
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
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23487931
add_20/PartitionedCallPartitionedCall*conv2d_62/StatefulPartitionedCall:output:07batch_normalization_48/StatefulPartitionedCall:output:0*
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
D__inference_add_20_layer_call_and_return_conditional_losses_23488286å
activation_48/PartitionedCallPartitionedCalladd_20/PartitionedCall:output:0*
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
K__inference_activation_48_layer_call_and_return_conditional_losses_23488293ø
#average_pooling2d_6/PartitionedCallPartitionedCall&activation_48/PartitionedCall:output:0*
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
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_23487951â
flatten_6/PartitionedCallPartitionedCall,average_pooling2d_6/PartitionedCall:output:0*
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
G__inference_flatten_6_layer_call_and_return_conditional_losses_23488302
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_6_23489435dense_6_23489437*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_23488314
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_54_23489315*&
_output_shapes
:*
dtype0
#conv2d_54/kernel/Regularizer/SquareSquare:conv2d_54/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_54/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_54/kernel/Regularizer/SumSum'conv2d_54/kernel/Regularizer/Square:y:0+conv2d_54/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_54/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_54/kernel/Regularizer/mulMul+conv2d_54/kernel/Regularizer/mul/x:output:0)conv2d_54/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_55_23489330*&
_output_shapes
:*
dtype0
#conv2d_55/kernel/Regularizer/SquareSquare:conv2d_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_55/kernel/Regularizer/SumSum'conv2d_55/kernel/Regularizer/Square:y:0+conv2d_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_55/kernel/Regularizer/mulMul+conv2d_55/kernel/Regularizer/mul/x:output:0)conv2d_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_56_23489345*&
_output_shapes
:*
dtype0
#conv2d_56/kernel/Regularizer/SquareSquare:conv2d_56/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_56/kernel/Regularizer/SumSum'conv2d_56/kernel/Regularizer/Square:y:0+conv2d_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_56/kernel/Regularizer/mulMul+conv2d_56/kernel/Regularizer/mul/x:output:0)conv2d_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_57_23489361*&
_output_shapes
: *
dtype0
#conv2d_57/kernel/Regularizer/SquareSquare:conv2d_57/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_57/kernel/Regularizer/SumSum'conv2d_57/kernel/Regularizer/Square:y:0+conv2d_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_57/kernel/Regularizer/mulMul+conv2d_57/kernel/Regularizer/mul/x:output:0)conv2d_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_58_23489376*&
_output_shapes
:  *
dtype0
#conv2d_58/kernel/Regularizer/SquareSquare:conv2d_58/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_58/kernel/Regularizer/SumSum'conv2d_58/kernel/Regularizer/Square:y:0+conv2d_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_58/kernel/Regularizer/mulMul+conv2d_58/kernel/Regularizer/mul/x:output:0)conv2d_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_59_23489381*&
_output_shapes
: *
dtype0
#conv2d_59/kernel/Regularizer/SquareSquare:conv2d_59/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_59/kernel/Regularizer/SumSum'conv2d_59/kernel/Regularizer/Square:y:0+conv2d_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_59/kernel/Regularizer/mulMul+conv2d_59/kernel/Regularizer/mul/x:output:0)conv2d_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_60/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_60_23489397*&
_output_shapes
: @*
dtype0
#conv2d_60/kernel/Regularizer/SquareSquare:conv2d_60/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_60/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_60/kernel/Regularizer/SumSum'conv2d_60/kernel/Regularizer/Square:y:0+conv2d_60/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_60/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_60/kernel/Regularizer/mulMul+conv2d_60/kernel/Regularizer/mul/x:output:0)conv2d_60/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_61/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_61_23489412*&
_output_shapes
:@@*
dtype0
#conv2d_61/kernel/Regularizer/SquareSquare:conv2d_61/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@{
"conv2d_61/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_61/kernel/Regularizer/SumSum'conv2d_61/kernel/Regularizer/Square:y:0+conv2d_61/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_61/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_61/kernel/Regularizer/mulMul+conv2d_61/kernel/Regularizer/mul/x:output:0)conv2d_61/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_62/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_62_23489417*&
_output_shapes
: @*
dtype0
#conv2d_62/kernel/Regularizer/SquareSquare:conv2d_62/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_62/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_62/kernel/Regularizer/SumSum'conv2d_62/kernel/Regularizer/Square:y:0+conv2d_62/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_62/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_62/kernel/Regularizer/mulMul+conv2d_62/kernel/Regularizer/mul/x:output:0)conv2d_62/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
à	
NoOpNoOp/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_46/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_48/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall3^conv2d_54/kernel/Regularizer/Square/ReadVariableOp"^conv2d_55/StatefulPartitionedCall3^conv2d_55/kernel/Regularizer/Square/ReadVariableOp"^conv2d_56/StatefulPartitionedCall3^conv2d_56/kernel/Regularizer/Square/ReadVariableOp"^conv2d_57/StatefulPartitionedCall3^conv2d_57/kernel/Regularizer/Square/ReadVariableOp"^conv2d_58/StatefulPartitionedCall3^conv2d_58/kernel/Regularizer/Square/ReadVariableOp"^conv2d_59/StatefulPartitionedCall3^conv2d_59/kernel/Regularizer/Square/ReadVariableOp"^conv2d_60/StatefulPartitionedCall3^conv2d_60/kernel/Regularizer/Square/ReadVariableOp"^conv2d_61/StatefulPartitionedCall3^conv2d_61/kernel/Regularizer/Square/ReadVariableOp"^conv2d_62/StatefulPartitionedCall3^conv2d_62/kernel/Regularizer/Square/ReadVariableOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_46/StatefulPartitionedCall.batch_normalization_46/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_48/StatefulPartitionedCall.batch_normalization_48/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2h
2conv2d_54/kernel/Regularizer/Square/ReadVariableOp2conv2d_54/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2h
2conv2d_55/kernel/Regularizer/Square/ReadVariableOp2conv2d_55/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_56/StatefulPartitionedCall!conv2d_56/StatefulPartitionedCall2h
2conv2d_56/kernel/Regularizer/Square/ReadVariableOp2conv2d_56/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_57/StatefulPartitionedCall!conv2d_57/StatefulPartitionedCall2h
2conv2d_57/kernel/Regularizer/Square/ReadVariableOp2conv2d_57/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2h
2conv2d_58/kernel/Regularizer/Square/ReadVariableOp2conv2d_58/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2h
2conv2d_59/kernel/Regularizer/Square/ReadVariableOp2conv2d_59/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2h
2conv2d_60/kernel/Regularizer/Square/ReadVariableOp2conv2d_60/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2h
2conv2d_61/kernel/Regularizer/Square/ReadVariableOp2conv2d_61/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2h
2conv2d_62/kernel/Regularizer/Square/ReadVariableOp2conv2d_62/kernel/Regularizer/Square/ReadVariableOp2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_7
Ï

T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23487900

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
·b
¿
!__inference__traced_save_23491437
file_prefix/
+savev2_conv2d_54_kernel_read_readvariableop-
)savev2_conv2d_54_bias_read_readvariableop;
7savev2_batch_normalization_42_gamma_read_readvariableop:
6savev2_batch_normalization_42_beta_read_readvariableopA
=savev2_batch_normalization_42_moving_mean_read_readvariableopE
Asavev2_batch_normalization_42_moving_variance_read_readvariableop/
+savev2_conv2d_55_kernel_read_readvariableop-
)savev2_conv2d_55_bias_read_readvariableop;
7savev2_batch_normalization_43_gamma_read_readvariableop:
6savev2_batch_normalization_43_beta_read_readvariableopA
=savev2_batch_normalization_43_moving_mean_read_readvariableopE
Asavev2_batch_normalization_43_moving_variance_read_readvariableop/
+savev2_conv2d_56_kernel_read_readvariableop-
)savev2_conv2d_56_bias_read_readvariableop;
7savev2_batch_normalization_44_gamma_read_readvariableop:
6savev2_batch_normalization_44_beta_read_readvariableopA
=savev2_batch_normalization_44_moving_mean_read_readvariableopE
Asavev2_batch_normalization_44_moving_variance_read_readvariableop/
+savev2_conv2d_57_kernel_read_readvariableop-
)savev2_conv2d_57_bias_read_readvariableop;
7savev2_batch_normalization_45_gamma_read_readvariableop:
6savev2_batch_normalization_45_beta_read_readvariableopA
=savev2_batch_normalization_45_moving_mean_read_readvariableopE
Asavev2_batch_normalization_45_moving_variance_read_readvariableop/
+savev2_conv2d_58_kernel_read_readvariableop-
)savev2_conv2d_58_bias_read_readvariableop/
+savev2_conv2d_59_kernel_read_readvariableop-
)savev2_conv2d_59_bias_read_readvariableop;
7savev2_batch_normalization_46_gamma_read_readvariableop:
6savev2_batch_normalization_46_beta_read_readvariableopA
=savev2_batch_normalization_46_moving_mean_read_readvariableopE
Asavev2_batch_normalization_46_moving_variance_read_readvariableop/
+savev2_conv2d_60_kernel_read_readvariableop-
)savev2_conv2d_60_bias_read_readvariableop;
7savev2_batch_normalization_47_gamma_read_readvariableop:
6savev2_batch_normalization_47_beta_read_readvariableopA
=savev2_batch_normalization_47_moving_mean_read_readvariableopE
Asavev2_batch_normalization_47_moving_variance_read_readvariableop/
+savev2_conv2d_61_kernel_read_readvariableop-
)savev2_conv2d_61_bias_read_readvariableop/
+savev2_conv2d_62_kernel_read_readvariableop-
)savev2_conv2d_62_bias_read_readvariableop;
7savev2_batch_normalization_48_gamma_read_readvariableop:
6savev2_batch_normalization_48_beta_read_readvariableopA
=savev2_batch_normalization_48_moving_mean_read_readvariableopE
Asavev2_batch_normalization_48_moving_variance_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_54_kernel_read_readvariableop)savev2_conv2d_54_bias_read_readvariableop7savev2_batch_normalization_42_gamma_read_readvariableop6savev2_batch_normalization_42_beta_read_readvariableop=savev2_batch_normalization_42_moving_mean_read_readvariableopAsavev2_batch_normalization_42_moving_variance_read_readvariableop+savev2_conv2d_55_kernel_read_readvariableop)savev2_conv2d_55_bias_read_readvariableop7savev2_batch_normalization_43_gamma_read_readvariableop6savev2_batch_normalization_43_beta_read_readvariableop=savev2_batch_normalization_43_moving_mean_read_readvariableopAsavev2_batch_normalization_43_moving_variance_read_readvariableop+savev2_conv2d_56_kernel_read_readvariableop)savev2_conv2d_56_bias_read_readvariableop7savev2_batch_normalization_44_gamma_read_readvariableop6savev2_batch_normalization_44_beta_read_readvariableop=savev2_batch_normalization_44_moving_mean_read_readvariableopAsavev2_batch_normalization_44_moving_variance_read_readvariableop+savev2_conv2d_57_kernel_read_readvariableop)savev2_conv2d_57_bias_read_readvariableop7savev2_batch_normalization_45_gamma_read_readvariableop6savev2_batch_normalization_45_beta_read_readvariableop=savev2_batch_normalization_45_moving_mean_read_readvariableopAsavev2_batch_normalization_45_moving_variance_read_readvariableop+savev2_conv2d_58_kernel_read_readvariableop)savev2_conv2d_58_bias_read_readvariableop+savev2_conv2d_59_kernel_read_readvariableop)savev2_conv2d_59_bias_read_readvariableop7savev2_batch_normalization_46_gamma_read_readvariableop6savev2_batch_normalization_46_beta_read_readvariableop=savev2_batch_normalization_46_moving_mean_read_readvariableopAsavev2_batch_normalization_46_moving_variance_read_readvariableop+savev2_conv2d_60_kernel_read_readvariableop)savev2_conv2d_60_bias_read_readvariableop7savev2_batch_normalization_47_gamma_read_readvariableop6savev2_batch_normalization_47_beta_read_readvariableop=savev2_batch_normalization_47_moving_mean_read_readvariableopAsavev2_batch_normalization_47_moving_variance_read_readvariableop+savev2_conv2d_61_kernel_read_readvariableop)savev2_conv2d_61_bias_read_readvariableop+savev2_conv2d_62_kernel_read_readvariableop)savev2_conv2d_62_bias_read_readvariableop7savev2_batch_normalization_48_gamma_read_readvariableop6savev2_batch_normalization_48_beta_read_readvariableop=savev2_batch_normalization_48_moving_mean_read_readvariableopAsavev2_batch_normalization_48_moving_variance_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
Ï

T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23490490

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
â
µ
G__inference_conv2d_58_layer_call_and_return_conditional_losses_23490767

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_58/kernel/Regularizer/Square/ReadVariableOp|
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
2conv2d_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0
#conv2d_58/kernel/Regularizer/SquareSquare:conv2d_58/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_58/kernel/Regularizer/SumSum'conv2d_58/kernel/Regularizer/Square:y:0+conv2d_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_58/kernel/Regularizer/mulMul+conv2d_58/kernel/Regularizer/mul/x:output:0)conv2d_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_58/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_58/kernel/Regularizer/Square/ReadVariableOp2conv2d_58/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23491091

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
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23490860

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
Ë
L
0__inference_activation_45_layer_call_fn_23490731

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
K__inference_activation_45_layer_call_and_return_conditional_losses_23488119h
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
ð
¡
,__inference_conv2d_58_layer_call_fn_23490751

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
G__inference_conv2d_58_layer_call_and_return_conditional_losses_23488137w
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
¢
m
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_23487951

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

£
&__inference_signature_wrapper_23490312
input_7!
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
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
#__inference__wrapped_model_23487494o
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
_user_specified_name	input_7
ï
g
K__inference_activation_42_layer_call_and_return_conditional_losses_23490415

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
õ×
¿2
E__inference_model_6_layer_call_and_return_conditional_losses_23490209

inputsB
(conv2d_54_conv2d_readvariableop_resource:7
)conv2d_54_biasadd_readvariableop_resource:<
.batch_normalization_42_readvariableop_resource:>
0batch_normalization_42_readvariableop_1_resource:M
?batch_normalization_42_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_42_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_55_conv2d_readvariableop_resource:7
)conv2d_55_biasadd_readvariableop_resource:<
.batch_normalization_43_readvariableop_resource:>
0batch_normalization_43_readvariableop_1_resource:M
?batch_normalization_43_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_56_conv2d_readvariableop_resource:7
)conv2d_56_biasadd_readvariableop_resource:<
.batch_normalization_44_readvariableop_resource:>
0batch_normalization_44_readvariableop_1_resource:M
?batch_normalization_44_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_57_conv2d_readvariableop_resource: 7
)conv2d_57_biasadd_readvariableop_resource: <
.batch_normalization_45_readvariableop_resource: >
0batch_normalization_45_readvariableop_1_resource: M
?batch_normalization_45_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_58_conv2d_readvariableop_resource:  7
)conv2d_58_biasadd_readvariableop_resource: B
(conv2d_59_conv2d_readvariableop_resource: 7
)conv2d_59_biasadd_readvariableop_resource: <
.batch_normalization_46_readvariableop_resource: >
0batch_normalization_46_readvariableop_1_resource: M
?batch_normalization_46_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_60_conv2d_readvariableop_resource: @7
)conv2d_60_biasadd_readvariableop_resource:@<
.batch_normalization_47_readvariableop_resource:@>
0batch_normalization_47_readvariableop_1_resource:@M
?batch_normalization_47_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource:@B
(conv2d_61_conv2d_readvariableop_resource:@@7
)conv2d_61_biasadd_readvariableop_resource:@B
(conv2d_62_conv2d_readvariableop_resource: @7
)conv2d_62_biasadd_readvariableop_resource:@<
.batch_normalization_48_readvariableop_resource:@>
0batch_normalization_48_readvariableop_1_resource:@M
?batch_normalization_48_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource:@8
&dense_6_matmul_readvariableop_resource:@
5
'dense_6_biasadd_readvariableop_resource:

identity¢%batch_normalization_42/AssignNewValue¢'batch_normalization_42/AssignNewValue_1¢6batch_normalization_42/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_42/ReadVariableOp¢'batch_normalization_42/ReadVariableOp_1¢%batch_normalization_43/AssignNewValue¢'batch_normalization_43/AssignNewValue_1¢6batch_normalization_43/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_43/ReadVariableOp¢'batch_normalization_43/ReadVariableOp_1¢%batch_normalization_44/AssignNewValue¢'batch_normalization_44/AssignNewValue_1¢6batch_normalization_44/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_44/ReadVariableOp¢'batch_normalization_44/ReadVariableOp_1¢%batch_normalization_45/AssignNewValue¢'batch_normalization_45/AssignNewValue_1¢6batch_normalization_45/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_45/ReadVariableOp¢'batch_normalization_45/ReadVariableOp_1¢%batch_normalization_46/AssignNewValue¢'batch_normalization_46/AssignNewValue_1¢6batch_normalization_46/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_46/ReadVariableOp¢'batch_normalization_46/ReadVariableOp_1¢%batch_normalization_47/AssignNewValue¢'batch_normalization_47/AssignNewValue_1¢6batch_normalization_47/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_47/ReadVariableOp¢'batch_normalization_47/ReadVariableOp_1¢%batch_normalization_48/AssignNewValue¢'batch_normalization_48/AssignNewValue_1¢6batch_normalization_48/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_48/ReadVariableOp¢'batch_normalization_48/ReadVariableOp_1¢ conv2d_54/BiasAdd/ReadVariableOp¢conv2d_54/Conv2D/ReadVariableOp¢2conv2d_54/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_55/BiasAdd/ReadVariableOp¢conv2d_55/Conv2D/ReadVariableOp¢2conv2d_55/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_56/BiasAdd/ReadVariableOp¢conv2d_56/Conv2D/ReadVariableOp¢2conv2d_56/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_57/BiasAdd/ReadVariableOp¢conv2d_57/Conv2D/ReadVariableOp¢2conv2d_57/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_58/BiasAdd/ReadVariableOp¢conv2d_58/Conv2D/ReadVariableOp¢2conv2d_58/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_59/BiasAdd/ReadVariableOp¢conv2d_59/Conv2D/ReadVariableOp¢2conv2d_59/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_60/BiasAdd/ReadVariableOp¢conv2d_60/Conv2D/ReadVariableOp¢2conv2d_60/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_61/BiasAdd/ReadVariableOp¢conv2d_61/Conv2D/ReadVariableOp¢2conv2d_61/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_62/BiasAdd/ReadVariableOp¢conv2d_62/Conv2D/ReadVariableOp¢2conv2d_62/kernel/Regularizer/Square/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp
conv2d_54/Conv2D/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0­
conv2d_54/Conv2DConv2Dinputs'conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

 conv2d_54/BiasAdd/ReadVariableOpReadVariableOp)conv2d_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_54/BiasAddBiasAddconv2d_54/Conv2D:output:0(conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
%batch_normalization_42/ReadVariableOpReadVariableOp.batch_normalization_42_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_42/ReadVariableOp_1ReadVariableOp0batch_normalization_42_readvariableop_1_resource*
_output_shapes
:*
dtype0²
6batch_normalization_42/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_42_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¶
8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_42_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ë
'batch_normalization_42/FusedBatchNormV3FusedBatchNormV3conv2d_54/BiasAdd:output:0-batch_normalization_42/ReadVariableOp:value:0/batch_normalization_42/ReadVariableOp_1:value:0>batch_normalization_42/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_42/AssignNewValueAssignVariableOp?batch_normalization_42_fusedbatchnormv3_readvariableop_resource4batch_normalization_42/FusedBatchNormV3:batch_mean:07^batch_normalization_42/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_42/AssignNewValue_1AssignVariableOpAbatch_normalization_42_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_42/FusedBatchNormV3:batch_variance:09^batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
activation_42/ReluRelu+batch_normalization_42/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
conv2d_55/Conv2D/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ç
conv2d_55/Conv2DConv2D activation_42/Relu:activations:0'conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

 conv2d_55/BiasAdd/ReadVariableOpReadVariableOp)conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_55/BiasAddBiasAddconv2d_55/Conv2D:output:0(conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
%batch_normalization_43/ReadVariableOpReadVariableOp.batch_normalization_43_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_43/ReadVariableOp_1ReadVariableOp0batch_normalization_43_readvariableop_1_resource*
_output_shapes
:*
dtype0²
6batch_normalization_43/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_43_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¶
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ë
'batch_normalization_43/FusedBatchNormV3FusedBatchNormV3conv2d_55/BiasAdd:output:0-batch_normalization_43/ReadVariableOp:value:0/batch_normalization_43/ReadVariableOp_1:value:0>batch_normalization_43/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_43/AssignNewValueAssignVariableOp?batch_normalization_43_fusedbatchnormv3_readvariableop_resource4batch_normalization_43/FusedBatchNormV3:batch_mean:07^batch_normalization_43/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_43/AssignNewValue_1AssignVariableOpAbatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_43/FusedBatchNormV3:batch_variance:09^batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
activation_43/ReluRelu+batch_normalization_43/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
conv2d_56/Conv2D/ReadVariableOpReadVariableOp(conv2d_56_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ç
conv2d_56/Conv2DConv2D activation_43/Relu:activations:0'conv2d_56/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

 conv2d_56/BiasAdd/ReadVariableOpReadVariableOp)conv2d_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_56/BiasAddBiasAddconv2d_56/Conv2D:output:0(conv2d_56/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
%batch_normalization_44/ReadVariableOpReadVariableOp.batch_normalization_44_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_44/ReadVariableOp_1ReadVariableOp0batch_normalization_44_readvariableop_1_resource*
_output_shapes
:*
dtype0²
6batch_normalization_44/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_44_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¶
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ë
'batch_normalization_44/FusedBatchNormV3FusedBatchNormV3conv2d_56/BiasAdd:output:0-batch_normalization_44/ReadVariableOp:value:0/batch_normalization_44/ReadVariableOp_1:value:0>batch_normalization_44/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_44/AssignNewValueAssignVariableOp?batch_normalization_44_fusedbatchnormv3_readvariableop_resource4batch_normalization_44/FusedBatchNormV3:batch_mean:07^batch_normalization_44/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_44/AssignNewValue_1AssignVariableOpAbatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_44/FusedBatchNormV3:batch_variance:09^batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0

add_18/addAddV2 activation_42/Relu:activations:0+batch_normalization_44/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  d
activation_44/ReluReluadd_18/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
conv2d_57/Conv2D/ReadVariableOpReadVariableOp(conv2d_57_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ç
conv2d_57/Conv2DConv2D activation_44/Relu:activations:0'conv2d_57/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

 conv2d_57/BiasAdd/ReadVariableOpReadVariableOp)conv2d_57_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_57/BiasAddBiasAddconv2d_57/Conv2D:output:0(conv2d_57/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%batch_normalization_45/ReadVariableOpReadVariableOp.batch_normalization_45_readvariableop_resource*
_output_shapes
: *
dtype0
'batch_normalization_45/ReadVariableOp_1ReadVariableOp0batch_normalization_45_readvariableop_1_resource*
_output_shapes
: *
dtype0²
6batch_normalization_45/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_45_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¶
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ë
'batch_normalization_45/FusedBatchNormV3FusedBatchNormV3conv2d_57/BiasAdd:output:0-batch_normalization_45/ReadVariableOp:value:0/batch_normalization_45/ReadVariableOp_1:value:0>batch_normalization_45/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_45/AssignNewValueAssignVariableOp?batch_normalization_45_fusedbatchnormv3_readvariableop_resource4batch_normalization_45/FusedBatchNormV3:batch_mean:07^batch_normalization_45/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_45/AssignNewValue_1AssignVariableOpAbatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_45/FusedBatchNormV3:batch_variance:09^batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
activation_45/ReluRelu+batch_normalization_45/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_58/Conv2D/ReadVariableOpReadVariableOp(conv2d_58_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Ç
conv2d_58/Conv2DConv2D activation_45/Relu:activations:0'conv2d_58/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

 conv2d_58/BiasAdd/ReadVariableOpReadVariableOp)conv2d_58_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_58/BiasAddBiasAddconv2d_58/Conv2D:output:0(conv2d_58/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_59/Conv2D/ReadVariableOpReadVariableOp(conv2d_59_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ç
conv2d_59/Conv2DConv2D activation_44/Relu:activations:0'conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

 conv2d_59/BiasAdd/ReadVariableOpReadVariableOp)conv2d_59_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_59/BiasAddBiasAddconv2d_59/Conv2D:output:0(conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%batch_normalization_46/ReadVariableOpReadVariableOp.batch_normalization_46_readvariableop_resource*
_output_shapes
: *
dtype0
'batch_normalization_46/ReadVariableOp_1ReadVariableOp0batch_normalization_46_readvariableop_1_resource*
_output_shapes
: *
dtype0²
6batch_normalization_46/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_46_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¶
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ë
'batch_normalization_46/FusedBatchNormV3FusedBatchNormV3conv2d_58/BiasAdd:output:0-batch_normalization_46/ReadVariableOp:value:0/batch_normalization_46/ReadVariableOp_1:value:0>batch_normalization_46/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_46/AssignNewValueAssignVariableOp?batch_normalization_46_fusedbatchnormv3_readvariableop_resource4batch_normalization_46/FusedBatchNormV3:batch_mean:07^batch_normalization_46/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_46/AssignNewValue_1AssignVariableOpAbatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_46/FusedBatchNormV3:batch_variance:09^batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0

add_19/addAddV2conv2d_59/BiasAdd:output:0+batch_normalization_46/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
activation_46/ReluReluadd_19/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ç
conv2d_60/Conv2DConv2D activation_46/Relu:activations:0'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%batch_normalization_47/ReadVariableOpReadVariableOp.batch_normalization_47_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_47/ReadVariableOp_1ReadVariableOp0batch_normalization_47_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ë
'batch_normalization_47/FusedBatchNormV3FusedBatchNormV3conv2d_60/BiasAdd:output:0-batch_normalization_47/ReadVariableOp:value:0/batch_normalization_47/ReadVariableOp_1:value:0>batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_47/AssignNewValueAssignVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource4batch_normalization_47/FusedBatchNormV3:batch_mean:07^batch_normalization_47/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_47/AssignNewValue_1AssignVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_47/FusedBatchNormV3:batch_variance:09^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
activation_47/ReluRelu+batch_normalization_47/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_61/Conv2D/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ç
conv2d_61/Conv2DConv2D activation_47/Relu:activations:0'conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

 conv2d_61/BiasAdd/ReadVariableOpReadVariableOp)conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_61/BiasAddBiasAddconv2d_61/Conv2D:output:0(conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ç
conv2d_62/Conv2DConv2D activation_46/Relu:activations:0'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%batch_normalization_48/ReadVariableOpReadVariableOp.batch_normalization_48_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_48/ReadVariableOp_1ReadVariableOp0batch_normalization_48_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ë
'batch_normalization_48/FusedBatchNormV3FusedBatchNormV3conv2d_61/BiasAdd:output:0-batch_normalization_48/ReadVariableOp:value:0/batch_normalization_48/ReadVariableOp_1:value:0>batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_48/AssignNewValueAssignVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource4batch_normalization_48/FusedBatchNormV3:batch_mean:07^batch_normalization_48/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_48/AssignNewValue_1AssignVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_48/FusedBatchNormV3:batch_variance:09^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0

add_20/addAddV2conv2d_62/BiasAdd:output:0+batch_normalization_48/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
activation_48/ReluReluadd_20/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
average_pooling2d_6/AvgPoolAvgPool activation_48/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   
flatten_6/ReshapeReshape$average_pooling2d_6/AvgPool:output:0flatten_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype0
dense_6/MatMulMatMulflatten_6/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
£
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_54/kernel/Regularizer/SquareSquare:conv2d_54/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_54/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_54/kernel/Regularizer/SumSum'conv2d_54/kernel/Regularizer/Square:y:0+conv2d_54/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_54/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_54/kernel/Regularizer/mulMul+conv2d_54/kernel/Regularizer/mul/x:output:0)conv2d_54/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_55/kernel/Regularizer/SquareSquare:conv2d_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_55/kernel/Regularizer/SumSum'conv2d_55/kernel/Regularizer/Square:y:0+conv2d_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_55/kernel/Regularizer/mulMul+conv2d_55/kernel/Regularizer/mul/x:output:0)conv2d_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_56_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_56/kernel/Regularizer/SquareSquare:conv2d_56/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_56/kernel/Regularizer/SumSum'conv2d_56/kernel/Regularizer/Square:y:0+conv2d_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_56/kernel/Regularizer/mulMul+conv2d_56/kernel/Regularizer/mul/x:output:0)conv2d_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_57_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_57/kernel/Regularizer/SquareSquare:conv2d_57/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_57/kernel/Regularizer/SumSum'conv2d_57/kernel/Regularizer/Square:y:0+conv2d_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_57/kernel/Regularizer/mulMul+conv2d_57/kernel/Regularizer/mul/x:output:0)conv2d_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_58_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0
#conv2d_58/kernel/Regularizer/SquareSquare:conv2d_58/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_58/kernel/Regularizer/SumSum'conv2d_58/kernel/Regularizer/Square:y:0+conv2d_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_58/kernel/Regularizer/mulMul+conv2d_58/kernel/Regularizer/mul/x:output:0)conv2d_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_59_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_59/kernel/Regularizer/SquareSquare:conv2d_59/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_59/kernel/Regularizer/SumSum'conv2d_59/kernel/Regularizer/Square:y:0+conv2d_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_59/kernel/Regularizer/mulMul+conv2d_59/kernel/Regularizer/mul/x:output:0)conv2d_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_60/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_60/kernel/Regularizer/SquareSquare:conv2d_60/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_60/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_60/kernel/Regularizer/SumSum'conv2d_60/kernel/Regularizer/Square:y:0+conv2d_60/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_60/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_60/kernel/Regularizer/mulMul+conv2d_60/kernel/Regularizer/mul/x:output:0)conv2d_60/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_61/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
#conv2d_61/kernel/Regularizer/SquareSquare:conv2d_61/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@{
"conv2d_61/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_61/kernel/Regularizer/SumSum'conv2d_61/kernel/Regularizer/Square:y:0+conv2d_61/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_61/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_61/kernel/Regularizer/mulMul+conv2d_61/kernel/Regularizer/mul/x:output:0)conv2d_61/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
2conv2d_62/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_62/kernel/Regularizer/SquareSquare:conv2d_62/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_62/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_62/kernel/Regularizer/SumSum'conv2d_62/kernel/Regularizer/Square:y:0+conv2d_62/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_62/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_62/kernel/Regularizer/mulMul+conv2d_62/kernel/Regularizer/mul/x:output:0)conv2d_62/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ù
NoOpNoOp&^batch_normalization_42/AssignNewValue(^batch_normalization_42/AssignNewValue_17^batch_normalization_42/FusedBatchNormV3/ReadVariableOp9^batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_42/ReadVariableOp(^batch_normalization_42/ReadVariableOp_1&^batch_normalization_43/AssignNewValue(^batch_normalization_43/AssignNewValue_17^batch_normalization_43/FusedBatchNormV3/ReadVariableOp9^batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_43/ReadVariableOp(^batch_normalization_43/ReadVariableOp_1&^batch_normalization_44/AssignNewValue(^batch_normalization_44/AssignNewValue_17^batch_normalization_44/FusedBatchNormV3/ReadVariableOp9^batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_44/ReadVariableOp(^batch_normalization_44/ReadVariableOp_1&^batch_normalization_45/AssignNewValue(^batch_normalization_45/AssignNewValue_17^batch_normalization_45/FusedBatchNormV3/ReadVariableOp9^batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_45/ReadVariableOp(^batch_normalization_45/ReadVariableOp_1&^batch_normalization_46/AssignNewValue(^batch_normalization_46/AssignNewValue_17^batch_normalization_46/FusedBatchNormV3/ReadVariableOp9^batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_46/ReadVariableOp(^batch_normalization_46/ReadVariableOp_1&^batch_normalization_47/AssignNewValue(^batch_normalization_47/AssignNewValue_17^batch_normalization_47/FusedBatchNormV3/ReadVariableOp9^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_47/ReadVariableOp(^batch_normalization_47/ReadVariableOp_1&^batch_normalization_48/AssignNewValue(^batch_normalization_48/AssignNewValue_17^batch_normalization_48/FusedBatchNormV3/ReadVariableOp9^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_48/ReadVariableOp(^batch_normalization_48/ReadVariableOp_1!^conv2d_54/BiasAdd/ReadVariableOp ^conv2d_54/Conv2D/ReadVariableOp3^conv2d_54/kernel/Regularizer/Square/ReadVariableOp!^conv2d_55/BiasAdd/ReadVariableOp ^conv2d_55/Conv2D/ReadVariableOp3^conv2d_55/kernel/Regularizer/Square/ReadVariableOp!^conv2d_56/BiasAdd/ReadVariableOp ^conv2d_56/Conv2D/ReadVariableOp3^conv2d_56/kernel/Regularizer/Square/ReadVariableOp!^conv2d_57/BiasAdd/ReadVariableOp ^conv2d_57/Conv2D/ReadVariableOp3^conv2d_57/kernel/Regularizer/Square/ReadVariableOp!^conv2d_58/BiasAdd/ReadVariableOp ^conv2d_58/Conv2D/ReadVariableOp3^conv2d_58/kernel/Regularizer/Square/ReadVariableOp!^conv2d_59/BiasAdd/ReadVariableOp ^conv2d_59/Conv2D/ReadVariableOp3^conv2d_59/kernel/Regularizer/Square/ReadVariableOp!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp3^conv2d_60/kernel/Regularizer/Square/ReadVariableOp!^conv2d_61/BiasAdd/ReadVariableOp ^conv2d_61/Conv2D/ReadVariableOp3^conv2d_61/kernel/Regularizer/Square/ReadVariableOp!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp3^conv2d_62/kernel/Regularizer/Square/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_42/AssignNewValue%batch_normalization_42/AssignNewValue2R
'batch_normalization_42/AssignNewValue_1'batch_normalization_42/AssignNewValue_12p
6batch_normalization_42/FusedBatchNormV3/ReadVariableOp6batch_normalization_42/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_42/FusedBatchNormV3/ReadVariableOp_18batch_normalization_42/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_42/ReadVariableOp%batch_normalization_42/ReadVariableOp2R
'batch_normalization_42/ReadVariableOp_1'batch_normalization_42/ReadVariableOp_12N
%batch_normalization_43/AssignNewValue%batch_normalization_43/AssignNewValue2R
'batch_normalization_43/AssignNewValue_1'batch_normalization_43/AssignNewValue_12p
6batch_normalization_43/FusedBatchNormV3/ReadVariableOp6batch_normalization_43/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_18batch_normalization_43/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_43/ReadVariableOp%batch_normalization_43/ReadVariableOp2R
'batch_normalization_43/ReadVariableOp_1'batch_normalization_43/ReadVariableOp_12N
%batch_normalization_44/AssignNewValue%batch_normalization_44/AssignNewValue2R
'batch_normalization_44/AssignNewValue_1'batch_normalization_44/AssignNewValue_12p
6batch_normalization_44/FusedBatchNormV3/ReadVariableOp6batch_normalization_44/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_18batch_normalization_44/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_44/ReadVariableOp%batch_normalization_44/ReadVariableOp2R
'batch_normalization_44/ReadVariableOp_1'batch_normalization_44/ReadVariableOp_12N
%batch_normalization_45/AssignNewValue%batch_normalization_45/AssignNewValue2R
'batch_normalization_45/AssignNewValue_1'batch_normalization_45/AssignNewValue_12p
6batch_normalization_45/FusedBatchNormV3/ReadVariableOp6batch_normalization_45/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_18batch_normalization_45/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_45/ReadVariableOp%batch_normalization_45/ReadVariableOp2R
'batch_normalization_45/ReadVariableOp_1'batch_normalization_45/ReadVariableOp_12N
%batch_normalization_46/AssignNewValue%batch_normalization_46/AssignNewValue2R
'batch_normalization_46/AssignNewValue_1'batch_normalization_46/AssignNewValue_12p
6batch_normalization_46/FusedBatchNormV3/ReadVariableOp6batch_normalization_46/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_18batch_normalization_46/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_46/ReadVariableOp%batch_normalization_46/ReadVariableOp2R
'batch_normalization_46/ReadVariableOp_1'batch_normalization_46/ReadVariableOp_12N
%batch_normalization_47/AssignNewValue%batch_normalization_47/AssignNewValue2R
'batch_normalization_47/AssignNewValue_1'batch_normalization_47/AssignNewValue_12p
6batch_normalization_47/FusedBatchNormV3/ReadVariableOp6batch_normalization_47/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_18batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_47/ReadVariableOp%batch_normalization_47/ReadVariableOp2R
'batch_normalization_47/ReadVariableOp_1'batch_normalization_47/ReadVariableOp_12N
%batch_normalization_48/AssignNewValue%batch_normalization_48/AssignNewValue2R
'batch_normalization_48/AssignNewValue_1'batch_normalization_48/AssignNewValue_12p
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp6batch_normalization_48/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_18batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_48/ReadVariableOp%batch_normalization_48/ReadVariableOp2R
'batch_normalization_48/ReadVariableOp_1'batch_normalization_48/ReadVariableOp_12D
 conv2d_54/BiasAdd/ReadVariableOp conv2d_54/BiasAdd/ReadVariableOp2B
conv2d_54/Conv2D/ReadVariableOpconv2d_54/Conv2D/ReadVariableOp2h
2conv2d_54/kernel/Regularizer/Square/ReadVariableOp2conv2d_54/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_55/BiasAdd/ReadVariableOp conv2d_55/BiasAdd/ReadVariableOp2B
conv2d_55/Conv2D/ReadVariableOpconv2d_55/Conv2D/ReadVariableOp2h
2conv2d_55/kernel/Regularizer/Square/ReadVariableOp2conv2d_55/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_56/BiasAdd/ReadVariableOp conv2d_56/BiasAdd/ReadVariableOp2B
conv2d_56/Conv2D/ReadVariableOpconv2d_56/Conv2D/ReadVariableOp2h
2conv2d_56/kernel/Regularizer/Square/ReadVariableOp2conv2d_56/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_57/BiasAdd/ReadVariableOp conv2d_57/BiasAdd/ReadVariableOp2B
conv2d_57/Conv2D/ReadVariableOpconv2d_57/Conv2D/ReadVariableOp2h
2conv2d_57/kernel/Regularizer/Square/ReadVariableOp2conv2d_57/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_58/BiasAdd/ReadVariableOp conv2d_58/BiasAdd/ReadVariableOp2B
conv2d_58/Conv2D/ReadVariableOpconv2d_58/Conv2D/ReadVariableOp2h
2conv2d_58/kernel/Regularizer/Square/ReadVariableOp2conv2d_58/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_59/BiasAdd/ReadVariableOp conv2d_59/BiasAdd/ReadVariableOp2B
conv2d_59/Conv2D/ReadVariableOpconv2d_59/Conv2D/ReadVariableOp2h
2conv2d_59/kernel/Regularizer/Square/ReadVariableOp2conv2d_59/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp2h
2conv2d_60/kernel/Regularizer/Square/ReadVariableOp2conv2d_60/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_61/BiasAdd/ReadVariableOp conv2d_61/BiasAdd/ReadVariableOp2B
conv2d_61/Conv2D/ReadVariableOpconv2d_61/Conv2D/ReadVariableOp2h
2conv2d_61/kernel/Regularizer/Square/ReadVariableOp2conv2d_61/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp2h
2conv2d_62/kernel/Regularizer/Square/ReadVariableOp2conv2d_62/kernel/Regularizer/Square/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Á÷
0
#__inference__wrapped_model_23487494
input_7J
0model_6_conv2d_54_conv2d_readvariableop_resource:?
1model_6_conv2d_54_biasadd_readvariableop_resource:D
6model_6_batch_normalization_42_readvariableop_resource:F
8model_6_batch_normalization_42_readvariableop_1_resource:U
Gmodel_6_batch_normalization_42_fusedbatchnormv3_readvariableop_resource:W
Imodel_6_batch_normalization_42_fusedbatchnormv3_readvariableop_1_resource:J
0model_6_conv2d_55_conv2d_readvariableop_resource:?
1model_6_conv2d_55_biasadd_readvariableop_resource:D
6model_6_batch_normalization_43_readvariableop_resource:F
8model_6_batch_normalization_43_readvariableop_1_resource:U
Gmodel_6_batch_normalization_43_fusedbatchnormv3_readvariableop_resource:W
Imodel_6_batch_normalization_43_fusedbatchnormv3_readvariableop_1_resource:J
0model_6_conv2d_56_conv2d_readvariableop_resource:?
1model_6_conv2d_56_biasadd_readvariableop_resource:D
6model_6_batch_normalization_44_readvariableop_resource:F
8model_6_batch_normalization_44_readvariableop_1_resource:U
Gmodel_6_batch_normalization_44_fusedbatchnormv3_readvariableop_resource:W
Imodel_6_batch_normalization_44_fusedbatchnormv3_readvariableop_1_resource:J
0model_6_conv2d_57_conv2d_readvariableop_resource: ?
1model_6_conv2d_57_biasadd_readvariableop_resource: D
6model_6_batch_normalization_45_readvariableop_resource: F
8model_6_batch_normalization_45_readvariableop_1_resource: U
Gmodel_6_batch_normalization_45_fusedbatchnormv3_readvariableop_resource: W
Imodel_6_batch_normalization_45_fusedbatchnormv3_readvariableop_1_resource: J
0model_6_conv2d_58_conv2d_readvariableop_resource:  ?
1model_6_conv2d_58_biasadd_readvariableop_resource: J
0model_6_conv2d_59_conv2d_readvariableop_resource: ?
1model_6_conv2d_59_biasadd_readvariableop_resource: D
6model_6_batch_normalization_46_readvariableop_resource: F
8model_6_batch_normalization_46_readvariableop_1_resource: U
Gmodel_6_batch_normalization_46_fusedbatchnormv3_readvariableop_resource: W
Imodel_6_batch_normalization_46_fusedbatchnormv3_readvariableop_1_resource: J
0model_6_conv2d_60_conv2d_readvariableop_resource: @?
1model_6_conv2d_60_biasadd_readvariableop_resource:@D
6model_6_batch_normalization_47_readvariableop_resource:@F
8model_6_batch_normalization_47_readvariableop_1_resource:@U
Gmodel_6_batch_normalization_47_fusedbatchnormv3_readvariableop_resource:@W
Imodel_6_batch_normalization_47_fusedbatchnormv3_readvariableop_1_resource:@J
0model_6_conv2d_61_conv2d_readvariableop_resource:@@?
1model_6_conv2d_61_biasadd_readvariableop_resource:@J
0model_6_conv2d_62_conv2d_readvariableop_resource: @?
1model_6_conv2d_62_biasadd_readvariableop_resource:@D
6model_6_batch_normalization_48_readvariableop_resource:@F
8model_6_batch_normalization_48_readvariableop_1_resource:@U
Gmodel_6_batch_normalization_48_fusedbatchnormv3_readvariableop_resource:@W
Imodel_6_batch_normalization_48_fusedbatchnormv3_readvariableop_1_resource:@@
.model_6_dense_6_matmul_readvariableop_resource:@
=
/model_6_dense_6_biasadd_readvariableop_resource:

identity¢>model_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOp¢@model_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1¢-model_6/batch_normalization_42/ReadVariableOp¢/model_6/batch_normalization_42/ReadVariableOp_1¢>model_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOp¢@model_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1¢-model_6/batch_normalization_43/ReadVariableOp¢/model_6/batch_normalization_43/ReadVariableOp_1¢>model_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOp¢@model_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1¢-model_6/batch_normalization_44/ReadVariableOp¢/model_6/batch_normalization_44/ReadVariableOp_1¢>model_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOp¢@model_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1¢-model_6/batch_normalization_45/ReadVariableOp¢/model_6/batch_normalization_45/ReadVariableOp_1¢>model_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOp¢@model_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1¢-model_6/batch_normalization_46/ReadVariableOp¢/model_6/batch_normalization_46/ReadVariableOp_1¢>model_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOp¢@model_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1¢-model_6/batch_normalization_47/ReadVariableOp¢/model_6/batch_normalization_47/ReadVariableOp_1¢>model_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOp¢@model_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1¢-model_6/batch_normalization_48/ReadVariableOp¢/model_6/batch_normalization_48/ReadVariableOp_1¢(model_6/conv2d_54/BiasAdd/ReadVariableOp¢'model_6/conv2d_54/Conv2D/ReadVariableOp¢(model_6/conv2d_55/BiasAdd/ReadVariableOp¢'model_6/conv2d_55/Conv2D/ReadVariableOp¢(model_6/conv2d_56/BiasAdd/ReadVariableOp¢'model_6/conv2d_56/Conv2D/ReadVariableOp¢(model_6/conv2d_57/BiasAdd/ReadVariableOp¢'model_6/conv2d_57/Conv2D/ReadVariableOp¢(model_6/conv2d_58/BiasAdd/ReadVariableOp¢'model_6/conv2d_58/Conv2D/ReadVariableOp¢(model_6/conv2d_59/BiasAdd/ReadVariableOp¢'model_6/conv2d_59/Conv2D/ReadVariableOp¢(model_6/conv2d_60/BiasAdd/ReadVariableOp¢'model_6/conv2d_60/Conv2D/ReadVariableOp¢(model_6/conv2d_61/BiasAdd/ReadVariableOp¢'model_6/conv2d_61/Conv2D/ReadVariableOp¢(model_6/conv2d_62/BiasAdd/ReadVariableOp¢'model_6/conv2d_62/Conv2D/ReadVariableOp¢&model_6/dense_6/BiasAdd/ReadVariableOp¢%model_6/dense_6/MatMul/ReadVariableOp 
'model_6/conv2d_54/Conv2D/ReadVariableOpReadVariableOp0model_6_conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¾
model_6/conv2d_54/Conv2DConv2Dinput_7/model_6/conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

(model_6/conv2d_54/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv2d_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0³
model_6/conv2d_54/BiasAddBiasAdd!model_6/conv2d_54/Conv2D:output:00model_6/conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   
-model_6/batch_normalization_42/ReadVariableOpReadVariableOp6model_6_batch_normalization_42_readvariableop_resource*
_output_shapes
:*
dtype0¤
/model_6/batch_normalization_42/ReadVariableOp_1ReadVariableOp8model_6_batch_normalization_42_readvariableop_1_resource*
_output_shapes
:*
dtype0Â
>model_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_6_batch_normalization_42_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Æ
@model_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_6_batch_normalization_42_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0í
/model_6/batch_normalization_42/FusedBatchNormV3FusedBatchNormV3"model_6/conv2d_54/BiasAdd:output:05model_6/batch_normalization_42/ReadVariableOp:value:07model_6/batch_normalization_42/ReadVariableOp_1:value:0Fmodel_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
is_training( 
model_6/activation_42/ReluRelu3model_6/batch_normalization_42/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   
'model_6/conv2d_55/Conv2D/ReadVariableOpReadVariableOp0model_6_conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ß
model_6/conv2d_55/Conv2DConv2D(model_6/activation_42/Relu:activations:0/model_6/conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

(model_6/conv2d_55/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0³
model_6/conv2d_55/BiasAddBiasAdd!model_6/conv2d_55/Conv2D:output:00model_6/conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   
-model_6/batch_normalization_43/ReadVariableOpReadVariableOp6model_6_batch_normalization_43_readvariableop_resource*
_output_shapes
:*
dtype0¤
/model_6/batch_normalization_43/ReadVariableOp_1ReadVariableOp8model_6_batch_normalization_43_readvariableop_1_resource*
_output_shapes
:*
dtype0Â
>model_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_6_batch_normalization_43_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Æ
@model_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_6_batch_normalization_43_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0í
/model_6/batch_normalization_43/FusedBatchNormV3FusedBatchNormV3"model_6/conv2d_55/BiasAdd:output:05model_6/batch_normalization_43/ReadVariableOp:value:07model_6/batch_normalization_43/ReadVariableOp_1:value:0Fmodel_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
is_training( 
model_6/activation_43/ReluRelu3model_6/batch_normalization_43/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   
'model_6/conv2d_56/Conv2D/ReadVariableOpReadVariableOp0model_6_conv2d_56_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ß
model_6/conv2d_56/Conv2DConv2D(model_6/activation_43/Relu:activations:0/model_6/conv2d_56/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

(model_6/conv2d_56/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv2d_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0³
model_6/conv2d_56/BiasAddBiasAdd!model_6/conv2d_56/Conv2D:output:00model_6/conv2d_56/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   
-model_6/batch_normalization_44/ReadVariableOpReadVariableOp6model_6_batch_normalization_44_readvariableop_resource*
_output_shapes
:*
dtype0¤
/model_6/batch_normalization_44/ReadVariableOp_1ReadVariableOp8model_6_batch_normalization_44_readvariableop_1_resource*
_output_shapes
:*
dtype0Â
>model_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_6_batch_normalization_44_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Æ
@model_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_6_batch_normalization_44_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0í
/model_6/batch_normalization_44/FusedBatchNormV3FusedBatchNormV3"model_6/conv2d_56/BiasAdd:output:05model_6/batch_normalization_44/ReadVariableOp:value:07model_6/batch_normalization_44/ReadVariableOp_1:value:0Fmodel_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  :::::*
epsilon%o:*
is_training( ´
model_6/add_18/addAddV2(model_6/activation_42/Relu:activations:03model_6/batch_normalization_44/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  t
model_6/activation_44/ReluRelumodel_6/add_18/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   
'model_6/conv2d_57/Conv2D/ReadVariableOpReadVariableOp0model_6_conv2d_57_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ß
model_6/conv2d_57/Conv2DConv2D(model_6/activation_44/Relu:activations:0/model_6/conv2d_57/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

(model_6/conv2d_57/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv2d_57_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0³
model_6/conv2d_57/BiasAddBiasAdd!model_6/conv2d_57/Conv2D:output:00model_6/conv2d_57/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
-model_6/batch_normalization_45/ReadVariableOpReadVariableOp6model_6_batch_normalization_45_readvariableop_resource*
_output_shapes
: *
dtype0¤
/model_6/batch_normalization_45/ReadVariableOp_1ReadVariableOp8model_6_batch_normalization_45_readvariableop_1_resource*
_output_shapes
: *
dtype0Â
>model_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_6_batch_normalization_45_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Æ
@model_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_6_batch_normalization_45_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0í
/model_6/batch_normalization_45/FusedBatchNormV3FusedBatchNormV3"model_6/conv2d_57/BiasAdd:output:05model_6/batch_normalization_45/ReadVariableOp:value:07model_6/batch_normalization_45/ReadVariableOp_1:value:0Fmodel_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 
model_6/activation_45/ReluRelu3model_6/batch_normalization_45/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
'model_6/conv2d_58/Conv2D/ReadVariableOpReadVariableOp0model_6_conv2d_58_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0ß
model_6/conv2d_58/Conv2DConv2D(model_6/activation_45/Relu:activations:0/model_6/conv2d_58/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

(model_6/conv2d_58/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv2d_58_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0³
model_6/conv2d_58/BiasAddBiasAdd!model_6/conv2d_58/Conv2D:output:00model_6/conv2d_58/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
'model_6/conv2d_59/Conv2D/ReadVariableOpReadVariableOp0model_6_conv2d_59_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ß
model_6/conv2d_59/Conv2DConv2D(model_6/activation_44/Relu:activations:0/model_6/conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

(model_6/conv2d_59/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv2d_59_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0³
model_6/conv2d_59/BiasAddBiasAdd!model_6/conv2d_59/Conv2D:output:00model_6/conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
-model_6/batch_normalization_46/ReadVariableOpReadVariableOp6model_6_batch_normalization_46_readvariableop_resource*
_output_shapes
: *
dtype0¤
/model_6/batch_normalization_46/ReadVariableOp_1ReadVariableOp8model_6_batch_normalization_46_readvariableop_1_resource*
_output_shapes
: *
dtype0Â
>model_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_6_batch_normalization_46_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Æ
@model_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_6_batch_normalization_46_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0í
/model_6/batch_normalization_46/FusedBatchNormV3FusedBatchNormV3"model_6/conv2d_58/BiasAdd:output:05model_6/batch_normalization_46/ReadVariableOp:value:07model_6/batch_normalization_46/ReadVariableOp_1:value:0Fmodel_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( ®
model_6/add_19/addAddV2"model_6/conv2d_59/BiasAdd:output:03model_6/batch_normalization_46/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ t
model_6/activation_46/ReluRelumodel_6/add_19/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
'model_6/conv2d_60/Conv2D/ReadVariableOpReadVariableOp0model_6_conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0ß
model_6/conv2d_60/Conv2DConv2D(model_6/activation_46/Relu:activations:0/model_6/conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

(model_6/conv2d_60/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
model_6/conv2d_60/BiasAddBiasAdd!model_6/conv2d_60/Conv2D:output:00model_6/conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
-model_6/batch_normalization_47/ReadVariableOpReadVariableOp6model_6_batch_normalization_47_readvariableop_resource*
_output_shapes
:@*
dtype0¤
/model_6/batch_normalization_47/ReadVariableOp_1ReadVariableOp8model_6_batch_normalization_47_readvariableop_1_resource*
_output_shapes
:@*
dtype0Â
>model_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_6_batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Æ
@model_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_6_batch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0í
/model_6/batch_normalization_47/FusedBatchNormV3FusedBatchNormV3"model_6/conv2d_60/BiasAdd:output:05model_6/batch_normalization_47/ReadVariableOp:value:07model_6/batch_normalization_47/ReadVariableOp_1:value:0Fmodel_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 
model_6/activation_47/ReluRelu3model_6/batch_normalization_47/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
'model_6/conv2d_61/Conv2D/ReadVariableOpReadVariableOp0model_6_conv2d_61_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ß
model_6/conv2d_61/Conv2DConv2D(model_6/activation_47/Relu:activations:0/model_6/conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

(model_6/conv2d_61/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
model_6/conv2d_61/BiasAddBiasAdd!model_6/conv2d_61/Conv2D:output:00model_6/conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
'model_6/conv2d_62/Conv2D/ReadVariableOpReadVariableOp0model_6_conv2d_62_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0ß
model_6/conv2d_62/Conv2DConv2D(model_6/activation_46/Relu:activations:0/model_6/conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

(model_6/conv2d_62/BiasAdd/ReadVariableOpReadVariableOp1model_6_conv2d_62_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
model_6/conv2d_62/BiasAddBiasAdd!model_6/conv2d_62/Conv2D:output:00model_6/conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
-model_6/batch_normalization_48/ReadVariableOpReadVariableOp6model_6_batch_normalization_48_readvariableop_resource*
_output_shapes
:@*
dtype0¤
/model_6/batch_normalization_48/ReadVariableOp_1ReadVariableOp8model_6_batch_normalization_48_readvariableop_1_resource*
_output_shapes
:@*
dtype0Â
>model_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_6_batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Æ
@model_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_6_batch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0í
/model_6/batch_normalization_48/FusedBatchNormV3FusedBatchNormV3"model_6/conv2d_61/BiasAdd:output:05model_6/batch_normalization_48/ReadVariableOp:value:07model_6/batch_normalization_48/ReadVariableOp_1:value:0Fmodel_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( ®
model_6/add_20/addAddV2"model_6/conv2d_62/BiasAdd:output:03model_6/batch_normalization_48/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
model_6/activation_48/ReluRelumodel_6/add_20/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
#model_6/average_pooling2d_6/AvgPoolAvgPool(model_6/activation_48/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
h
model_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¦
model_6/flatten_6/ReshapeReshape,model_6/average_pooling2d_6/AvgPool:output:0 model_6/flatten_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%model_6/dense_6/MatMul/ReadVariableOpReadVariableOp.model_6_dense_6_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype0¥
model_6/dense_6/MatMulMatMul"model_6/flatten_6/Reshape:output:0-model_6/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&model_6/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_6_dense_6_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0¦
model_6/dense_6/BiasAddBiasAdd model_6/dense_6/MatMul:product:0.model_6/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
IdentityIdentity model_6/dense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Þ
NoOpNoOp?^model_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOpA^model_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1.^model_6/batch_normalization_42/ReadVariableOp0^model_6/batch_normalization_42/ReadVariableOp_1?^model_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOpA^model_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1.^model_6/batch_normalization_43/ReadVariableOp0^model_6/batch_normalization_43/ReadVariableOp_1?^model_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOpA^model_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1.^model_6/batch_normalization_44/ReadVariableOp0^model_6/batch_normalization_44/ReadVariableOp_1?^model_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOpA^model_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1.^model_6/batch_normalization_45/ReadVariableOp0^model_6/batch_normalization_45/ReadVariableOp_1?^model_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOpA^model_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1.^model_6/batch_normalization_46/ReadVariableOp0^model_6/batch_normalization_46/ReadVariableOp_1?^model_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOpA^model_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1.^model_6/batch_normalization_47/ReadVariableOp0^model_6/batch_normalization_47/ReadVariableOp_1?^model_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOpA^model_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1.^model_6/batch_normalization_48/ReadVariableOp0^model_6/batch_normalization_48/ReadVariableOp_1)^model_6/conv2d_54/BiasAdd/ReadVariableOp(^model_6/conv2d_54/Conv2D/ReadVariableOp)^model_6/conv2d_55/BiasAdd/ReadVariableOp(^model_6/conv2d_55/Conv2D/ReadVariableOp)^model_6/conv2d_56/BiasAdd/ReadVariableOp(^model_6/conv2d_56/Conv2D/ReadVariableOp)^model_6/conv2d_57/BiasAdd/ReadVariableOp(^model_6/conv2d_57/Conv2D/ReadVariableOp)^model_6/conv2d_58/BiasAdd/ReadVariableOp(^model_6/conv2d_58/Conv2D/ReadVariableOp)^model_6/conv2d_59/BiasAdd/ReadVariableOp(^model_6/conv2d_59/Conv2D/ReadVariableOp)^model_6/conv2d_60/BiasAdd/ReadVariableOp(^model_6/conv2d_60/Conv2D/ReadVariableOp)^model_6/conv2d_61/BiasAdd/ReadVariableOp(^model_6/conv2d_61/Conv2D/ReadVariableOp)^model_6/conv2d_62/BiasAdd/ReadVariableOp(^model_6/conv2d_62/Conv2D/ReadVariableOp'^model_6/dense_6/BiasAdd/ReadVariableOp&^model_6/dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
>model_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOp>model_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOp2
@model_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOp_1@model_6/batch_normalization_42/FusedBatchNormV3/ReadVariableOp_12^
-model_6/batch_normalization_42/ReadVariableOp-model_6/batch_normalization_42/ReadVariableOp2b
/model_6/batch_normalization_42/ReadVariableOp_1/model_6/batch_normalization_42/ReadVariableOp_12
>model_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOp>model_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOp2
@model_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1@model_6/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_12^
-model_6/batch_normalization_43/ReadVariableOp-model_6/batch_normalization_43/ReadVariableOp2b
/model_6/batch_normalization_43/ReadVariableOp_1/model_6/batch_normalization_43/ReadVariableOp_12
>model_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOp>model_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOp2
@model_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1@model_6/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_12^
-model_6/batch_normalization_44/ReadVariableOp-model_6/batch_normalization_44/ReadVariableOp2b
/model_6/batch_normalization_44/ReadVariableOp_1/model_6/batch_normalization_44/ReadVariableOp_12
>model_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOp>model_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOp2
@model_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1@model_6/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_12^
-model_6/batch_normalization_45/ReadVariableOp-model_6/batch_normalization_45/ReadVariableOp2b
/model_6/batch_normalization_45/ReadVariableOp_1/model_6/batch_normalization_45/ReadVariableOp_12
>model_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOp>model_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOp2
@model_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1@model_6/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_12^
-model_6/batch_normalization_46/ReadVariableOp-model_6/batch_normalization_46/ReadVariableOp2b
/model_6/batch_normalization_46/ReadVariableOp_1/model_6/batch_normalization_46/ReadVariableOp_12
>model_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOp>model_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOp2
@model_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1@model_6/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12^
-model_6/batch_normalization_47/ReadVariableOp-model_6/batch_normalization_47/ReadVariableOp2b
/model_6/batch_normalization_47/ReadVariableOp_1/model_6/batch_normalization_47/ReadVariableOp_12
>model_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOp>model_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOp2
@model_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1@model_6/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12^
-model_6/batch_normalization_48/ReadVariableOp-model_6/batch_normalization_48/ReadVariableOp2b
/model_6/batch_normalization_48/ReadVariableOp_1/model_6/batch_normalization_48/ReadVariableOp_12T
(model_6/conv2d_54/BiasAdd/ReadVariableOp(model_6/conv2d_54/BiasAdd/ReadVariableOp2R
'model_6/conv2d_54/Conv2D/ReadVariableOp'model_6/conv2d_54/Conv2D/ReadVariableOp2T
(model_6/conv2d_55/BiasAdd/ReadVariableOp(model_6/conv2d_55/BiasAdd/ReadVariableOp2R
'model_6/conv2d_55/Conv2D/ReadVariableOp'model_6/conv2d_55/Conv2D/ReadVariableOp2T
(model_6/conv2d_56/BiasAdd/ReadVariableOp(model_6/conv2d_56/BiasAdd/ReadVariableOp2R
'model_6/conv2d_56/Conv2D/ReadVariableOp'model_6/conv2d_56/Conv2D/ReadVariableOp2T
(model_6/conv2d_57/BiasAdd/ReadVariableOp(model_6/conv2d_57/BiasAdd/ReadVariableOp2R
'model_6/conv2d_57/Conv2D/ReadVariableOp'model_6/conv2d_57/Conv2D/ReadVariableOp2T
(model_6/conv2d_58/BiasAdd/ReadVariableOp(model_6/conv2d_58/BiasAdd/ReadVariableOp2R
'model_6/conv2d_58/Conv2D/ReadVariableOp'model_6/conv2d_58/Conv2D/ReadVariableOp2T
(model_6/conv2d_59/BiasAdd/ReadVariableOp(model_6/conv2d_59/BiasAdd/ReadVariableOp2R
'model_6/conv2d_59/Conv2D/ReadVariableOp'model_6/conv2d_59/Conv2D/ReadVariableOp2T
(model_6/conv2d_60/BiasAdd/ReadVariableOp(model_6/conv2d_60/BiasAdd/ReadVariableOp2R
'model_6/conv2d_60/Conv2D/ReadVariableOp'model_6/conv2d_60/Conv2D/ReadVariableOp2T
(model_6/conv2d_61/BiasAdd/ReadVariableOp(model_6/conv2d_61/BiasAdd/ReadVariableOp2R
'model_6/conv2d_61/Conv2D/ReadVariableOp'model_6/conv2d_61/Conv2D/ReadVariableOp2T
(model_6/conv2d_62/BiasAdd/ReadVariableOp(model_6/conv2d_62/BiasAdd/ReadVariableOp2R
'model_6/conv2d_62/Conv2D/ReadVariableOp'model_6/conv2d_62/Conv2D/ReadVariableOp2P
&model_6/dense_6/BiasAdd/ReadVariableOp&model_6/dense_6/BiasAdd/ReadVariableOp2N
%model_6/dense_6/MatMul/ReadVariableOp%model_6/dense_6/MatMul/ReadVariableOp:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_7
ï
g
K__inference_activation_43_layer_call_and_return_conditional_losses_23490518

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
	
Ô
9__inference_batch_normalization_47_layer_call_fn_23490939

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
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23487867
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
Ë
L
0__inference_activation_47_layer_call_fn_23490980

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
K__inference_activation_47_layer_call_and_return_conditional_losses_23488225h
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
â
µ
G__inference_conv2d_55_layer_call_and_return_conditional_losses_23488015

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_55/kernel/Regularizer/Square/ReadVariableOp|
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
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_55/kernel/Regularizer/SquareSquare:conv2d_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_55/kernel/Regularizer/SumSum'conv2d_55/kernel/Regularizer/Square:y:0+conv2d_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_55/kernel/Regularizer/mulMul+conv2d_55/kernel/Regularizer/mul/x:output:0)conv2d_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_55/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_55/kernel/Regularizer/Square/ReadVariableOp2conv2d_55/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
é
n
D__inference_add_19_layer_call_and_return_conditional_losses_23488180

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
¡Ø
¶
E__inference_model_6_layer_call_and_return_conditional_losses_23488929

inputs,
conv2d_54_23488749: 
conv2d_54_23488751:-
batch_normalization_42_23488754:-
batch_normalization_42_23488756:-
batch_normalization_42_23488758:-
batch_normalization_42_23488760:,
conv2d_55_23488764: 
conv2d_55_23488766:-
batch_normalization_43_23488769:-
batch_normalization_43_23488771:-
batch_normalization_43_23488773:-
batch_normalization_43_23488775:,
conv2d_56_23488779: 
conv2d_56_23488781:-
batch_normalization_44_23488784:-
batch_normalization_44_23488786:-
batch_normalization_44_23488788:-
batch_normalization_44_23488790:,
conv2d_57_23488795:  
conv2d_57_23488797: -
batch_normalization_45_23488800: -
batch_normalization_45_23488802: -
batch_normalization_45_23488804: -
batch_normalization_45_23488806: ,
conv2d_58_23488810:   
conv2d_58_23488812: ,
conv2d_59_23488815:  
conv2d_59_23488817: -
batch_normalization_46_23488820: -
batch_normalization_46_23488822: -
batch_normalization_46_23488824: -
batch_normalization_46_23488826: ,
conv2d_60_23488831: @ 
conv2d_60_23488833:@-
batch_normalization_47_23488836:@-
batch_normalization_47_23488838:@-
batch_normalization_47_23488840:@-
batch_normalization_47_23488842:@,
conv2d_61_23488846:@@ 
conv2d_61_23488848:@,
conv2d_62_23488851: @ 
conv2d_62_23488853:@-
batch_normalization_48_23488856:@-
batch_normalization_48_23488858:@-
batch_normalization_48_23488860:@-
batch_normalization_48_23488862:@"
dense_6_23488869:@

dense_6_23488871:

identity¢.batch_normalization_42/StatefulPartitionedCall¢.batch_normalization_43/StatefulPartitionedCall¢.batch_normalization_44/StatefulPartitionedCall¢.batch_normalization_45/StatefulPartitionedCall¢.batch_normalization_46/StatefulPartitionedCall¢.batch_normalization_47/StatefulPartitionedCall¢.batch_normalization_48/StatefulPartitionedCall¢!conv2d_54/StatefulPartitionedCall¢2conv2d_54/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_55/StatefulPartitionedCall¢2conv2d_55/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_56/StatefulPartitionedCall¢2conv2d_56/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_57/StatefulPartitionedCall¢2conv2d_57/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_58/StatefulPartitionedCall¢2conv2d_58/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_59/StatefulPartitionedCall¢2conv2d_59/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_60/StatefulPartitionedCall¢2conv2d_60/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_61/StatefulPartitionedCall¢2conv2d_61/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_62/StatefulPartitionedCall¢2conv2d_62/kernel/Regularizer/Square/ReadVariableOp¢dense_6/StatefulPartitionedCall
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_54_23488749conv2d_54_23488751*
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
G__inference_conv2d_54_layer_call_and_return_conditional_losses_23487977
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_42_23488754batch_normalization_42_23488756batch_normalization_42_23488758batch_normalization_42_23488760*
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
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23487547ý
activation_42/PartitionedCallPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0*
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
K__inference_activation_42_layer_call_and_return_conditional_losses_23487997¢
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall&activation_42/PartitionedCall:output:0conv2d_55_23488764conv2d_55_23488766*
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
G__inference_conv2d_55_layer_call_and_return_conditional_losses_23488015
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0batch_normalization_43_23488769batch_normalization_43_23488771batch_normalization_43_23488773batch_normalization_43_23488775*
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
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23487611ý
activation_43/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
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
K__inference_activation_43_layer_call_and_return_conditional_losses_23488035¢
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall&activation_43/PartitionedCall:output:0conv2d_56_23488779conv2d_56_23488781*
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
G__inference_conv2d_56_layer_call_and_return_conditional_losses_23488053
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_56/StatefulPartitionedCall:output:0batch_normalization_44_23488784batch_normalization_44_23488786batch_normalization_44_23488788batch_normalization_44_23488790*
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
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23487675
add_18/PartitionedCallPartitionedCall&activation_42/PartitionedCall:output:07batch_normalization_44/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *M
fHRF
D__inference_add_18_layer_call_and_return_conditional_losses_23488074å
activation_44/PartitionedCallPartitionedCalladd_18/PartitionedCall:output:0*
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
K__inference_activation_44_layer_call_and_return_conditional_losses_23488081¢
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCall&activation_44/PartitionedCall:output:0conv2d_57_23488795conv2d_57_23488797*
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
G__inference_conv2d_57_layer_call_and_return_conditional_losses_23488099
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0batch_normalization_45_23488800batch_normalization_45_23488802batch_normalization_45_23488804batch_normalization_45_23488806*
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
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23487739ý
activation_45/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
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
K__inference_activation_45_layer_call_and_return_conditional_losses_23488119¢
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall&activation_45/PartitionedCall:output:0conv2d_58_23488810conv2d_58_23488812*
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
G__inference_conv2d_58_layer_call_and_return_conditional_losses_23488137¢
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall&activation_44/PartitionedCall:output:0conv2d_59_23488815conv2d_59_23488817*
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
G__inference_conv2d_59_layer_call_and_return_conditional_losses_23488159
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0batch_normalization_46_23488820batch_normalization_46_23488822batch_normalization_46_23488824batch_normalization_46_23488826*
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
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23487803
add_19/PartitionedCallPartitionedCall*conv2d_59/StatefulPartitionedCall:output:07batch_normalization_46/StatefulPartitionedCall:output:0*
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
D__inference_add_19_layer_call_and_return_conditional_losses_23488180å
activation_46/PartitionedCallPartitionedCalladd_19/PartitionedCall:output:0*
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
K__inference_activation_46_layer_call_and_return_conditional_losses_23488187¢
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall&activation_46/PartitionedCall:output:0conv2d_60_23488831conv2d_60_23488833*
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
G__inference_conv2d_60_layer_call_and_return_conditional_losses_23488205
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0batch_normalization_47_23488836batch_normalization_47_23488838batch_normalization_47_23488840batch_normalization_47_23488842*
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
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23487867ý
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
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
K__inference_activation_47_layer_call_and_return_conditional_losses_23488225¢
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall&activation_47/PartitionedCall:output:0conv2d_61_23488846conv2d_61_23488848*
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
G__inference_conv2d_61_layer_call_and_return_conditional_losses_23488243¢
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall&activation_46/PartitionedCall:output:0conv2d_62_23488851conv2d_62_23488853*
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
G__inference_conv2d_62_layer_call_and_return_conditional_losses_23488265
.batch_normalization_48/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0batch_normalization_48_23488856batch_normalization_48_23488858batch_normalization_48_23488860batch_normalization_48_23488862*
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
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23487931
add_20/PartitionedCallPartitionedCall*conv2d_62/StatefulPartitionedCall:output:07batch_normalization_48/StatefulPartitionedCall:output:0*
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
D__inference_add_20_layer_call_and_return_conditional_losses_23488286å
activation_48/PartitionedCallPartitionedCalladd_20/PartitionedCall:output:0*
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
K__inference_activation_48_layer_call_and_return_conditional_losses_23488293ø
#average_pooling2d_6/PartitionedCallPartitionedCall&activation_48/PartitionedCall:output:0*
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
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_23487951â
flatten_6/PartitionedCallPartitionedCall,average_pooling2d_6/PartitionedCall:output:0*
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
G__inference_flatten_6_layer_call_and_return_conditional_losses_23488302
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_6_23488869dense_6_23488871*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_23488314
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_54_23488749*&
_output_shapes
:*
dtype0
#conv2d_54/kernel/Regularizer/SquareSquare:conv2d_54/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_54/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_54/kernel/Regularizer/SumSum'conv2d_54/kernel/Regularizer/Square:y:0+conv2d_54/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_54/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_54/kernel/Regularizer/mulMul+conv2d_54/kernel/Regularizer/mul/x:output:0)conv2d_54/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_55_23488764*&
_output_shapes
:*
dtype0
#conv2d_55/kernel/Regularizer/SquareSquare:conv2d_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_55/kernel/Regularizer/SumSum'conv2d_55/kernel/Regularizer/Square:y:0+conv2d_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_55/kernel/Regularizer/mulMul+conv2d_55/kernel/Regularizer/mul/x:output:0)conv2d_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_56_23488779*&
_output_shapes
:*
dtype0
#conv2d_56/kernel/Regularizer/SquareSquare:conv2d_56/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_56/kernel/Regularizer/SumSum'conv2d_56/kernel/Regularizer/Square:y:0+conv2d_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_56/kernel/Regularizer/mulMul+conv2d_56/kernel/Regularizer/mul/x:output:0)conv2d_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_57_23488795*&
_output_shapes
: *
dtype0
#conv2d_57/kernel/Regularizer/SquareSquare:conv2d_57/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_57/kernel/Regularizer/SumSum'conv2d_57/kernel/Regularizer/Square:y:0+conv2d_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_57/kernel/Regularizer/mulMul+conv2d_57/kernel/Regularizer/mul/x:output:0)conv2d_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_58_23488810*&
_output_shapes
:  *
dtype0
#conv2d_58/kernel/Regularizer/SquareSquare:conv2d_58/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_58/kernel/Regularizer/SumSum'conv2d_58/kernel/Regularizer/Square:y:0+conv2d_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_58/kernel/Regularizer/mulMul+conv2d_58/kernel/Regularizer/mul/x:output:0)conv2d_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_59_23488815*&
_output_shapes
: *
dtype0
#conv2d_59/kernel/Regularizer/SquareSquare:conv2d_59/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_59/kernel/Regularizer/SumSum'conv2d_59/kernel/Regularizer/Square:y:0+conv2d_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_59/kernel/Regularizer/mulMul+conv2d_59/kernel/Regularizer/mul/x:output:0)conv2d_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_60/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_60_23488831*&
_output_shapes
: @*
dtype0
#conv2d_60/kernel/Regularizer/SquareSquare:conv2d_60/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_60/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_60/kernel/Regularizer/SumSum'conv2d_60/kernel/Regularizer/Square:y:0+conv2d_60/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_60/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_60/kernel/Regularizer/mulMul+conv2d_60/kernel/Regularizer/mul/x:output:0)conv2d_60/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_61/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_61_23488846*&
_output_shapes
:@@*
dtype0
#conv2d_61/kernel/Regularizer/SquareSquare:conv2d_61/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@{
"conv2d_61/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_61/kernel/Regularizer/SumSum'conv2d_61/kernel/Regularizer/Square:y:0+conv2d_61/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_61/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_61/kernel/Regularizer/mulMul+conv2d_61/kernel/Regularizer/mul/x:output:0)conv2d_61/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_62/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_62_23488851*&
_output_shapes
: @*
dtype0
#conv2d_62/kernel/Regularizer/SquareSquare:conv2d_62/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_62/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_62/kernel/Regularizer/SumSum'conv2d_62/kernel/Regularizer/Square:y:0+conv2d_62/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_62/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_62/kernel/Regularizer/mulMul+conv2d_62/kernel/Regularizer/mul/x:output:0)conv2d_62/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
à	
NoOpNoOp/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_46/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_48/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall3^conv2d_54/kernel/Regularizer/Square/ReadVariableOp"^conv2d_55/StatefulPartitionedCall3^conv2d_55/kernel/Regularizer/Square/ReadVariableOp"^conv2d_56/StatefulPartitionedCall3^conv2d_56/kernel/Regularizer/Square/ReadVariableOp"^conv2d_57/StatefulPartitionedCall3^conv2d_57/kernel/Regularizer/Square/ReadVariableOp"^conv2d_58/StatefulPartitionedCall3^conv2d_58/kernel/Regularizer/Square/ReadVariableOp"^conv2d_59/StatefulPartitionedCall3^conv2d_59/kernel/Regularizer/Square/ReadVariableOp"^conv2d_60/StatefulPartitionedCall3^conv2d_60/kernel/Regularizer/Square/ReadVariableOp"^conv2d_61/StatefulPartitionedCall3^conv2d_61/kernel/Regularizer/Square/ReadVariableOp"^conv2d_62/StatefulPartitionedCall3^conv2d_62/kernel/Regularizer/Square/ReadVariableOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_46/StatefulPartitionedCall.batch_normalization_46/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_48/StatefulPartitionedCall.batch_normalization_48/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2h
2conv2d_54/kernel/Regularizer/Square/ReadVariableOp2conv2d_54/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2h
2conv2d_55/kernel/Regularizer/Square/ReadVariableOp2conv2d_55/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_56/StatefulPartitionedCall!conv2d_56/StatefulPartitionedCall2h
2conv2d_56/kernel/Regularizer/Square/ReadVariableOp2conv2d_56/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_57/StatefulPartitionedCall!conv2d_57/StatefulPartitionedCall2h
2conv2d_57/kernel/Regularizer/Square/ReadVariableOp2conv2d_57/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2h
2conv2d_58/kernel/Regularizer/Square/ReadVariableOp2conv2d_58/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2h
2conv2d_59/kernel/Regularizer/Square/ReadVariableOp2conv2d_59/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2h
2conv2d_60/kernel/Regularizer/Square/ReadVariableOp2conv2d_60/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2h
2conv2d_61/kernel/Regularizer/Square/ReadVariableOp2conv2d_61/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2h
2conv2d_62/kernel/Regularizer/Square/ReadVariableOp2conv2d_62/kernel/Regularizer/Square/ReadVariableOp2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
â
µ
G__inference_conv2d_57_layer_call_and_return_conditional_losses_23488099

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_57/kernel/Regularizer/Square/ReadVariableOp|
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
2conv2d_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_57/kernel/Regularizer/SquareSquare:conv2d_57/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_57/kernel/Regularizer/SumSum'conv2d_57/kernel/Regularizer/Square:y:0+conv2d_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_57/kernel/Regularizer/mulMul+conv2d_57/kernel/Regularizer/mul/x:output:0)conv2d_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_57/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_57/kernel/Regularizer/Square/ReadVariableOp2conv2d_57/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ë
½
__inference_loss_fn_0_23491182U
;conv2d_54_kernel_regularizer_square_readvariableop_resource:
identity¢2conv2d_54/kernel/Regularizer/Square/ReadVariableOp¶
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_54_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_54/kernel/Regularizer/SquareSquare:conv2d_54/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_54/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_54/kernel/Regularizer/SumSum'conv2d_54/kernel/Regularizer/Square:y:0+conv2d_54/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_54/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_54/kernel/Regularizer/mulMul+conv2d_54/kernel/Regularizer/mul/x:output:0)conv2d_54/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_54/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_54/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_54/kernel/Regularizer/Square/ReadVariableOp2conv2d_54/kernel/Regularizer/Square/ReadVariableOp
Ï

T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23487708

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
­
¦
*__inference_model_6_layer_call_fn_23489751

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
E__inference_model_6_layer_call_and_return_conditional_losses_23488929o
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
â
µ
G__inference_conv2d_55_layer_call_and_return_conditional_losses_23490446

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_55/kernel/Regularizer/Square/ReadVariableOp|
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
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_55/kernel/Regularizer/SquareSquare:conv2d_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_55/kernel/Regularizer/SumSum'conv2d_55/kernel/Regularizer/Square:y:0+conv2d_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_55/kernel/Regularizer/mulMul+conv2d_55/kernel/Regularizer/mul/x:output:0)conv2d_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_55/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_55/kernel/Regularizer/Square/ReadVariableOp2conv2d_55/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23487580

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
ð
¡
,__inference_conv2d_62_layer_call_fn_23491031

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
G__inference_conv2d_62_layer_call_and_return_conditional_losses_23488265w
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
Ò
U
)__inference_add_20_layer_call_fn_23491115
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
D__inference_add_20_layer_call_and_return_conditional_losses_23488286h
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
Ý
Ã
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23487931

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
Ç
c
G__inference_flatten_6_layer_call_and_return_conditional_losses_23488302

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
Ï

T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23490387

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
K__inference_activation_46_layer_call_and_return_conditional_losses_23488187

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
0__inference_activation_44_layer_call_fn_23490628

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
K__inference_activation_44_layer_call_and_return_conditional_losses_23488081h
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
ï
g
K__inference_activation_44_layer_call_and_return_conditional_losses_23488081

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
0__inference_activation_48_layer_call_fn_23491126

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
K__inference_activation_48_layer_call_and_return_conditional_losses_23488293h
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
ï
g
K__inference_activation_46_layer_call_and_return_conditional_losses_23490882

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
â
µ
G__inference_conv2d_54_layer_call_and_return_conditional_losses_23487977

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_54/kernel/Regularizer/Square/ReadVariableOp|
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
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_54/kernel/Regularizer/SquareSquare:conv2d_54/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_54/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_54/kernel/Regularizer/SumSum'conv2d_54/kernel/Regularizer/Square:y:0+conv2d_54/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_54/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_54/kernel/Regularizer/mulMul+conv2d_54/kernel/Regularizer/mul/x:output:0)conv2d_54/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_54/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_54/kernel/Regularizer/Square/ReadVariableOp2conv2d_54/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ï
g
K__inference_activation_45_layer_call_and_return_conditional_losses_23488119

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
ð
¡
,__inference_conv2d_57_layer_call_fn_23490648

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
G__inference_conv2d_57_layer_call_and_return_conditional_losses_23488099w
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
	
Ô
9__inference_batch_normalization_44_layer_call_fn_23490562

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
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23487644
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
	
Ô
9__inference_batch_normalization_43_layer_call_fn_23490472

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
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23487611
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
G__inference_conv2d_56_layer_call_and_return_conditional_losses_23488053

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_56/kernel/Regularizer/Square/ReadVariableOp|
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
2conv2d_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_56/kernel/Regularizer/SquareSquare:conv2d_56/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_56/kernel/Regularizer/SumSum'conv2d_56/kernel/Regularizer/Square:y:0+conv2d_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_56/kernel/Regularizer/mulMul+conv2d_56/kernel/Regularizer/mul/x:output:0)conv2d_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_56/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_56/kernel/Regularizer/Square/ReadVariableOp2conv2d_56/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ë
½
__inference_loss_fn_1_23491193U
;conv2d_55_kernel_regularizer_square_readvariableop_resource:
identity¢2conv2d_55/kernel/Regularizer/Square/ReadVariableOp¶
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_55_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_55/kernel/Regularizer/SquareSquare:conv2d_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_55/kernel/Regularizer/SumSum'conv2d_55/kernel/Regularizer/Square:y:0+conv2d_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_55/kernel/Regularizer/mulMul+conv2d_55/kernel/Regularizer/mul/x:output:0)conv2d_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_55/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_55/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_55/kernel/Regularizer/Square/ReadVariableOp2conv2d_55/kernel/Regularizer/Square/ReadVariableOp
Ë
L
0__inference_activation_46_layer_call_fn_23490877

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
K__inference_activation_46_layer_call_and_return_conditional_losses_23488187h
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
ë
½
__inference_loss_fn_6_23491248U
;conv2d_60_kernel_regularizer_square_readvariableop_resource: @
identity¢2conv2d_60/kernel/Regularizer/Square/ReadVariableOp¶
2conv2d_60/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_60_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_60/kernel/Regularizer/SquareSquare:conv2d_60/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_60/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_60/kernel/Regularizer/SumSum'conv2d_60/kernel/Regularizer/Square:y:0+conv2d_60/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_60/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_60/kernel/Regularizer/mulMul+conv2d_60/kernel/Regularizer/mul/x:output:0)conv2d_60/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_60/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_60/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_60/kernel/Regularizer/Square/ReadVariableOp2conv2d_60/kernel/Regularizer/Square/ReadVariableOp
Ë
L
0__inference_activation_42_layer_call_fn_23490410

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
K__inference_activation_42_layer_call_and_return_conditional_losses_23487997h
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
)__inference_add_19_layer_call_fn_23490866
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
D__inference_add_19_layer_call_and_return_conditional_losses_23488180h
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
â
µ
G__inference_conv2d_60_layer_call_and_return_conditional_losses_23488205

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_60/kernel/Regularizer/Square/ReadVariableOp|
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
2conv2d_60/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_60/kernel/Regularizer/SquareSquare:conv2d_60/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_60/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_60/kernel/Regularizer/SumSum'conv2d_60/kernel/Regularizer/Square:y:0+conv2d_60/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_60/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_60/kernel/Regularizer/mulMul+conv2d_60/kernel/Regularizer/mul/x:output:0)conv2d_60/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_60/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_60/kernel/Regularizer/Square/ReadVariableOp2conv2d_60/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ë
½
__inference_loss_fn_4_23491226U
;conv2d_58_kernel_regularizer_square_readvariableop_resource:  
identity¢2conv2d_58/kernel/Regularizer/Square/ReadVariableOp¶
2conv2d_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_58_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype0
#conv2d_58/kernel/Regularizer/SquareSquare:conv2d_58/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_58/kernel/Regularizer/SumSum'conv2d_58/kernel/Regularizer/Square:y:0+conv2d_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_58/kernel/Regularizer/mulMul+conv2d_58/kernel/Regularizer/mul/x:output:0)conv2d_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_58/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_58/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_58/kernel/Regularizer/Square/ReadVariableOp2conv2d_58/kernel/Regularizer/Square/ReadVariableOp
¢
m
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_23491141

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
	
Ô
9__inference_batch_normalization_44_layer_call_fn_23490575

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
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23487675
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
Ï

T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23487836

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
â
µ
G__inference_conv2d_61_layer_call_and_return_conditional_losses_23488243

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_61/kernel/Regularizer/Square/ReadVariableOp|
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
2conv2d_61/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
#conv2d_61/kernel/Regularizer/SquareSquare:conv2d_61/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@{
"conv2d_61/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_61/kernel/Regularizer/SumSum'conv2d_61/kernel/Regularizer/Square:y:0+conv2d_61/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_61/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_61/kernel/Regularizer/mulMul+conv2d_61/kernel/Regularizer/mul/x:output:0)conv2d_61/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_61/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_61/kernel/Regularizer/Square/ReadVariableOp2conv2d_61/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ð
¡
,__inference_conv2d_56_layer_call_fn_23490533

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
G__inference_conv2d_56_layer_call_and_return_conditional_losses_23488053w
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
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23490726

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
	
Ô
9__inference_batch_normalization_46_layer_call_fn_23490824

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
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23487803
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
G__inference_conv2d_58_layer_call_and_return_conditional_losses_23488137

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_58/kernel/Regularizer/Square/ReadVariableOp|
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
2conv2d_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0
#conv2d_58/kernel/Regularizer/SquareSquare:conv2d_58/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_58/kernel/Regularizer/SumSum'conv2d_58/kernel/Regularizer/Square:y:0+conv2d_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_58/kernel/Regularizer/mulMul+conv2d_58/kernel/Regularizer/mul/x:output:0)conv2d_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_58/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_58/kernel/Regularizer/Square/ReadVariableOp2conv2d_58/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ï

T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23490593

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
Ò
U
)__inference_add_18_layer_call_fn_23490617
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
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_add_18_layer_call_and_return_conditional_losses_23488074h
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
Ý
Ã
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23490508

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
Ý
Ã
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23487547

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
ñ
p
D__inference_add_18_layer_call_and_return_conditional_losses_23490623
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
ë
½
__inference_loss_fn_3_23491215U
;conv2d_57_kernel_regularizer_square_readvariableop_resource: 
identity¢2conv2d_57/kernel/Regularizer/Square/ReadVariableOp¶
2conv2d_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_57_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_57/kernel/Regularizer/SquareSquare:conv2d_57/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_57/kernel/Regularizer/SumSum'conv2d_57/kernel/Regularizer/Square:y:0+conv2d_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_57/kernel/Regularizer/mulMul+conv2d_57/kernel/Regularizer/mul/x:output:0)conv2d_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_57/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_57/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_57/kernel/Regularizer/Square/ReadVariableOp2conv2d_57/kernel/Regularizer/Square/ReadVariableOp
²Ø
·
E__inference_model_6_layer_call_and_return_conditional_losses_23489312
input_7,
conv2d_54_23489132: 
conv2d_54_23489134:-
batch_normalization_42_23489137:-
batch_normalization_42_23489139:-
batch_normalization_42_23489141:-
batch_normalization_42_23489143:,
conv2d_55_23489147: 
conv2d_55_23489149:-
batch_normalization_43_23489152:-
batch_normalization_43_23489154:-
batch_normalization_43_23489156:-
batch_normalization_43_23489158:,
conv2d_56_23489162: 
conv2d_56_23489164:-
batch_normalization_44_23489167:-
batch_normalization_44_23489169:-
batch_normalization_44_23489171:-
batch_normalization_44_23489173:,
conv2d_57_23489178:  
conv2d_57_23489180: -
batch_normalization_45_23489183: -
batch_normalization_45_23489185: -
batch_normalization_45_23489187: -
batch_normalization_45_23489189: ,
conv2d_58_23489193:   
conv2d_58_23489195: ,
conv2d_59_23489198:  
conv2d_59_23489200: -
batch_normalization_46_23489203: -
batch_normalization_46_23489205: -
batch_normalization_46_23489207: -
batch_normalization_46_23489209: ,
conv2d_60_23489214: @ 
conv2d_60_23489216:@-
batch_normalization_47_23489219:@-
batch_normalization_47_23489221:@-
batch_normalization_47_23489223:@-
batch_normalization_47_23489225:@,
conv2d_61_23489229:@@ 
conv2d_61_23489231:@,
conv2d_62_23489234: @ 
conv2d_62_23489236:@-
batch_normalization_48_23489239:@-
batch_normalization_48_23489241:@-
batch_normalization_48_23489243:@-
batch_normalization_48_23489245:@"
dense_6_23489252:@

dense_6_23489254:

identity¢.batch_normalization_42/StatefulPartitionedCall¢.batch_normalization_43/StatefulPartitionedCall¢.batch_normalization_44/StatefulPartitionedCall¢.batch_normalization_45/StatefulPartitionedCall¢.batch_normalization_46/StatefulPartitionedCall¢.batch_normalization_47/StatefulPartitionedCall¢.batch_normalization_48/StatefulPartitionedCall¢!conv2d_54/StatefulPartitionedCall¢2conv2d_54/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_55/StatefulPartitionedCall¢2conv2d_55/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_56/StatefulPartitionedCall¢2conv2d_56/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_57/StatefulPartitionedCall¢2conv2d_57/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_58/StatefulPartitionedCall¢2conv2d_58/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_59/StatefulPartitionedCall¢2conv2d_59/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_60/StatefulPartitionedCall¢2conv2d_60/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_61/StatefulPartitionedCall¢2conv2d_61/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_62/StatefulPartitionedCall¢2conv2d_62/kernel/Regularizer/Square/ReadVariableOp¢dense_6/StatefulPartitionedCall
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCallinput_7conv2d_54_23489132conv2d_54_23489134*
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
G__inference_conv2d_54_layer_call_and_return_conditional_losses_23487977 
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_42_23489137batch_normalization_42_23489139batch_normalization_42_23489141batch_normalization_42_23489143*
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
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23487516ý
activation_42/PartitionedCallPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0*
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
K__inference_activation_42_layer_call_and_return_conditional_losses_23487997¢
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall&activation_42/PartitionedCall:output:0conv2d_55_23489147conv2d_55_23489149*
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
G__inference_conv2d_55_layer_call_and_return_conditional_losses_23488015 
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0batch_normalization_43_23489152batch_normalization_43_23489154batch_normalization_43_23489156batch_normalization_43_23489158*
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
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23487580ý
activation_43/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
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
K__inference_activation_43_layer_call_and_return_conditional_losses_23488035¢
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall&activation_43/PartitionedCall:output:0conv2d_56_23489162conv2d_56_23489164*
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
G__inference_conv2d_56_layer_call_and_return_conditional_losses_23488053 
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_56/StatefulPartitionedCall:output:0batch_normalization_44_23489167batch_normalization_44_23489169batch_normalization_44_23489171batch_normalization_44_23489173*
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
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23487644
add_18/PartitionedCallPartitionedCall&activation_42/PartitionedCall:output:07batch_normalization_44/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *M
fHRF
D__inference_add_18_layer_call_and_return_conditional_losses_23488074å
activation_44/PartitionedCallPartitionedCalladd_18/PartitionedCall:output:0*
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
K__inference_activation_44_layer_call_and_return_conditional_losses_23488081¢
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCall&activation_44/PartitionedCall:output:0conv2d_57_23489178conv2d_57_23489180*
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
G__inference_conv2d_57_layer_call_and_return_conditional_losses_23488099 
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0batch_normalization_45_23489183batch_normalization_45_23489185batch_normalization_45_23489187batch_normalization_45_23489189*
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
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23487708ý
activation_45/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
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
K__inference_activation_45_layer_call_and_return_conditional_losses_23488119¢
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall&activation_45/PartitionedCall:output:0conv2d_58_23489193conv2d_58_23489195*
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
G__inference_conv2d_58_layer_call_and_return_conditional_losses_23488137¢
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall&activation_44/PartitionedCall:output:0conv2d_59_23489198conv2d_59_23489200*
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
G__inference_conv2d_59_layer_call_and_return_conditional_losses_23488159 
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0batch_normalization_46_23489203batch_normalization_46_23489205batch_normalization_46_23489207batch_normalization_46_23489209*
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
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23487772
add_19/PartitionedCallPartitionedCall*conv2d_59/StatefulPartitionedCall:output:07batch_normalization_46/StatefulPartitionedCall:output:0*
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
D__inference_add_19_layer_call_and_return_conditional_losses_23488180å
activation_46/PartitionedCallPartitionedCalladd_19/PartitionedCall:output:0*
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
K__inference_activation_46_layer_call_and_return_conditional_losses_23488187¢
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall&activation_46/PartitionedCall:output:0conv2d_60_23489214conv2d_60_23489216*
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
G__inference_conv2d_60_layer_call_and_return_conditional_losses_23488205 
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0batch_normalization_47_23489219batch_normalization_47_23489221batch_normalization_47_23489223batch_normalization_47_23489225*
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
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23487836ý
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
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
K__inference_activation_47_layer_call_and_return_conditional_losses_23488225¢
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall&activation_47/PartitionedCall:output:0conv2d_61_23489229conv2d_61_23489231*
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
G__inference_conv2d_61_layer_call_and_return_conditional_losses_23488243¢
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall&activation_46/PartitionedCall:output:0conv2d_62_23489234conv2d_62_23489236*
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
G__inference_conv2d_62_layer_call_and_return_conditional_losses_23488265 
.batch_normalization_48/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0batch_normalization_48_23489239batch_normalization_48_23489241batch_normalization_48_23489243batch_normalization_48_23489245*
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
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23487900
add_20/PartitionedCallPartitionedCall*conv2d_62/StatefulPartitionedCall:output:07batch_normalization_48/StatefulPartitionedCall:output:0*
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
D__inference_add_20_layer_call_and_return_conditional_losses_23488286å
activation_48/PartitionedCallPartitionedCalladd_20/PartitionedCall:output:0*
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
K__inference_activation_48_layer_call_and_return_conditional_losses_23488293ø
#average_pooling2d_6/PartitionedCallPartitionedCall&activation_48/PartitionedCall:output:0*
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
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_23487951â
flatten_6/PartitionedCallPartitionedCall,average_pooling2d_6/PartitionedCall:output:0*
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
G__inference_flatten_6_layer_call_and_return_conditional_losses_23488302
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_6_23489252dense_6_23489254*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_23488314
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_54_23489132*&
_output_shapes
:*
dtype0
#conv2d_54/kernel/Regularizer/SquareSquare:conv2d_54/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_54/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_54/kernel/Regularizer/SumSum'conv2d_54/kernel/Regularizer/Square:y:0+conv2d_54/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_54/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_54/kernel/Regularizer/mulMul+conv2d_54/kernel/Regularizer/mul/x:output:0)conv2d_54/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_55_23489147*&
_output_shapes
:*
dtype0
#conv2d_55/kernel/Regularizer/SquareSquare:conv2d_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_55/kernel/Regularizer/SumSum'conv2d_55/kernel/Regularizer/Square:y:0+conv2d_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_55/kernel/Regularizer/mulMul+conv2d_55/kernel/Regularizer/mul/x:output:0)conv2d_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_56_23489162*&
_output_shapes
:*
dtype0
#conv2d_56/kernel/Regularizer/SquareSquare:conv2d_56/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_56/kernel/Regularizer/SumSum'conv2d_56/kernel/Regularizer/Square:y:0+conv2d_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_56/kernel/Regularizer/mulMul+conv2d_56/kernel/Regularizer/mul/x:output:0)conv2d_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_57_23489178*&
_output_shapes
: *
dtype0
#conv2d_57/kernel/Regularizer/SquareSquare:conv2d_57/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_57/kernel/Regularizer/SumSum'conv2d_57/kernel/Regularizer/Square:y:0+conv2d_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_57/kernel/Regularizer/mulMul+conv2d_57/kernel/Regularizer/mul/x:output:0)conv2d_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_58_23489193*&
_output_shapes
:  *
dtype0
#conv2d_58/kernel/Regularizer/SquareSquare:conv2d_58/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_58/kernel/Regularizer/SumSum'conv2d_58/kernel/Regularizer/Square:y:0+conv2d_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_58/kernel/Regularizer/mulMul+conv2d_58/kernel/Regularizer/mul/x:output:0)conv2d_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_59_23489198*&
_output_shapes
: *
dtype0
#conv2d_59/kernel/Regularizer/SquareSquare:conv2d_59/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_59/kernel/Regularizer/SumSum'conv2d_59/kernel/Regularizer/Square:y:0+conv2d_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_59/kernel/Regularizer/mulMul+conv2d_59/kernel/Regularizer/mul/x:output:0)conv2d_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_60/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_60_23489214*&
_output_shapes
: @*
dtype0
#conv2d_60/kernel/Regularizer/SquareSquare:conv2d_60/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_60/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_60/kernel/Regularizer/SumSum'conv2d_60/kernel/Regularizer/Square:y:0+conv2d_60/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_60/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_60/kernel/Regularizer/mulMul+conv2d_60/kernel/Regularizer/mul/x:output:0)conv2d_60/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_61/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_61_23489229*&
_output_shapes
:@@*
dtype0
#conv2d_61/kernel/Regularizer/SquareSquare:conv2d_61/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@{
"conv2d_61/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_61/kernel/Regularizer/SumSum'conv2d_61/kernel/Regularizer/Square:y:0+conv2d_61/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_61/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_61/kernel/Regularizer/mulMul+conv2d_61/kernel/Regularizer/mul/x:output:0)conv2d_61/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
2conv2d_62/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_62_23489234*&
_output_shapes
: @*
dtype0
#conv2d_62/kernel/Regularizer/SquareSquare:conv2d_62/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_62/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_62/kernel/Regularizer/SumSum'conv2d_62/kernel/Regularizer/Square:y:0+conv2d_62/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_62/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_62/kernel/Regularizer/mulMul+conv2d_62/kernel/Regularizer/mul/x:output:0)conv2d_62/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
à	
NoOpNoOp/^batch_normalization_42/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_46/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_48/StatefulPartitionedCall"^conv2d_54/StatefulPartitionedCall3^conv2d_54/kernel/Regularizer/Square/ReadVariableOp"^conv2d_55/StatefulPartitionedCall3^conv2d_55/kernel/Regularizer/Square/ReadVariableOp"^conv2d_56/StatefulPartitionedCall3^conv2d_56/kernel/Regularizer/Square/ReadVariableOp"^conv2d_57/StatefulPartitionedCall3^conv2d_57/kernel/Regularizer/Square/ReadVariableOp"^conv2d_58/StatefulPartitionedCall3^conv2d_58/kernel/Regularizer/Square/ReadVariableOp"^conv2d_59/StatefulPartitionedCall3^conv2d_59/kernel/Regularizer/Square/ReadVariableOp"^conv2d_60/StatefulPartitionedCall3^conv2d_60/kernel/Regularizer/Square/ReadVariableOp"^conv2d_61/StatefulPartitionedCall3^conv2d_61/kernel/Regularizer/Square/ReadVariableOp"^conv2d_62/StatefulPartitionedCall3^conv2d_62/kernel/Regularizer/Square/ReadVariableOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_46/StatefulPartitionedCall.batch_normalization_46/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_48/StatefulPartitionedCall.batch_normalization_48/StatefulPartitionedCall2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2h
2conv2d_54/kernel/Regularizer/Square/ReadVariableOp2conv2d_54/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2h
2conv2d_55/kernel/Regularizer/Square/ReadVariableOp2conv2d_55/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_56/StatefulPartitionedCall!conv2d_56/StatefulPartitionedCall2h
2conv2d_56/kernel/Regularizer/Square/ReadVariableOp2conv2d_56/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_57/StatefulPartitionedCall!conv2d_57/StatefulPartitionedCall2h
2conv2d_57/kernel/Regularizer/Square/ReadVariableOp2conv2d_57/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2h
2conv2d_58/kernel/Regularizer/Square/ReadVariableOp2conv2d_58/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2h
2conv2d_59/kernel/Regularizer/Square/ReadVariableOp2conv2d_59/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2h
2conv2d_60/kernel/Regularizer/Square/ReadVariableOp2conv2d_60/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2h
2conv2d_61/kernel/Regularizer/Square/ReadVariableOp2conv2d_61/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2h
2conv2d_62/kernel/Regularizer/Square/ReadVariableOp2conv2d_62/kernel/Regularizer/Square/ReadVariableOp2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_7
	
Ô
9__inference_batch_normalization_42_layer_call_fn_23490369

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
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23487547
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
K__inference_activation_47_layer_call_and_return_conditional_losses_23490985

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
¾
§
*__inference_model_6_layer_call_fn_23488474
input_7!
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
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
E__inference_model_6_layer_call_and_return_conditional_losses_23488375o
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
_user_specified_name	input_7
È	
ö
E__inference_dense_6_layer_call_and_return_conditional_losses_23491171

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
9__inference_batch_normalization_46_layer_call_fn_23490811

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
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23487772
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
³
H
,__inference_flatten_6_layer_call_fn_23491146

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
G__inference_flatten_6_layer_call_and_return_conditional_losses_23488302`
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
Ý
Ã
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23487675

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
G__inference_conv2d_62_layer_call_and_return_conditional_losses_23488265

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_62/kernel/Regularizer/Square/ReadVariableOp|
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
2conv2d_62/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_62/kernel/Regularizer/SquareSquare:conv2d_62/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_62/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_62/kernel/Regularizer/SumSum'conv2d_62/kernel/Regularizer/Square:y:0+conv2d_62/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_62/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_62/kernel/Regularizer/mulMul+conv2d_62/kernel/Regularizer/mul/x:output:0)conv2d_62/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_62/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_62/kernel/Regularizer/Square/ReadVariableOp2conv2d_62/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
Ô
9__inference_batch_normalization_43_layer_call_fn_23490459

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
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23487580
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
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23491109

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
G__inference_conv2d_59_layer_call_and_return_conditional_losses_23488159

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_59/kernel/Regularizer/Square/ReadVariableOp|
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
2conv2d_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_59/kernel/Regularizer/SquareSquare:conv2d_59/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_59/kernel/Regularizer/SumSum'conv2d_59/kernel/Regularizer/Square:y:0+conv2d_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_59/kernel/Regularizer/mulMul+conv2d_59/kernel/Regularizer/mul/x:output:0)conv2d_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_59/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_59/kernel/Regularizer/Square/ReadVariableOp2conv2d_59/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ð
¡
,__inference_conv2d_54_layer_call_fn_23490327

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
G__inference_conv2d_54_layer_call_and_return_conditional_losses_23487977w
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
â
µ
G__inference_conv2d_56_layer_call_and_return_conditional_losses_23490549

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_56/kernel/Regularizer/Square/ReadVariableOp|
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
2conv2d_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_56/kernel/Regularizer/SquareSquare:conv2d_56/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_56/kernel/Regularizer/SumSum'conv2d_56/kernel/Regularizer/Square:y:0+conv2d_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_56/kernel/Regularizer/mulMul+conv2d_56/kernel/Regularizer/mul/x:output:0)conv2d_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_56/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_56/kernel/Regularizer/Square/ReadVariableOp2conv2d_56/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ñ
p
D__inference_add_20_layer_call_and_return_conditional_losses_23491121
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
ï
g
K__inference_activation_42_layer_call_and_return_conditional_losses_23487997

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
	
Ô
9__inference_batch_normalization_45_layer_call_fn_23490690

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
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23487739
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
È	
ö
E__inference_dense_6_layer_call_and_return_conditional_losses_23488314

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
é
n
D__inference_add_20_layer_call_and_return_conditional_losses_23488286

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
ë
½
__inference_loss_fn_7_23491259U
;conv2d_61_kernel_regularizer_square_readvariableop_resource:@@
identity¢2conv2d_61/kernel/Regularizer/Square/ReadVariableOp¶
2conv2d_61/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_61_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype0
#conv2d_61/kernel/Regularizer/SquareSquare:conv2d_61/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@{
"conv2d_61/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_61/kernel/Regularizer/SumSum'conv2d_61/kernel/Regularizer/Square:y:0+conv2d_61/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_61/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_61/kernel/Regularizer/mulMul+conv2d_61/kernel/Regularizer/mul/x:output:0)conv2d_61/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_61/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_61/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_61/kernel/Regularizer/Square/ReadVariableOp2conv2d_61/kernel/Regularizer/Square/ReadVariableOp
ë
½
__inference_loss_fn_2_23491204U
;conv2d_56_kernel_regularizer_square_readvariableop_resource:
identity¢2conv2d_56/kernel/Regularizer/Square/ReadVariableOp¶
2conv2d_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_56_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_56/kernel/Regularizer/SquareSquare:conv2d_56/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_56/kernel/Regularizer/SumSum'conv2d_56/kernel/Regularizer/Square:y:0+conv2d_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_56/kernel/Regularizer/mulMul+conv2d_56/kernel/Regularizer/mul/x:output:0)conv2d_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_56/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_56/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_56/kernel/Regularizer/Square/ReadVariableOp2conv2d_56/kernel/Regularizer/Square/ReadVariableOp
Ý
Ã
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23490405

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
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23487516

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
Ï

T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23490708

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
	
Ô
9__inference_batch_normalization_48_layer_call_fn_23491073

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
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23487931
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
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23487739

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
G__inference_conv2d_54_layer_call_and_return_conditional_losses_23490343

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢2conv2d_54/kernel/Regularizer/Square/ReadVariableOp|
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
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_54/kernel/Regularizer/SquareSquare:conv2d_54/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_54/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_54/kernel/Regularizer/SumSum'conv2d_54/kernel/Regularizer/Square:y:0+conv2d_54/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_54/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_54/kernel/Regularizer/mulMul+conv2d_54/kernel/Regularizer/mul/x:output:0)conv2d_54/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¬
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_54/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_54/kernel/Regularizer/Square/ReadVariableOp2conv2d_54/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ç
c
G__inference_flatten_6_layer_call_and_return_conditional_losses_23491152

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
ë
½
__inference_loss_fn_8_23491270U
;conv2d_62_kernel_regularizer_square_readvariableop_resource: @
identity¢2conv2d_62/kernel/Regularizer/Square/ReadVariableOp¶
2conv2d_62/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_62_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_62/kernel/Regularizer/SquareSquare:conv2d_62/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @{
"conv2d_62/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_62/kernel/Regularizer/SumSum'conv2d_62/kernel/Regularizer/Square:y:0+conv2d_62/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_62/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 conv2d_62/kernel/Regularizer/mulMul+conv2d_62/kernel/Regularizer/mul/x:output:0)conv2d_62/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_62/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_62/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_62/kernel/Regularizer/Square/ReadVariableOp2conv2d_62/kernel/Regularizer/Square/ReadVariableOp"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*²
serving_default
C
input_78
serving_default_input_7:0ÿÿÿÿÿÿÿÿÿ  ;
dense_60
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:×ï
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
*__inference_model_6_layer_call_fn_23488474
*__inference_model_6_layer_call_fn_23489650
*__inference_model_6_layer_call_fn_23489751
*__inference_model_6_layer_call_fn_23489129À
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
E__inference_model_6_layer_call_and_return_conditional_losses_23489980
E__inference_model_6_layer_call_and_return_conditional_losses_23490209
E__inference_model_6_layer_call_and_return_conditional_losses_23489312
E__inference_model_6_layer_call_and_return_conditional_losses_23489495À
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
#__inference__wrapped_model_23487494input_7"
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
*:(2conv2d_54/kernel
:2conv2d_54/bias
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
,__inference_conv2d_54_layer_call_fn_23490327¢
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
G__inference_conv2d_54_layer_call_and_return_conditional_losses_23490343¢
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
*:(2batch_normalization_42/gamma
):'2batch_normalization_42/beta
2:0 (2"batch_normalization_42/moving_mean
6:4 (2&batch_normalization_42/moving_variance
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
9__inference_batch_normalization_42_layer_call_fn_23490356
9__inference_batch_normalization_42_layer_call_fn_23490369´
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
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23490387
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23490405´
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
0__inference_activation_42_layer_call_fn_23490410¢
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
K__inference_activation_42_layer_call_and_return_conditional_losses_23490415¢
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
*:(2conv2d_55/kernel
:2conv2d_55/bias
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
,__inference_conv2d_55_layer_call_fn_23490430¢
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
G__inference_conv2d_55_layer_call_and_return_conditional_losses_23490446¢
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
*:(2batch_normalization_43/gamma
):'2batch_normalization_43/beta
2:0 (2"batch_normalization_43/moving_mean
6:4 (2&batch_normalization_43/moving_variance
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
9__inference_batch_normalization_43_layer_call_fn_23490459
9__inference_batch_normalization_43_layer_call_fn_23490472´
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
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23490490
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23490508´
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
0__inference_activation_43_layer_call_fn_23490513¢
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
K__inference_activation_43_layer_call_and_return_conditional_losses_23490518¢
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
*:(2conv2d_56/kernel
:2conv2d_56/bias
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
,__inference_conv2d_56_layer_call_fn_23490533¢
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
G__inference_conv2d_56_layer_call_and_return_conditional_losses_23490549¢
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
*:(2batch_normalization_44/gamma
):'2batch_normalization_44/beta
2:0 (2"batch_normalization_44/moving_mean
6:4 (2&batch_normalization_44/moving_variance
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
9__inference_batch_normalization_44_layer_call_fn_23490562
9__inference_batch_normalization_44_layer_call_fn_23490575´
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
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23490593
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23490611´
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
Ó2Ð
)__inference_add_18_layer_call_fn_23490617¢
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
D__inference_add_18_layer_call_and_return_conditional_losses_23490623¢
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
0__inference_activation_44_layer_call_fn_23490628¢
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
K__inference_activation_44_layer_call_and_return_conditional_losses_23490633¢
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
*:( 2conv2d_57/kernel
: 2conv2d_57/bias
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
,__inference_conv2d_57_layer_call_fn_23490648¢
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
G__inference_conv2d_57_layer_call_and_return_conditional_losses_23490664¢
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
*:( 2batch_normalization_45/gamma
):' 2batch_normalization_45/beta
2:0  (2"batch_normalization_45/moving_mean
6:4  (2&batch_normalization_45/moving_variance
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
9__inference_batch_normalization_45_layer_call_fn_23490677
9__inference_batch_normalization_45_layer_call_fn_23490690´
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
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23490708
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23490726´
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
0__inference_activation_45_layer_call_fn_23490731¢
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
K__inference_activation_45_layer_call_and_return_conditional_losses_23490736¢
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
*:(  2conv2d_58/kernel
: 2conv2d_58/bias
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
,__inference_conv2d_58_layer_call_fn_23490751¢
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
G__inference_conv2d_58_layer_call_and_return_conditional_losses_23490767¢
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
*:( 2conv2d_59/kernel
: 2conv2d_59/bias
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
,__inference_conv2d_59_layer_call_fn_23490782¢
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
G__inference_conv2d_59_layer_call_and_return_conditional_losses_23490798¢
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
*:( 2batch_normalization_46/gamma
):' 2batch_normalization_46/beta
2:0  (2"batch_normalization_46/moving_mean
6:4  (2&batch_normalization_46/moving_variance
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
9__inference_batch_normalization_46_layer_call_fn_23490811
9__inference_batch_normalization_46_layer_call_fn_23490824´
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
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23490842
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23490860´
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
)__inference_add_19_layer_call_fn_23490866¢
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
D__inference_add_19_layer_call_and_return_conditional_losses_23490872¢
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
0__inference_activation_46_layer_call_fn_23490877¢
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
K__inference_activation_46_layer_call_and_return_conditional_losses_23490882¢
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
*:( @2conv2d_60/kernel
:@2conv2d_60/bias
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
,__inference_conv2d_60_layer_call_fn_23490897¢
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
G__inference_conv2d_60_layer_call_and_return_conditional_losses_23490913¢
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
*:(@2batch_normalization_47/gamma
):'@2batch_normalization_47/beta
2:0@ (2"batch_normalization_47/moving_mean
6:4@ (2&batch_normalization_47/moving_variance
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
9__inference_batch_normalization_47_layer_call_fn_23490926
9__inference_batch_normalization_47_layer_call_fn_23490939´
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
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23490957
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23490975´
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
0__inference_activation_47_layer_call_fn_23490980¢
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
K__inference_activation_47_layer_call_and_return_conditional_losses_23490985¢
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
*:(@@2conv2d_61/kernel
:@2conv2d_61/bias
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
,__inference_conv2d_61_layer_call_fn_23491000¢
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
G__inference_conv2d_61_layer_call_and_return_conditional_losses_23491016¢
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
*:( @2conv2d_62/kernel
:@2conv2d_62/bias
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
,__inference_conv2d_62_layer_call_fn_23491031¢
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
G__inference_conv2d_62_layer_call_and_return_conditional_losses_23491047¢
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
*:(@2batch_normalization_48/gamma
):'@2batch_normalization_48/beta
2:0@ (2"batch_normalization_48/moving_mean
6:4@ (2&batch_normalization_48/moving_variance
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
9__inference_batch_normalization_48_layer_call_fn_23491060
9__inference_batch_normalization_48_layer_call_fn_23491073´
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
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23491091
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23491109´
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
)__inference_add_20_layer_call_fn_23491115¢
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
D__inference_add_20_layer_call_and_return_conditional_losses_23491121¢
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
0__inference_activation_48_layer_call_fn_23491126¢
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
K__inference_activation_48_layer_call_and_return_conditional_losses_23491131¢
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
6__inference_average_pooling2d_6_layer_call_fn_23491136¢
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
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_23491141¢
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
,__inference_flatten_6_layer_call_fn_23491146¢
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
G__inference_flatten_6_layer_call_and_return_conditional_losses_23491152¢
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
2dense_6/kernel
:
2dense_6/bias
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
*__inference_dense_6_layer_call_fn_23491161¢
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
E__inference_dense_6_layer_call_and_return_conditional_losses_23491171¢
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
__inference_loss_fn_0_23491182
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
__inference_loss_fn_1_23491193
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
__inference_loss_fn_2_23491204
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
__inference_loss_fn_3_23491215
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
__inference_loss_fn_4_23491226
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
__inference_loss_fn_5_23491237
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
__inference_loss_fn_6_23491248
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
__inference_loss_fn_7_23491259
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
__inference_loss_fn_8_23491270
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
&__inference_signature_wrapper_23490312input_7"
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
#__inference__wrapped_model_23487494»L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå8¢5
.¢+
)&
input_7ÿÿÿÿÿÿÿÿÿ  
ª "1ª.
,
dense_6!
dense_6ÿÿÿÿÿÿÿÿÿ
·
K__inference_activation_42_layer_call_and_return_conditional_losses_23490415h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
0__inference_activation_42_layer_call_fn_23490410[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ·
K__inference_activation_43_layer_call_and_return_conditional_losses_23490518h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
0__inference_activation_43_layer_call_fn_23490513[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ·
K__inference_activation_44_layer_call_and_return_conditional_losses_23490633h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
0__inference_activation_44_layer_call_fn_23490628[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ·
K__inference_activation_45_layer_call_and_return_conditional_losses_23490736h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
0__inference_activation_45_layer_call_fn_23490731[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ ·
K__inference_activation_46_layer_call_and_return_conditional_losses_23490882h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
0__inference_activation_46_layer_call_fn_23490877[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ ·
K__inference_activation_47_layer_call_and_return_conditional_losses_23490985h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
0__inference_activation_47_layer_call_fn_23490980[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@·
K__inference_activation_48_layer_call_and_return_conditional_losses_23491131h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
0__inference_activation_48_layer_call_fn_23491126[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@ä
D__inference_add_18_layer_call_and_return_conditional_losses_23490623j¢g
`¢]
[X
*'
inputs/0ÿÿÿÿÿÿÿÿÿ  
*'
inputs/1ÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 ¼
)__inference_add_18_layer_call_fn_23490617j¢g
`¢]
[X
*'
inputs/0ÿÿÿÿÿÿÿÿÿ  
*'
inputs/1ÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ä
D__inference_add_19_layer_call_and_return_conditional_losses_23490872j¢g
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
)__inference_add_19_layer_call_fn_23490866j¢g
`¢]
[X
*'
inputs/0ÿÿÿÿÿÿÿÿÿ 
*'
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ ä
D__inference_add_20_layer_call_and_return_conditional_losses_23491121j¢g
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
)__inference_add_20_layer_call_fn_23491115j¢g
`¢]
[X
*'
inputs/0ÿÿÿÿÿÿÿÿÿ@
*'
inputs/1ÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@ô
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_23491141R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ì
6__inference_average_pooling2d_6_layer_call_fn_23491136R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_234903870123M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ï
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_234904050123M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
9__inference_batch_normalization_42_layer_call_fn_234903560123M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÇ
9__inference_batch_normalization_42_layer_call_fn_234903690123M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23490490IJKLM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ï
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23490508IJKLM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
9__inference_batch_normalization_43_layer_call_fn_23490459IJKLM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÇ
9__inference_batch_normalization_43_layer_call_fn_23490472IJKLM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23490593bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ï
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23490611bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
9__inference_batch_normalization_44_layer_call_fn_23490562bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÇ
9__inference_batch_normalization_44_layer_call_fn_23490575bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿó
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23490708M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ó
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23490726M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ë
9__inference_batch_normalization_45_layer_call_fn_23490677M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ë
9__inference_batch_normalization_45_layer_call_fn_23490690M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ó
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23490842¢£¤¥M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ó
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23490860¢£¤¥M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ë
9__inference_batch_normalization_46_layer_call_fn_23490811¢£¤¥M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ë
9__inference_batch_normalization_46_layer_call_fn_23490824¢£¤¥M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ó
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23490957ÁÂÃÄM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ó
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23490975ÁÂÃÄM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ë
9__inference_batch_normalization_47_layer_call_fn_23490926ÁÂÃÄM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ë
9__inference_batch_normalization_47_layer_call_fn_23490939ÁÂÃÄM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ó
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23491091âãäåM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ó
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23491109âãäåM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ë
9__inference_batch_normalization_48_layer_call_fn_23491060âãäåM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ë
9__inference_batch_normalization_48_layer_call_fn_23491073âãäåM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@·
G__inference_conv2d_54_layer_call_and_return_conditional_losses_23490343l'(7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
,__inference_conv2d_54_layer_call_fn_23490327_'(7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ·
G__inference_conv2d_55_layer_call_and_return_conditional_losses_23490446l@A7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
,__inference_conv2d_55_layer_call_fn_23490430_@A7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ·
G__inference_conv2d_56_layer_call_and_return_conditional_losses_23490549lYZ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
,__inference_conv2d_56_layer_call_fn_23490533_YZ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ·
G__inference_conv2d_57_layer_call_and_return_conditional_losses_23490664lxy7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_conv2d_57_layer_call_fn_23490648_xy7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ ¹
G__inference_conv2d_58_layer_call_and_return_conditional_losses_23490767n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_conv2d_58_layer_call_fn_23490751a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ ¹
G__inference_conv2d_59_layer_call_and_return_conditional_losses_23490798n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_conv2d_59_layer_call_fn_23490782a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ ¹
G__inference_conv2d_60_layer_call_and_return_conditional_losses_23490913n¸¹7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_conv2d_60_layer_call_fn_23490897a¸¹7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ@¹
G__inference_conv2d_61_layer_call_and_return_conditional_losses_23491016nÑÒ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_conv2d_61_layer_call_fn_23491000aÑÒ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@¹
G__inference_conv2d_62_layer_call_and_return_conditional_losses_23491047nÙÚ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_conv2d_62_layer_call_fn_23491031aÙÚ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ@§
E__inference_dense_6_layer_call_and_return_conditional_losses_23491171^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
*__inference_dense_6_layer_call_fn_23491161Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ
«
G__inference_flatten_6_layer_call_and_return_conditional_losses_23491152`7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_flatten_6_layer_call_fn_23491146S7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@=
__inference_loss_fn_0_23491182'¢

¢ 
ª " =
__inference_loss_fn_1_23491193@¢

¢ 
ª " =
__inference_loss_fn_2_23491204Y¢

¢ 
ª " =
__inference_loss_fn_3_23491215x¢

¢ 
ª " >
__inference_loss_fn_4_23491226¢

¢ 
ª " >
__inference_loss_fn_5_23491237¢

¢ 
ª " >
__inference_loss_fn_6_23491248¸¢

¢ 
ª " >
__inference_loss_fn_7_23491259Ñ¢

¢ 
ª " >
__inference_loss_fn_8_23491270Ù¢

¢ 
ª " 
E__inference_model_6_layer_call_and_return_conditional_losses_23489312·L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå@¢=
6¢3
)&
input_7ÿÿÿÿÿÿÿÿÿ  
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
E__inference_model_6_layer_call_and_return_conditional_losses_23489495·L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå@¢=
6¢3
)&
input_7ÿÿÿÿÿÿÿÿÿ  
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
E__inference_model_6_layer_call_and_return_conditional_losses_23489980¶L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå?¢<
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
E__inference_model_6_layer_call_and_return_conditional_losses_23490209¶L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå?¢<
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
*__inference_model_6_layer_call_fn_23488474ªL'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå@¢=
6¢3
)&
input_7ÿÿÿÿÿÿÿÿÿ  
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
Ù
*__inference_model_6_layer_call_fn_23489129ªL'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå@¢=
6¢3
)&
input_7ÿÿÿÿÿÿÿÿÿ  
p

 
ª "ÿÿÿÿÿÿÿÿÿ
Ø
*__inference_model_6_layer_call_fn_23489650©L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
Ø
*__inference_model_6_layer_call_fn_23489751©L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "ÿÿÿÿÿÿÿÿÿ
ñ
&__inference_signature_wrapper_23490312ÆL'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäåC¢@
¢ 
9ª6
4
input_7)&
input_7ÿÿÿÿÿÿÿÿÿ  "1ª.
,
dense_6!
dense_6ÿÿÿÿÿÿÿÿÿ
