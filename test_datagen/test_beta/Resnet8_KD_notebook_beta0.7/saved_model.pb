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
&__inference_signature_wrapper_23490200
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
!__inference__traced_save_23491325
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
$__inference__traced_restore_23491479÷­
Ý
Ã
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23487691

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
,__inference_conv2d_55_layer_call_fn_23490318

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
G__inference_conv2d_55_layer_call_and_return_conditional_losses_23487903w
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
õ×
¿2
E__inference_model_6_layer_call_and_return_conditional_losses_23490097

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
Ý
Ã
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23487435

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
­
¦
*__inference_model_6_layer_call_fn_23489639

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
E__inference_model_6_layer_call_and_return_conditional_losses_23488817o
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
Ë
L
0__inference_activation_43_layer_call_fn_23490401

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
K__inference_activation_43_layer_call_and_return_conditional_losses_23487923h
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
ñ
p
D__inference_add_19_layer_call_and_return_conditional_losses_23490760
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
â
µ
G__inference_conv2d_54_layer_call_and_return_conditional_losses_23487865

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
K__inference_activation_48_layer_call_and_return_conditional_losses_23488181

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
,__inference_conv2d_54_layer_call_fn_23490215

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
G__inference_conv2d_54_layer_call_and_return_conditional_losses_23487865w
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
Ý
Ã
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23490997

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
	
Ô
9__inference_batch_normalization_42_layer_call_fn_23490257

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
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23487435
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
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23490378

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
G__inference_conv2d_57_layer_call_and_return_conditional_losses_23487987

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
â
µ
G__inference_conv2d_59_layer_call_and_return_conditional_losses_23488047

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
â
µ
G__inference_conv2d_60_layer_call_and_return_conditional_losses_23488093

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
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23487532

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
__inference_loss_fn_4_23491114U
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
Ï

T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23490275

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
0__inference_activation_42_layer_call_fn_23490298

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
K__inference_activation_42_layer_call_and_return_conditional_losses_23487885h
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
Ç
c
G__inference_flatten_6_layer_call_and_return_conditional_losses_23491040

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
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23490979

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
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23490614

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
__inference_loss_fn_3_23491103U
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
	
Ô
9__inference_batch_normalization_42_layer_call_fn_23490244

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
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23487404
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
K__inference_activation_42_layer_call_and_return_conditional_losses_23490303

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
Á÷
0
#__inference__wrapped_model_23487382
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
Ò
U
)__inference_add_18_layer_call_fn_23490505
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
D__inference_add_18_layer_call_and_return_conditional_losses_23487962h
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
ë
½
__inference_loss_fn_0_23491070U
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
â
µ
G__inference_conv2d_60_layer_call_and_return_conditional_losses_23490801

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
Ý
Ã
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23487499

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
,__inference_conv2d_58_layer_call_fn_23490639

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
G__inference_conv2d_58_layer_call_and_return_conditional_losses_23488025w
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
Ý
Ã
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23490293

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
ë
½
__inference_loss_fn_5_23491125U
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
Ï

T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23487596

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
ï
g
K__inference_activation_44_layer_call_and_return_conditional_losses_23490521

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
ï
g
K__inference_activation_43_layer_call_and_return_conditional_losses_23487923

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
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23487563

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
,__inference_conv2d_61_layer_call_fn_23490888

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
G__inference_conv2d_61_layer_call_and_return_conditional_losses_23488131w
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
ï
g
K__inference_activation_43_layer_call_and_return_conditional_losses_23490406

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
ñ
p
D__inference_add_20_layer_call_and_return_conditional_losses_23491009
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
Ë
L
0__inference_activation_47_layer_call_fn_23490868

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
K__inference_activation_47_layer_call_and_return_conditional_losses_23488113h
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
K__inference_activation_44_layer_call_and_return_conditional_losses_23487969

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

£
&__inference_signature_wrapper_23490200
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
#__inference__wrapped_model_23487382o
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
	
Ô
9__inference_batch_normalization_44_layer_call_fn_23490463

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
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23487563
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
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23490596

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
ë
½
__inference_loss_fn_7_23491147U
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
Ý
Ã
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23490748

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
9__inference_batch_normalization_46_layer_call_fn_23490712

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
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23487691
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
Ï

T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23487660

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
G__inference_conv2d_61_layer_call_and_return_conditional_losses_23488131

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
Ý
Ã
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23490396

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
	
Ô
9__inference_batch_normalization_47_layer_call_fn_23490814

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
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23487724
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
â
µ
G__inference_conv2d_57_layer_call_and_return_conditional_losses_23490552

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
³
H
,__inference_flatten_6_layer_call_fn_23491034

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
G__inference_flatten_6_layer_call_and_return_conditional_losses_23488190`
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
â
µ
G__inference_conv2d_56_layer_call_and_return_conditional_losses_23490437

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
	
Ô
9__inference_batch_normalization_47_layer_call_fn_23490827

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
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23487755
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
Ç
c
G__inference_flatten_6_layer_call_and_return_conditional_losses_23488190

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
G__inference_conv2d_55_layer_call_and_return_conditional_losses_23490334

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
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23487468

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
D__inference_add_18_layer_call_and_return_conditional_losses_23487962

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
¯Ø
¶
E__inference_model_6_layer_call_and_return_conditional_losses_23488263

inputs,
conv2d_54_23487866: 
conv2d_54_23487868:-
batch_normalization_42_23487871:-
batch_normalization_42_23487873:-
batch_normalization_42_23487875:-
batch_normalization_42_23487877:,
conv2d_55_23487904: 
conv2d_55_23487906:-
batch_normalization_43_23487909:-
batch_normalization_43_23487911:-
batch_normalization_43_23487913:-
batch_normalization_43_23487915:,
conv2d_56_23487942: 
conv2d_56_23487944:-
batch_normalization_44_23487947:-
batch_normalization_44_23487949:-
batch_normalization_44_23487951:-
batch_normalization_44_23487953:,
conv2d_57_23487988:  
conv2d_57_23487990: -
batch_normalization_45_23487993: -
batch_normalization_45_23487995: -
batch_normalization_45_23487997: -
batch_normalization_45_23487999: ,
conv2d_58_23488026:   
conv2d_58_23488028: ,
conv2d_59_23488048:  
conv2d_59_23488050: -
batch_normalization_46_23488053: -
batch_normalization_46_23488055: -
batch_normalization_46_23488057: -
batch_normalization_46_23488059: ,
conv2d_60_23488094: @ 
conv2d_60_23488096:@-
batch_normalization_47_23488099:@-
batch_normalization_47_23488101:@-
batch_normalization_47_23488103:@-
batch_normalization_47_23488105:@,
conv2d_61_23488132:@@ 
conv2d_61_23488134:@,
conv2d_62_23488154: @ 
conv2d_62_23488156:@-
batch_normalization_48_23488159:@-
batch_normalization_48_23488161:@-
batch_normalization_48_23488163:@-
batch_normalization_48_23488165:@"
dense_6_23488203:@

dense_6_23488205:

identity¢.batch_normalization_42/StatefulPartitionedCall¢.batch_normalization_43/StatefulPartitionedCall¢.batch_normalization_44/StatefulPartitionedCall¢.batch_normalization_45/StatefulPartitionedCall¢.batch_normalization_46/StatefulPartitionedCall¢.batch_normalization_47/StatefulPartitionedCall¢.batch_normalization_48/StatefulPartitionedCall¢!conv2d_54/StatefulPartitionedCall¢2conv2d_54/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_55/StatefulPartitionedCall¢2conv2d_55/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_56/StatefulPartitionedCall¢2conv2d_56/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_57/StatefulPartitionedCall¢2conv2d_57/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_58/StatefulPartitionedCall¢2conv2d_58/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_59/StatefulPartitionedCall¢2conv2d_59/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_60/StatefulPartitionedCall¢2conv2d_60/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_61/StatefulPartitionedCall¢2conv2d_61/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_62/StatefulPartitionedCall¢2conv2d_62/kernel/Regularizer/Square/ReadVariableOp¢dense_6/StatefulPartitionedCall
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_54_23487866conv2d_54_23487868*
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
G__inference_conv2d_54_layer_call_and_return_conditional_losses_23487865 
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_42_23487871batch_normalization_42_23487873batch_normalization_42_23487875batch_normalization_42_23487877*
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
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23487404ý
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
K__inference_activation_42_layer_call_and_return_conditional_losses_23487885¢
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall&activation_42/PartitionedCall:output:0conv2d_55_23487904conv2d_55_23487906*
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
G__inference_conv2d_55_layer_call_and_return_conditional_losses_23487903 
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0batch_normalization_43_23487909batch_normalization_43_23487911batch_normalization_43_23487913batch_normalization_43_23487915*
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
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23487468ý
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
K__inference_activation_43_layer_call_and_return_conditional_losses_23487923¢
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall&activation_43/PartitionedCall:output:0conv2d_56_23487942conv2d_56_23487944*
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
G__inference_conv2d_56_layer_call_and_return_conditional_losses_23487941 
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_56/StatefulPartitionedCall:output:0batch_normalization_44_23487947batch_normalization_44_23487949batch_normalization_44_23487951batch_normalization_44_23487953*
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
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23487532
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
D__inference_add_18_layer_call_and_return_conditional_losses_23487962å
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
K__inference_activation_44_layer_call_and_return_conditional_losses_23487969¢
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCall&activation_44/PartitionedCall:output:0conv2d_57_23487988conv2d_57_23487990*
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
G__inference_conv2d_57_layer_call_and_return_conditional_losses_23487987 
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0batch_normalization_45_23487993batch_normalization_45_23487995batch_normalization_45_23487997batch_normalization_45_23487999*
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
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23487596ý
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
K__inference_activation_45_layer_call_and_return_conditional_losses_23488007¢
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall&activation_45/PartitionedCall:output:0conv2d_58_23488026conv2d_58_23488028*
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
G__inference_conv2d_58_layer_call_and_return_conditional_losses_23488025¢
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall&activation_44/PartitionedCall:output:0conv2d_59_23488048conv2d_59_23488050*
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
G__inference_conv2d_59_layer_call_and_return_conditional_losses_23488047 
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0batch_normalization_46_23488053batch_normalization_46_23488055batch_normalization_46_23488057batch_normalization_46_23488059*
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
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23487660
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
D__inference_add_19_layer_call_and_return_conditional_losses_23488068å
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
K__inference_activation_46_layer_call_and_return_conditional_losses_23488075¢
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall&activation_46/PartitionedCall:output:0conv2d_60_23488094conv2d_60_23488096*
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
G__inference_conv2d_60_layer_call_and_return_conditional_losses_23488093 
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0batch_normalization_47_23488099batch_normalization_47_23488101batch_normalization_47_23488103batch_normalization_47_23488105*
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
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23487724ý
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
K__inference_activation_47_layer_call_and_return_conditional_losses_23488113¢
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall&activation_47/PartitionedCall:output:0conv2d_61_23488132conv2d_61_23488134*
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
G__inference_conv2d_61_layer_call_and_return_conditional_losses_23488131¢
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall&activation_46/PartitionedCall:output:0conv2d_62_23488154conv2d_62_23488156*
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
G__inference_conv2d_62_layer_call_and_return_conditional_losses_23488153 
.batch_normalization_48/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0batch_normalization_48_23488159batch_normalization_48_23488161batch_normalization_48_23488163batch_normalization_48_23488165*
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
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23487788
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
D__inference_add_20_layer_call_and_return_conditional_losses_23488174å
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
K__inference_activation_48_layer_call_and_return_conditional_losses_23488181ø
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
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_23487839â
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
G__inference_flatten_6_layer_call_and_return_conditional_losses_23488190
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_6_23488203dense_6_23488205*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_23488202
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_54_23487866*&
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
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_55_23487904*&
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
2conv2d_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_56_23487942*&
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
2conv2d_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_57_23487988*&
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
2conv2d_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_58_23488026*&
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
2conv2d_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_59_23488048*&
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
2conv2d_60/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_60_23488094*&
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
2conv2d_61/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_61_23488132*&
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
2conv2d_62/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_62_23488154*&
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
ð
¡
,__inference_conv2d_59_layer_call_fn_23490670

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
G__inference_conv2d_59_layer_call_and_return_conditional_losses_23488047w
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
Ë
L
0__inference_activation_46_layer_call_fn_23490765

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
K__inference_activation_46_layer_call_and_return_conditional_losses_23488075h
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
ï
g
K__inference_activation_46_layer_call_and_return_conditional_losses_23490770

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
Ä

*__inference_dense_6_layer_call_fn_23491049

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
E__inference_dense_6_layer_call_and_return_conditional_losses_23488202o
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
ð
¡
,__inference_conv2d_56_layer_call_fn_23490421

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
G__inference_conv2d_56_layer_call_and_return_conditional_losses_23487941w
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
ë
½
__inference_loss_fn_2_23491092U
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
Ë
L
0__inference_activation_48_layer_call_fn_23491014

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
K__inference_activation_48_layer_call_and_return_conditional_losses_23488181h
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
ð
¡
,__inference_conv2d_57_layer_call_fn_23490536

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
G__inference_conv2d_57_layer_call_and_return_conditional_losses_23487987w
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
¾
§
*__inference_model_6_layer_call_fn_23488362
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
E__inference_model_6_layer_call_and_return_conditional_losses_23488263o
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
K__inference_activation_46_layer_call_and_return_conditional_losses_23488075

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
»
¦
*__inference_model_6_layer_call_fn_23489538

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
E__inference_model_6_layer_call_and_return_conditional_losses_23488263o
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
¡Ø
¶
E__inference_model_6_layer_call_and_return_conditional_losses_23488817

inputs,
conv2d_54_23488637: 
conv2d_54_23488639:-
batch_normalization_42_23488642:-
batch_normalization_42_23488644:-
batch_normalization_42_23488646:-
batch_normalization_42_23488648:,
conv2d_55_23488652: 
conv2d_55_23488654:-
batch_normalization_43_23488657:-
batch_normalization_43_23488659:-
batch_normalization_43_23488661:-
batch_normalization_43_23488663:,
conv2d_56_23488667: 
conv2d_56_23488669:-
batch_normalization_44_23488672:-
batch_normalization_44_23488674:-
batch_normalization_44_23488676:-
batch_normalization_44_23488678:,
conv2d_57_23488683:  
conv2d_57_23488685: -
batch_normalization_45_23488688: -
batch_normalization_45_23488690: -
batch_normalization_45_23488692: -
batch_normalization_45_23488694: ,
conv2d_58_23488698:   
conv2d_58_23488700: ,
conv2d_59_23488703:  
conv2d_59_23488705: -
batch_normalization_46_23488708: -
batch_normalization_46_23488710: -
batch_normalization_46_23488712: -
batch_normalization_46_23488714: ,
conv2d_60_23488719: @ 
conv2d_60_23488721:@-
batch_normalization_47_23488724:@-
batch_normalization_47_23488726:@-
batch_normalization_47_23488728:@-
batch_normalization_47_23488730:@,
conv2d_61_23488734:@@ 
conv2d_61_23488736:@,
conv2d_62_23488739: @ 
conv2d_62_23488741:@-
batch_normalization_48_23488744:@-
batch_normalization_48_23488746:@-
batch_normalization_48_23488748:@-
batch_normalization_48_23488750:@"
dense_6_23488757:@

dense_6_23488759:

identity¢.batch_normalization_42/StatefulPartitionedCall¢.batch_normalization_43/StatefulPartitionedCall¢.batch_normalization_44/StatefulPartitionedCall¢.batch_normalization_45/StatefulPartitionedCall¢.batch_normalization_46/StatefulPartitionedCall¢.batch_normalization_47/StatefulPartitionedCall¢.batch_normalization_48/StatefulPartitionedCall¢!conv2d_54/StatefulPartitionedCall¢2conv2d_54/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_55/StatefulPartitionedCall¢2conv2d_55/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_56/StatefulPartitionedCall¢2conv2d_56/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_57/StatefulPartitionedCall¢2conv2d_57/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_58/StatefulPartitionedCall¢2conv2d_58/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_59/StatefulPartitionedCall¢2conv2d_59/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_60/StatefulPartitionedCall¢2conv2d_60/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_61/StatefulPartitionedCall¢2conv2d_61/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_62/StatefulPartitionedCall¢2conv2d_62/kernel/Regularizer/Square/ReadVariableOp¢dense_6/StatefulPartitionedCall
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_54_23488637conv2d_54_23488639*
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
G__inference_conv2d_54_layer_call_and_return_conditional_losses_23487865
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_42_23488642batch_normalization_42_23488644batch_normalization_42_23488646batch_normalization_42_23488648*
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
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23487435ý
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
K__inference_activation_42_layer_call_and_return_conditional_losses_23487885¢
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall&activation_42/PartitionedCall:output:0conv2d_55_23488652conv2d_55_23488654*
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
G__inference_conv2d_55_layer_call_and_return_conditional_losses_23487903
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0batch_normalization_43_23488657batch_normalization_43_23488659batch_normalization_43_23488661batch_normalization_43_23488663*
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
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23487499ý
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
K__inference_activation_43_layer_call_and_return_conditional_losses_23487923¢
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall&activation_43/PartitionedCall:output:0conv2d_56_23488667conv2d_56_23488669*
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
G__inference_conv2d_56_layer_call_and_return_conditional_losses_23487941
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_56/StatefulPartitionedCall:output:0batch_normalization_44_23488672batch_normalization_44_23488674batch_normalization_44_23488676batch_normalization_44_23488678*
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
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23487563
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
D__inference_add_18_layer_call_and_return_conditional_losses_23487962å
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
K__inference_activation_44_layer_call_and_return_conditional_losses_23487969¢
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCall&activation_44/PartitionedCall:output:0conv2d_57_23488683conv2d_57_23488685*
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
G__inference_conv2d_57_layer_call_and_return_conditional_losses_23487987
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0batch_normalization_45_23488688batch_normalization_45_23488690batch_normalization_45_23488692batch_normalization_45_23488694*
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
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23487627ý
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
K__inference_activation_45_layer_call_and_return_conditional_losses_23488007¢
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall&activation_45/PartitionedCall:output:0conv2d_58_23488698conv2d_58_23488700*
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
G__inference_conv2d_58_layer_call_and_return_conditional_losses_23488025¢
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall&activation_44/PartitionedCall:output:0conv2d_59_23488703conv2d_59_23488705*
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
G__inference_conv2d_59_layer_call_and_return_conditional_losses_23488047
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0batch_normalization_46_23488708batch_normalization_46_23488710batch_normalization_46_23488712batch_normalization_46_23488714*
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
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23487691
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
D__inference_add_19_layer_call_and_return_conditional_losses_23488068å
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
K__inference_activation_46_layer_call_and_return_conditional_losses_23488075¢
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall&activation_46/PartitionedCall:output:0conv2d_60_23488719conv2d_60_23488721*
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
G__inference_conv2d_60_layer_call_and_return_conditional_losses_23488093
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0batch_normalization_47_23488724batch_normalization_47_23488726batch_normalization_47_23488728batch_normalization_47_23488730*
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
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23487755ý
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
K__inference_activation_47_layer_call_and_return_conditional_losses_23488113¢
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall&activation_47/PartitionedCall:output:0conv2d_61_23488734conv2d_61_23488736*
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
G__inference_conv2d_61_layer_call_and_return_conditional_losses_23488131¢
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall&activation_46/PartitionedCall:output:0conv2d_62_23488739conv2d_62_23488741*
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
G__inference_conv2d_62_layer_call_and_return_conditional_losses_23488153
.batch_normalization_48/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0batch_normalization_48_23488744batch_normalization_48_23488746batch_normalization_48_23488748batch_normalization_48_23488750*
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
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23487819
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
D__inference_add_20_layer_call_and_return_conditional_losses_23488174å
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
K__inference_activation_48_layer_call_and_return_conditional_losses_23488181ø
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
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_23487839â
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
G__inference_flatten_6_layer_call_and_return_conditional_losses_23488190
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_6_23488757dense_6_23488759*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_23488202
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_54_23488637*&
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
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_55_23488652*&
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
2conv2d_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_56_23488667*&
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
2conv2d_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_57_23488683*&
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
2conv2d_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_58_23488698*&
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
2conv2d_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_59_23488703*&
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
2conv2d_60/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_60_23488719*&
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
2conv2d_61/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_61_23488734*&
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
2conv2d_62/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_62_23488739*&
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
	
Ô
9__inference_batch_normalization_44_layer_call_fn_23490450

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
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23487532
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
G__inference_conv2d_62_layer_call_and_return_conditional_losses_23490935

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
ï
g
K__inference_activation_47_layer_call_and_return_conditional_losses_23490873

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
K__inference_activation_48_layer_call_and_return_conditional_losses_23491019

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
6__inference_average_pooling2d_6_layer_call_fn_23491024

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
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_23487839
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
G__inference_conv2d_58_layer_call_and_return_conditional_losses_23488025

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
È	
ö
E__inference_dense_6_layer_call_and_return_conditional_losses_23488202

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
Ï

T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23490481

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
G__inference_conv2d_58_layer_call_and_return_conditional_losses_23490655

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
¢
m
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_23491029

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
ë
½
__inference_loss_fn_6_23491136U
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
Ý
Ã
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23487627

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
°
§
*__inference_model_6_layer_call_fn_23489017
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
E__inference_model_6_layer_call_and_return_conditional_losses_23488817o
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
	
Ô
9__inference_batch_normalization_45_layer_call_fn_23490578

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
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23487627
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
Ë
L
0__inference_activation_45_layer_call_fn_23490619

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
K__inference_activation_45_layer_call_and_return_conditional_losses_23488007h
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
Ò
U
)__inference_add_19_layer_call_fn_23490754
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
D__inference_add_19_layer_call_and_return_conditional_losses_23488068h
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
G__inference_conv2d_62_layer_call_and_return_conditional_losses_23488153

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
Ë
L
0__inference_activation_44_layer_call_fn_23490516

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
K__inference_activation_44_layer_call_and_return_conditional_losses_23487969h
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
Ï

T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23487724

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
ïÀ
¤ 
$__inference__traced_restore_23491479
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
ï
g
K__inference_activation_45_layer_call_and_return_conditional_losses_23490624

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
G__inference_conv2d_54_layer_call_and_return_conditional_losses_23490231

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
Ï

T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23487788

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
§
.
E__inference_model_6_layer_call_and_return_conditional_losses_23489868

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
â
µ
G__inference_conv2d_55_layer_call_and_return_conditional_losses_23487903

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
	
Ô
9__inference_batch_normalization_46_layer_call_fn_23490699

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
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23487660
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
Ï

T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23490845

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
G__inference_conv2d_56_layer_call_and_return_conditional_losses_23487941

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
é
n
D__inference_add_19_layer_call_and_return_conditional_losses_23488068

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
	
Ô
9__inference_batch_normalization_43_layer_call_fn_23490360

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
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23487499
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
ë
½
__inference_loss_fn_8_23491158U
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
2conv2d_62/kernel/Regularizer/Square/ReadVariableOp2conv2d_62/kernel/Regularizer/Square/ReadVariableOp
Ý
Ã
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23487755

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
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23490499

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
ë
½
__inference_loss_fn_1_23491081U
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
·b
¿
!__inference__traced_save_23491325
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
²Ø
·
E__inference_model_6_layer_call_and_return_conditional_losses_23489200
input_7,
conv2d_54_23489020: 
conv2d_54_23489022:-
batch_normalization_42_23489025:-
batch_normalization_42_23489027:-
batch_normalization_42_23489029:-
batch_normalization_42_23489031:,
conv2d_55_23489035: 
conv2d_55_23489037:-
batch_normalization_43_23489040:-
batch_normalization_43_23489042:-
batch_normalization_43_23489044:-
batch_normalization_43_23489046:,
conv2d_56_23489050: 
conv2d_56_23489052:-
batch_normalization_44_23489055:-
batch_normalization_44_23489057:-
batch_normalization_44_23489059:-
batch_normalization_44_23489061:,
conv2d_57_23489066:  
conv2d_57_23489068: -
batch_normalization_45_23489071: -
batch_normalization_45_23489073: -
batch_normalization_45_23489075: -
batch_normalization_45_23489077: ,
conv2d_58_23489081:   
conv2d_58_23489083: ,
conv2d_59_23489086:  
conv2d_59_23489088: -
batch_normalization_46_23489091: -
batch_normalization_46_23489093: -
batch_normalization_46_23489095: -
batch_normalization_46_23489097: ,
conv2d_60_23489102: @ 
conv2d_60_23489104:@-
batch_normalization_47_23489107:@-
batch_normalization_47_23489109:@-
batch_normalization_47_23489111:@-
batch_normalization_47_23489113:@,
conv2d_61_23489117:@@ 
conv2d_61_23489119:@,
conv2d_62_23489122: @ 
conv2d_62_23489124:@-
batch_normalization_48_23489127:@-
batch_normalization_48_23489129:@-
batch_normalization_48_23489131:@-
batch_normalization_48_23489133:@"
dense_6_23489140:@

dense_6_23489142:

identity¢.batch_normalization_42/StatefulPartitionedCall¢.batch_normalization_43/StatefulPartitionedCall¢.batch_normalization_44/StatefulPartitionedCall¢.batch_normalization_45/StatefulPartitionedCall¢.batch_normalization_46/StatefulPartitionedCall¢.batch_normalization_47/StatefulPartitionedCall¢.batch_normalization_48/StatefulPartitionedCall¢!conv2d_54/StatefulPartitionedCall¢2conv2d_54/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_55/StatefulPartitionedCall¢2conv2d_55/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_56/StatefulPartitionedCall¢2conv2d_56/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_57/StatefulPartitionedCall¢2conv2d_57/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_58/StatefulPartitionedCall¢2conv2d_58/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_59/StatefulPartitionedCall¢2conv2d_59/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_60/StatefulPartitionedCall¢2conv2d_60/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_61/StatefulPartitionedCall¢2conv2d_61/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_62/StatefulPartitionedCall¢2conv2d_62/kernel/Regularizer/Square/ReadVariableOp¢dense_6/StatefulPartitionedCall
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCallinput_7conv2d_54_23489020conv2d_54_23489022*
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
G__inference_conv2d_54_layer_call_and_return_conditional_losses_23487865 
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_42_23489025batch_normalization_42_23489027batch_normalization_42_23489029batch_normalization_42_23489031*
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
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23487404ý
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
K__inference_activation_42_layer_call_and_return_conditional_losses_23487885¢
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall&activation_42/PartitionedCall:output:0conv2d_55_23489035conv2d_55_23489037*
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
G__inference_conv2d_55_layer_call_and_return_conditional_losses_23487903 
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0batch_normalization_43_23489040batch_normalization_43_23489042batch_normalization_43_23489044batch_normalization_43_23489046*
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
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23487468ý
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
K__inference_activation_43_layer_call_and_return_conditional_losses_23487923¢
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall&activation_43/PartitionedCall:output:0conv2d_56_23489050conv2d_56_23489052*
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
G__inference_conv2d_56_layer_call_and_return_conditional_losses_23487941 
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_56/StatefulPartitionedCall:output:0batch_normalization_44_23489055batch_normalization_44_23489057batch_normalization_44_23489059batch_normalization_44_23489061*
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
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23487532
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
D__inference_add_18_layer_call_and_return_conditional_losses_23487962å
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
K__inference_activation_44_layer_call_and_return_conditional_losses_23487969¢
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCall&activation_44/PartitionedCall:output:0conv2d_57_23489066conv2d_57_23489068*
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
G__inference_conv2d_57_layer_call_and_return_conditional_losses_23487987 
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0batch_normalization_45_23489071batch_normalization_45_23489073batch_normalization_45_23489075batch_normalization_45_23489077*
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
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23487596ý
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
K__inference_activation_45_layer_call_and_return_conditional_losses_23488007¢
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall&activation_45/PartitionedCall:output:0conv2d_58_23489081conv2d_58_23489083*
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
G__inference_conv2d_58_layer_call_and_return_conditional_losses_23488025¢
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall&activation_44/PartitionedCall:output:0conv2d_59_23489086conv2d_59_23489088*
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
G__inference_conv2d_59_layer_call_and_return_conditional_losses_23488047 
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0batch_normalization_46_23489091batch_normalization_46_23489093batch_normalization_46_23489095batch_normalization_46_23489097*
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
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23487660
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
D__inference_add_19_layer_call_and_return_conditional_losses_23488068å
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
K__inference_activation_46_layer_call_and_return_conditional_losses_23488075¢
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall&activation_46/PartitionedCall:output:0conv2d_60_23489102conv2d_60_23489104*
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
G__inference_conv2d_60_layer_call_and_return_conditional_losses_23488093 
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0batch_normalization_47_23489107batch_normalization_47_23489109batch_normalization_47_23489111batch_normalization_47_23489113*
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
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23487724ý
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
K__inference_activation_47_layer_call_and_return_conditional_losses_23488113¢
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall&activation_47/PartitionedCall:output:0conv2d_61_23489117conv2d_61_23489119*
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
G__inference_conv2d_61_layer_call_and_return_conditional_losses_23488131¢
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall&activation_46/PartitionedCall:output:0conv2d_62_23489122conv2d_62_23489124*
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
G__inference_conv2d_62_layer_call_and_return_conditional_losses_23488153 
.batch_normalization_48/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0batch_normalization_48_23489127batch_normalization_48_23489129batch_normalization_48_23489131batch_normalization_48_23489133*
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
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23487788
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
D__inference_add_20_layer_call_and_return_conditional_losses_23488174å
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
K__inference_activation_48_layer_call_and_return_conditional_losses_23488181ø
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
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_23487839â
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
G__inference_flatten_6_layer_call_and_return_conditional_losses_23488190
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_6_23489140dense_6_23489142*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_23488202
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_54_23489020*&
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
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_55_23489035*&
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
2conv2d_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_56_23489050*&
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
2conv2d_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_57_23489066*&
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
2conv2d_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_58_23489081*&
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
2conv2d_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_59_23489086*&
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
2conv2d_60/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_60_23489102*&
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
2conv2d_61/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_61_23489117*&
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
2conv2d_62/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_62_23489122*&
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
ð
¡
,__inference_conv2d_62_layer_call_fn_23490919

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
G__inference_conv2d_62_layer_call_and_return_conditional_losses_23488153w
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
	
Ô
9__inference_batch_normalization_48_layer_call_fn_23490948

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
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23487788
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
K__inference_activation_42_layer_call_and_return_conditional_losses_23487885

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
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23490863

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
é
n
D__inference_add_20_layer_call_and_return_conditional_losses_23488174

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
Ï

T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23490730

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
Ò
U
)__inference_add_20_layer_call_fn_23491003
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
D__inference_add_20_layer_call_and_return_conditional_losses_23488174h
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
È	
ö
E__inference_dense_6_layer_call_and_return_conditional_losses_23491059

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
Ï

T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23487404

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
G__inference_conv2d_61_layer_call_and_return_conditional_losses_23490904

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
¢
m
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_23487839

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
¤Ø
·
E__inference_model_6_layer_call_and_return_conditional_losses_23489383
input_7,
conv2d_54_23489203: 
conv2d_54_23489205:-
batch_normalization_42_23489208:-
batch_normalization_42_23489210:-
batch_normalization_42_23489212:-
batch_normalization_42_23489214:,
conv2d_55_23489218: 
conv2d_55_23489220:-
batch_normalization_43_23489223:-
batch_normalization_43_23489225:-
batch_normalization_43_23489227:-
batch_normalization_43_23489229:,
conv2d_56_23489233: 
conv2d_56_23489235:-
batch_normalization_44_23489238:-
batch_normalization_44_23489240:-
batch_normalization_44_23489242:-
batch_normalization_44_23489244:,
conv2d_57_23489249:  
conv2d_57_23489251: -
batch_normalization_45_23489254: -
batch_normalization_45_23489256: -
batch_normalization_45_23489258: -
batch_normalization_45_23489260: ,
conv2d_58_23489264:   
conv2d_58_23489266: ,
conv2d_59_23489269:  
conv2d_59_23489271: -
batch_normalization_46_23489274: -
batch_normalization_46_23489276: -
batch_normalization_46_23489278: -
batch_normalization_46_23489280: ,
conv2d_60_23489285: @ 
conv2d_60_23489287:@-
batch_normalization_47_23489290:@-
batch_normalization_47_23489292:@-
batch_normalization_47_23489294:@-
batch_normalization_47_23489296:@,
conv2d_61_23489300:@@ 
conv2d_61_23489302:@,
conv2d_62_23489305: @ 
conv2d_62_23489307:@-
batch_normalization_48_23489310:@-
batch_normalization_48_23489312:@-
batch_normalization_48_23489314:@-
batch_normalization_48_23489316:@"
dense_6_23489323:@

dense_6_23489325:

identity¢.batch_normalization_42/StatefulPartitionedCall¢.batch_normalization_43/StatefulPartitionedCall¢.batch_normalization_44/StatefulPartitionedCall¢.batch_normalization_45/StatefulPartitionedCall¢.batch_normalization_46/StatefulPartitionedCall¢.batch_normalization_47/StatefulPartitionedCall¢.batch_normalization_48/StatefulPartitionedCall¢!conv2d_54/StatefulPartitionedCall¢2conv2d_54/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_55/StatefulPartitionedCall¢2conv2d_55/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_56/StatefulPartitionedCall¢2conv2d_56/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_57/StatefulPartitionedCall¢2conv2d_57/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_58/StatefulPartitionedCall¢2conv2d_58/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_59/StatefulPartitionedCall¢2conv2d_59/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_60/StatefulPartitionedCall¢2conv2d_60/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_61/StatefulPartitionedCall¢2conv2d_61/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_62/StatefulPartitionedCall¢2conv2d_62/kernel/Regularizer/Square/ReadVariableOp¢dense_6/StatefulPartitionedCall
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCallinput_7conv2d_54_23489203conv2d_54_23489205*
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
G__inference_conv2d_54_layer_call_and_return_conditional_losses_23487865
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0batch_normalization_42_23489208batch_normalization_42_23489210batch_normalization_42_23489212batch_normalization_42_23489214*
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
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23487435ý
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
K__inference_activation_42_layer_call_and_return_conditional_losses_23487885¢
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall&activation_42/PartitionedCall:output:0conv2d_55_23489218conv2d_55_23489220*
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
G__inference_conv2d_55_layer_call_and_return_conditional_losses_23487903
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0batch_normalization_43_23489223batch_normalization_43_23489225batch_normalization_43_23489227batch_normalization_43_23489229*
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
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23487499ý
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
K__inference_activation_43_layer_call_and_return_conditional_losses_23487923¢
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall&activation_43/PartitionedCall:output:0conv2d_56_23489233conv2d_56_23489235*
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
G__inference_conv2d_56_layer_call_and_return_conditional_losses_23487941
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_56/StatefulPartitionedCall:output:0batch_normalization_44_23489238batch_normalization_44_23489240batch_normalization_44_23489242batch_normalization_44_23489244*
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
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23487563
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
D__inference_add_18_layer_call_and_return_conditional_losses_23487962å
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
K__inference_activation_44_layer_call_and_return_conditional_losses_23487969¢
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCall&activation_44/PartitionedCall:output:0conv2d_57_23489249conv2d_57_23489251*
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
G__inference_conv2d_57_layer_call_and_return_conditional_losses_23487987
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0batch_normalization_45_23489254batch_normalization_45_23489256batch_normalization_45_23489258batch_normalization_45_23489260*
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
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23487627ý
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
K__inference_activation_45_layer_call_and_return_conditional_losses_23488007¢
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall&activation_45/PartitionedCall:output:0conv2d_58_23489264conv2d_58_23489266*
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
G__inference_conv2d_58_layer_call_and_return_conditional_losses_23488025¢
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall&activation_44/PartitionedCall:output:0conv2d_59_23489269conv2d_59_23489271*
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
G__inference_conv2d_59_layer_call_and_return_conditional_losses_23488047
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0batch_normalization_46_23489274batch_normalization_46_23489276batch_normalization_46_23489278batch_normalization_46_23489280*
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
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23487691
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
D__inference_add_19_layer_call_and_return_conditional_losses_23488068å
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
K__inference_activation_46_layer_call_and_return_conditional_losses_23488075¢
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall&activation_46/PartitionedCall:output:0conv2d_60_23489285conv2d_60_23489287*
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
G__inference_conv2d_60_layer_call_and_return_conditional_losses_23488093
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0batch_normalization_47_23489290batch_normalization_47_23489292batch_normalization_47_23489294batch_normalization_47_23489296*
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
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23487755ý
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
K__inference_activation_47_layer_call_and_return_conditional_losses_23488113¢
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall&activation_47/PartitionedCall:output:0conv2d_61_23489300conv2d_61_23489302*
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
G__inference_conv2d_61_layer_call_and_return_conditional_losses_23488131¢
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall&activation_46/PartitionedCall:output:0conv2d_62_23489305conv2d_62_23489307*
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
G__inference_conv2d_62_layer_call_and_return_conditional_losses_23488153
.batch_normalization_48/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0batch_normalization_48_23489310batch_normalization_48_23489312batch_normalization_48_23489314batch_normalization_48_23489316*
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
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23487819
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
D__inference_add_20_layer_call_and_return_conditional_losses_23488174å
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
K__inference_activation_48_layer_call_and_return_conditional_losses_23488181ø
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
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_23487839â
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
G__inference_flatten_6_layer_call_and_return_conditional_losses_23488190
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_6_23489323dense_6_23489325*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_23488202
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_54_23489203*&
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
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_55_23489218*&
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
2conv2d_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_56_23489233*&
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
2conv2d_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_57_23489249*&
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
2conv2d_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_58_23489264*&
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
2conv2d_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_59_23489269*&
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
2conv2d_60/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_60_23489285*&
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
2conv2d_61/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_61_23489300*&
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
2conv2d_62/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_62_23489305*&
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
9__inference_batch_normalization_48_layer_call_fn_23490961

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
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23487819
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
K__inference_activation_45_layer_call_and_return_conditional_losses_23488007

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
Ý
Ã
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23487819

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
	
Ô
9__inference_batch_normalization_45_layer_call_fn_23490565

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
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23487596
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
ð
¡
,__inference_conv2d_60_layer_call_fn_23490785

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
G__inference_conv2d_60_layer_call_and_return_conditional_losses_23488093w
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
	
Ô
9__inference_batch_normalization_43_layer_call_fn_23490347

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
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23487468
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
K__inference_activation_47_layer_call_and_return_conditional_losses_23488113

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
D__inference_add_18_layer_call_and_return_conditional_losses_23490511
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
â
µ
G__inference_conv2d_59_layer_call_and_return_conditional_losses_23490686

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
 
_user_specified_nameinputs"ÛL
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
*__inference_model_6_layer_call_fn_23488362
*__inference_model_6_layer_call_fn_23489538
*__inference_model_6_layer_call_fn_23489639
*__inference_model_6_layer_call_fn_23489017À
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
E__inference_model_6_layer_call_and_return_conditional_losses_23489868
E__inference_model_6_layer_call_and_return_conditional_losses_23490097
E__inference_model_6_layer_call_and_return_conditional_losses_23489200
E__inference_model_6_layer_call_and_return_conditional_losses_23489383À
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
#__inference__wrapped_model_23487382input_7"
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
,__inference_conv2d_54_layer_call_fn_23490215¢
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
G__inference_conv2d_54_layer_call_and_return_conditional_losses_23490231¢
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
9__inference_batch_normalization_42_layer_call_fn_23490244
9__inference_batch_normalization_42_layer_call_fn_23490257´
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
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23490275
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_23490293´
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
0__inference_activation_42_layer_call_fn_23490298¢
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
K__inference_activation_42_layer_call_and_return_conditional_losses_23490303¢
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
,__inference_conv2d_55_layer_call_fn_23490318¢
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
G__inference_conv2d_55_layer_call_and_return_conditional_losses_23490334¢
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
9__inference_batch_normalization_43_layer_call_fn_23490347
9__inference_batch_normalization_43_layer_call_fn_23490360´
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
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23490378
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23490396´
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
0__inference_activation_43_layer_call_fn_23490401¢
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
K__inference_activation_43_layer_call_and_return_conditional_losses_23490406¢
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
,__inference_conv2d_56_layer_call_fn_23490421¢
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
G__inference_conv2d_56_layer_call_and_return_conditional_losses_23490437¢
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
9__inference_batch_normalization_44_layer_call_fn_23490450
9__inference_batch_normalization_44_layer_call_fn_23490463´
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
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23490481
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23490499´
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
)__inference_add_18_layer_call_fn_23490505¢
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
D__inference_add_18_layer_call_and_return_conditional_losses_23490511¢
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
0__inference_activation_44_layer_call_fn_23490516¢
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
K__inference_activation_44_layer_call_and_return_conditional_losses_23490521¢
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
,__inference_conv2d_57_layer_call_fn_23490536¢
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
G__inference_conv2d_57_layer_call_and_return_conditional_losses_23490552¢
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
9__inference_batch_normalization_45_layer_call_fn_23490565
9__inference_batch_normalization_45_layer_call_fn_23490578´
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
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23490596
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23490614´
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
0__inference_activation_45_layer_call_fn_23490619¢
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
K__inference_activation_45_layer_call_and_return_conditional_losses_23490624¢
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
,__inference_conv2d_58_layer_call_fn_23490639¢
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
G__inference_conv2d_58_layer_call_and_return_conditional_losses_23490655¢
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
,__inference_conv2d_59_layer_call_fn_23490670¢
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
G__inference_conv2d_59_layer_call_and_return_conditional_losses_23490686¢
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
9__inference_batch_normalization_46_layer_call_fn_23490699
9__inference_batch_normalization_46_layer_call_fn_23490712´
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
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23490730
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23490748´
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
)__inference_add_19_layer_call_fn_23490754¢
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
D__inference_add_19_layer_call_and_return_conditional_losses_23490760¢
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
0__inference_activation_46_layer_call_fn_23490765¢
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
K__inference_activation_46_layer_call_and_return_conditional_losses_23490770¢
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
,__inference_conv2d_60_layer_call_fn_23490785¢
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
G__inference_conv2d_60_layer_call_and_return_conditional_losses_23490801¢
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
9__inference_batch_normalization_47_layer_call_fn_23490814
9__inference_batch_normalization_47_layer_call_fn_23490827´
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
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23490845
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23490863´
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
0__inference_activation_47_layer_call_fn_23490868¢
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
K__inference_activation_47_layer_call_and_return_conditional_losses_23490873¢
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
,__inference_conv2d_61_layer_call_fn_23490888¢
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
G__inference_conv2d_61_layer_call_and_return_conditional_losses_23490904¢
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
,__inference_conv2d_62_layer_call_fn_23490919¢
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
G__inference_conv2d_62_layer_call_and_return_conditional_losses_23490935¢
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
9__inference_batch_normalization_48_layer_call_fn_23490948
9__inference_batch_normalization_48_layer_call_fn_23490961´
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
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23490979
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23490997´
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
)__inference_add_20_layer_call_fn_23491003¢
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
D__inference_add_20_layer_call_and_return_conditional_losses_23491009¢
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
0__inference_activation_48_layer_call_fn_23491014¢
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
K__inference_activation_48_layer_call_and_return_conditional_losses_23491019¢
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
6__inference_average_pooling2d_6_layer_call_fn_23491024¢
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
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_23491029¢
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
,__inference_flatten_6_layer_call_fn_23491034¢
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
G__inference_flatten_6_layer_call_and_return_conditional_losses_23491040¢
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
*__inference_dense_6_layer_call_fn_23491049¢
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
E__inference_dense_6_layer_call_and_return_conditional_losses_23491059¢
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
__inference_loss_fn_0_23491070
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
__inference_loss_fn_1_23491081
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
__inference_loss_fn_2_23491092
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
__inference_loss_fn_3_23491103
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
__inference_loss_fn_4_23491114
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
__inference_loss_fn_5_23491125
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
__inference_loss_fn_6_23491136
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
__inference_loss_fn_7_23491147
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
__inference_loss_fn_8_23491158
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
&__inference_signature_wrapper_23490200input_7"
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
#__inference__wrapped_model_23487382»L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå8¢5
.¢+
)&
input_7ÿÿÿÿÿÿÿÿÿ  
ª "1ª.
,
dense_6!
dense_6ÿÿÿÿÿÿÿÿÿ
·
K__inference_activation_42_layer_call_and_return_conditional_losses_23490303h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
0__inference_activation_42_layer_call_fn_23490298[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ·
K__inference_activation_43_layer_call_and_return_conditional_losses_23490406h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
0__inference_activation_43_layer_call_fn_23490401[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ·
K__inference_activation_44_layer_call_and_return_conditional_losses_23490521h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
0__inference_activation_44_layer_call_fn_23490516[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ·
K__inference_activation_45_layer_call_and_return_conditional_losses_23490624h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
0__inference_activation_45_layer_call_fn_23490619[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ ·
K__inference_activation_46_layer_call_and_return_conditional_losses_23490770h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
0__inference_activation_46_layer_call_fn_23490765[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ ·
K__inference_activation_47_layer_call_and_return_conditional_losses_23490873h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
0__inference_activation_47_layer_call_fn_23490868[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@·
K__inference_activation_48_layer_call_and_return_conditional_losses_23491019h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
0__inference_activation_48_layer_call_fn_23491014[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@ä
D__inference_add_18_layer_call_and_return_conditional_losses_23490511j¢g
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
)__inference_add_18_layer_call_fn_23490505j¢g
`¢]
[X
*'
inputs/0ÿÿÿÿÿÿÿÿÿ  
*'
inputs/1ÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ä
D__inference_add_19_layer_call_and_return_conditional_losses_23490760j¢g
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
)__inference_add_19_layer_call_fn_23490754j¢g
`¢]
[X
*'
inputs/0ÿÿÿÿÿÿÿÿÿ 
*'
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ ä
D__inference_add_20_layer_call_and_return_conditional_losses_23491009j¢g
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
)__inference_add_20_layer_call_fn_23491003j¢g
`¢]
[X
*'
inputs/0ÿÿÿÿÿÿÿÿÿ@
*'
inputs/1ÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@ô
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_23491029R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ì
6__inference_average_pooling2d_6_layer_call_fn_23491024R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_234902750123M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ï
T__inference_batch_normalization_42_layer_call_and_return_conditional_losses_234902930123M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
9__inference_batch_normalization_42_layer_call_fn_234902440123M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÇ
9__inference_batch_normalization_42_layer_call_fn_234902570123M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23490378IJKLM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ï
T__inference_batch_normalization_43_layer_call_and_return_conditional_losses_23490396IJKLM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
9__inference_batch_normalization_43_layer_call_fn_23490347IJKLM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÇ
9__inference_batch_normalization_43_layer_call_fn_23490360IJKLM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23490481bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ï
T__inference_batch_normalization_44_layer_call_and_return_conditional_losses_23490499bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
9__inference_batch_normalization_44_layer_call_fn_23490450bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÇ
9__inference_batch_normalization_44_layer_call_fn_23490463bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿó
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23490596M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ó
T__inference_batch_normalization_45_layer_call_and_return_conditional_losses_23490614M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ë
9__inference_batch_normalization_45_layer_call_fn_23490565M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ë
9__inference_batch_normalization_45_layer_call_fn_23490578M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ó
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23490730¢£¤¥M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ó
T__inference_batch_normalization_46_layer_call_and_return_conditional_losses_23490748¢£¤¥M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ë
9__inference_batch_normalization_46_layer_call_fn_23490699¢£¤¥M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ë
9__inference_batch_normalization_46_layer_call_fn_23490712¢£¤¥M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ó
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23490845ÁÂÃÄM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ó
T__inference_batch_normalization_47_layer_call_and_return_conditional_losses_23490863ÁÂÃÄM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ë
9__inference_batch_normalization_47_layer_call_fn_23490814ÁÂÃÄM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ë
9__inference_batch_normalization_47_layer_call_fn_23490827ÁÂÃÄM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ó
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23490979âãäåM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ó
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_23490997âãäåM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ë
9__inference_batch_normalization_48_layer_call_fn_23490948âãäåM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ë
9__inference_batch_normalization_48_layer_call_fn_23490961âãäåM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@·
G__inference_conv2d_54_layer_call_and_return_conditional_losses_23490231l'(7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
,__inference_conv2d_54_layer_call_fn_23490215_'(7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ·
G__inference_conv2d_55_layer_call_and_return_conditional_losses_23490334l@A7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
,__inference_conv2d_55_layer_call_fn_23490318_@A7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ·
G__inference_conv2d_56_layer_call_and_return_conditional_losses_23490437lYZ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  
 
,__inference_conv2d_56_layer_call_fn_23490421_YZ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ  ·
G__inference_conv2d_57_layer_call_and_return_conditional_losses_23490552lxy7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_conv2d_57_layer_call_fn_23490536_xy7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ ¹
G__inference_conv2d_58_layer_call_and_return_conditional_losses_23490655n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_conv2d_58_layer_call_fn_23490639a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ ¹
G__inference_conv2d_59_layer_call_and_return_conditional_losses_23490686n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_conv2d_59_layer_call_fn_23490670a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ ¹
G__inference_conv2d_60_layer_call_and_return_conditional_losses_23490801n¸¹7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_conv2d_60_layer_call_fn_23490785a¸¹7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ@¹
G__inference_conv2d_61_layer_call_and_return_conditional_losses_23490904nÑÒ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_conv2d_61_layer_call_fn_23490888aÑÒ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@¹
G__inference_conv2d_62_layer_call_and_return_conditional_losses_23490935nÙÚ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_conv2d_62_layer_call_fn_23490919aÙÚ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ@§
E__inference_dense_6_layer_call_and_return_conditional_losses_23491059^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
*__inference_dense_6_layer_call_fn_23491049Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ
«
G__inference_flatten_6_layer_call_and_return_conditional_losses_23491040`7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_flatten_6_layer_call_fn_23491034S7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@=
__inference_loss_fn_0_23491070'¢

¢ 
ª " =
__inference_loss_fn_1_23491081@¢

¢ 
ª " =
__inference_loss_fn_2_23491092Y¢

¢ 
ª " =
__inference_loss_fn_3_23491103x¢

¢ 
ª " >
__inference_loss_fn_4_23491114¢

¢ 
ª " >
__inference_loss_fn_5_23491125¢

¢ 
ª " >
__inference_loss_fn_6_23491136¸¢

¢ 
ª " >
__inference_loss_fn_7_23491147Ñ¢

¢ 
ª " >
__inference_loss_fn_8_23491158Ù¢

¢ 
ª " 
E__inference_model_6_layer_call_and_return_conditional_losses_23489200·L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå@¢=
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
E__inference_model_6_layer_call_and_return_conditional_losses_23489383·L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå@¢=
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
E__inference_model_6_layer_call_and_return_conditional_losses_23489868¶L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå?¢<
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
E__inference_model_6_layer_call_and_return_conditional_losses_23490097¶L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå?¢<
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
*__inference_model_6_layer_call_fn_23488362ªL'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå@¢=
6¢3
)&
input_7ÿÿÿÿÿÿÿÿÿ  
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
Ù
*__inference_model_6_layer_call_fn_23489017ªL'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå@¢=
6¢3
)&
input_7ÿÿÿÿÿÿÿÿÿ  
p

 
ª "ÿÿÿÿÿÿÿÿÿ
Ø
*__inference_model_6_layer_call_fn_23489538©L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
Ø
*__inference_model_6_layer_call_fn_23489639©L'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäå?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "ÿÿÿÿÿÿÿÿÿ
ñ
&__inference_signature_wrapper_23490200ÆL'(0123@AIJKLYZbcdexy¢£¤¥¸¹ÁÂÃÄÑÒÙÚâãäåC¢@
¢ 
9ª6
4
input_7)&
input_7ÿÿÿÿÿÿÿÿÿ  "1ª.
,
dense_6!
dense_6ÿÿÿÿÿÿÿÿÿ
