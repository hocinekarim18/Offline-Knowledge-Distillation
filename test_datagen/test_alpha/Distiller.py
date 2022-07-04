import tensorflow as tf


# Adaptive Instance Normalization Knowledge distillation
class Distiller_AdaIn(tf.keras.Model):
  def __init__(self, teacher, student):
      super(Distiller_AdaIn, self).__init__()
      
      # Attributs de la classe Distiller
      self.teacher = teacher
      self.student = student
      
  # Compilation du model
  def compile( self, optimizer, metrics, student_loss_fn, alpha = 0.1, beta= 1):

    super(Distiller_AdaIn,self).compile(optimizer = optimizer, metrics= metrics )
    # losses
    self.student_loss_fn = student_loss_fn

    # Hyperparameters
    self.beta = beta
    self.alpha = alpha
  
  # Training Step

  def train_step(self, data):
    # Unpack data
    x, y = data

    # Forward pass of teacher
    teacher_predictions = self.teacher(x, training=False)
    batch_size = tf.cast(tf.shape(teacher_predictions)[0], tf.float32)

    with tf.GradientTape() as tape:
      # student forward
      student_predictions = self.student(x, training= True)

      # Compute losses
      Lce = self.student_loss_fn(y, student_predictions)

      # Compute student stats
      mean_student = tf.reduce_mean(student_predictions, axis = 1)
      std_student = tf.math.reduce_std(student_predictions, axis = 1)

      # Compute teacher stats
      mean_teacher = tf.reduce_mean(teacher_predictions, axis = 1)
      std_teacher =  tf.math.reduce_std(student_predictions, axis = 1)

      # Compute delta
      delta_mean = tf.pow(tf.math.subtract(mean_teacher, mean_student), 2)
      delta_std = tf.pow(tf.math.subtract(std_teacher, std_student), 2)

  
      # COmpute statistic loss
      Lsm =  (1.0/ batch_size)*tf.math.reduce_sum( tf.math.add( delta_mean , delta_std))


      # Compute similarity LOss
      sub = tf.math.subtract(teacher_predictions , tf.reduce_mean(mean_teacher))
      teacher_norm = tf.math.divide( sub , tf.reduce_mean(std_teacher) ) 
      F_t = teacher_norm * tf.reduce_mean(std_student) + tf.reduce_mean(mean_student) 

      L_AdaIn = tf.pow(tf.norm(tf.math.subtract(teacher_predictions , F_t)), 2)

      # Compute total loss
      loss = Lce + self.alpha* Lsm + self.beta* L_AdaIn   

    # Compute gradients
    trainable_vars = self.student.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    # Update the metrics configured in `compile()`.
    self.compiled_metrics.update_state(y, student_predictions)

    # Return a dict of performance
    results = {m.name: m.result() for m in self.metrics}
    results.update(
        {"student_loss": Lce, "KD_loss": loss}
    )
    return results

  # Test Step
  def test_step(self, data):
    
    # Unpack the data
    x, y = data

    # Compute predictions
    y_prediction = self.student(x, training=False)

    # Calculate the loss
    student_loss = self.student_loss_fn(y, y_prediction)

    # Update the metrics.
    self.compiled_metrics.update_state(y, y_prediction)

    # Return a dict of performance
    results = {m.name: m.result() for m in self.metrics}
    results.update({"student_loss": student_loss})
    return results
  
  
  
# Offline Standard Distillation
class Distiller(tf.keras.Model):
  def __init__(self, teacher, student):
      super(Distiller, self).__init__()
      
      # Attributs de la classe Distiller
      self.teacher = teacher
      self.student = student
      
  # Compilation du model
  def compile( self, optimizer, metrics, distillation_loss_fn, student_loss_fn, alpha = 0.1, temperature= 20):

    super(Distiller,self).compile(optimizer = optimizer, metrics= metrics )
    # losses
    self.distillation_loss_fn = distillation_loss_fn
    self.student_loss_fn = student_loss_fn

    # Hyperparameters
    self.temperature = temperature
    self.alpha = alpha
  
  # Training Step
  def train_step(self, data):
    # Unpack data
    x, y = data

    # Forward pass of teacher
    teacher_predictions = self.teacher(x, training=False)
    with tf.GradientTape() as tape:
      # student forward
      student_predictions = self.student(x, training= True)

      # Compute losses
      student_loss = self.student_loss_fn(y, tf.nn.softmax(student_predictions))
      distillation_loss = self.distillation_loss_fn(
          tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
          tf.nn.softmax(student_predictions / self.temperature, axis=1),
        )

      loss = self.alpha * student_loss + (1- self.alpha)* distillation_loss

    # Compute gradients
    trainable_vars = self.student.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    # Update the metrics configured in `compile()`.
    self.compiled_metrics.update_state(y, student_predictions)

    # Return a dict of performance
    results = {m.name: m.result() for m in self.metrics}
    results.update(
        {"student_loss": student_loss, "loss": loss}
    )
    return results

  # Test Step
  def test_step(self, data):
    
    # Unpack the data
    x, y = data

    # Compute predictions
    y_prediction = self.student(x, training=False)

    # Calculate the loss
    student_loss = self.student_loss_fn(y, y_prediction)

    # Update the metrics.
    self.compiled_metrics.update_state(y, y_prediction)

    # Return a dict of performance
    results = {m.name: m.result() for m in self.metrics}
    results.update({"student_loss": student_loss})
    return results
