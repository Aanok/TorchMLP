require("torch")

-- module
local mlp = {}

-- class
local MLP = {}


--- Generate randomized Tensor.
-- The function generates a Tensor, initializing its values with a uniform distribution.
-- @param size longStorage defining the Tensor's sizes.
-- @param low Lower end of the value distribution.
-- @param up Upper end of the value distribution.
-- @return The new Tensor.
local function rand_range(size, low, up)
  local t = torch.rand(size)
  t:mul(up-low)
  t:add(low)
  return t
end


local exp = math.exp --performance
--- Logistic sigmoid.
-- Please note exponentiation is accessed through a local to save on an indirection from the global table.
-- @param x Input scalar.
-- @return The image of x.
function sigmoid(x)
  return 1 / (1 + exp(-x))
end


--- Derivative of logistic sigmoid.
-- Please note exponentiation is accessed through a local to save on an indirection from the global table.
-- @param x Input scalar.
-- @return The image of x.
function sigmoid_der(x)
  return exp(-x) / (1 + exp(-x))^2
end


--- Squared error contribution.
-- Returns the squared 2-norm of x - y.
-- @param x Input Tensor.
-- @param y Input Tensor.
-- @return torch.norm(x - y)^2
function squared_error(x, y)
    return torch.norm(x - y)^2
end


--- Euclidean error contribution.
-- Returns the 2-norm of x - y.
-- @param x Input Tensor.
-- @param y Input Tensor.
-- @return torch.norm(x - y)
function euclidean_error(x, y)
  return torch.norm(x - y)
end


--- Multi-Layer Perceptron constructor.
-- @param input_size Dimension of input patterns. Mandatory.
-- @param output_size Dimension of output patterns. Mandatory.
-- @param options Table containing optional fields specifying parameters and hyperparameters. All values have defaults to fall back on. The table must be present but it may be empty.
-- <li>options.neurons: Number of hidden units. Default: 10.</li>
-- <li>options.init_range: Connection weights are initialized uniformly in [-init_range, init_range]. Default: 0.5.</li>
-- <li>options.learning_rate: Multiplicative coefficient for delta in gradient descent (eta). Default: 1.</li>
-- <li>options.momentum: Multiplicative coefficient for momentum contribution in update. Default: 0.</li>
-- <li>options.penalty: Multiplicative coefficient for L2 regularization term. Default: 0.</li>
-- <li>options.early_stop_threshold: TR error threshold (after update) that triggers early stop. Default: 0.001.</li>
-- <li>options.act_fun: Activation function of input-to-hidden layer (real -> real). Default: sigmoid.</li>
-- <li>options.act_fun_der: Derivative of act_fun (real -> real). Default: sigmoid_der.</li>
-- &emsp; Note: both act_fun and act_fun_der must be passed or neither will be assigned.
-- <li>out_fun: Activation function of hidden-to-output layer (real -> real). Default: sigmoid.</li>
-- <li>out_fun_der: Derivative of out_fun (real -> real)</li>
-- &emsp; Note: both out_fun and out_fun_der must be passed or neither will be assigned.
-- <li>postprocess: Postprocessing function to apply to the output (e.g. thresholding or a linear transformation) (Tensor -> Tensor). Default: identity.</li>
-- &emsp; Note: it is not used during training.</li>
-- <li>max_epochs: Epoch iteration high end cutoff. Default: 100.</li>
-- <li>error_metric: Error function, accumulated over all outputs during the epoch, averaged at the end of epoch for final aggregate result ((Tensor, Tensor) -> real). Default: squared_error.</li>
-- @return The new MLP instance.
function mlp.new(input_size, output_size, options)
  local self = {}
  setmetatable(self, { __index = MLP })
  
  local neurons = options.neurons or 10
  self.init_range = options.init_range or 0.5
  self.W_in = rand_range(torch.LongStorage({neurons, input_size +1}), -self.init_range,self.init_range)
  self.W_out = rand_range(torch.LongStorage({output_size, neurons +1}), -self.init_range,self.init_range)
  self.learning_rate = options.learning_rate or 1
  self.momentum = options.momentum or 0
  self.penalty = options.penalty or 0
  self.early_stop_threshold = options.early_stop_threshold or 0.001
  if options.act_fun and options.act_fun_der then
    self.act_fun, self.act_fun_der = options.act_fun, options.act_fun_der
  else
    self.act_fun, self.act_fun_der = sigmoid, sigmoid_der
  end
  if options.out_fun and options.out_fun_der then
    self.out_fun, self.out_fun_der = options.out_fun, options.out_fun_der
  else
    self.out_fun, self.out_fun_der = sigmoid, sigmoid_der
  end
  self.postprocess = options.postprocess or function(x) return x end
  self.max_epochs = options.max_epochs or 100
  self.error_metric = options.error_metric or squared_error
  
  return self
end


--- Compute mean error and accuracy over labelled input set.
-- Iterate over (pattern,label) pairs in the set, accruing an error according to self.error_metric(pattern,label)
-- and an accuracy measure as per equality between postprocessed outputs and targets.
-- The error is then averaged, the accuracy normalized to [0,1].
-- @param set Labelled input set, that is:
-- &emsp; set[1]: array of input Tensors.
-- &emsp; set[2]: array of target Tensors, aligned with input.
-- @return Error, Accuracy
function MLP:mean_error(set)
  -- accumulate over patterns
  local error_acc = 0
  local well_classified = 0
  for i,p in ipairs(set[1]) do
    local raw_output = self:sim_raw(p)
    error_acc = error_acc + self.error_metric(raw_output, set[2][i])
    if torch.equal(self.postprocess(raw_output), set[2][i]) then well_classified = well_classified + 1 end
  end
  -- then average/reduce
  return error_acc/#set[1], well_classified/#set[1]
end


--- Compute Gradient Descent deltas as per backpropagation.
-- Single (pattern,label) pair.
-- @param input Input Tensor.
-- @param target Target Tensor.
-- @return hidden-to-output delta, input-to-hidden delta
function MLP:gd_deltas(input, target)
  local out_out, field_out, out_in, field_in = self:sim_raw(input)
  
  -- hidden-to-output weight matrix
  -- schematically:
  -- delta_w = lr * (target - out_out) * out_fun_der(field_out) * out_in
  
  local e_out = torch.add(target, -1, out_out) -- e_out = target - out_out
  local field_der_out = field_out:apply(self.out_fun_der) -- field_der_out = out_fun_der(field_out)
  -- note: apply works in place. i'm adding an alias for readability
  local delta_out = torch.cmul(e_out, field_der_out) -- delta_out = e_out .* field_der_out
  local W_out_delta = torch.ger(delta_out, out_in) -- W_out_delta = delta_out [outer] out_in (lacks lr)
  
  -- input-to-hidden weight matrix
  -- schematically:
  -- delta_w = lr * actFunDer(field_in) .* ((delta_out)T * W_out) * input
  
  -- NB torch vectors lack orientation. row * matrix operations are run as matrix:t() * vector
  -- in cmul, the result will always have the same shape as the first argument
  local field_der_in = field_in:apply(self.act_fun_der) -- field_der_in = actFunDer(field_in)
  local e_in = torch.mv(self.W_out:t(), delta_out):sub(1,-2) -- e_in = delta_out:t() * W_out
  -- NB :t() has no side effects, it just returns a "view"
  -- also we remove the last element of e_in, corresponding to the bias from the output layer
  local delta_in =  torch.cmul(field_der_in, e_in) -- delta_in = field_der_in .* e_in
  local W_in_delta = torch.ger(delta_in, torch.cat(input, torch.ones(1))) -- W_in_delta = delta_in [outer] input (lacks lr)
  
  return W_out_delta, W_in_delta
end


--- Apply updates to self's weight matrices.
-- Applies GD, momentum and L2 regularization as per configuration.
-- @param member_str Must be one of "W_in" or "W_out" to apply the updates to the respective matrices. Undefined behavior otherwise.
-- @param W_delta Plain GD delta for current epoch.
-- @param W_delta_old GD delta from previous epoch, for momentum.
function MLP:update_step(member_str, W_delta, W_delta_old)
  local W = self[member_str]
  -- apply L2 penalty term
  -- take care to ignore the last column: it's the bias
  W:sub(1,-1, 1,-2):add(-self.penalty, W:sub(1,-1, 1,-2)) -- W = W - p * W
  -- apply delta
  W:add(self.learning_rate, W_delta) -- W = W + lr * W_delta
  -- apply momentum
  W:add(self.momentum, W_delta_old)
end


--- Check conditions for early stop.
-- Conditions: TR error < self.early_stop_threshold or TR error increasing or VL error increasing or VL accuracy 100%.
-- @param epoch Counter for the current epoch.
-- @param traces Table of error traces, such that:
-- <li>traces.training.error_trace: Array-like table with error measurements over TR set for past epochs. Mandatory.</li>
-- <li>traces.validation: Table with error and accuracy measurements for the VL set. Optional. If present, the following must be defined and valid:
-- <ul><li>traces.validation.error_trace: Array-like table with error measurements over VL set for past epochs.</li>
-- <li>traces.validation.accuracy_trace: Array-like table with accuracy measurements over VL set for past epochs.</li></ul></li>
-- @return true/false
function MLP:early_stop(epoch, traces)
  -- aliases help readability and performance
  local tr = traces.training
  local val = traces.validation
  
  -- tr_error < threshold
  -- moving average over last 10 epochs
  tr.new_error = tr.new_error + tr.error_trace[epoch]/10
  if epoch > 10 then
    tr.new_error = tr.new_error - tr.error_trace[epoch - 10]/10
    tr.old_error = tr.old_error + tr.error_trace[epoch - 10]/10
    if tr.new_error < self.early_stop_threshold then return true end
  end
  -- tr_error incresing
  -- moving averages over last 20 epochs, 10 and 10
  if epoch > 20 then
    tr.old_error = tr.old_error - tr.error_trace[epoch - 20]/10
    if tr.new_error > tr.old_error then return true end
  end
  
  -- validation error is increasing
  -- moving averages over the last 20 epochs, 10 and 10
  if val then
    val.new_error = val.new_error + val.error_trace[epoch]
    if epoch > 10 then
      val.new_error = val.new_error - val.error_trace[epoch - 10]
      val.old_error = val.old_error + val.error_trace[epoch - 10]
    end
    if epoch > 20 then
      val.old_error = val.old_error - val.error_trace[epoch - 20]
      if val.new_error > val.old_error then return true end
    end

  end
  
  -- validation accuracy is 100%
  if val and val.accuracy_trace[epoch] == 1 then return true end
  return false
end


--- Train self to convergence over a labelled training set.
-- Runs main GD with momentum and L2 regularization.
-- @param training_set Labelled TR set, that is:
-- &emsp; training_set[1]: array-like table of input Tensors.
-- &emsp; training_set[2]: array-like table of target Tensors, aligned with input.
-- @param validation_set Labelled VL set, formatted like training_set. Optional.
-- @return Table with error traces for TR and, if present, VL, that is:
-- <li>traces.training.error_trace: Tensor of error measures computed at each epoch.</li>
-- <li>traces.training.accuracy_trace: Tensor of accuracy measures computed at each epoch.</li>
-- <li>traces.validation: Likewise.</li>
function MLP:train_once(training_set, validation_set)
  -- conveniency aliases (also useful for perfomance)
  local tr_input = training_set[1]
  local tr_targets = training_set[2]
  local val_input, val_targets = nil,nil
  
  -- trace of error metric at each epoch
  local traces = { training = { error_trace = {}, accuracy_trace = {}, new_error = 0, old_error = 0 }}
  
  -- validation related values are defined only if appropriate
  if validation_set then
    traces.validation = { error_trace = {}, accuracy_trace = {}, new_error = 0, old_error = 0 }
    val_input = validation_set[1]
    val_targets = validation_set[2]
  end
  
  -- old deltas for momentum
  local W_out_delta_old = torch.zeros(self.W_out:size())
  local W_in_delta_old = torch.zeros(self.W_in:size())
  
  -- epoch iteration loop
  for epoch = 1,self.max_epochs do
    local access = torch.randperm(#tr_input)
    local W_out_delta_acc = torch.zeros(self.W_out:size())
    local W_in_delta_acc = torch.zeros(self.W_in:size())

    --accumulate deltas
    for i = 1,#tr_input do
      local W_out_delta, W_in_delta = self:gd_deltas(tr_input[access[i]], tr_targets[access[i]])      
      W_out_delta_acc:add(W_out_delta)
      W_in_delta_acc:add(W_in_delta)
    end
    
    -- average deltas
    W_out_delta_acc:div(#tr_input)
    W_in_delta_acc:div(#tr_input)
    -- apply deltas
    self:update_step("W_out", W_out_delta_acc, W_out_delta_old)
    self:update_step("W_in", W_in_delta_acc, W_in_delta_old)
    -- update momentum deltas
    W_out_delta_old = W_out_delta_acc
    W_in_delta_old = W_in_delta_acc
    
    -- compute errors for the epoch
    traces.training.error_trace[#traces.training.error_trace + 1], traces.training.accuracy_trace[#traces.training.accuracy_trace + 1] = self:mean_error(training_set)
    if validation_set then
      traces.validation.error_trace[#traces.validation.error_trace + 1], traces.validation.accuracy_trace[#traces.validation.accuracy_trace + 1] = self:mean_error(validation_set)
    end
    
    -- check for early stop
    if self:early_stop(epoch, traces) then break end
  end
  
  -- format output for gnuplot
  for _,set in pairs(traces) do
    for trace_name,trace in pairs(set) do
      set[trace_name] = torch.Tensor(trace)
    end
  end
  
  return traces
end


--- Reinitialize self's weight matrices to new random values.
-- Matrices keep their size. New values uniformly distributed in [-self.init_range,self.init_range].
function MLP:randomize_weights()
  self.W_in = rand_range(self.W_in:size(), -self.init_range,self.init_range)
  self.W_out = rand_range(self.W_out:size(), -self.init_range,self.init_range)
end


--- Train self five times with random starts, take best trial.
-- Calls self:train_once five times. Chooses model with least final VL error if VL provided, otherwise least final TR error.
-- @param training_set Labelled TR set, that is:
-- &emsp; training_set[1]: array-like table of input Tensors.
-- &emsp; training_set[2]: array-like table of target Tensors, aligned with input.
-- @param validation_set Labelled VL set, formatted like training_set. Optional.
-- @return Table with error traces for TR and, if present, VL, that is:
-- <li>traces.training.error_trace: Tensor of error measures computed at each epoch.</li>
-- <li>traces.training.accuracy_trace: Tensor of accuracy measures computed at each epoch.</li>
-- <li>traces.validation: Likewise.</li>
function MLP:train(training_set, validation_set)
  local run_length = 5
  local W_in_best, W_out_best
  local traces_best = { training = { error_trace = { math.huge } } }
  if validation_set then traces_best.validation = { error_trace = { math.huge } } end
  for i = 1,run_length do
    self:randomize_weights()
    local traces = self:train_once(training_set, validation_set)
    if validation_set then
      -- ignore TR performance if VL available
      if traces.validation.error_trace[#traces.validation.error_trace] < traces_best.validation.error_trace[#traces_best.validation.error_trace] then
        traces_best = traces
        W_in_best = self.W_in
        W_out_best = self.W_out
      end
    elseif traces.training.error_trace[#traces.training.error_trace] < traces_best.training.error_trace[#traces_best.training.error_trace] then
      traces_best = traces
      W_in_best = self.W_in
      W_out_best = self.W_out
    end
  end
  self.W_in = W_in_best
  self.W_out = W_out_best
  return traces_best
end


--- Extends a vector to length len.
-- New elements will be copies of the last element of x.
-- REQUIRES: len >= x:size(1)
-- @param x Torch one-dimensional Tensor.
-- @param len Integer length of new vector.
-- @return The new extended vector.
local function trail_vector_to(x, len)
  return x:cat(torch.Tensor(len - x:size(1)):fill(x[x:size(1)]))
end


--- Adds error and accuracy traces from addendum to accumulator.
-- Private conveniency function.
-- @param accumulator Formatted like a standard trace table.
-- @param addendum Formatted like a standard trace table.
local function join_traces(accumulator, addendum)
  for k,acc_err in pairs(accumulator) do
    acc_err.error_trace:add(trail_vector_to(addendum[k].error_trace, acc_err.error_trace:size(1)))
    acc_err.accuracy_trace:add(trail_vector_to(addendum[k].accuracy_trace, acc_err.accuracy_trace:size(1)))
  end
end


--- Run K-fold Cross Validation.
-- Defines a random partition of set over k regions; runs self:train using each of the regions in turn as VL, the rest TR.
-- Leaves self trained over the last fold. Computes mean and standard deviation of final VL error.
-- @param	set Tabel of labelled patterns, that is:
-- &emsp; set[1] Array-like table of input Tensors.
-- &emsp; set[2] Array-like table of target Tensors, aligned with input
-- @param k number of folds
-- @return Table with error traces for the CV averaged over the folds, that is:
-- <li>traces.training.error_trace: Tensor of error measures computed at each epoch.</li>
-- <li>traces.training.accuracy_trace: Tensor of accuracy measures computed at each epoch.</li>
-- <li>traces.validation: Likewise.</li>
-- <li>traces.validation.error_mean: final mean error for VS.</li>
-- <li>traces.validation.error_sd: standard deviation of final mean error for VS.</li>
function MLP:k_fold_cross_validate(set, k)
  local fold_size = math.ceil(#set[1]/k) -- rounding up means favouring VL
  local access = torch.randperm(#set[1]) -- to shuffle the pattern set
  local val_errors = torch.Tensor(k) -- store values to compute standard deviation
  local longest_run = 0 -- highes epoch cutoff; we'll trim the results up to this
  local mean_traces = {
    training = {
      error_trace = torch.zeros(self.max_epochs),
      accuracy_trace = torch.zeros(self.max_epochs)
    },
    validation = {
      error_trace = torch.zeros(self.max_epochs),
      accuracy_trace = torch.zeros(self.max_epochs)
    }
  }
  -- iterate folds
  for i = 1,k do
    local tr_fold = {{},{}}
    local val_fold = {{},{}}
    -- iterate over set and partition it into TR,VL
    for j = 1,#set[1] do
      if (i-1)*fold_size < j and j <= i*fold_size then
        val_fold[1][#val_fold[1] +1] = set[1][access[j]]
        val_fold[2][#val_fold[2] +1] = set[2][access[j]]
      else
        tr_fold[1][#tr_fold[1] +1] = set[1][access[j]]
        tr_fold[2][#tr_fold[2] +1] = set[2][access[j]]
      end
    end
    -- retrain from new random start
    self:randomize_weights()
    local fold_traces = self:train(tr_fold, val_fold)
    -- bookkeeping for statistical analysis
    longest_run = math.max(longest_run, fold_traces.training.error_trace:size(1))
    val_errors[k] = fold_traces.validation.error_trace[#fold_traces.validation.error_trace]
    join_traces(mean_traces, fold_traces)
  end
  -- finalize measurements
  for tr_vs_name,tr_vs in pairs(mean_traces) do
    for err_acc_name,err_acc in pairs(tr_vs) do
      -- averages
      mean_traces[tr_vs_name][err_acc_name] = err_acc:sub(1, longest_run):div(k)
    end
  end
  local val_error_sd = 0
  local val_error_mean = mean_traces.validation.error_trace[#mean_traces.validation.error_trace]
  val_errors:apply(function(x) val_error_sd = val_error_sd + (x - val_error_mean)^2 end)
  val_error_sd = math.sqrt(val_error_sd) / k
  mean_traces.validation.error_mean = val_error_mean
  mean_traces.validation.error_sd = val_error_sd
  return mean_traces
end


--- Propagate signal forwards to output. Don't postprocess.
-- @param input Input Tensor.
-- @return hidden-to-output output, hidden-to-output activation, input-to-hidden output, input-to-hidden activation
function MLP:sim_raw(input)
  -- out_fun(W_out * act_fun(W_in * input))
  
  -- add fake firing neuron for bias
  local field_in = torch.mv(self.W_in, torch.cat(input, torch.ones(1)))
  -- add fake firing neuron for bias
  local out_in = torch.cat(field_in:clone():apply(self.act_fun), torch.ones(1))
  local field_out = torch.mv(self.W_out, out_in)
  local out_out = field_out:clone():apply(self.out_fun)
  
  return out_out, field_out, out_in, field_in
end


--- Propagate signal forwards to output. Postprocess in full.
-- @param input Input Tensor.
-- @return postprocessed hidden-to-output output, hidden-to-output activation, input-to-hidden output, input-to-hidden activation
function MLP:sim(input)
  -- postprocess(out_fun(W_out * act_fun(W_in * input)))
  local out_out, field_out, out_in, field_in = self:sim_raw(input)
  self.postprocess(out_out)
	return out_out, field_out, out_in, field_in
end


return mlp
