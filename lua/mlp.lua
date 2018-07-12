-- MLP class
require("torch")

-- @module mlp
local mlp = {}

-- class table
local MLP = {}


-- utility function to generate a 2D Tensor with uniform random values in the specified interval
local function rand_range(size, low, up)
  local t = torch.rand(size)
  t:mul(up-low)
  t:add(low)
  return t
end

-- activation function: logistic function
function sigmoid(x)
  return 1 / (1 + math.exp(-x))
end

-- derivative of activation function above
local exp = math.exp --performance
function sigmoid_der(x)
  return exp(-x) / (1 + exp(-x))^2
end


-- squared error for MSE purposes
-- arguments are torch Tensors
function squared_error(x, y)
    return torch.norm(x - y)^2
end


-- euclidean error for MEE purposes
-- arguments are torch Tensors
function euclidean_error(x, y)
  return torch.norm(x - y)
end


-- constructor
function mlp.new(input_size, output_size, options)
--[[ ARGS:
  input_size: dimension of input patterns -- mandatory
  output_size: dimension count of output patterns -- mandatory
  options: table containing optional fields specifying parameters and hyperparameters
  all values have defaults to fall back on
    neurons: # of neurons
    init_range: connection weights are initialized uniformly in [-init_range, init_range]
    learning_rate: gradient coefficient for descent
    momentum: momentum coefficient
    penalty: L2 regularization term
    early_stop_threshold: TR error threshold (at epoch) that triggers early stop
    act_fun: activation function of hidden layer (real -> real)
    act_fun_der: derivative of act_fun (real -> real)
      both act_fun and act_fun_der must be passed or neither will be assigned
    out_fun: activation function of output layer (real -> real)
    out_fun_der: derivative of out_fun (real -> real)
      both out_fun and out_fun_der must be passed or neither will be assigned
    postprocess: postprocessing function to apply to the output (e.g. thresholding or a linear transformation) (real -> real)
      -- NB it is not used during training, of course
    max_epochs: epoch iteration high end cutoff
    error_metric: error function, accumulated over all outputs during the epoch, averaged at the end of epoch for final aggregate result ((Tensor, Tensor) -> real)
]]--
  local self = {}
  setmetatable(self, { __index = MLP })
  
  local neurons = options.neurons or 10
  self.init_range = options.init_range or 0.5
  self.W_in = rand_range(torch.LongStorage({neurons, input_size +1}), -self.init_range,self.init_range)
  self.W_out = rand_range(torch.LongStorage({output_size, neurons +1}), -self.init_range,self.init_range)
  self.learning_rate = options.learning_rate or 0.1
  self.momentum = options.momentum or 0.01
  self.penalty = options.penalty or 0.01
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


function MLP:mean_error(set)
  -- accumulate over patterns
  local error_acc = 0
  local well_classified = 0
  for i,p in ipairs(set[1]) do
    local raw_output = self:sim_raw(p)
    error_acc = error_acc + self.error_metric(raw_output, set[2][i])
    if torch.equal(raw_output:apply(self.postprocess), set[2][i]) then well_classified = well_classified + 1 end
  end
  -- then average/reduce
  return error_acc/#set[1], well_classified/#set[1]
end


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


function MLP:early_stop(epoch, traces)
  -- aliases help readability and performance
  local tr = traces.training
  local val = traces.validation
  
  -- tr_error < threshold
  -- averaged over last 10 epochs
  tr.new_error = tr.new_error + tr.error_trace[epoch]/10
  if epoch > 10 then
    tr.new_error = tr.new_error - tr.error_trace[epoch - 10]/10
    tr.old_error = tr.old_error + tr.error_trace[epoch - 10]/10
    if tr.new_error < self.early_stop_threshold then return true end
  end
  -- tr_error incresing
  -- averaged over last 20 epochs, 10 and 10
  if epoch > 20 then
    tr.old_error = tr.old_error - tr.error_trace[epoch - 20]/10
    if tr.new_error > tr.old_error then return true end
  end
  
  -- validation error is increasing
  -- averaged over the last 20 epochs, 10 and 10
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


function MLP:train_once(training_set, validation_set)
--[[ ARGS:
	training_set: mandatory
    training_set[1]: array of input Tensors
    training_set[2]: array of target Tensors, aligned with input
  validation_set: likewise, but it is optional
--]]
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


function MLP:alike()
  return mlp.new(self.W_in:size(2)-1, self.W_out:size(1), {
      neurons = self.W_in:size(1),
      init_range = self.init_range,
      learning_rate = self.learning_rate,
      momentum = self.momentum,
      penalty = self.penalty,
      early_stop_threshold = self.early_stop_threshold,
      act_fun = self.act_fun,
      act_fun_der = self.act_fun_der,
      out_fun = self.out_fun,
      out_fun_der = self.out_fun_der,
      postprocess = self.postprocess,
      max_epochs = self.max_epochs,
      error_metric = self.error_metric
    })
end


function MLP:randomize_weights()
  self.W_in = rand_range(self.W_in:size(), -self.init_range,self.init_range)
  self.W_out = rand_range(self.W_out:size(), -self.init_range,self.init_range)
end


function MLP:train(training_set, validation_set)
  -- runs 5 training sessions, takes average (avg. error too)
  local run_length = 5
  local W_in_best, W_out_best
  local traces_best = { training = { error_trace = { math.huge } } }
  if validation_set then traces_best.validation = { error_trace = { math.huge } } end
  for i = 1,run_length do
    self:randomize_weights()
    local traces = self:train_once(training_set, validation_set)
    --print(traces.training)
    if validation_set and traces.validation.error_trace[#traces.validation.error_trace] < traces_best.validation.error_trace[#traces_best.validation.error_trace] then
      traces_best = traces
      W_in_best = self.W_in
      W_out_best = self.W_out
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


function MLP:k_fold_cross_validate(set, k)
--[[ ARGS:
	set: labelled patterns
    set[1]: array of input Tensors
    set[2]: array of target Tensors, aligned with input
  k: number of folds
--]]
  local fold_size = math.ceil(#set[1]/k) -- rounding up means favouring VL
  local access = torch.randperm(#set[1]) -- to shuffle the pattern set
  local val_error_mean = 0 -- mean
  local val_error_sd = 0 -- standard deviation
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
    self:train(tr_fold)
    -- assess and compound measurements
    local val_error = self:mean_error(val_fold)
    val_error_mean = val_error_mean + val_error
    val_error_sd = val_error_sd + (val_error - val_error_mean)^2
  end
  -- finalize measurements
  val_error_mean = val_error_mean / k
  val_error_sd = val_error_sd / (k - 1)
  val_error_sd = math.sqrt(val_error_sd)
  return val_error_mean, val_error_sd
end


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


function MLP:sim(input)
  -- postprocess(out_fun(W_out * act_fun(W_in * input)))
  local out_out, field_out, out_in, field_in = self:sim_raw(input)
  out_out:apply(self.postprocess)
	return out_out, field_out, out_in, field_in
end


return mlp
