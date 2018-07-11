-- MLP class
require("torch")

-- @module mlp
local mlp = {}

-- class table
local MLP = {}


-- general architecture: output = W_out * W_in * input
-- both output and input are column vectors

-- utility function to generate a 2D Tensor with uniform random values in the specified interval
local function rand_range(rows, cols, low, up)
  local t = torch.rand(rows, cols)
  t:mul(up-low)
  t:add(low)
  return t
end

local exp = math.exp --performance
-- activation function: logistic function
function sigmoid(x)
  return 1 / (1 + exp(-x))
end

-- derivative of activation function above
function sigmoid_der(x)
  return exp(-x) / (1 + exp(-x))^2
end


-- squared error for MSE purposes
-- arguments are torch Tensors
function squared_error(x, y)
    return torch.norm(x - y)^2
end


-- constructor
function mlp.new(input_size, output_size, options)
--[[ ARGS:
  input_size: feature count of input patterns
  output_size: feature count of output patterns
  options: table containing optional fields specifying parameters and hyperparameters
  all values have defaults to fall back on
    neurons: # of neurons
    init_range: connection weights are initialized uniformly in [-init_range, init_range]
    learning_rate: gradient coefficient for descent
    momentum: momentum coefficient
    penalty: L1 regularization term
    early_stop_threshold: error threshold (at epoch) that triggers early stop
    act_fun: activation function of hidden layer (real -> real)
    act_fun_der: derivative of act_fun (real -> real)
      both act_fun and act_fun_der must be passed or neither will be assigned
    out_fun: activation function of output layer (real -> real)
    out_fun_der: derivative of out_fun (real -> real)
      both out_fun and out_fun_der must be passed or neither will be assigned
    out_threshold: thresholding function to apply to the output (real -> real)
    max_epochs: epoch iteration high end cutoff
    error_metric: error function, accumulated over all outputs during the epoch, averaged at the end of epoch for final aggregate result ((Tensor, Tensor) -> real)
]]--
  local self = {}
  setmetatable(self, { __index = MLP })
  
  local neurons = options.neurons or 10
  local init_range = options.init_range or 0.5
  self.W_in = rand_range(neurons,input_size +1, -init_range,init_range)
  self.W_out = rand_range(output_size,neurons +1, -init_range,init_range)
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
  self.out_threshold = options.out_threshold or function(x) return x end
  self.max_epochs = options.max_epochs or 100
  self.error_metric = options.error_metric or squared_error
  
  return self
end


function MLP:train(input_set, target_set)
--[[ ARGS:
	input_set: array of input Tensors
	target_set: array of target Tensors, aligned with input
--]]
-- TODO: penalization
  -- trace of error metric at each epoch
  local error_trace = {}
  
  -- old deltas for momentum
  local W_out_delta_old = torch.zeros(self.W_out:size())
  local W_in_delta_old = torch.zeros(self.W_in:size())
  
  for _ = 1,self.max_epochs do
    local access = torch.randperm(#input_set)
    local error_acc = 0
    for i = 1,#input_set do
      local out_out, field_out, out_in, field_in = self:sim(input_set[access[i]])
      
      -- hidden-to-output weight matrix
      -- schematically:
      -- delta_w = lr * (target - out_out) * outFunDer(field_out) * out_in
      
      local e_out = torch.add(target_set[access[i]], -1, out_out) -- e_out = target - out_out
			local field_der_out = field_out:apply(self.out_fun_der) -- field_der_out = outFunDer(field_out)
      -- note: apply works in place. i'm adding an alias for readability
      local delta_out = torch.cmul(e_out, field_der_out) -- delta_out = e_out .* field_der_out
      local W_out_delta = torch.ger(delta_out, out_in) -- W_out_delta = delta_out [outer] out_in (lacks lr)
      self.W_out:add(self.learning_rate, W_out_delta) -- W_out = W_out + lr * W_out_delta
      -- apply momentum
      self.W_out:add(self.momentum, W_out_delta_old)
      W_out_delta_old = W_out_delta
      
			-- input-to-hidden weight matrix
      -- schematically:
      -- delta_w = lr * actFunDer(field_in) .* ((delta_out)T * W_out) * input
			
			-- NB torch vectors lack orientation. row * matrix operations are run as matrix:t() * vector
			-- in cmul, the result will always have the same shape as the first argument
			local field_der_in = field_in:apply(self.act_fun_der) -- field_der_in = actFunDer(field_in)
      local e_in = torch.mv(self.W_out:t(), delta_out) -- e_in = delta_out:t() * W_out
      -- NB :t() has no side effects, it just returns a "view"
      e_in = torch.split(e_in, e_in:size()[1]-1)[1] -- we remove the last element of e_in, corresponding to the bias from the output layer
			local delta_in =  torch.cmul(field_der_in, e_in) -- delta_in = field_der_in .* e_in
			local W_in_delta = torch.ger(delta_in, torch.cat(input_set[access[i]], torch.ones(1))) -- W_in_delta = delta_in [outer] input (lacks lr)
			self.W_in:add(self.learning_rate, W_in_delta) -- W_in = W_in + lr * W_in_delta
      -- apply momentum
      self.W_in:add(self.momentum, W_in_delta_old)
      W_in_delta_old = W_in_delta
      
      -- accrue error contribution for the epoch
      error_acc = error_acc + self.error_metric(out_out, target_set[access[i]])
    end
    -- average and record error
    error_acc = error_acc / #input_set
    error_trace[#error_trace + 1] = error_acc
    
    -- early stopping: error < threshold
    if error_acc < self.early_stop_threshold then break end
    
    -- end of epoch, a lot of data was instantiated and lost: do a run of GC for good luck
    --collectgarbage()
  end
  return torch.Tensor(error_trace)
end


function MLP:sim(input)
  -- out_threshold(out_fun(W_out * act_fun(W_in * input)))
  
  -- add fake firing neuron for bias
  local field_in = torch.mv(self.W_in, torch.cat(input, torch.ones(1)))
  -- add fake firing neuron for bias
  local out_in = torch.cat(field_in:clone():apply(self.act_fun), torch.ones(1))
  local field_out = torch.mv(self.W_out, out_in)
  local out_out = field_out:clone():apply(self.out_fun)
  out_out:apply(self.out_threshold)
	return out_out, field_out, out_in, field_in
end


return mlp
