-- MLP class
require "torch"
MLP = {
  maxEpochs = 10,
}


-- general architecture: output = W_out * W_in * input
-- both output and input are column vectors

-- activation function: a sigmoid
function actFun(x)
  return 1 / (1 + math.exp(-1))
end

-- derivative of activation function above
function actFunDer(x)
  return math.exp(-1) / (1 + math.exp(-1))^2
end


-- utility function to generate a matrix with uniform random values in the specified interval
function torch.randRange(rows, cols, low, up)
  t = torch.rand(rows, cols)
  t:mul(up-low)
  t:add(low)
  return t
end

-- constructor
function MLP:new(inputSize, neurons, outputSize, learningRate)
  o = {}
  setmetatable(o, self)
  self.__index = self
  o.W_in = torch.randRange(neurons,input, -0.5,0.5)
  o.W_out = torch.randRange(outputSize,neurons, -0.5,0.5)
  o.lr = learningRate
  return o
end


function MLP:train(inputSet, targetSet, epochs)
--[[ ARGS:
	inputSet: array of input column Tensors
	targetSet: array of target column Tensors, aligned with input
	epochs: maximum epochs of trainings, optional, defaults to MLP attribute
--]]
  maxEpoch = epochs or self.maxEpochs
  
  for e = 1,maxEpoch do
    for (i,v) in ipairs(inputSet) do
      field_in = torch.mv(self.W_in, v)
      out_in = field_in:clone():apply(actFun)
      field_out = torch.mv(self.W_out, out_in)
      out_out = field_out:clone():apply(actFun)
      
      -- hidden-to-output weight matrix
      -- schematically:
      -- delta_w = lr * (target - out_out) * actFunDer(field_out) * out_in
      
      -- NOTA BENE I MIGHT HAVE MESSED UP A -1 IN FRONT OF IT!!
      e_out = torch.add(out_out, -1, targetSet[i]) -- e_out = out_out - target
			field_der_out = field_out:clone():apply(actFunDer) -- field_der_out = actFunDer(field_out)
      delta_out = torch.cmul(e_out, field_der_out) -- delta_out = e_out .* field_der_out
      W_out_delta = torch.mm(delta_out, out_in:t()) -- W_out_delta = delta_out [outer] out_in (lacks lr)
      W_out:add(self.lr, W_out_delta) -- W_out = W_out + lr * W_out_delta
      
			-- input-to-hidden weight matrix
      -- schematically:
      -- delta_w = - lr * g'(field_in)' * sum(delta_out)
			
			-- NB torch.cmul doesn't care about the vectors' shape (row or col) as long as they have the same length
			-- the result will always have the same shape as the first argument
			-- so delta_in is a column vector
			field_der_in = field_in:clone():apply(actFunDer) -- field_der_in = actFunDer(field_in)
			e_in = torch.mm(delta_out:t(), W_out) -- e_in = delta_out:t() * W_out
			delta_in =  torch.cmul(field_der_in, e_in) -- delta_in = field_der_in .* e_in
			W_in_delta = torch.mm(delta_in, v:t()) -- W_in_delta = delta_in [outer] v (lacks lr)
			W_in:add(-self.lr, W_in_delta) -- W_in = W_in - lr * W_in_delta
    end
    -- end of epoch, a lot of data was instantiated and lost: do a run of GC for good luck
    collectgarbage()
  end
end


function MLP:sim (input)
  -- W_out * W_in * input
	return torch.mv(W_out, torch.mv(W_in, input))
end