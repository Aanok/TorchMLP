-- MLP class
require "torch"
MLP = {
  maxEpochs = 10,
}


-- general architecture: output = W_out * W_hid * W_in * input
-- both output and input are column vectors

-- activation function: a sigmoid
function act(x)
  return 1 / (1 + math.exp(-1))
end

-- derivative of activation function above
function act_der(x)
  return math.exp(-1) / (1 + math.exp(-1))^2
end


-- constructor
function MLP:new (inputSize, neurons, outputSize, learningRate)
  o = {}
  setmetatable(o, self)
  self.__index = self
  o.W_in = torch.Tensor(neurons,input)
  o.W_in:apply(function(x) return math.random() - 0.5 end)
  o.W_hid = torch.Tensor(1,neurons)
  o.W_hid:apply(function(x) return math.random() - 0.5 end)
  o.W_out = torch.Tensor(outputSize,1)
  o.W_out:apply(function(x) return math.random() - 0.5 end)
  o.lr = learningRate
  return o
end

function MLP:train (inputSet, targetSet, epochs)
  maxEpoch = epohcs or self.maxEpochs
  
  for e = 1,maxEpoch do
    for (i,v) in ipairs(inputSet) do
      field_in = torch.cmul(self.Win, v)
      out_in = field_in:clone():apply(act)
      field_hid = torch.cmul(self.Whid, out_in)
      out_hid = field_hid:clone():apply(act)
      field_out = torch.cmul(self.Wout, out_hid)
      out_out = field_out:clone():apply(act)
      
      -- output layer
      -- NOTA BENE I MIGHT HAVE MESSED UP A -1 IN FRONT OF IT!!
      --e_out = torch.add(out_out, -1, targetSet[i]) -- e = d - y
      --delta_out = torch.cmul(e_out, out_out:apply(act_der)) -- delta = e * g'(v)
      --w_out_delta = torch.cmul(delta_out, out_out) -- delta_w = delta * y
      --w_out_delta:mul(self.lr)
      W_out:add(torch.mul(torch.cmul(torch.cmul(torch.add(out_out, -1, targetSet[i]), out_out:clone():apply(act_der)), out_out), lr))
      
      -- hidden layer
      e_hid = ...
      
      -- input layer
      e_in = ...
    end
    -- end of epoch, a lot of data was instantiated and lost: do a run of GC
    collectgarbage()
  end
end

function MLP:sim (input)
  -- W_out * W_hid * W_in * input
  return torch.cmul(self.W_out, torch.cmul(self.W_hid, torch.cmul(self.W_in, input)))
end