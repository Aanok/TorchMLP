package.path = package.path .. ";../lua/?.lua"
local mlp = require("mlp")
require("utils")
require("gnuplot")

local training = {}
training.input, training.targets = parseMonk("../monk/monks-1.train")
training.name = "monks-1.train"

local test = {}
test.input, test.targets = parseMonk("../monk/monks-1.test")
test.name = "monks-1.test"

local ceil = math.ceil --peformance
local m1 = mlp.new(17, 1, {
  neurons = 10,
  learning_rate = 0.1,
  momentum = 0.01,
  penalty = 0.01,
  out_threshold = function(x) return ceil(x - 0.5) end, -- works because logistic sigmoid outputs in (0,1)
  max_epochs = 500})

for k,v in pairs(m1) do
  if k ~= "W_out" and k ~= "W_in" then
    print(k .. ": ")
    print(v)
  end
end

--os.exit()

local error_trace = m1:train(training.input, training.targets)
gnuplot.plot(torch.range(1,error_trace:size()[1]), error_trace)

local function assess(set)
  local correct = 0
  for i,p in ipairs(set.input) do
    if torch.equal(m1:sim(p), set.targets[i]) then correct = correct + 1 end
  end
  print(string.format("Set %s: %g%% prediction accuracy.", set.name, correct/#(set.targets)*100))
end

assess(training)
assess(test)