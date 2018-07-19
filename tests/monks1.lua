package.path = package.path .. ";../lua/?.lua"
local mlp = require("mlp")
require("utils")

local training = {}
training[1], training[2] = parse_monks("../monks/monks-1.train")
training.name = "monks-1.train"

local test = {}
test[1], test[2] = parse_monks("../monks/monks-1.test")
test.name = "monks-1.test"

local ceil = math.ceil --peformance
local nn = mlp.new(17, 1, {
  neurons = 4,
  learning_rate = 10,
  momentum = 0,
  penalty = 0,
  postprocess = function(x) return x:add(-0.5):ceil() end, -- works because logistic sigmoid outputs in (0,1)
  max_epochs = 500
  })

local timer = torch.Timer()
local best = { e_mean = math.huge }
local trace = {}
for _,lr in ipairs({15,10,1,0.5,0.1}) do
  for _,m in ipairs({1,0.1,0.01,0.001,0.0001}) do
    for _,p in ipairs({1,0.1,0.01,0.001,0.0001}) do
      if m < lr and p <= m then
        -- the other combinations are fairly nonsensical
        nn.learning_rate = lr
        nn.momentum = m
        nn.penalty = p
        local t1 = timer:time().real
        local fold_trace = nn:k_fold_cross_validate(training, 5)
        local t2 = timer:time().real
        print("5-fold CV completed in " ..  t2 - t1 .. " seconds. Data:")
        trace[#trace + 1] = {lr = lr, m = m, p = p, e_mean = fold_trace.validation.error_mean, e_sd = fold_trace.validation.error_sd, time = t2 - t1}
        print(trace[#trace])
        gnuplot_monks(trace, nn, 1, false)
        if fold_trace.validation.error_mean < best.e_mean then
          best = trace[#trace]
          print("Selected as current best hyperparameters.")
        end
      end
    end
  end
end

print("===== BEST HYPERPARAMETERS =====")
print(best)

nn.learning_rate = best.lr
nn.penalty = best.p
nn.momentum = best.m

local traces = nn:train(training, test)

gnuplot_monks(traces, nn, 1, true)
print("TR final error: " .. traces.training.error_trace[#traces.training.error_trace])
print("TS final error: " .. traces.validation.error_trace[#traces.validation.error_trace])
print("TR final accuracy: " .. traces.training.accuracy_trace[#traces.training.accuracy_trace])
print("TS final accuracy: " .. traces.validation.accuracy_trace[#traces.validation.accuracy_trace])