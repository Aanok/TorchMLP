package.path = package.path .. ";../lua/?.lua"
local mlp = require("mlp")
require("utils")
require("gnuplot")

local training = {}
local t_min, t_max
training[1], training[2], t_min, t_max = parse_cup("../cup/ML-CUP17-TR.csv", true)
training.name = "ML-CUP17-TR"

local test, _, _, _ = parse_cup("../cup/ML-CUP17-TS.csv", false)
test.name = "ML-CUP17-TS"

local range = t_max - t_min

local nn = mlp.new(10, 2, {
  neurons = 2,
  learning_rate = 1,
  momentum = 0,
  penalty = 0,
  max_epochs = 150,
  postprocess = function(x) return x:mul(range):add(t_min) end,
  error_metric = euclidean_error,
  postprocess_error = true
  })

local timer = torch.Timer()
local best = { e_mean = math.huge }
local trace = {}
for _,lr in ipairs({32,16,8,4,2,1}) do
  for _,m in ipairs({16,8,4,2,1}) do
    for _,p in ipairs({0.001,0.0001,0.00001}) do
      nn.learning_rate = lr
      nn.momentum = m
      nn.penalty = p
      local t1 = timer:time().real
      local fold_trace = nn:k_fold_cross_validate(training, 5)
      local t2 = timer:time().real
      print("5-fold CV completed in " ..  t2 - t1 .. " seconds. Data:")
      trace[#trace + 1] = {lr = lr, m = m, p = p, e_mean = fold_trace.validation.pp_error_mean, e_sd = fold_trace.validation.pp_error_sd, time = t2 - t1}
      print(trace[#trace])
      gnuplot_cup(fold_trace, nn)
      if fold_trace.validation.pp_error_mean < best.e_mean then
        best = trace[#trace]
        print("Selected as current best hyperparameters.")
      end
    end
  end
end

print("===== BEST HYPERPARAMETERS =====")
print(best)

nn.learning_rate = best.lr
nn.penalty = best.p
nn.momentum = best.m
nn.diminishing_returns_threshold = 0.00001
nn.max_epochs = 200
local traces = nn:train(training)
gnuplot_cup(traces, nn)
local results = {}
for i,v in ipairs(test) do
  results[i] = nn:sim(v)
end
record_cup({ full_name = "Fabrizio Baldini", team_name = "FB-18", date = "August 26th 2018" }, results)
