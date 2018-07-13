package.path = package.path .. ";../lua/?.lua"
local mlp = require("mlp")
require("utils")
require("gnuplot")

local training = {}
local t_min, t_max
training[1], training[2], t_min, t_max = parse_cup("../cup/ML-CUP17-TR.csv", true)
training.name = "ML-CUP17-TR"

local test = {}
test, a1, a2, a3 = parse_cup("../cup/ML-CUP17-TS.csv", false)
test.name = "ML-CUP17-TS"

local range = t_max - t_min

local nn = mlp.new(10, 2, {
  neurons = 1,
  learning_rate = 1,
  momentum = 0,
  penalty = 0,
  max_epochs = 200,
  postprocess = function(x) return x:mul(range):add(t_min) end,
  error_metric = euclidean_error
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
        local e_mean, e_sd = nn:k_fold_cross_validate(training, 5)
        local t2 = timer:time().real
        print("5-fold CV completed in " ..  t2 - t1 .. " seconds. Data:")
        trace[#trace + 1] = {lr = lr, m = m, p = p, e_mean = e_mean, e_sd = e_sd, time = t2 - t1}
        print(trace[#trace])
        if e_mean < best.e_mean then
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
local traces = nn:train(training)
gnuplot.epsfigure("cup_error.eps")
gnuplot.raw('set title "ML-CUP17" font ",20"')
gnuplot.raw('set xlabel "Epochs" font ",20"')
gnuplot.raw('set key font ",20"')
gnuplot.raw('set xtics font ",20"')
gnuplot.raw('set ytics font ",20"')
gnuplot.plot({"Training MEE", traces.training.error_trace, 'with lines lw 1 lc "red"'})
gnuplot.plotflush()

local results = {}
for i,v in ipairs(test) do
  results[i] = nn:sim(v)
end
record_cup({ full_name = "Fabrizio Baldini", team_name = "FB-18", date = "July 13th 2018" }, results)