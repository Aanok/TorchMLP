-- A collection of utility functions

local csv = require("csv")
local torch = require("torch")
require("gnuplot")

--- Parse MONKS data file
-- Parses provided CSV file, applying one-hot encoding to input.
-- @param source_file Path to file with Monk data to parse.
-- @return input,targets where
-- <li>input: Array-like table with Tensors of input patterns.</li>
-- <li>targets: Array-like table with Tensors of target patterns.</li>
function parse_monks(source_file)
  if (source_file == nil) then error("Missing file argument") end

  -- separator isn't actually a comma but blankspace
  local f = csv.open(source_file, {separator = ' '})
  
  -- one-hot encoding of input features
  local onehot = {}
  -- idx3: 1 of 3
  onehot[3] = { {1,0,0}, {0,1,0}, {0,0,1} }
  -- idx4: 1 of 3
  onehot[4] = { {1,0,0}, {0,1,0}, {0,0,1} }
  -- idx5: 1 of 2
  onehot[5] = { {1,0}, {0,1}}
  -- idx6: 1 of 3
  onehot[6] = { {1,0,0}, {0,1,0}, {0,0,1} }
  -- idx7: 1 of 4
  onehot[7] = { {1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1} }
  -- idx8: 1 of 2
  onehot[8] = { {1,0}, {0,1} }
  
  -- parsed return values
  local input = {}
  local targets = {}
  for fields in f:lines() do
    -- new pattern
    input[#input + 1] = torch.Tensor(17)
    
    -- index 1 is the separator, so we ignore it
    -- index 2 is the first valid field: the target class
    targets[#targets + 1] = torch.Tensor(1):fill(tonumber(fields[2]))
    -- indices from 3 to 8 are integer features of the input pattern
    -- each representing a class feature
    local y = 1
    for i = 3,8 do
      local val = tonumber(fields[i])
      for _,v in ipairs(onehot[i][val]) do
        input[#input][y] = v
        y = y + 1
      end
    end
    -- index 9 is a unique string identifier for the pattern, which we don't need
  end
  
  return input,targets
end


--- Parse ML-CUP data file
-- Parses provided CSV file, applying normalization to input and targets.
-- @param source_file Path to file with ML-CUP data to parse.
-- @param has_labels Boolean to distinguish labelled (TR) and unlabelled (TS) sources.
-- @return input,targets, t_min, t_max where
-- <li>input: Array-like table with Tensors of input patterns.</li>
-- <li>targets: Array-like table with Tensors of target patterns. nil if has_labels == false.</li>
-- <li>t_min: Minimum of target patterns. nil if has_labels == false.</li>
-- <li>t_max: Maximum of target patterns. nil if has_labels == false.</li>
function parse_cup(source_file, has_labels)
  if (source_file == nil) then error("Missing file argument") end

  local f = csv.open(source_file)
  
  -- parsed return values
  local input = {}
  local targets = {}
  -- metadata for normalization
  local in_min = math.huge
  local in_max = -math.huge
  local t_min =  math.huge
  local t_max = -math.huge
  --line counter
  local line = 0
  for fields in f:lines() do
    line = line + 1
    -- ignore the header
    if line >= 10 then
      input[#input + 1] = torch.Tensor(10)
      -- first field is the separator
      -- second field is the ID, which we don't need (it is encoded by the order)
      for i = 2,11 do
        local value = tonumber(fields[i])
        input[#input][i-1] = value
        if value > in_max then in_max = value end
        if value < in_min then in_min = value end
      end
      if has_labels then
        targets[#targets + 1] = torch.Tensor(2)
        for i = 12,13 do
          local value = tonumber(fields[i])
          targets[#targets][i-11] = value
          if value > t_max then t_max = value end
          if value < t_min then t_min = value end
        end
      end
    end
  end
  
  -- normalization
  local range = in_max -in_min
  for _,v in ipairs(input) do
    v:add(-in_min)
    v:div(range)
  end
  if has_labels then
    range = t_max -t_min
    for _,v in ipairs(targets) do
      v:add(-t_min)
      v:div(range)
    end
  end
  
  return input, targets, t_min, t_max
end


--- Write an array of output Tensors to a well-formatted CSV file.
-- The file will be decorated with the appropriate header and named as specs.team_name.._ML-CUP17-TS.csv.
-- @param specs Tabel of specifications for the output. That is:
-- <li> spec.dest_folder Destination folder. Optional. Default: CWD.
-- <li> spec.full_name Full name of CUP participant(s).
-- <li> spec.team_name Team name of CUP participant(s).
-- <li> spec.date Date to attach to results.
-- @param targets Array-like table of output Tensors. Note the ID corresponds to the ordering.
function record_cup(specs, targets)

  local root = specs.dest_folder or "."
  local f = io.open(root .. "/" .. specs.team_name .. "_ML-CUP17-TS.csv", "w+b")
  io.output(f)
  
  -- header
  print("# " .. specs.full_name)
  print('# ' .. specs.team_name)
  print('# ML-CUP17 v1')
  print("# " .. specs.date)
  
  -- patterns; NB identifier is encoded by order!
  for i,v in ipairs(targets) do
    print(i .. "," .. v[1] .. "," .. v[2])
  end
  
  print()
  io.close()
  io.output(io.stdout)
end


--- Generate EPS plots for a MONKS problem.
-- The file name will tell about the employed hyperparameters.
-- @param traces Standard error/accuracy trace table. traces.validation is required.
-- @param nn Neural network that generated the traces. Hyperparameters are read from it.
-- @param num MONKS dataset enumerator (1, 2 or 3)
-- @param is_test Boolean indicating if traces.validation represents a Test set or Validation set.
function gnuplot_monks(traces, nn, num, is_test)
  local params_string = "_lr=" .. nn.learning_rate .. "_p=" .. nn.penalty .. "_m=" .. nn.momentum
  local test_validation = is_test and "Test" or "Validation"
  local filename = "monks-" .. num .. (is_test and "_test" or "_validation") .. "_error" .. params_string .. ".eps"
  gnuplot.epsfigure(filename)
  gnuplot.raw('set title "Monks-' .. num .. '" font ",20"')
  gnuplot.raw('set xlabel "Epochs" font ",20"')
  gnuplot.raw('set key font ",20"')
  gnuplot.raw('set xtics font ",20"')
  gnuplot.raw('set ytics font ",20"')
  gnuplot.plot( {"Training MSE", traces.training.error_trace, 'with lines lw 2 lc "red"'},
                {test_validation .. " MSE", traces.validation.error_trace, 'with lines lw 2 dt "." lc "blue"'})
  gnuplot.plotflush()
  filename = "monks-" .. num .. (is_test and "_test" or "_validation") .. "_accuracy" .. params_string .. ".eps"
  gnuplot.epsfigure(filename)
  gnuplot.movelegend("right", "bottom")
  gnuplot.raw('set title "Monks-' .. num .. '" font ",20"')
  gnuplot.raw('set xlabel "Epochs" font ",20"')
  gnuplot.raw('set key font ",20"')
  gnuplot.raw('set xtics font ",20"')
  gnuplot.raw('set ytics font ",20"')
  gnuplot.plot( {"Training accuracy", traces.training.accuracy_trace, 'with lines lw 2 lc "red"'},
                {test_validation .. " accuracy", traces.validation.accuracy_trace, 'with lines lw 2 dt "." lc "blue"'})
  gnuplot.plotflush()
end


--- Generate EPS plots for ML-CUP.
-- The file name will tell about the employed hyperparameters.
-- @param traces Standard error/accuracy trace table. traces.validation is optional.
-- @param nn Neural network that generated the traces. Hyperparameters are read from it.
function gnuplot_cup(traces, nn)
  local params_string = (traces.validation and "_validation" or "_test") .. "_lr=" .. nn.learning_rate .. "_p=" .. nn.penalty .. "_m=" .. nn.momentum
  gnuplot.epsfigure("cup_error" .. params_string .. ".eps")
  gnuplot.raw('set title "ML-CUP17" font ",20"')
  gnuplot.raw('set xlabel "Epochs" font ",20"')
  gnuplot.raw('set key font ",20"')
  gnuplot.raw('set xtics font ",20"')
  gnuplot.raw('set ytics font ",20"')
  if traces.validation then
    gnuplot.plot({"Training MEE", traces.training.pp_error_trace, 'with lines lw 2 lc "red"'},
                 {"Validation MEE", traces.validation.pp_error_trace, 'with lines lw 2 dt "." lc "blue"'})
  else
    gnuplot.plot({"Training MEE", traces.training.pp_error_trace, 'with lines lw 2 lc "red"'})
  end
  gnuplot.plotflush()
end