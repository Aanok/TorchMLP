-- A collection of utility functions

local csv = require("csv")
local torch = require("torch")


-- This will parse a Monk data file
function parse_monks(source_file)
--[[
  ARGS:
  source_file: file with Monk data to parse
  RETURNS:
  (input,targets)
  input: Lua table with Torch tensors of input patterns
  targets: Lua table with Torch tensors of target patterns
--]]
  if (source_file == nil) then error "Missing file argument" end

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


-- this will parse an ML-CUP data file
function parse_cup(source_file)
--[[
  ARGS:
  source_file: file with CUP data to parse
  RETURNS:
  (input,targets)
  input: Lua table with Torch tensors of input patterns
  targets: Lua table with Torch tensors of target patterns
--]]
  if (source_file == nil) then error "Missing file argument" end

  local f = csv.open(source_file)
  
  -- parsed return values
  local input = {}
  local targets = {}
  -- metadata for normalization
  local in_min, in_max = math.huge, -math.huge
  local t_min, t_max = math.huge, -math.huge
  --line counter
  local line = 0
  for fields in f:lines() do
    line = line + 1
    -- ignore the header
    if line >= 10 then
      input[#input + 1] = torch.Tensor(10)
      targets[#targets + 1] = torch.Tensor(2)
      -- first field is the separator
      -- second field is the ID, which we don't need (it is encoded by the order)
      for i = 2,11 do
        local value = tonumber(fields[i])
        input[#input][i-1] = value
        if value > in_max then in_max = value end
        if value < in_min then in_min = value end
      end
      for i = 12,13 do
        local value = tonumber(fields[i])
        targets[#targets][i-11] = value
        if value > t_max then t_max = value end
        if value < t_min then t_min = value end
      end
    end
  end
  
  -- normalization
  local range = in_max - in_min
  for _,v in ipairs(input) do
    v:add(-in_min)
    v:div(range)
  end
  range = t_max - t_min
  for _,v in ipairs(targets) do
    v:add(-t_min)
    v:div(range)
  end
  
  return input, targets, t_min, t_max
end