-- A collection of utility functions

local csv = require("csv")
local torch = require("torch")


-- This will parse a Monk data file
function parseMonk(sourceFile)
--[[
  ARGS:
  sourceFile: file with Monk data to parse
  RETURNS:
  (input,targets,patterno)
  input: Lua table with Torch tensors of input patterns
  targets: Lua table with Torch tensors of target patterns
  patterno: integer with total count of patterns in the dataset
--]]
  if (sourceFile == nil) then error "Missing file argument" end

  -- separator isn't actually a comma but blankspace
  local f = csv.open(sourceFile, {separator = ' '})
  
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