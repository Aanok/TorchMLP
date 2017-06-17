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

  input = {}
  targets = {}
  -- separator isn't actually a comma but blankspace
  local f = csv.open(sourceFile, {separator = ' '})
  
  patterno = 0
  for fields in f:lines() do
    -- new pattern
    patterno = patterno + 1
    input[patterno] = {}
    
    -- index 1 is the separator, so we ignore it
    -- index 2 is the first valid field: the target class
    targets[patterno] = torch.Tensor({tonumber(fields[2])})
    -- indices from 3 to 8 are integer features of the input pattern
    for i = 3,8 do
      input[patterno][i-2] = tonumber(fields[i])
    end
    -- index 9 is a unique string identifier for the pattern, which we don't need
    
    input[patterno] = torch.Tensor(input[patterno])
  end
  
  return input,targets,patterno
end