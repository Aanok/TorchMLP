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
    pattern = {}
    for i,v in ipairs(fields) do
      -- indices start from 1
      -- index 1 is the separator, so we ignore it
      if (i == 2) then
        -- index 2 is the first valid field: the target class
        targets[patterno] = torch.Tensor({tonumber(v)})
      elseif (3 <= i and i <= 8) then
        -- indices from 3 to 6 are integer features of the input pattern
        pattern[i-2] = tonumber(v)
      end
      -- index 9 is a unique string identifier for the pattern, which we don't need
    end
    input[patterno] = torch.Tensor(pattern)
    patterno = patterno + 1
  end
  
  return input,targets,patterno
end