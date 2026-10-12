local NBSP = "\u{00A0}"
local SINGLE = {}
for letter in ("ksvzouai"):gmatch(".") do
  SINGLE[letter] = true
  SINGLE[letter:upper()] = true
end

local function is_number(text)
  local plain = text:gsub(NBSP, "")
  return plain:match("^%d$") ~= nil or plain:match("^%d[%d,.]*%d$") ~= nil
end

local function is_single(text)
  local bare = text:gsub("^%(", ""):gsub("^„", "")
  return SINGLE[bare] == true
end

local function joins(element)
  return element.t == "Str" and (is_single(element.text) or is_number(element.text))
end

local function breaks(element)
  return element.t == "Space" or element.t == "SoftBreak"
end

function Inlines(inlines)
  for index = #inlines - 1, 1, -1 do
    if breaks(inlines[index + 1]) and joins(inlines[index]) then
      inlines[index + 1] = pandoc.Str(NBSP)
    end
  end
  return inlines
end
