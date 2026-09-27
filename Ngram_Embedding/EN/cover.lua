-- Render the article title and contents using the reference notes style.
local has_cover = false
function Header(block)
  if FORMAT:match('latex') and block.level == 1 and not has_cover then
    has_cover = true
    local title = pandoc.write(pandoc.Pandoc({pandoc.Plain(block.content)}), 'latex')
    return pandoc.RawBlock('latex', '\\NotesCover{' .. title .. '}')
  end
end

-- Center standalone images without captions or paragraph indentation.
function Para(block)
  if FORMAT:match('latex') and #block.content == 1 and block.content[1].t == 'Image' then
    return {
      pandoc.RawBlock('latex', '\\begin{center}'),
      pandoc.Plain(block.content),
      pandoc.RawBlock('latex', '\\end{center}')
    }
  end
end

-- Keep a displayed formula with its introductory paragraph.
function Blocks(blocks)
  if not FORMAT:match('latex') then return nil end
  local result = pandoc.List()
  for _, block in ipairs(blocks) do
    local display = block.t == 'Para' and #block.content == 1
      and block.content[1].t == 'Math'
      and block.content[1].mathtype == 'DisplayMath'
    local previous = result[#result]
    if display and previous and previous.t == 'Para' then
      result:remove(#result)
      result:insert(pandoc.RawBlock('latex', '\\begin{NotesFormula}'))
      result:insert(previous)
      result:insert(block)
      result:insert(pandoc.RawBlock('latex', '\\end{NotesFormula}'))
    else
      result:insert(block)
    end
  end
  return result
end


-- Place the overview directly below the contents, at the regular text width.
function Div(block)
  if FORMAT:match('latex') and block.identifier == 'embedding-overview' then
    block = block:walk({
      Str = function(el)
        if el.text == '✓' then
          return pandoc.Math('InlineMath', '\\checkmark')
        end

      end,
      Table = function(tbl)
        tbl.colspecs = {
          {pandoc.AlignCenter, 0.160}, {pandoc.AlignCenter, 0.280},
          {pandoc.AlignCenter, 0.080}, {pandoc.AlignCenter, 0.160},
          {pandoc.AlignCenter, 0.090}, {pandoc.AlignCenter, 0.090},
          {pandoc.AlignCenter, 0.140}
        }
        return tbl
      end
    })
    local blocks = {}
    for _, content in ipairs(block.content) do
      if content.t == 'Table' then
        table.insert(blocks, pandoc.RawBlock('latex', '\\begingroup\\OverviewTableStyle'))
        table.insert(blocks, content)
        table.insert(blocks, pandoc.RawBlock('latex', '\\endgroup'))
      else
        table.insert(blocks, content)
      end
    end
    table.insert(blocks, pandoc.RawBlock('latex', '\\NotesCoverEnd'))
    return blocks
  end
end
