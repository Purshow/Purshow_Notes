-- Keep the Markdown title; render it as a title-and-contents page in LaTeX.
local has_cover = false
function Header(block)
  if FORMAT:match('latex') and block.level == 1 and not has_cover then
    has_cover = true
    local title = pandoc.write(pandoc.Pandoc({pandoc.Plain(block.content)}), 'latex')
    return pandoc.RawBlock('latex', '\\NotesCover{' .. title .. '}')
  end
end

-- Keep the editable overview in Markdown and place it below the contents.
function Div(block)
  if FORMAT:match('latex') and block.identifier == 'residual-overview' then
    block = block:walk({Table = function(tbl)
      tbl.colspecs = {
        {pandoc.AlignLeft, 0.14}, {pandoc.AlignLeft, 0.23},
        {pandoc.AlignLeft, 0.24}, {pandoc.AlignLeft, 0.20},
        {pandoc.AlignLeft, 0.19}
      }
      return tbl
    end})
    local blocks = {pandoc.RawBlock('latex', '\\begingroup\\OverviewTableStyle')}
    for _, content in ipairs(block.content) do
      table.insert(blocks, content)
    end
    table.insert(blocks, pandoc.RawBlock('latex', '\\endgroup\\NotesCoverEnd'))
    return blocks
  end
end
