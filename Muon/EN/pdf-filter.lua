-- Keep the source table editable in Markdown; use a landscape PDF layout.
function Pandoc(doc)
  if not FORMAT:match('latex') then return nil end
  local blocks = pandoc.List()
  for i, block in ipairs(doc.blocks) do
    if block.t == 'Para' and #block.content == 1
      and block.content[1].t == 'Math'
      and block.content[1].mathtype == 'DisplayMath'
      and i > 1 and doc.blocks[i - 1].t == 'Para' then
      blocks:insert(pandoc.RawBlock('latex', '\\nopagebreak[4]'))
    end
    blocks:insert(block)
  end
  doc.blocks = blocks
  return doc
end

function Table(tbl)
  if not FORMAT:match('latex') then return nil end
  tbl.colspecs = {
    {pandoc.AlignLeft, 0.14},
    {pandoc.AlignLeft, 0.30},
    {pandoc.AlignLeft, 0.28},
    {pandoc.AlignLeft, 0.28}
  }
  return {
    pandoc.RawBlock('latex', '\\ComparisonTableStart'),
    tbl,
    pandoc.RawBlock('latex', '\\ComparisonTableEnd')
  }
end
