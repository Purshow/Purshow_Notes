-- Render the article title and contents using the reference notes style.
local has_cover = false
function Header(block)
  if FORMAT:match('latex') and block.level == 1 and not has_cover then
    has_cover = true
    local title = pandoc.write(pandoc.Pandoc({pandoc.Plain(block.content)}), 'latex')
    return pandoc.RawBlock('latex', '\\NotesCover{' .. title .. '}\\NotesCoverEnd')
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
