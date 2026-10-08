-- One Markdown source: landscape overview followed by portrait VAE details.
function RawInline(el)
  if el.format == 'html' and el.text:match('<br%s*/?>') then
    return pandoc.LineBreak()
  end
end

function Str(el)
  if el.text:find('→') then
    local parts = pandoc.List()
    local first = true
    for piece in (el.text .. '→'):gmatch('(.-)→') do
      if not first then parts:insert(pandoc.Math('InlineMath', '\\rightarrow')) end
      if piece ~= '' then parts:insert(pandoc.Str(piece)) end
      first = false
    end
    return parts
  end
end

function Header(el)
  if el.level == 1 then
    local title = pandoc.write(pandoc.Pandoc({pandoc.Plain(el.content)}), 'latex')
    return pandoc.RawBlock('latex', '\\setlength{\\parskip}{3pt}{\\sffamily\\bfseries\\color{notesblue}\\fontsize{23}{27}\\selectfont ' .. title .. '\\par}\\vspace{1mm}{\\color{notesblue}\\rule{\\linewidth}{0.8pt}}\\vspace{1mm}')
  end
  if el.identifier == 'wan21' then
    el.level = 1
    return {
      pandoc.RawBlock('latex', '\\NotesWanLayout'),
      el
    }
  end
  if el.identifier == 'minimax-h3' then
    el.level = 1
    return {pandoc.RawBlock('latex', '\\NotesHThreeLayout'), el}
  end
  if el.identifier == 'ltx25' then
    el.level = 1
    return {pandoc.RawBlock('latex', '\\NotesLTXLayout'), el}
  end
  if el.identifier == 'flux3' then
    el.level = 1
    return {pandoc.RawBlock('latex', '\\NotesFluxLayout'), el}
  end
  if el.identifier == 'ltx25-conv-decoder' or el.identifier == 'ltx25-pixel-decode'
      or el.identifier == 'ltx25-sampling' or el.identifier == 'flux3-sampling' then
    el.level = el.level - 1
    return {pandoc.RawBlock('latex', '\\clearpage'), el}
  end
  if el.level >= 2 then el.level = el.level - 1 end
  if el.level >= 2 then
    local heading_space = el.identifier:match('^flux3%-') and 6 or 8
    return {pandoc.RawBlock('latex', '\\Needspace{' .. heading_space .. '\\baselineskip}'), el}
  end
  return el
end

function Table(el)
  local overview = #el.colspecs == 7
  local attention = #el.colspecs == 6
  local compact_attention = #el.colspecs == 4
  local heading = pandoc.utils.stringify(el.head)
  local shapes = heading:find('顺序') ~= nil or heading:find('Step') ~= nil
  local widths = overview and {0.12,0.35,0.085,0.065,0.085,0.085,0.21}
    or (attention and {0.22,0.12,0.15,0.14,0.23,0.14})
    or (compact_attention and {0.20,0.25,0.20,0.35})
    or (shapes and {0.17,0.55,0.28} or {0.20,0.50,0.30})
  local style = overview and '\\begingroup\\NotesTableStyle\\fontsize{8.25pt}{9.5pt}\\selectfont\\renewcommand{\\arraystretch}{1.0}\\setlength{\\extrarowheight}{1pt}'
    or '\\begingroup\\NotesTableStyle\\fontsize{10pt}{12pt}\\selectfont\\renewcommand{\\arraystretch}{1.02}\\setlength{\\extrarowheight}{1pt}'
  for i,w in ipairs(widths) do el.colspecs[i] = {pandoc.AlignLeft,w} end
  return {
    pandoc.RawBlock('latex', style),
    el,
    pandoc.RawBlock('latex', '\\endgroup')
  }
end
