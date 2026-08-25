-- Keep the Markdown title for normal renderers, but begin the PDF with body content.
local title_removed = false
local landscape_table_pending = false
local overview_table_done = false
local landscape_closed = false

function Header(header)
  local header_text = pandoc.utils.stringify(header.content)

  if not title_removed and header.level == 1 then
    title_removed = true
    return {}
  end

  if title_removed and header.level > 1 then
    header.level = header.level - 1
  end

  if header_text:match("Overview of Attention Architectures and Parameters") then
    landscape_table_pending = true
    return {
      pandoc.RawBlock("latex", "\\clearpage\\newgeometry{top=10mm,bottom=10mm,left=8mm,right=8mm,headheight=15pt}\\begin{landscape}\\thispagestyle{empty}\\noindent\\hfill{\\small\\sffamily\\color{attentiongray}Purshow’ Notes}\\par\\vspace{-0.6em}"),
      header,
    }
  end

  if overview_table_done and not landscape_closed then
    landscape_closed = true
    return {
      pandoc.RawBlock("latex", "\\end{landscape}\\restoregeometry\\clearpage"),
      header,
    }
  end

  return header
end

function Table(table)
  if landscape_table_pending and not overview_table_done then
    local widths = {0.10, 0.19, 0.16, 0.24, 0.31}
    for index, spec in ipairs(table.colspecs) do
      table.colspecs[index] = {spec[1], widths[index] or spec[2]}
    end

    landscape_table_pending = false
    overview_table_done = true
  end

  return table
end

-- Suppress paragraph indentation before standalone images.
function Para(para)
  if #para.content == 1 and para.content[1].t == "Image" then
    return pandoc.Para({
      pandoc.RawInline("latex", "\\noindent"),
      para.content[1],
    })
  end

  return para
end
