local compactmath = pandoc.RawBlock('latex', '\\fontsize{10.5}{13.5}\\selectfont\\setlength{\\parskip}{2pt}\\setlength{\\abovedisplayskip}{4pt plus 1pt minus 1pt}\\setlength{\\belowdisplayskip}{4pt plus 1pt minus 1pt}\\setlength{\\belowdisplayshortskip}{4pt plus 1pt minus 1pt}')

local derivationlayout = pandoc.RawBlock('latex', '\\begingroup\\fontsize{10.5}{13.2}\\selectfont\\setlength{\\parskip}{2pt}\\setlength{\\abovedisplayskip}{4pt plus 1pt minus 1pt}\\setlength{\\belowdisplayskip}{4pt plus 1pt minus 1pt}\\setlength{\\abovedisplayshortskip}{2pt}\\setlength{\\belowdisplayshortskip}{4pt plus 1pt minus 1pt}')

function Header(el)
  local title = pandoc.utils.stringify(el)
  if el.level == 1 then
    return {derivationlayout, el}
  end
  if el.level == 2 then
    if title:match('^7%.') then
      return {el, compactmath}
    end
    if title:match('^6%.') then
      return {pandoc.RawBlock('latex', '\\MuonLandscapeStart\\begingroup\\fontsize{9.5}{12}\\selectfont\\setlength{\\parskip}{3pt}'), el}
    end
    if title:match('^3%.') then
      return {pandoc.RawBlock('latex', '\\clearpage'), el}
    end
    if title:match('^5%.') then
      return {pandoc.RawBlock('latex', '\\clearpage\\endgroup'), el}
    end
  end
  if el.level == 3 and title:match('^7%.') then
    if title:match('^7%.3') or title:match('^7%.5') then
      return {pandoc.RawBlock('latex', '\\clearpage'), el, compactmath}
    end
    return {el, compactmath}
  end
end

function Table(el)
  local widths = {0.13, 0.29, 0.24, 0.34}
  for i, width in ipairs(widths) do
    el.colspecs[i] = {pandoc.AlignLeft, width}
  end
  return {
    pandoc.RawBlock('latex', '\\begingroup\\fontsize{9.5}{12}\\selectfont\\renewcommand{\\arraystretch}{1.02}'),
    el,
    pandoc.RawBlock('latex', '\\endgroup\\endgroup\\MuonLandscapeEnd')
  }
end

function Para(el)
  local text = pandoc.utils.stringify(el)
  if text:match('^关于') and text:find('非常非常推荐', 1, true) then
    return {pandoc.RawBlock('latex', '\\begingroup\\small'), el, pandoc.RawBlock('latex', '\\endgroup')}
  end
  if text == '参考资料' then
    return {pandoc.RawBlock('latex', '\\clearpage\\small\\setlength{\\parskip}{4pt}'), el}
  end
end
