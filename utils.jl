function hfun_bar(vname)
  val = Meta.parse(vname[1])
  return round(sqrt(val), digits=2)
end

# {{cssver}} — content hash of the CSS files, appended as ?v=... to the
# stylesheet links so browsers refetch only when the CSS actually changes.
function hfun_cssver(args...)
  s = ""
  for f in ("_css/franklin.css", "_css/pure.css", "_css/side-menu.css")
    isfile(f) && (s *= read(f, String))
  end
  return string(abs(hash(s)) % 100000000)
end

# {{cards path1 path2 ...}} — render a grid of entry cards (thumbnail + title +
# blurb) from each page's :title, :rss and :thumbnail page-variables.
function hfun_cards(params)
  io = IOBuffer()
  write(io, "<div class=\"card-grid\">")
  for path in params
    p = strip(path, '/')
    title = pagevar(p, :title); title = title === nothing ? p : title
    desc  = pagevar(p, :rss)
    desc === nothing && (desc = pagevar(p, :rss_description))
    desc === nothing && (desc = pagevar(p, :descr))
    desc === nothing && (desc = "")
    thumb = pagevar(p, :thumbnail)
    url = "/" * p * "/"
    img = thumb === nothing ? "" : "<img src=\"$thumb\" alt=\"\" loading=\"lazy\">"
    write(io, "<a class=\"card\" href=\"$url\">" *
              "<span class=\"card-bar\">~/$p</span>" *
              "<span class=\"card-thumb\">$img</span>" *
              "<span class=\"card-body\">" *
              "<span class=\"card-title\">$title</span>" *
              "<span class=\"card-desc\">$desc</span>" *
              "</span></a>")
  end
  write(io, "</div>")
  return String(take!(io))
end

function hfun_m1fill(vname)
  var = vname[1]
  return pagevar("index", var)
end

function lx_baz(com, _)
  # keep this first line
  brace_content = Franklin.content(com.braces[1]) # input string
  # do whatever you want here
  return uppercase(brace_content)
end
