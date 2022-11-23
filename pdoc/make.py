#!/usr/bin/env python3
"""Create HTML documentation from the source code using `pdoc`."""

# Note: Invoke this script from the parent directory as "pdoc/make.py" !

import pathlib
import pdoc
import re

if 0:  # https://github.com/mitmproxy/pdoc/issues/420
  pdoc.doc_types.simplify_annotation.replacements['AAAAAA'] = 'B'
  pdoc.doc_types.simplify_annotation.recompile()

MODULES = ['./resampler']  # One or more module names or file paths.
FAVICON = 'https://github.com/hhoppe/resampler/raw/main/media/spiral_resampled_with_alpha_scaletox64.ico'
FOOTER_TEXT = ''
LOGO = 'https://github.com/hhoppe/resampler/raw/main/media/spiral_resampled_with_alpha.png'
LOGO_LINK = 'https://hhoppe.github.io/resampler/'
TEMPLATE_DIRECTORY = pathlib.Path('./pdoc')
OUTPUT_DIRECTORY = pathlib.Path('./docs')


def main() -> None:
  """Invoke `pdoc` on the module source files."""
  # See https://github.com/mitmproxy/pdoc/blob/main/pdoc/__main__.py
  pdoc.render.configure(
    docformat='restructuredtext',  # 'google' is inferred from __docformat__.
    edit_url_map=None,
    favicon=FAVICON,
    footer_text=FOOTER_TEXT,
    logo=LOGO,
    logo_link=LOGO_LINK,
    math=True,  # Default is False.
    search=True,  # Default is True.
    show_source=True,  # Default is True.
    template_directory=TEMPLATE_DIRECTORY,
  )

  pdoc.pdoc(
    *MODULES,
    output_directory=OUTPUT_DIRECTORY,
  )

  if 1:
    output_file = OUTPUT_DIRECTORY / 'resampler.html'
    text = output_file.read_text()
    # collections.abc.Iterable -> Iterable.
    text = text.replace('<span class="n">collections</span><span class="o">'
                        '.</span><span class="n">abc</span><span class="o">.</span>', '')
    # resampler.Filter, resampler.Boundary, etc. -> Filter, Boundary, etc.
    text = re.sub(r'resampler\.([A-Z][a-z]+)', r'\1', text)
    output_file.write_text(text)


def main2() -> None:
  """Invoke `pdoc` on the imported module."""
  import resampler

  # Put the README.md text inline; otherwise the README.md file is not found in site-packages/resampler/.
  readme_text = pathlib.Path('README.md').read_text()
  resampler.__doc__ = resampler.__doc__.replace('.. include:: ../README.md', readme_text)

  doc = pdoc.doc.Module(resampler)

  # We can override most pdoc doc attributes by just assigning to them.
  if 0:
    doc.get('Foo.A').docstring = 'I am a docstring for Foo.A.'

  pdoc.render.configure(
    docformat='restructuredtext',  # 'google' is inferred from __docformat__.
    favicon=FAVICON,
    footer_text=FOOTER_TEXT,
    logo=LOGO,
    logo_link=LOGO_LINK,
    math=True,    # Default is False.
    search=True,  # Default is True.
    show_source=True,  # Default is True.
    template_directory=TEMPLATE_DIRECTORY,
  )

  # This creates just resampler.html, not index.html and search.js which are also in ./docs.
  text = pdoc.render.html_module(module=doc, all_modules={'resampler': doc})
  pathlib.Path('resampler.html').write_text(text)


if __name__ == '__main__':
  main()
