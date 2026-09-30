#!/usr/bin/env python3
"""Create HTML documentation from the source code using `pdoc`."""

# Note: Invoke this from the parent directory as "python3 pdoc_files/make.py".

import pathlib
import re

import pdoc

MODULE_NAME = 'resampler'
FAVICON = 'https://github.com/hhoppe/resampler/raw/main/pdoc_files/favicon.ico'
FOOTER_TEXT = ''
LOGO = 'https://github.com/hhoppe/resampler/raw/main/pdoc_files/logo.png'
LOGO_LINK = 'https://hhoppe.github.io/resampler/'
TEMPLATE_DIRECTORY = pathlib.Path('./pdoc_files')
OUTPUT_DIRECTORY = pathlib.Path('./pdoc_files/html')
APPLY_POSTPROCESS = True


def main() -> None:
  """Invoke `pdoc` on the module source files."""
  # See https://github.com/mitmproxy/pdoc/blob/main/pdoc/__main__.py
  pdoc.render.configure(
      docformat='google',
      edit_url_map=None,
      favicon=FAVICON,
      footer_text=FOOTER_TEXT,
      logo=LOGO,
      logo_link=LOGO_LINK,
      math=True,
      search=True,
      show_source=True,
      template_directory=TEMPLATE_DIRECTORY,
  )

  pdoc.pdoc(
      f'./{MODULE_NAME}',
      output_directory=OUTPUT_DIRECTORY,
  )

  if APPLY_POSTPROCESS:
    output_file = OUTPUT_DIRECTORY / f'{MODULE_NAME}.html'
    text = output_file.read_text(encoding='utf-8')

    # Deal with, e.g., "_ArrayLike = typing.TypeVar('_ArrayLike')", shown as "~_ArrayLike".
    for src, dst in [
        ('ArrayLike', None),
        ('DTypeLike', None),
        ('NDArray', 'np.ndarray'),
        ('Array', None),
    ]:
      dst = dst or src
      text = re.sub(
          rf'(?s)<span class="o">~</span>\s*<span class="n">_{src}<',
          rf'<span class="n">{dst}<',
          text,
      )

    output_file.write_text(text, encoding='utf-8', newline='\n')


if __name__ == '__main__':
  main()

# Local Variables:
# fill-column: 100
# End:
