#!/usr/bin/env python3
"""Create HTML documentation from the source code using `pdoc`."""
# Note: Invoke this from the parent directory as "python3 pdoc_files/make.py".

import pathlib
import re

import pdoc

if 0:  # https://github.com/mitmproxy/pdoc/issues/420
  pdoc.doc_types.simplify_annotation.replacements['AAAAAA'] = 'B'
  pdoc.doc_types.simplify_annotation.recompile()

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
    text = output_file.read_text()

    # collections.abc.* -> * (Iterable, Mapping, Callable, etc.).
    text = text.replace(
        (
            '<span class="n">collections</span><span class="o">'
            '.</span><span class="n">abc</span><span class="o">.</span>'
        ),
        '',
    )

    # typing.* -> * (e.g. typing.Any).
    text = text.replace(
        '<span class="n">typing</span><span class="o">.</span>',
        '',
    )

    # Deal with, e.g., "_ArrayLike = typing.TypeVar('_ArrayLike')".
    for src, dst in [
        ('ArrayLike', None),
        ('DTypeLike', None),
        ('NDArray', 'np.ndarray'),
        ('DType', 'np.dtype'),
        ('TensorflowTensor', None),
        ('TorchTensor', None),
        ('JaxArray', None),
        ('Array', None),
        ('AnyArray', None),
    ]:
      dst = dst or src
      text = re.sub(
          rf'(?s)<span class="o">~</span>\s*<span class="n">_{src}<',
          rf'<span class="n">{dst}<',
          text,
      )
      text = text.replace(f'~_{src}', dst)

    # resampler.Filter, resampler.Boundary, etc. -> Filter, Boundary, etc.
    text = re.sub(r'resampler\.([A-Z][a-z]+)', r'\1', text)

    output_file.write_text(text)


def main2() -> None:
  """Invoke `pdoc` on the imported module."""
  import resampler

  # Put the README.md text inline; otherwise the README.md file is not found
  # in site-packages/resampler/.
  readme_text = pathlib.Path('README.md').read_text()
  resampler.__doc__ = resampler.__doc__.replace('.. include:: ../README.md', readme_text)

  doc = pdoc.doc.Module(resampler)

  # We can override most pdoc doc attributes by just assigning to them.
  if 0:
    doc.get('Foo.A').docstring = 'I am a docstring for Foo.A.'

  pdoc.render.configure(
      docformat='google',
      favicon=FAVICON,
      footer_text=FOOTER_TEXT,
      logo=LOGO,
      logo_link=LOGO_LINK,
      math=True,
      search=True,
      show_source=True,
      template_directory=TEMPLATE_DIRECTORY,
  )

  # This creates just resampler.html, not index.html and search.js which are
  # also in ./docs.
  text = pdoc.render.html_module(module=doc, all_modules={'resampler': doc})
  pathlib.Path('resampler.html').write_text(text)


if __name__ == '__main__':
  main()

# Local Variables:
# fill-column: 80
# End:
