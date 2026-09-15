"""hwocr - proof-of-concept OCR for Hebrew handwriting fonts.

The package recognises Hebrew text that was rendered with handwriting-style
fonts (e.g. "Gveret Levin" from AlefAlefAlef).  It is built around a small
CRNN + CTC recogniser trained on synthetic renderings of the fonts that are
registered in ``fonts/``.

Keep this module free of heavy imports (torch, cv2) so that the command line
interface starts up quickly.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
