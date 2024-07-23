from colorama import Fore, Back, Style, just_fix_windows_console
just_fix_windows_console()
import sys
import logging
from typing import Optional, Dict
# Important that colorama is initialized before import sys,
# otherwise there is no color output in the Windows powershell
# See https://stackoverflow.com/a/61069032


class ColoredFormatter(logging.Formatter):
    """Colored log formatter.
    Taken from github-user: joshbode
    Source: https://gist.github.com/joshbode/58fac7ababc700f51e2a9ecdebe563ad
    """

    def __init__(self, *args, colors: Optional[Dict[str, str]] = None, **kwargs) -> None:
        """Initialize the formatter with specified format strings."""

        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record) -> str:
        """Format the specified record as text."""

        record.color = self.colors.get(record.levelname, '')
        record.reset = Style.RESET_ALL

        return super().format(record)


formatter = ColoredFormatter(
    '{color}[{levelname}] {message}{reset}',
    style='{', datefmt='%Y-%m-%d %H:%M:%S',
    colors={
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.RED,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
    }
)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.handlers[:] = []
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def set_debug(DEBUG=False):
    if DEBUG is False:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)


def get_debug():
    if logger.getEffectiveLevel() is logging.WARNING:
        return False
    else:
        return True
