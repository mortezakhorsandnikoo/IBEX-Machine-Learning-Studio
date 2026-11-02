"""Entry point for the illustrated, printable cheat sheet."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name('build_visual_cheat_sheet.py')), run_name='__main__')
