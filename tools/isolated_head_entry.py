"""Test-only normal script-directory import semantics with -I -S isolation."""
import pathlib,runpy,sys
script=pathlib.Path(sys.argv[1]).resolve();sys.argv=sys.argv[1:];sys.path.insert(0,str(script.parent));runpy.run_path(str(script),run_name='__main__')
