"""Bounded full-pallet runtime; invoked only through guarded_run.py."""
from task_runtime import main
if __name__=="__main__":
    raise SystemExit(main())
