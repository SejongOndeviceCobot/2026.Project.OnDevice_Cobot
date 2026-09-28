"""Low-volume wall/simulation timing journal independent of the controller.

Timing events cannot issue commands, infer success, or contribute dataset hours.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time


def _finite(value):
    return not isinstance(value,bool) and isinstance(value,(float,int)) and math.isfinite(value)


class TimingJournal:
    def __init__(self, output, *, clock=time.monotonic, origin=None):
        self.clock=clock
        self.origin=clock() if origin is None else origin
        self.folder=Path(output)/"timing"
        self.folder.mkdir(exist_ok=False)
        self.file=self.folder/"events.jsonl"
        self.stream=self.file.open("x", buffering=1)
        self.seq=0; self.next_span=0; self.last_wall=-1.; self.active=[]; self.closed=False
        self.write("start", scope="runtime_start_to_pre_app_shutdown", accepted_dataset_hours=0)

    def write(self, kind, *, simulation_time_s=None, **fields):
        if self.closed: raise RuntimeError("timing journal finalized")
        wall=self.clock()-self.origin
        if not _finite(wall) or wall<0 or wall<self.last_wall:
            raise ValueError("timing wall clock reversed/nonfinite")
        if simulation_time_s is not None and (not _finite(simulation_time_s) or simulation_time_s<0):
            raise ValueError("invalid simulation clock")
        row={"seq":self.seq,"kind":kind,"wall_elapsed_s":wall,"simulation_time_s":simulation_time_s,**fields}
        self.stream.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n")
        self.seq+=1; self.last_wall=wall
        return row

    @contextmanager
    def span(self, name, *, transfer_id=None, simulation_clock=None):
        ident=self.next_span; self.next_span+=1
        sim=lambda: None if simulation_clock is None else simulation_clock()
        self.write("span_start",span_id=ident,name=name,transfer_id=transfer_id,simulation_time_s=sim())
        self.active.append(ident)
        try:
            yield
        except BaseException as error:
            self.write("span_end",span_id=ident,status="raised",error=repr(error),simulation_time_s=sim())
            raise
        else:
            self.write("span_end",span_id=ident,status="returned",simulation_time_s=sim())
        finally:
            if self.active and self.active[-1]==ident:self.active.pop()

    def phase(self, phase, transfer_id, simulation_time_s):
        return self.write("phase",phase=phase,transfer_id=transfer_id,simulation_time_s=simulation_time_s)

    def finish(self, status):
        if self.closed:return
        self.write("end",status=status,unclosed_spans=list(self.active))
        self.stream.flush(); self.stream.close(); self.closed=True
        report=analyze_timing(self.file)
        report.update(source_sha256=hashlib.sha256(self.file.read_bytes()).hexdigest())
        (self.folder/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        return report


class BestEffortTimingJournal:
    """Disable optional timing after its first failure; never mask controller errors."""
    def __init__(self, output, *, clock=time.monotonic, origin=None):
        self.output=Path(output); self.journal=None; self.failure=None
        try:self.journal=TimingJournal(output,clock=clock,origin=origin)
        except Exception as error:self._disable(error)

    def _disable(self,error):
        if self.failure is not None:return
        self.failure={"schema":"depallet.runtime_timing_failure.v1","timing_complete":False,
            "error":repr(error),"controller_outcome_affected":False,"accepted_dataset_hours":0}
        if self.journal is not None:
            try:self.journal.stream.close()
            except Exception:pass
        try:
            folder=self.output/"timing";folder.mkdir(exist_ok=True)
            (folder/"failure.json").write_text(json.dumps(self.failure,indent=2)+"\n")
        except Exception:pass
        print("TIMING_DISABLED",repr(error),flush=True)

    def phase(self,*args):
        if self.failure is not None:return
        try:return self.journal.phase(*args)
        except Exception as error:self._disable(error)

    @contextmanager
    def span(self,name,*,transfer_id=None,simulation_clock=None):
        ident=None
        sim=lambda:None if simulation_clock is None else simulation_clock()
        if self.failure is None:
            try:
                ident=self.journal.next_span;self.journal.next_span+=1
                self.journal.write("span_start",span_id=ident,name=name,transfer_id=transfer_id,simulation_time_s=sim())
                self.journal.active.append(ident)
            except Exception as error:self._disable(error)
        def end(status,error=None):
            if self.failure is not None or ident is None:return
            try:
                self.journal.write("span_end",span_id=ident,status=status,error=error,simulation_time_s=sim())
                self.journal.active.pop()
            except Exception as timing_error:self._disable(timing_error)
        try:yield
        except BaseException as error:
            end("raised",repr(error))
            raise
        else:end("returned")

    def finish(self,status):
        if self.failure is not None:return self.failure
        try:return self.journal.finish(status)
        except Exception as error:
            self._disable(error)
            return self.failure


def analyze_timing(path):
    events=[json.loads(line) for line in Path(path).read_text().splitlines()]
    if not events or events[0].get("kind")!="start":raise ValueError("missing timing start")
    pending={}; finished=[]; stack=[]; previous=-1.; last_sim=-1.; phases=[]
    for seq,row in enumerate(events):
        wall=row.get("wall_elapsed_s"); sim=row.get("simulation_time_s")
        if row.get("seq")!=seq or not _finite(wall) or wall<0 or wall<previous:
            raise ValueError("timing sequence/clock invalid")
        previous=wall
        if sim is not None:
            if not _finite(sim) or sim<last_sim:raise ValueError("simulation clock reversed")
            last_sim=sim
        kind=row.get("kind")
        if kind=="start" and seq!=0:raise ValueError("duplicate start")
        if kind=="end" and seq!=len(events)-1:raise ValueError("events after end")
        if kind=="span_start":
            ident=row["span_id"]
            if ident in pending or any(x["span_id"]==ident for x in finished):raise ValueError("duplicate span")
            pending[ident]=row; stack.append(ident)
        elif kind=="span_end":
            ident=row["span_id"]
            if not stack or stack[-1]!=ident:raise ValueError("unmatched or crossed span")
            start=pending.pop(ident); stack.pop()
            if row.get("status") not in ("returned","raised"):raise ValueError("invalid span status")
            finished.append({"span_id":ident,"name":start["name"],"transfer_id":start.get("transfer_id"),
                "start_wall_s":start["wall_elapsed_s"],"end_wall_s":wall,
                "wall_seconds":wall-start["wall_elapsed_s"],"status":row["status"],
                "simulation_seconds":None if sim is None or start.get("simulation_time_s") is None else sim-start["simulation_time_s"]})
        elif kind=="phase":phases.append(row)
        elif kind not in ("start","end"):raise ValueError("unknown timing kind")
    finalized=events[-1].get("kind")=="end"
    if finalized and events[-1].get("unclosed_spans")!=stack:raise ValueError("end span inventory mismatch")
    complete=finalized and not pending
    by_name={}
    for item in finished:
        bucket=by_name.setdefault(item["name"],{"calls":0,"wall_seconds":0.,"raised_calls":0})
        bucket["calls"]+=1; bucket["wall_seconds"]+=item["wall_seconds"]
        bucket["raised_calls"]+=item["status"]=="raised"
    return {"schema":"depallet.runtime_timing.v1","finalized":finalized,"complete":complete,
        "runtime_status":events[-1].get("status") if finalized else "interrupted",
        "observed_wall_seconds":events[-1]["wall_elapsed_s"]-events[0]["wall_elapsed_s"],
        "clock_scope":"runtime_start_to_pre_app_shutdown; guard preflight and app shutdown excluded",
        "spans":finished,"span_totals":by_name,"phase_events":phases,"open_spans":list(pending),
        "timing_only":True,"physical_success_inferred":False,"accepted_dataset_hours":0,
        "limits":["Nested span totals can overlap; do not sum categories as an episode duration.",
                  "A returned planner call can still report infeasible; returned is not planning success.",
                  "Simulation may be frozen during planning; wall time preserves this cost."]}
