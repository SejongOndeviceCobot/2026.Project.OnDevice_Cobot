"""Optional accumulated wall-time measurements; never control or dataset evidence."""
from collections import defaultdict
from contextlib import contextmanager
import json
import math
import time
from pathlib import Path


class RuntimeCosts:
    """Accumulate exclusive and inclusive times without per-step disk writes."""
    def __init__(self, *, clock=time.perf_counter):
        self.clock=clock; self.stack=[]; self.failure=None
        self.totals=defaultdict(lambda:dict(calls=0,raised_calls=0,inclusive_wall_s=0.,exclusive_wall_s=0.))

    def _now(self):
        value=self.clock()
        if not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Invalid cost clock')
        return value

    @contextmanager
    def measure(self,name):
        if self.failure is not None:
            yield;return
        try:
            frame=[self._now(),0.];self.stack.append(frame)
        except Exception as error:
            self.failure=repr(error);yield;return
        raised=False
        try:yield
        except BaseException:
            raised=True;raise
        finally:
            try:
                duration=self._now()-frame[0]
                if duration<0 or duration+1e-9<frame[1]:raise ValueError('Cost clock reversed')
                self.stack.pop()
                if self.stack:self.stack[-1][1]+=duration
                row=self.totals[name];row['calls']+=1;row['raised_calls']+=int(raised)
                row['inclusive_wall_s']+=duration;row['exclusive_wall_s']+=max(0.,duration-frame[1])
            except Exception as error:self.failure=repr(error)

    def call(self,name,fn,*args,**kwargs):
        with self.measure(name):return fn(*args,**kwargs)

    def finish(self,output):
        result=dict(schema='depallet.runtime_costs.v1',complete=self.failure is None and not self.stack,
            failure=self.failure,components=dict(self.totals),
            scope='Instrumented Python call wall times; CUDA asynchronous work may finish in another call.',
            uninstrumented_work_present=True,physical_success_inferred=False,accepted_dataset_hours=0)
        try:
            folder=Path(output)/'timing';folder.mkdir(exist_ok=True)
            (folder/'components.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        except Exception as error:print('COST_REPORT_FAILED',repr(error),flush=True)
        return result


class PreviewCadence:
    """Throttle only latest.jpg/live-state.json; policy and observer recordings stay native."""
    def __init__(self,hz=30):
        if type(hz) is not int or not 1<=hz<=30:raise ValueError('Preview Hz must be integer 1..30')
        self.hz=hz;self.last_seen=None;self.last_published=None

    def due(self,image_time):
        if isinstance(image_time,bool) or not isinstance(image_time,(int,float)) or not math.isfinite(image_time) or image_time<0:
            raise ValueError('Finite nonnegative image timestamp required')
        if self.last_seen is not None and image_time<self.last_seen:raise ValueError('Preview time reversed')
        self.last_seen=image_time
        if self.last_published is not None and image_time-self.last_published<1/self.hz-2e-9:return False
        self.last_published=image_time;return True
