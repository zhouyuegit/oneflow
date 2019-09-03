from __future__ import absolute_import

import functools
import oneflow.python.framework.job_set_util as job_set_util
from oneflow.python.framework.session import Session
from oneflow.python.oneflow_export import oneflow_export

_job_name2job_func = {}
_sess = Session()

@oneflow_export("function")
def function(job_func):
    @functools.wraps(job_func)
    def Func(*args, **kwargs):
        return _sess.run(job_func)
    for x in dir(job_func):
        if x.startswith('__oneflow_'):
            setattr(Func, x, getattr(job_func, x))
    if job_func.__name__ not in _job_name2job_func:
        job_set_util.add_job(job_func)
        _job_name2job_func[job_func.__name__] = job_func
    return Func
