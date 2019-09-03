from __future__ import absolute_import

import functools
import oneflow.python.framework.job_set_util as job_set_util
from oneflow.python.framework.session import Session

_job_name2job_func = {}

@oneflow_export("function")
def function(job_func):
    @functools.wraps(job_func)
    def Func(*args, **kwargs):
        Session().run(job_func(*args, **kwargs))
    for x in dir(job_func):
        if x.startwith('__oneflow_'):
            setattr(Func, x, getattr(func, x))
    if job_func.__name__ not in _job_name2job_func:
        job_set_util.add_job(func(*args, **kwargs))
        _job_name2job_func[func.__name__] = job_func
    return Func
