from __future__ import absolute_import

import functools
import oneflow.python.framework.job_set_util as job_set_util

@oneflow_export("function")
def function(func):
    @functools.wraps(func)
    def Func(*args, **kwargs):
        job_set_util.add_job(func(*args, **kwargs))
    for x in dir(func):
        if x.startwith('__oneflow_'):
            setattr(Func, x, getattr(func, x))
    return Func
