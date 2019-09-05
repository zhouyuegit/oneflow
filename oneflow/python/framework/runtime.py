from __future__ import absolute_import

import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.runtime_context as runtime_ctx
import oneflow.python.framework.job_instance as job_instance
import oneflow.python.framework.inter_user_job_util as inter_user_job_util
import numpy as np

def GetMachineRuntimeEnv():
    return MasterRuntimeEnv()

class MasterRuntimeEnv(object):
    def __init__(self):
        self._env_inited = False
        self._global_inited = False

    @property
    def env_inited(self):
        return self._env_inited

    @property
    def global_inited(self):
        return self._global_inited

    def InitRuntimeEnv(self):
        assert self._env_inited == False
        runtime_ctx.InitInterUserJobInfo(c_api_util.GetInterUserJobInfo())
        self._env_inited = True

    def InitGlobalOneflow(self):
        assert self._global_inited == False
        c_api_util.InitGlobalOneflow()
        self._global_inited = True


def LaunchJob(job_func, *arg):
    job_name = job_func.__name__
    assert len(arg) == len(job_func.__oneflow_input_blob_defs__)
    for i in range(len(arg)):
        assert isinstance(arg[i], np.ndarray)
        input_op_name = job_func.__oneflow_input_blob_defs__[i].op_name
        inter_user_job_util.AsyncPush(input_op_name, inter_user_job_util.MakePushCallback(arg[i]))
    c_api_util.LaunchJob(job_instance.MakeUserJobInstance(job_name))
    return job_func.__oneflow_output_remote_blobs__
