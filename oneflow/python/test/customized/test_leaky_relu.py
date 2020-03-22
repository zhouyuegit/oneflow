import oneflow as flow
import numpy as np

flow.config.gpu_device_num(1)

func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.default_data_type(flow.float)

def leaky_relu(x, name, alpha=0.2):
    return flow.user_op_builder(name).Op("leaky_relu").Input("x",[x]).Output("y") \
           .SetAttr("alpha", alpha, "AttrTypeFloat").Build().RemoteBlobList()[0]

@flow.function(func_config)
def LeakyReluJob(x = flow.FixedTensorDef((2, 2))):
    return leaky_relu(x, "my_leaky_relu_op")

index = [-2, -1, 0, 1, 2]
data = []
for i in index: data.append(np.ones((2, 2,), dtype=np.float32) * i)
for x in data:  print(LeakyReluJob(x).get())
