from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("onerec.FieldConf")
class FieldConf(object):
    def __init__(self, key, shape, dtype):
        assert isinstance(key, str)
        assert isinstance(shape, (list, tuple))

        self.key = key
        self.shape = shape
        self.dtype = dtype

    def to_proto(self):
        field_conf = op_conf_util.DecodeOneRecFieldConf()
        field_conf.key = self.key
        field_conf.output_shape.dim.extend(self.shape)
        field_conf.output_data_type = self.dtype
        return field_conf


@oneflow_export("onerec.decode_onerec")
def decode_onerec(files, fields,
                  batch_size=1,
                  name=None):
    if name is None:
        name = id_util.UniqueStr("DecodeOneRec_")

    lbis = []

    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name

    op_conf.decode_onerec_conf.file.extend(files)
    op_conf.decode_onerec_conf.batch_size = batch_size
    for field in fields:
        op_conf.decode_onerec_conf.field.extend([field.to_proto()])
        op_conf.decode_onerec_conf.out.extend(field.key)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = name
        lbi.blob_name = field.key
        lbis.append(lbi)

    compile_context.CurJobAddOp(op_conf)
    return tuple(map(lambda x: remote_blob_util.RemoteBlob(x), lbis))
