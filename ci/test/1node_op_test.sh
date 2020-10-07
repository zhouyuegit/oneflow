#!/bin/bash
set -xe

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
test_tmp_dir=${ONEFLOW_TEST_TMP_DIR:-"/test_tmp_dir"}


rm -rf $test_tmp_dir
mkdir -p $test_tmp_dir
cp -r $src_dir/oneflow/python/test $test_tmp_dir
cd $test_tmp_dir

cd test/ops/

export ONEFLOW_TEST_DEVICE_NUM=1
python3 -m unittest --failfast --verbose

export ONEFLOW_TEST_DEVICE_NUM=2
python3 -m unittest --failfast --verbose

export ONEFLOW_TEST_DEVICE_NUM=4
python3 -m unittest --failfast --verbose
