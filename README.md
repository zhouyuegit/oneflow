# Oneflow

### 1.1 Linux 

```
export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
mkdir build && cd build
cmake -DBUILD_THIRD_PARTY=ON .. && make
cmake -DBUILD_THIRD_PARTY=OFF .. && make
```

### 1.2 Glossary 

abbreviation|original
:-----------|:-------
acc         |accumulate
bn          |blob name
comp        |compute
comm        |communication
ctrl        |control
ctx         |context
desc        |descriptor
diff        |differential
dtbn        |data tmp blob name
gph         |graph
ibn         |input blob name
idbn        |input diff blob name
lbi         |logical blob id
lbn         |logical blob name
md          |model
mgr         |manager
mthd        |method
obn         |output blob name
odbn        |output diff blob name
op          |operator
regst       |register
