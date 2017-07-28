# nd4j-conv-benchmarks

In each directory there is a `run.sh` file that can be used to benchmark the respective implementation.

For example:

```sh
# ./run.sh mulsum-test [repeat] [height] [width]
./run.sh mulsum-test 24000 50 5

# ./run.sh run-nn-test [repeat] [numFilters] [embeddingH] [embeddingW] [filterW] [padding] [numExternalFeatures] [numHiddenLayerUnits]
./run.sh run-nn-test 500 100 50 20 5 4 4 201
```
