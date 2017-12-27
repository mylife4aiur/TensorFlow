# Google I/O
## From Research to Production with TensorFlow Serving
Serving is how you apply a ML model after you've trained it

RPC server

### Goals

* Answer online requests at low latency
* Serve multiple models and versions in a single process
* Achieve efficiency of mini-batching from training ... with requests arriving asynchronously

### TensorFlow Serving
Tree major pillars
* C++ Libraries
    - Standard support for saving and loading of models
    - Generic core platform
* Binaries
    - Best practices out of the box
    - Docker containers, K8s tutorial
* Hosted Service
