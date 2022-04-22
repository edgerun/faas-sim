.. _system:

========
System
========

In the following we describe the inner workings of our *FaasSystem* implementation.
The API of the `FaasSystem` is designed around real life requirements and represents typical operations that can be found in a typical API Gateway (such as in `OpenFaaS`_).
We provide a default implementation of ``FaasSystem``, called ``DefaultFaasSystme`` in ``sim.faas.system.py``.
The following explains the inner workings of our implementations, which components are used and how you can configure the system.

We recall the methods a ``FaasSystem`` has to implement:

.. code-block:: python

    class FaasSystem(abc.ABC):

        def deploy(self, fn: FunctionDeployment): ...

        def invoke(self, request: FunctionRequest): ...

        def remove(self, fn: FunctionDeployment): ...

        def discover(self, fn_name: str) -> List[FunctionReplica]: ...

        def scale_down(self, fn_name: str, remove: int): ...

        def scale_up(self, fn_name: str, replicas: int): ...

        def suspend(self, fn_name: str): ...

        # ... and several other lookup methods

To implement these functions, our system contains the following state:

.. attention::

    This section provides insights into the current implementation of ``FaasSystem``.
    Be aware that this is subject to change and using lookup methods is much safer with respect to updates.


* ``env: Environment``: used to access global configured components (i.e., ``SimMetrics``, ``SimulatorFactory``, ``ClusterContext``)
* ``function_containers: Dict[str, FunctionContainer]``: stores all available function containers from the deployed functions
* ``replicas: Dict[str, List[FunctionReplica]``: collects all FunctionReplicas under the name of the corresponding FunctionDeployment
* ``scheduler_queue: simpy.Store``: contains function replicas that need to be scheduled. ``scale_up`` puts replicas into the queue and ``run_schedule_worker`` polls from it.
* ``load_balancer: LoadBalancer``: called upon ``invoke`` to select replica that handles the invocation. (currently round-robin)
* ``functions_deployments: Dict[str, FunctionDeployment``: stores the deployed functions and gets modified by ``deploy`` and ``remove``.
* ``replica_count: Dict[str, int]``: counts the number of active replica per ``FunctionDeployment``
* ``functions_definitions: Counter``: counts the number of replica per ``FunctionContainer``

.. _OpenFaaS: https://docs.openfaas.com/


.. _Resources:

Resources
=========

Simulation of resources has to be implemented by users due to necessary flexibility regarding the implementation of a ``FunctionSimulator``. In example, the execution of a function can be delayed through queuing.
Therefore, resources are not immediately used and it's the ``FunctionSimulator's`` responsibility to consume them at the right time.

*faas-sim* offers a standardized interface to manage resources which is based on dictionaries.
This allows *faas-sim* to implement common componnents (such as resource monitoring for nodes & functions, as well as an implementation of `Kubernetes' HPA`_)
Resources get added up.

The following code shows an example on consuming resources:

.. code-block:: python

    class CpuConsumingSim(FunctionSimulator):

        def __init__(self, queue: simpy.Resource):
            self.queue = queue

        def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
            token = self.queue.request()
            yield token

            # definition of resources is up to users
            # here we assume that a function call needs 20% cpu usage of the whole call
            env.resource_state.put_resource(replica, 'cpu', 0.2)

            yield env.timeout(1)

            # release resources
            env.resource_state.remove_resource(replica, 'cpu', 0.2)


The ``Environment`` object contains a resource monitor which continuously collects the momentary resource utilization and puts into the ``MetricsServer`` which can be used to query the average usage of a certain resource.

.. _Kubernetes' HPA: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/

