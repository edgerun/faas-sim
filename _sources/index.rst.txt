.. only:: html

    .. figure:: _static/logo.png
        :height: 150px
        :align: center

        A framework for trace-driven simulation of serverless Function-as-a-Service platforms.

*faas-sim* Overview
===================

*faas-sim* is a trace-driven simulation framework to simulate container-based function-as-a-service platforms.
It can be used to develop, and evaluate the performance of operational strategies for such systems, like scheduling, autoscaling, load balancing, and others.

Architecture
------------

*faas-sim* is based on the `SimPy <https://simpy.readthedocs.io>`_ discrete-event simulation framework.
It uses `Ether <https://github.com/edgerun/ether>`_ as network simulation layer, and as source for cluster configurations and network topologies.
By default, it uses the `Skippy <https://github.com/edgerun/skippy-core>`_ scheduling system for serverless resource scheduling,
but schedulers, autoscalers, and load-balancers can be plugged in by the user.
*faas-sim* is trace-driven, and uses profiling data from real workloads and devices to simulate function execution.
It comes pre-packaged with traces from several common computing devices and representative cluster workloads.
The following figure shows a high-level overview:

.. figure:: figures/architecture-overview.png
    :align: center


Background
----------

*faas-sim* was developed at the `Distributed Systems Group <https://dsg.tuwien.ac.at>`_ at `TU Wien <https://tuwien.at>`_ as part of a larger research effort surrounding serverless edge computing systems.
