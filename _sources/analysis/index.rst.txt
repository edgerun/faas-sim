.. _analysis:

========
Analysis
========

Analysis of simulation results is done by extracting pandas DataFrames upon completion (``sim.env.metrics.extract_dataframe(<name>)``).
The environment of the simulation contains a ``Metrics`` object used throughout the simulation to log events.
Those events describe different aspects of a FaaS platform (``FaasSystem``), such as scheduling process, data flow or invocations.

Default logs
============

The default implementation of a FaasSystem (``DefaultFaasSystem``) logs events of the following processes and can be extracted as dataframe with the associated names:

* Allocation (``'allocation'``)
* Invocations (``'invocations'``)
* Scaling (``'scale'``)
* Scheduling (``'schedule'``)
* Function Replica Deployment (``'replica_deployment'``)
* Function Deployments (``'function_deployments'``)
* Function Deployment (``'function_deployment'``)
* Function Deployment lifecycle (``'function_deployment_lifecycle'``)
* Functions (``'functions'``)
* Flow (``'flow'``)
* Network (``'network'``)
* Node utilization (``'node_utilization'``)
* Function utilization (``'function_utilization'``)
* Function Execution Times (``'fets'``)

.. hint::

	We provide a basic example in ``examples/analysis/main.py`` and details for each dataframe can be found in the documentation to the corresponding aspect.

Logging
=======

During the simulation various aspects of the system are being logged.
Logging happens mainly from the core implementation but some aspects are left to the users.
Details about those aspects follow later.

``Metrics`` defines a general log function and different out-of-the-box log functions that target specific events in the lifecycle of a FaaS platform.


The ``Metrics`` constructor takes a ``RuntimeLogger`` object as initialisation parameter.
The *logger* stores all records and can be configured by providing a ``Clock`` object, which determines the time of each log event.

.. hint::
	Checkout ``sim.logging`` for different implementations!
