Examples
========

This package contains examples that demonstrate various ways to work with *faas-sim*.
Each example has a runnable `main.py`.
You can run them from the faas-sim root directory with:

    make venv
    source .venv/bin/activate
    python -m examples.<example>.main

Replace `<example>` with one of the example packages:

* `basic`: Shows how to set up and run a simulation.
* `custom_scheduler`: builds on `basic` to demonstrate how to replace the default scheduler with a custom implementation
