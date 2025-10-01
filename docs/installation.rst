Installation
============

Prerequisites
-------------

LinkMotion requires **Python 3.12 or higher**.

Basic Installation
------------------

Install LinkMotion from PyPI:

.. code-block:: bash

   pip install linkmotion

Development Installation
------------------------

For development or to get the latest features:

.. code-block:: bash

   git clone https://github.com/hshrg-kw/linkmotion.git
   cd linkmotion
   uv sync

With Jupyter Support
--------------------

For interactive notebooks and visualization:

.. code-block:: bash

   pip install linkmotion[jup]

This includes:

* ``jupyter`` - Interactive notebooks
* ``k3d`` - 3D visualization in Jupyter
* ``plotly`` - Interactive plotting

Dependencies
------------

**Core Dependencies:**

* ``joblib`` - Parallel computing utilities
* ``manifold3d`` - 3D geometry processing
* ``python-fcl`` - Collision detection library
* ``scipy`` - Scientific computing
* ``shapely`` - Geometric operations
* ``trimesh`` - 3D mesh processing
* ``numpy`` - Numerical computing

**Optional Dependencies:**

* ``jupyter`` - Interactive notebooks (with ``[jup]`` extra)
* ``k3d`` - 3D visualization (with ``[jup]`` extra)
* ``plotly`` - Interactive plotting (with ``[jup]`` extra)

Verification
------------

Verify your installation:

.. code-block:: python

   import linkmotion
   from linkmotion import Robot, Link
   import numpy as np

   # Create a simple robot
   robot = Robot()
   base = Link.from_box("base", extents=np.array([1, 1, 0.5]))
   robot.add_link(base)
   print(f"Robot created with {len(robot.links)} link(s)")
