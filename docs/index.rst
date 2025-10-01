LinkMotion Documentation
=========================

**LinkMotion** is a comprehensive Python library for robotics applications, providing tools for robot modeling, joint control, collision detection, visualization, and URDF import/export functionality.

🚀 Quick Start
--------------

Install LinkMotion:

.. code-block:: bash

   pip install linkmotion

Basic robot construction:

.. code-block:: python

   import numpy as np
   from linkmotion import Robot, Link

   robot = Robot()
   base = Link.from_box("base", extents=np.array([1, 1, 0.5]))
   robot.add_link(base)

📚 User Guide
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

📖 API Reference
----------------

.. toctree::
   :maxdepth: 3
   :caption: API Documentation

   api/modules

🔧 Core Components
------------------

**Robot System**
   Build and manipulate robot models with hierarchical structures

**Transform System**
   Handle 3D spatial transformations and coordinate frames

**Movement Control**
   Control robot joint positions and animate movements

**Collision Detection**
   Real-time collision checking using FCL library

**Workspace Analysis**
   Calculate reachable areas and joint limits

**3D Visualization**
   Interactive visualization using K3D

**URDF Support**
   Import/export URDF files with mesh support

🎯 Examples
-----------

* `Robot Construction Examples <https://github.com/hshrg-kw/linkmotion/tree/main/examples/robot>`_
* `Transform Operations <https://github.com/hshrg-kw/linkmotion/tree/main/examples/transform>`_
* `Movement Control <https://github.com/hshrg-kw/linkmotion/tree/main/examples/move>`_
* `Collision Detection <https://github.com/hshrg-kw/linkmotion/tree/main/examples/collision>`_
* `Interactive Notebooks <https://github.com/hshrg-kw/linkmotion/tree/main/notebooks>`_

🔗 Links
--------

* **GitHub Repository**: https://github.com/hshrg-kw/linkmotion
* **Issue Tracker**: https://github.com/hshrg-kw/linkmotion/issues
* **PyPI Package**: https://pypi.org/project/linkmotion/

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`